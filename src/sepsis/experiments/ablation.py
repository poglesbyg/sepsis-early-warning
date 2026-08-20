"""What each feature block actually buys.

345 features derived from 40 channels invite one obvious question: how much of
that is doing work? This answers it two ways, because the two answers disagree and
the disagreement is the interesting part.

**Leave-one-out** drops a block and refits. It measures what a block contributes
*on top of everything else*, so a block that duplicates information available
elsewhere scores near zero even if it is individually strong. That is the number
that tells you what is safe to delete.

**Solo** fits on a block alone. It measures standalone power, ignoring redundancy.
That is the number that tells you what actually carries signal.

A block that is strong solo and worthless leave-one-out is redundant, not useless.
Reporting only leave-one-out would quietly recommend deleting it, and reporting
only solo would quietly recommend keeping everything.
"""

from __future__ import annotations

import pandas as pd

from ..config import CFG, Config
from ..features import build_features
from ..features.builder import FEATURE_BLOCKS, feature_columns
from .common import ExperimentResult, admission_split, assert_partition, block_columns, fit_and_score

# Blocks derived purely from measurement behaviour: which channel was sampled,
# how recently, and how often. None of them contains a measured value.
ORDERING_ONLY_BLOCKS = ("recency", "intensity", "missing")

BLOCK_DESCRIPTIONS = {
    "locf": "carried-forward channel values",
    "recency": "hours since each channel was last measured",
    "intensity": "how often each channel has been sampled",
    "deviation": "each channel against the patient's own running mean",
    "rolling": "6h and 24h level, spread and trend",
    "missing": "panel-level ordering activity",
    "clinical": "SIRS, qSOFA, partial SOFA, shock index",
}


def run(cfg: Config = CFG, seed: int = 0, quiet: bool = False) -> ExperimentResult:
    def log(msg: str) -> None:
        if not quiet:
            print(f"[ablation] {msg}", flush=True)

    raw = pd.read_parquet(cfg.interim_dir / "setA.parquet")
    dev_ids = set(pd.read_parquet(cfg.processed_dir / "train.parquet", columns=["patient_id"])["patient_id"])
    dev_ids |= set(pd.read_parquet(cfg.processed_dir / "val.parquet", columns=["patient_id"])["patient_id"])
    raw = raw[raw["patient_id"].isin(dev_ids)].reset_index(drop=True)

    features = build_features(raw, cfg)
    all_cols = feature_columns(features)

    # Membership comes from the builder, not from a regex over column names, and
    # the partition is asserted before a single number is computed. A block set
    # that overlaps or leaves orphans makes every row below meaningless.
    membership = block_columns(raw[raw["patient_id"].isin(list(dev_ids)[:6])].reset_index(drop=True), cfg)
    assert_partition(membership, all_cols)
    log(f"blocks partition {len(all_cols)} features: "
        + ", ".join(f"{k}={len(v)}" for k, v in membership.items()))

    train, test = admission_split(features, seed=seed)
    full = fit_and_score(train, test, all_cols, seed=seed)
    log(f"all features: AUROC {full['auroc']:.4f}, utility {full['utility']:.4f}")

    rows = []
    for block in FEATURE_BLOCKS:
        held = membership[block]
        without = [c for c in all_cols if c not in set(held)]

        loo = fit_and_score(train, test, without, seed=seed)
        solo = fit_and_score(train, test, held, seed=seed)
        log(f"{block:<10} drop -> {loo['auroc']:.4f}   solo -> {solo['auroc']:.4f}")

        rows.append(
            {
                "block": block,
                "what it is": BLOCK_DESCRIPTIONS[block],
                "n_features": len(held),
                "auroc_without": loo["auroc"],
                "loo_cost": full["auroc"] - loo["auroc"],
                "auroc_solo": solo["auroc"],
                "utility_without": loo["utility"],
                "utility_loo_cost": full["utility"] - loo["utility"],
            }
        )

    # Blocks that contain no channel VALUE at all -- only what was measured and
    # when. If these carry most of the signal on their own, missingness is not a
    # nuisance to impute away, and that is worth one extra fit to establish.
    ordering_only = [c for b in ORDERING_ONLY_BLOCKS for c in membership[b]]
    ordering = fit_and_score(train, test, ordering_only, seed=seed)
    log(f"{'ordering-only':<10} solo -> {ordering['auroc']:.4f} "
        f"({len(ordering_only)} features, zero measured values)")

    table = pd.DataFrame(rows).sort_values("loo_cost", ascending=False, ignore_index=True)
    _assert_blocks_are_real(table, len(all_cols), membership)

    top = table.iloc[0]
    strongest_solo = table.sort_values("auroc_solo", ascending=False).iloc[0]
    ordering_share = ordering["auroc"] / full["auroc"]

    prose = (
        f"With all {len(all_cols)} features the model scores AUROC "
        f"{full['auroc']:.4f}. **No single block costs more than "
        f"{top['loo_cost']:.4f} AUROC to remove** — the most expensive is "
        f"`{top['block']}` ({top['what it is']}). Yet every block scores between "
        f"{table['auroc_solo'].min():.3f} and {table['auroc_solo'].max():.3f} on its "
        f"own. That combination has one explanation: the matrix is enormously "
        f"redundant, and the same physiology is reachable through several different "
        f"encodings of it.\n\n"
        f"The sharpest number here is the ordering-only row. Using **{len(ordering_only)} "
        f"features that contain no measured value whatsoever** — only which channel "
        f"was sampled, how recently, and how often — the model reaches AUROC "
        f"**{ordering['auroc']:.4f}**, or {ordering_share:.0%} of what the full "
        f"matrix achieves. Nothing about the patient's physiology is in that subset. "
        f"It is a record of what the care team chose to look at, and it is nearly as "
        f"predictive as the measurements themselves.\n\n"
        f"Two readings follow. The optimistic one: missingness is signal, and the "
        f"recency and intensity blocks earn their place rather than padding the "
        f"matrix. The uncomfortable one: a model this dependent on ordering "
        f"behaviour is partly learning clinical suspicion rather than physiology, so "
        f"it would degrade wherever ordering habits differ — which is exactly what "
        f"the drop from hospital A to hospital B looks like."
    )

    return ExperimentResult(
        name="ablation",
        title="What each feature block buys",
        table=table,
        prose=prose,
        metadata={
            "full_auroc": full["auroc"],
            "full_utility": full["utility"],
            "n_features": len(all_cols),
            "block_sizes": {k: len(v) for k, v in membership.items()},
            "most_costly_to_drop": top["block"],
            "max_loo_cost": float(top["loo_cost"]),
            "strongest_solo": strongest_solo["block"],
            "ordering_only_auroc": ordering["auroc"],
            "ordering_only_features": len(ordering_only),
            "ordering_only_share_of_full": float(ordering_share),
        },
    ).validate()


def _assert_blocks_are_real(
    table: pd.DataFrame, n_features: int, membership: dict[str, list[str]]
) -> None:
    """Guard the ways this table can be plausible and wrong."""
    if len(table) != len(FEATURE_BLOCKS):
        raise ValueError(
            f"ablation covered {len(table)} blocks, expected {len(FEATURE_BLOCKS)}"
        )
    if int(table["n_features"].sum()) != n_features:
        raise ValueError(
            f"block sizes sum to {int(table['n_features'].sum())}, not {n_features}; "
            f"the ablation is not measuring a partition"
        )
    empty = [b for b, cols in membership.items() if not cols]
    if empty:
        raise ValueError(f"blocks produced no features and cannot be ablated: {empty}")
    # A solo fit on any real block should beat a coin flip. Below that, the column
    # subset is almost certainly wrong rather than the block being uninformative.
    weak = table.loc[table["auroc_solo"] < 0.5, "block"].tolist()
    if weak:
        raise ValueError(
            f"blocks scored below chance on their own: {weak}. A real feature block "
            f"does not do this; the column selection is wrong."
        )
