"""Does the hospital gap actually come from ordering behaviour?

The feature-block ablation established that 109 features containing no measured
value -- only which channel was sampled, how recently, and how often -- reach 97% of
the full matrix's AUROC at hospital A. That invites an explanation for why the model
transfers poorly to hospital B: it is substantially reading what the care team chose
to measure, and charting habits are exactly what differs between health systems.

**That explanation was asserted before it was tested.** It is a plausible story that
fits the numbers, which is the most dangerous kind, so this experiment tests it
directly rather than leaving it as narrative.

The test: fit two models on hospital A, identical in every respect except that one
cannot see ordering behaviour at all. Score both at hospital A and hospital B. If
ordering behaviour is what fails to cross the boundary, the model that never used it
should lose less when it crosses.

Falsifiable, and it can come back either way:

* **Gap shrinks without ordering features** -- the story survives. Ordering behaviour
  buys performance at home and costs it away.
* **Gap unchanged or wider** -- the story is wrong. Whatever fails to transfer is
  somewhere else, and the ablation result, while true, does not explain the drop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..config import CFG, Config
from ..evaluate.metrics import UtilityScorer
from ..features.builder import feature_columns
from .ablation import ORDERING_ONLY_BLOCKS
from .common import (
    ExperimentResult,
    admission_utility_parts,
    assert_contiguous_admissions,
    block_columns,
    fit_booster,
    predict,
)

# A gap difference smaller than this is not worth a mechanism claim in either
# direction: it is within the noise of two independent fits.
NOISE_FLOOR = 0.005


def run(cfg: Config = CFG, seed: int = 0, quiet: bool = False, n_boot: int = 200) -> ExperimentResult:
    def log(msg: str) -> None:
        if not quiet:
            print(f"[mechanism] {msg}", flush=True)

    train = pd.read_parquet(cfg.processed_dir / "train.parquet")
    all_features = feature_columns(train)

    # Membership comes from the builder, as everywhere else, so the "no measured
    # value" claim is structural rather than a guess about column names.
    sample_ids = train["patient_id"].drop_duplicates().head(6)
    raw = pd.read_parquet(cfg.interim_dir / "setA.parquet")
    membership = block_columns(
        raw[raw["patient_id"].isin(set(sample_ids))].reset_index(drop=True), cfg
    )
    ordering = sorted({c for b in ORDERING_ONLY_BLOCKS for c in membership[b]})
    physiology = [c for c in all_features if c not in set(ordering)]
    _assert_subsets(ordering, physiology, all_features)
    log(f"{len(ordering)} ordering features withheld from the second model, "
        f"{len(physiology)} remain")

    arms = {
        "everything": all_features,
        "no ordering behaviour": physiology,
    }
    boosters = {name: fit_booster(train, cols, seed=seed) for name, cols in arms.items()}
    del train, raw

    scored = {}
    for split, label in (("test", "hospital A (test)"), ("external", "hospital B")):
        frame = pd.read_parquet(cfg.processed_dir / f"{split}.parquet")
        assert_contiguous_admissions(frame["patient_id"].to_numpy())
        for name, booster in boosters.items():
            scored[(name, split)] = _score(frame, predict(booster, frame, arms[name]))
            log(f"{name:<22} {label:<18} AUROC {scored[(name, split)]['auroc']:.4f}")
        del frame

    rows, gaps = [], {}
    for name in arms:
        a, b = scored[(name, "test")], scored[(name, "external")]
        gaps[name] = a["auroc"] - b["auroc"]
        rows.append({
            "model": name,
            "n_features": len(arms[name]),
            "auroc_hospital_a": a["auroc"],
            "auroc_hospital_b": b["auroc"],
            "transfer_gap": gaps[name],
            "utility_hospital_a": a["utility"],
            "utility_hospital_b": b["utility"],
        })

    difference = gaps["everything"] - gaps["no ordering behaviour"]
    ci = _difference_ci(scored, seed=seed, n_boot=n_boot)
    log(f"transfer gap {gaps['everything']:.4f} with ordering, "
        f"{gaps['no ordering behaviour']:.4f} without; difference {difference:+.4f} "
        f"(95% CI {ci[0]:+.4f} to {ci[1]:+.4f})")

    table = pd.DataFrame(rows)
    supported = ci[0] > 0 and difference > NOISE_FLOOR

    return ExperimentResult(
        name="mechanism",
        title="Is ordering behaviour what fails to transfer?",
        table=table,
        prose=_prose(rows, gaps, difference, ci, supported, len(ordering)),
        metadata={
            "n_ordering_features": len(ordering),
            "n_physiology_features": len(physiology),
            "transfer_gap_with_ordering": gaps["everything"],
            "transfer_gap_without_ordering": gaps["no ordering behaviour"],
            "gap_difference": difference,
            "gap_difference_ci": ci,
            "mechanism_supported": bool(supported),
            "noise_floor": NOISE_FLOOR,
            "n_boot": n_boot,
        },
    ).validate()


def _score(frame: pd.DataFrame, scores: np.ndarray) -> dict:
    y = frame["SepsisLabel"].to_numpy()
    groups = frame["patient_id"].to_numpy()
    scorer = UtilityScorer(y, groups)
    _, utility = scorer.best_threshold(scores)
    return {
        "auroc": float(roc_auc_score(y, scores)),
        "utility": float(utility),
        "y": y,
        "groups": groups,
        "scores": scores,
    }


def _difference_ci(scored: dict, seed: int, n_boot: int) -> list[float]:
    """Interval on how much the transfer gap changes when ordering is withheld.

    Both models score the identical rows within a split, so the two gaps are
    paired and the bootstrap resamples admissions once per split per replicate,
    then recomputes every AUROC on that same resample. Treating them as
    independent would inflate the interval and hide a real difference.
    """
    rng = np.random.default_rng(seed)
    index = {}
    for split in ("test", "external"):
        groups = scored[("everything", split)]["groups"]
        order = pd.Series(range(len(groups))).groupby(pd.Series(groups), sort=False).apply(list)
        index[split] = [np.asarray(v) for v in order]

    draws = np.empty(n_boot)
    for b in range(n_boot):
        aurocs = {}
        for split in ("test", "external"):
            stays = index[split]
            picks = rng.integers(0, len(stays), size=len(stays))
            rows = np.concatenate([stays[i] for i in picks])
            for name in ("everything", "no ordering behaviour"):
                s = scored[(name, split)]
                y = s["y"][rows]
                aurocs[(name, split)] = (
                    roc_auc_score(y, s["scores"][rows]) if 0 < y.sum() < len(y) else np.nan
                )
        with_ordering = aurocs[("everything", "test")] - aurocs[("everything", "external")]
        without = (
            aurocs[("no ordering behaviour", "test")]
            - aurocs[("no ordering behaviour", "external")]
        )
        draws[b] = with_ordering - without
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return [float(lo), float(hi)]


def _assert_subsets(ordering: list[str], physiology: list[str], all_features: list[str]) -> None:
    """The two arms must differ in exactly the ordering features and nothing else."""
    if set(ordering) & set(physiology):
        raise ValueError("the two feature sets overlap; the arms differ in more than one thing")
    if set(ordering) | set(physiology) != set(all_features):
        missing = set(all_features) - (set(ordering) | set(physiology))
        raise ValueError(f"{len(missing)} features belong to neither arm: {sorted(missing)[:5]}")
    if not ordering:
        raise ValueError("no ordering features identified; the experiment would compare a model to itself")


def _prose(rows, gaps, difference, ci, supported, n_ordering) -> str:
    full, without = rows[0], rows[1]
    verdict = (
        f"**The mechanism story survives its own test.** Withholding ordering "
        f"behaviour shrinks the transfer gap by {difference:.4f} AUROC (95% CI "
        f"{ci[0]:+.4f} to {ci[1]:+.4f}), so a meaningful part of what fails to cross "
        f"the boundary is the model's dependence on what the care team chose to "
        f"measure. The interval is tight against zero, so this is a real effect "
        f"rather than a large one: it establishes the direction, not the size."
        if supported
        else f"**The mechanism story does not survive its own test.** Withholding "
             f"ordering behaviour changes the transfer gap by {difference:+.4f} AUROC "
             f"(95% CI {ci[0]:+.4f} to {ci[1]:+.4f}), which does not establish that "
             f"ordering behaviour is what fails to cross the boundary. The ablation "
             f"result it was built on is still true — those {n_ordering} features do "
             f"reach most of the full matrix's performance at hospital A — but true "
             f"and explanatory are different claims, and only the first one is "
             f"supported here."
    )
    return (
        f"The feature-block ablation showed that {n_ordering} features containing no "
        f"measured value reach most of the full matrix's AUROC. The obvious reading is "
        f"that the model is partly learning clinical suspicion, which would explain why "
        f"it transfers poorly to a hospital with different charting habits. That reading "
        f"is a story that fits the numbers, so it is tested here rather than repeated.\n\n"
        f"Two models, identical but for the {n_ordering} withheld features. With "
        f"everything, AUROC falls {full['auroc_hospital_a']:.4f} to "
        f"{full['auroc_hospital_b']:.4f} across the hospital boundary, a gap of "
        f"{gaps['everything']:.4f}. Without ordering behaviour, it falls "
        f"{without['auroc_hospital_a']:.4f} to {without['auroc_hospital_b']:.4f}, a gap "
        f"of {gaps['no ordering behaviour']:.4f}.\n\n"
        f"{verdict}\n\n"
        f"Either way the price of removing them is visible in the first column: "
        f"hospital A performance drops from {full['auroc_hospital_a']:.4f} to "
        f"{without['auroc_hospital_a']:.4f}. Ordering behaviour is not noise to be "
        f"regularised away — it is real signal about a real thing, which is precisely "
        f"why its portability is worth knowing."
    )
