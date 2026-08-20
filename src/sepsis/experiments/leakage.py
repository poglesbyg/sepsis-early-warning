"""What leakage is actually worth, in AUROC.

The repository already proves it does not leak. This turns that from a claim into
a number, by deliberately committing each mistake and measuring the inflation.

Two distinct mistakes are measured **separately and never pooled**, because they
are different errors with different fixes and conflating them would hide which one
matters:

  A. splitting ICU hours at random instead of splitting admissions
  B. filling gaps from the future (one ``bfill``) so features see ahead

Mistake A is the common one in the wild. It requires no exotic bug, only a call to
``train_test_split`` on a dataframe of rows, which is the default thing to do with
a dataframe of rows.

Both variants hold everything else fixed: identical features where the variant is
about splitting, identical splits where the variant is about features, identical
hyperparameters, identical boosting rounds, identical seed. The only moving part
is the mistake.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CFG, CHANNELS, Config
from ..features import build_features
from ..features.builder import feature_columns
from .common import (
    ExperimentResult,
    admission_split,
    fit_and_score,
    row_split,
)

# Below this, the measured inflation is indistinguishable from fit noise and the
# experiment is not entitled to claim a direction.
NOISE_FLOOR = 0.002


def bidirectional_fill(raw: pd.DataFrame) -> pd.DataFrame:
    """The lookahead mistake, in one line of pandas.

    Production carries the last observation *forward* only. Adding a backward fill
    means a gap at hour 5 is filled with a measurement taken at hour 9, so every
    downstream feature at hour 5 has seen hour 9. This is the single most common
    way a clinical time-series pipeline leaks, and it looks like tidiness.
    """
    out = raw.copy()
    g = out.groupby("patient_id", sort=False)
    out[CHANNELS] = g[CHANNELS].ffill()
    out[CHANNELS] = out.groupby("patient_id", sort=False)[CHANNELS].bfill()
    return out


def run(cfg: Config = CFG, seed: int = 0, quiet: bool = False) -> ExperimentResult:
    """Measure both leakage variants against an honest baseline."""

    def log(msg: str) -> None:
        if not quiet:
            print(f"[leakage] {msg}", flush=True)

    # Hospital A development pool. The competition test split is untouched: these
    # experiments never look at it, so nothing here can contaminate the headline.
    raw = pd.read_parquet(cfg.interim_dir / "setA.parquet")
    dev_ids = set(pd.read_parquet(cfg.processed_dir / "train.parquet", columns=["patient_id"])["patient_id"])
    dev_ids |= set(pd.read_parquet(cfg.processed_dir / "val.parquet", columns=["patient_id"])["patient_id"])
    raw = raw[raw["patient_id"].isin(dev_ids)].reset_index(drop=True)
    n_admissions = int(raw["patient_id"].nunique())
    log(f"development pool: {len(raw):,} hours, {n_admissions:,} admissions")

    honest_features = build_features(raw, cfg)
    cols = feature_columns(honest_features)

    # --- baseline: correct features, correct split ------------------------
    tr, te = admission_split(honest_features, seed=seed)
    baseline = fit_and_score(tr, te, cols, seed=seed)
    log(f"honest baseline: AUROC {baseline['auroc']:.4f}, utility {baseline['utility']:.4f}")

    # --- mistake A: split rows, not admissions ----------------------------
    tr_a, te_a = row_split(honest_features, seed=seed)
    variant_a = fit_and_score(tr_a, te_a, cols, seed=seed)
    shared = len(set(tr_a["patient_id"]) & set(te_a["patient_id"]))
    log(f"row-split: AUROC {variant_a['auroc']:.4f} "
        f"({shared:,} admissions appear on both sides)")

    # --- mistake B: fill gaps from the future -----------------------------
    leaky_features = build_features(bidirectional_fill(raw), cfg)
    tr_b = leaky_features[leaky_features["patient_id"].isin(set(tr["patient_id"]))].reset_index(drop=True)
    te_b = leaky_features[leaky_features["patient_id"].isin(set(te["patient_id"]))].reset_index(drop=True)
    variant_b = fit_and_score(tr_b, te_b, cols, seed=seed)
    log(f"lookahead fill: AUROC {variant_b['auroc']:.4f}")

    rows = [
        {"variant": "honest (admission split, causal features)", **baseline,
         "auroc_inflation": 0.0, "utility_inflation": 0.0},
        {"variant": "A: split ICU hours at random", **variant_a,
         "auroc_inflation": variant_a["auroc"] - baseline["auroc"],
         "utility_inflation": variant_a["utility"] - baseline["utility"]},
        {"variant": "B: fill gaps from the future (bfill)", **variant_b,
         "auroc_inflation": variant_b["auroc"] - baseline["auroc"],
         "utility_inflation": variant_b["utility"] - baseline["utility"]},
    ]
    table = pd.DataFrame(rows)[
        ["variant", "auroc", "auroc_inflation", "utility", "utility_inflation",
         "n_train_rows", "n_test_rows"]
    ]

    _assert_direction(table, shared)

    a_inf = table.loc[1, "auroc_inflation"]
    b_inf = table.loc[2, "auroc_inflation"]
    a_util = table.loc[1, "utility_inflation"]
    b_util = table.loc[2, "utility_inflation"]

    prose = (
        f"Splitting ICU hours at random instead of splitting admissions inflates "
        f"AUROC by **{a_inf:+.4f}** ({baseline['auroc']:.3f} to "
        f"{variant_a['auroc']:.3f}). It is not an exotic bug: it is what "
        f"`train_test_split` does to a dataframe of rows, and it puts "
        f"{shared:,} of {n_admissions:,} admissions on both sides of the split at "
        f"once. Filling gaps from the future with a single `bfill` inflates AUROC "
        f"by **{b_inf:+.4f}**. Everything else is held fixed across the three rows: "
        f"same hyperparameters, same boosting rounds, same seed, and for variant B "
        f"the same admissions in train and test as the baseline.\n\n"
        f"The more useful number is the one underneath. Clinical utility inflates by "
        f"**{a_util:+.4f}** and **{b_util:+.4f}** for the two mistakes, roughly "
        f"{a_util / a_inf:.0f}x and {b_util / b_inf:.0f}x the corresponding AUROC "
        f"movement. Leakage does not merely make the ranking look better; it moves "
        f"the whole risk distribution, so the threshold sweep finds an operating "
        f"point that does not exist on honest data. A project reporting AUROC alone "
        f"would see a fraction of the damage it is actually doing."
    )

    return ExperimentResult(
        name="leakage",
        title="What leakage is worth, measured",
        table=table,
        prose=prose,
        metadata={
            "baseline_auroc": baseline["auroc"],
            "row_split_inflation": float(a_inf),
            "lookahead_inflation": float(b_inf),
            "admissions_on_both_sides": shared,
            "admissions_in_pool": n_admissions,
            "noise_floor": NOISE_FLOOR,
        },
    ).validate()


def _assert_direction(table: pd.DataFrame, shared_admissions: int) -> None:
    """Leakage must help the leaky model. If it does not, the experiment is broken.

    A leaky variant scoring at or below the honest baseline means the mistake was
    not actually committed -- a split that did not overlap, a fill that did not
    fire -- and the resulting table would look entirely plausible while measuring
    nothing. Nothing else in the pipeline would catch that, so it is caught here.
    """
    if shared_admissions <= 0:
        raise ValueError(
            "row-split variant put no admission on both sides of the split; "
            "the leak was never committed and its number is meaningless"
        )

    for _, row in table.iloc[1:].iterrows():
        inflation = row["auroc_inflation"]
        if inflation < -NOISE_FLOOR:
            raise ValueError(
                f"{row['variant']!r} scored {inflation:+.4f} AUROC BELOW the honest "
                f"baseline. Leakage cannot hurt; this variant is not doing what it "
                f"claims and its number must not be published."
            )
        if abs(inflation) <= NOISE_FLOOR:
            raise ValueError(
                f"{row['variant']!r} moved AUROC by only {inflation:+.4f}, within the "
                f"{NOISE_FLOOR} noise floor. Either the mistake was not committed or "
                f"this dataset does not exhibit it; publishing a direction would "
                f"overstate the evidence."
            )
