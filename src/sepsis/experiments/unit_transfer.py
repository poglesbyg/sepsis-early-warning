"""Does a model trained in the medical ICU work in the surgical ICU?

This is the shift experiment this cohort can actually support. PhysioNet/CinC 2019
carries no calendar time, so an admission-ordered split is not buildable here (see
``DECISIONS.md``). What the data does carry is the admitting unit, so the claim
under test is stated in those terms and fixed in advance:

    **A model trained on medical ICU admissions transfers to surgical ICU
    admissions within the same hospital.**

* **Direction.** Train on ``Unit1`` (MICU), test on ``Unit2`` (SICU). One
  direction, chosen before the numbers existed, so the better-looking direction
  cannot be picked afterwards.
* **Missing units.** Admissions with neither indicator are a third bucket,
  reported separately. They are never silently dropped, and in hospital A they are
  the largest group of the three.
* **Additive.** Nothing here replaces the random-split results. Those stay
  published unchanged.
* **What it is not.** Not temporal validation and not external validation. Both
  units sit inside one hospital, so this measures care-pathway shift, and the two
  units differ sharply in septic rate -- which is reported in the table because it
  is a large part of what any drop will be made of.

The three unit indicator columns are removed from the feature matrix. They are
constant within the training cohort and take an unseen value in every evaluation
bucket, so leaving them in would measure the model's reaction to a dead column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from ..config import CFG, Config
from ..evaluate.metrics import UtilityScorer, cluster_bootstrap_ci
from ..features.builder import feature_columns
from .common import (
    ExperimentResult,
    assert_contiguous_admissions,
    fit_booster,
    predict,
)

UNIT_COLUMNS = ["unit_micu", "unit_sicu", "unit_unknown"]

BUCKETS = {
    "unit_micu": "MICU (medical, the training unit)",
    "unit_sicu": "SICU (surgical, the transfer target)",
    "unit_unknown": "unit not recorded",
}


def run(
    cfg: Config = CFG, seed: int = 0, quiet: bool = False, n_boot: int = 200
) -> ExperimentResult:
    def log(msg: str) -> None:
        if not quiet:
            print(f"[unit_transfer] {msg}", flush=True)

    # Hospital A's development pool only. The competition test split is never
    # touched here, so nothing in this experiment can contaminate the headline.
    pool = pd.concat(
        [pd.read_parquet(cfg.processed_dir / f"{s}.parquet") for s in ("train", "val")],
        ignore_index=True,
    ).sort_values(["patient_id", "hour"], ignore_index=True)
    assert_contiguous_admissions(pool["patient_id"].to_numpy())

    features = transfer_features(pool)
    buckets = _bucket_admissions(pool)
    _assert_buckets(buckets, pool)
    log("; ".join(f"{k}={len(v):,} admissions" for k, v in buckets.items()))

    # MICU is split three ways: fit, freeze the threshold, and a held-out slice
    # that answers "how well does this model do at home?". Without that within-unit
    # number the SICU result is unreadable -- a drop could just as easily be the
    # cost of training on a third of the pool.
    labels = pool.groupby("patient_id", observed=True)["SepsisLabel"].max()
    micu = buckets["unit_micu"]
    fit_ids, rest = train_test_split(
        micu, test_size=0.4, random_state=seed, stratify=labels.reindex(micu).to_numpy()
    )
    thr_ids, home_ids = train_test_split(
        rest, test_size=0.625, random_state=seed, stratify=labels.reindex(rest).to_numpy()
    )
    log(f"MICU split: {len(fit_ids):,} fit / {len(thr_ids):,} threshold / {len(home_ids):,} held out")

    train = pool[pool["patient_id"].isin(set(fit_ids))]
    booster = fit_booster(train, features, seed=seed)

    threshold = _freeze_threshold(booster, pool, thr_ids, features)
    log(f"threshold frozen on held-out MICU admissions: {threshold:.4f}")

    evaluation = {
        "MICU (held out, same unit)": home_ids,
        BUCKETS["unit_sicu"]: buckets["unit_sicu"],
        BUCKETS["unit_unknown"]: buckets["unit_unknown"],
    }
    _assert_no_overlap(fit_ids, thr_ids, evaluation)

    rows = []
    for name, ids in evaluation.items():
        rows.append(_score_bucket(booster, pool, ids, features, threshold, name, seed, n_boot))
        log(f"{name}: AUROC {rows[-1]['auroc']:.4f}, utility {rows[-1]['utility_frozen']:.4f} "
            f"(prevalence {rows[-1]['septic_admissions_pct']:.1f}%)")

    table = pd.DataFrame(rows)
    _assert_scored(table)

    home, sicu, unknown = table.iloc[0], table.iloc[1], table.iloc[2]
    prose = _prose(home, sicu, unknown, threshold, len(fit_ids), n_boot)

    return ExperimentResult(
        name="unit_transfer",
        title="MICU to SICU: a shift this cohort can actually support",
        table=table,
        prose=prose,
        metadata={
            "direction": "train Unit1 (MICU), test Unit2 (SICU)",
            "threshold": threshold,
            "n_fit_admissions": int(len(fit_ids)),
            "n_threshold_admissions": int(len(thr_ids)),
            "excluded_features": UNIT_COLUMNS,
            "auroc_home": float(home["auroc"]),
            "auroc_sicu": float(sicu["auroc"]),
            "auroc_drop": float(home["auroc"] - sicu["auroc"]),
            "utility_home": float(home["utility_frozen"]),
            "utility_sicu": float(sicu["utility_frozen"]),
            "utility_sicu_retuned": float(sicu["utility_retuned"]),
            "prevalence_home": float(home["septic_admissions_pct"]),
            "prevalence_sicu": float(sicu["septic_admissions_pct"]),
            "n_boot": n_boot,
        },
    ).validate()


# --------------------------------------------------------------------------
def transfer_features(frame: pd.DataFrame) -> list[str]:
    """The feature matrix minus the unit indicators. See the module docstring."""
    return [c for c in feature_columns(frame) if c not in UNIT_COLUMNS]


def _bucket_admissions(pool: pd.DataFrame) -> dict[str, np.ndarray]:
    """One bucket per admitting unit, keyed on the admission's first hour."""
    first = pool.groupby("patient_id", observed=True)[UNIT_COLUMNS].first()
    return {col: first.index[first[col] == 1].to_numpy() for col in UNIT_COLUMNS}


def _freeze_threshold(booster, pool, ids, features) -> float:
    frame = pool[pool["patient_id"].isin(set(ids))]
    scorer = UtilityScorer(frame["SepsisLabel"].to_numpy(), frame["patient_id"].to_numpy())
    threshold, _ = scorer.best_threshold(predict(booster, frame, features))
    return threshold


def _score_bucket(booster, pool, ids, features, threshold, name, seed, n_boot) -> dict:
    frame = pool[pool["patient_id"].isin(set(ids))]
    y = frame["SepsisLabel"].to_numpy()
    groups = frame["patient_id"].to_numpy()
    scores = predict(booster, frame, features)

    scorer = UtilityScorer(y, groups)
    _, lo, hi = cluster_bootstrap_ci(y, scores, groups, n_boot=n_boot, seed=seed)
    _, retuned = scorer.best_threshold(scores)

    return {
        "cohort": name,
        "n_admissions": int(len(ids)),
        "n_hours": int(len(frame)),
        "septic_admissions_pct": 100 * float(
            pd.Series(y).groupby(pd.Series(groups), observed=True).max().mean()
        ),
        "auroc": float(roc_auc_score(y, scores)),
        "auroc_lo": float(lo),
        "auroc_hi": float(hi),
        "utility_frozen": scorer.score((scores >= threshold).astype(np.float64)),
        "utility_retuned": float(retuned),
    }


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------
def _assert_buckets(buckets: dict[str, np.ndarray], pool: pd.DataFrame) -> None:
    """The three unit buckets must partition the pool: no admission in two, none lost."""
    sizes = {k: len(v) for k, v in buckets.items()}
    empty = [k for k, n in sizes.items() if n == 0]
    if empty:
        raise ValueError(f"unit buckets are empty and cannot be compared: {empty}")

    seen: set = set()
    for name, ids in buckets.items():
        overlap = seen & set(ids)
        if overlap:
            raise ValueError(
                f"{len(overlap)} admissions are in more than one unit bucket "
                f"(e.g. {sorted(overlap)[:3]}); the unit indicators are not exclusive "
                f"and every number below would double-count them"
            )
        seen |= set(ids)

    total = int(pool["patient_id"].nunique())
    if len(seen) != total:
        raise ValueError(
            f"unit buckets cover {len(seen)} of {total} admissions; "
            f"{total - len(seen)} would be silently dropped"
        )


def _assert_no_overlap(fit_ids, thr_ids, evaluation: dict[str, np.ndarray]) -> None:
    """Nothing the model was fitted or tuned on may appear in an evaluation bucket."""
    spent = set(fit_ids) | set(thr_ids)
    for name, ids in evaluation.items():
        shared = spent & set(ids)
        if shared:
            raise ValueError(
                f"{len(shared)} admissions appear both in training/tuning and in "
                f"the {name!r} bucket; that bucket's score is not held out"
            )


def _assert_scored(table: pd.DataFrame) -> None:
    """A bucket with no septic admission cannot be scored, and a below-chance AUROC
    on a real bucket means the cohort was assembled wrongly rather than that the
    model inverted."""
    dead = table.loc[table["septic_admissions_pct"] == 0, "cohort"].tolist()
    if dead:
        raise ValueError(f"buckets contain no septic admission and cannot be scored: {dead}")
    inverted = table.loc[table["auroc"] < 0.5, "cohort"].tolist()
    if inverted:
        raise ValueError(
            f"AUROC below chance on {inverted}; a transfer this bad is a cohort "
            f"assembly error, not a shift result"
        )


def _prose(home, sicu, unknown, threshold, n_fit, n_boot) -> str:
    auroc_drop = home["auroc"] - sicu["auroc"]
    overlap = sicu["auroc_hi"] >= home["auroc_lo"] and home["auroc_hi"] >= sicu["auroc_lo"]
    discrimination = (
        "the two intervals overlap, so discrimination is not measurably worse across "
        "the unit boundary"
        if overlap
        else "the intervals are disjoint, so discrimination is measurably worse across "
             "the unit boundary"
    )
    threshold_cost = sicu["utility_retuned"] - sicu["utility_frozen"]

    return (
        f"Trained on {n_fit:,} medical ICU admissions and evaluated at a threshold "
        f"frozen on held-out MICU patients ({threshold:.3f}), the model scores AUROC "
        f"**{home['auroc']:.4f}** (95% CI {home['auroc_lo']:.4f} to "
        f"{home['auroc_hi']:.4f}) on MICU admissions it has not seen, and "
        f"**{sicu['auroc']:.4f}** ({sicu['auroc_lo']:.4f} to {sicu['auroc_hi']:.4f}) "
        f"on surgical ICU admissions: a drop of {auroc_drop:+.4f}, and "
        f"{discrimination}.\n\n"
        f"Clinical utility tells a harsher story: {home['utility_frozen']:.4f} at home "
        f"against **{sicu['utility_frozen']:.4f}** in the SICU. Most of that is not "
        f"the model getting worse at ranking patients. The septic rate is "
        f"{home['septic_admissions_pct']:.1f}% in the MICU and "
        f"{sicu['septic_admissions_pct']:.1f}% in the SICU, and an alert threshold "
        f"tuned where sepsis is common fires far too often where it is rare. Retuning "
        f"the threshold on the SICU alone recovers {threshold_cost:+.4f} to "
        f"{sicu['utility_retuned']:.4f} — that recovery is the price of shipping one "
        f"operating point across a boundary the units do not share, and it is "
        f"available in production by re-picking a threshold, without retraining.\n\n"
        f"The third bucket is the one it would be convenient to omit: "
        f"{unknown['n_admissions']:,} admissions where neither unit indicator was "
        f"recorded — more than either named unit — scoring AUROC "
        f"{unknown['auroc']:.4f} at {unknown['septic_admissions_pct']:.1f}% "
        f"prevalence. Dropping them would have made the transfer claim cleaner and "
        f"the cohort unrepresentative of the hospital it came from."
    )
