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
    admission_utility_parts,
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

    # Every evaluation bucket is split again: half to choose a local threshold on,
    # half to score it. Choosing a threshold on the same admissions it is then
    # graded on reports the maximum of a sweep as though it were a result, which
    # is the mistake this repository exists to argue against -- and the first
    # version of this experiment made it.
    rows = []
    for name, ids in evaluation.items():
        rows.append(
            _score_bucket(booster, pool, ids, features, threshold, name, labels, seed, n_boot)
        )
        r = rows[-1]
        log(f"{name}: AUROC {r['auroc']:.4f} | utility frozen {r['utility_frozen']:.4f}, "
            f"local {r['utility_local']:.4f} (prevalence {r['septic_admissions_pct']:.1f}%)")

    table = pd.DataFrame(rows)
    _assert_scored(table)

    home, sicu, unknown = table.iloc[0], table.iloc[1], table.iloc[2]

    # MICU and SICU are disjoint cohorts, so the AUROC difference is unpaired and
    # DeLong does not apply. An independent two-sample cluster bootstrap does.
    auroc_diff_ci = _auroc_difference_ci(
        booster, pool, evaluation["MICU (held out, same unit)"],
        buckets["unit_sicu"], features, seed=seed, n_boot=n_boot,
    )
    log(f"AUROC difference MICU - SICU: {home['auroc'] - sicu['auroc']:+.4f} "
        f"(95% CI {auroc_diff_ci[0]:+.4f} to {auroc_diff_ci[1]:+.4f})")

    prose = _prose(home, sicu, unknown, threshold, len(fit_ids), n_boot, auroc_diff_ci)

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
            "auroc_difference": float(home["auroc"] - sicu["auroc"]),
            "auroc_difference_ci": auroc_diff_ci,
            "utility_home": float(home["utility_frozen"]),
            "utility_sicu": float(sicu["utility_frozen"]),
            "utility_sicu_ci": [float(sicu["utility_frozen_lo"]), float(sicu["utility_frozen_hi"])],
            "utility_sicu_local": float(sicu["utility_local"]),
            "utility_sicu_local_ci": [float(sicu["utility_local_lo"]), float(sicu["utility_local_hi"])],
            "local_threshold_gain": float(sicu["utility_local"] - sicu["utility_frozen"]),
            "local_threshold_gain_ci": [float(sicu["gain_lo"]), float(sicu["gain_hi"])],
            "alerted_admissions_pct_home": float(home["alerted_admissions_pct"]),
            "alerted_admissions_pct_sicu": float(sicu["alerted_admissions_pct"]),
            "mean_score_home": float(home["mean_score"]),
            "mean_score_sicu": float(sicu["mean_score"]),
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


def _score_bucket(booster, pool, ids, features, threshold, name, labels, seed, n_boot) -> dict:
    """One bucket, scored honestly.

    AUROC uses the whole bucket: it is threshold-free, so nothing is spent by
    computing it on everything. The utilities use only the evaluation half,
    because the local threshold is chosen on the other half and a threshold graded
    on its own selection set is a maximum, not a measurement.
    """
    frame = pool[pool["patient_id"].isin(set(ids))]
    y = frame["SepsisLabel"].to_numpy()
    groups = frame["patient_id"].to_numpy()
    scores = predict(booster, frame, features)

    _, lo, hi = cluster_bootstrap_ci(y, scores, groups, n_boot=n_boot, seed=seed)

    tune_ids, eval_ids = train_test_split(
        ids, test_size=0.5, random_state=seed, stratify=labels.reindex(ids).to_numpy()
    )
    _assert_local_threshold_is_held_out(tune_ids, eval_ids, name)
    local = _local_threshold(frame, scores, set(tune_ids))
    evaluated = _utility_on(frame, scores, set(eval_ids), threshold, local, seed, n_boot)

    alerts = scores >= threshold
    alerted = pd.Series(alerts, index=groups).groupby(level=0, sort=False).any()

    return {
        "cohort": name,
        "n_admissions": int(len(ids)),
        "n_eval_admissions": int(len(eval_ids)),
        "n_hours": int(len(frame)),
        "septic_admissions_pct": 100 * float(
            pd.Series(y).groupby(pd.Series(groups), observed=True).max().mean()
        ),
        "auroc": float(roc_auc_score(y, scores)),
        "auroc_lo": float(lo),
        "auroc_hi": float(hi),
        # Score-distribution diagnostics for the same model the utilities come
        # from. Borrowing them from a different model, as an earlier draft of the
        # write-up did, describes a different system.
        "mean_score": float(scores.mean()),
        "alert_hours_pct": 100 * float(alerts.mean()),
        "alerted_admissions_pct": 100 * float(alerted.mean()),
        "local_threshold": local,
        **evaluated,
    }


def _local_threshold(frame: pd.DataFrame, scores: np.ndarray, tune_ids: set) -> float:
    """The threshold a site would pick for itself, chosen on the tuning half only.

    Choosing it requires this unit's own labelled outcomes. That is not a free
    operation and the write-up must not describe it as one.
    """
    mask = frame["patient_id"].isin(tune_ids).to_numpy()
    sub = frame[mask]
    scorer = UtilityScorer(sub["SepsisLabel"].to_numpy(), sub["patient_id"].to_numpy())
    threshold, _ = scorer.best_threshold(scores[mask])
    return float(threshold)


def _utility_on(frame, scores, eval_ids, frozen, local, seed, n_boot) -> dict:
    """Both operating points on the held-out half, with paired intervals.

    The two thresholds are scored on the same admissions, so the bootstrap
    resamples once per replicate and evaluates both. An unpaired interval on the
    difference would be wider for no reason.
    """
    mask = frame["patient_id"].isin(eval_ids).to_numpy()
    sub, sub_scores = frame[mask], scores[mask]
    y, groups = sub["SepsisLabel"].to_numpy(), sub["patient_id"].to_numpy()
    scorer = UtilityScorer(y, groups)

    parts = {
        key: admission_utility_parts(scorer, (sub_scores >= t).astype(float), groups)
        for key, t in (("frozen", frozen), ("local", local))
    }
    point = {k: float(v["num"].sum() / v["den"].sum()) for k, v in parts.items()}

    rng = np.random.default_rng(seed)
    n = len(parts["frozen"])
    draws = {"frozen": np.empty(n_boot), "local": np.empty(n_boot), "gain": np.empty(n_boot)}
    for b in range(n_boot):
        counts = rng.multinomial(n, np.full(n, 1 / n)).astype(float)
        vals = {
            k: float(counts @ v["num"].to_numpy() / (counts @ v["den"].to_numpy()))
            for k, v in parts.items()
        }
        draws["frozen"][b], draws["local"][b] = vals["frozen"], vals["local"]
        draws["gain"][b] = vals["local"] - vals["frozen"]

    ci = {k: np.nanpercentile(v, [2.5, 97.5]) for k, v in draws.items()}
    return {
        "utility_frozen": point["frozen"],
        "utility_frozen_lo": float(ci["frozen"][0]),
        "utility_frozen_hi": float(ci["frozen"][1]),
        "utility_local": point["local"],
        "utility_local_lo": float(ci["local"][0]),
        "utility_local_hi": float(ci["local"][1]),
        "gain_lo": float(ci["gain"][0]),
        "gain_hi": float(ci["gain"][1]),
    }


def _auroc_difference_ci(booster, pool, home_ids, sicu_ids, features, seed, n_boot) -> list[float]:
    """Interval on the AUROC difference between two disjoint cohorts.

    DeLong assumes both scores are computed on the same rows, which is exactly
    what these are not: different patients in different units. Each cohort is
    resampled independently at the admission level and the difference recomputed,
    which is the unpaired analogue.
    """
    cohorts = []
    for ids in (home_ids, sicu_ids):
        frame = pool[pool["patient_id"].isin(set(ids))]
        scores = predict(booster, frame, features)
        cohorts.append(
            pd.DataFrame({
                "patient_id": frame["patient_id"].to_numpy(),
                "y": frame["SepsisLabel"].to_numpy(),
                "score": scores,
            })
        )

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    grouped = [list(c.groupby("patient_id", observed=True, sort=False)) for c in cohorts]
    for b in range(n_boot):
        aurocs = []
        for stays in grouped:
            picks = rng.integers(0, len(stays), size=len(stays))
            sample = pd.concat([stays[i][1] for i in picks], ignore_index=True)
            y = sample["y"].to_numpy()
            aurocs.append(roc_auc_score(y, sample["score"]) if 0 < y.sum() < len(y) else np.nan)
        draws[b] = aurocs[0] - aurocs[1]
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return [float(lo), float(hi)]


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


def _assert_local_threshold_is_held_out(tune_ids, eval_ids, bucket: str) -> None:
    """The half a local threshold is chosen on may not be the half it is scored on.

    Without this, ``utility_local`` is the maximum of a sweep over the same
    admissions it is reported on -- an optimistic number that looks like a
    measured recovery. The first version of this experiment published exactly
    that, and nothing in the pipeline noticed.
    """
    overlap = set(tune_ids) & set(eval_ids)
    if overlap:
        raise ValueError(
            f"{bucket}: {len(overlap)} admissions are in both the threshold-selection "
            f"and evaluation halves; the local threshold would be graded on its own "
            f"selection set"
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


def _prose(home, sicu, unknown, threshold, n_fit, n_boot, auroc_diff_ci) -> str:
    diff = home["auroc"] - sicu["auroc"]
    separable = auroc_diff_ci[0] > 0
    gain = sicu["utility_local"] - sicu["utility_frozen"]
    gain_real = sicu["gain_lo"] > 0

    discrimination = (
        f"the interval on that difference excludes zero, so ranking is measurably "
        f"worse in the surgical unit, though by an amount ({diff:.3f} AUROC) that "
        f"no operating point would notice"
        if separable
        else f"the interval on that difference spans zero, so this experiment does "
             f"not establish a ranking loss across the unit boundary. That is not the "
             f"same as showing there is none: with {sicu['n_admissions']:,} SICU "
             f"admissions it is a statement about what this test could detect"
    )

    return (
        f"Trained on {n_fit:,} medical ICU admissions and evaluated at a threshold "
        f"frozen on held-out MICU patients ({threshold:.3f}), the model scores AUROC "
        f"**{home['auroc']:.4f}** (95% CI {home['auroc_lo']:.4f} to "
        f"{home['auroc_hi']:.4f}) on MICU admissions it has not seen and "
        f"**{sicu['auroc']:.4f}** ({sicu['auroc_lo']:.4f} to {sicu['auroc_hi']:.4f}) "
        f"on surgical ICU admissions. The difference is {diff:+.4f} with a 95% "
        f"interval of {auroc_diff_ci[0]:+.4f} to {auroc_diff_ci[1]:+.4f}, from an "
        f"independent two-sample cluster bootstrap — DeLong does not apply here, "
        f"because the two cohorts are different patients rather than two scores on "
        f"the same rows. So {discrimination}.\n\n"
        f"Clinical utility is a different story, and it is measured on a held-out "
        f"half of each bucket so that no threshold is graded on the admissions it "
        f"was chosen from. At the MICU threshold the model scores "
        f"{home['utility_frozen']:.4f} at home against **{sicu['utility_frozen']:.4f}** "
        f"in the SICU (95% CI {sicu['utility_frozen_lo']:.4f} to "
        f"{sicu['utility_frozen_hi']:.4f}). Choosing a threshold on a separate half "
        f"of the SICU and scoring it here recovers **{gain:+.4f}** to "
        f"{sicu['utility_local']:.4f} (95% CI on the gain: {sicu['gain_lo']:+.4f} to "
        f"{sicu['gain_hi']:+.4f}"
        f"{', which excludes zero' if gain_real else ', which spans zero'}).\n\n"
        f"Two cautions on reading that as a free fix. Choosing a local threshold "
        f"requires this unit's own labelled outcomes — utility is optimised against "
        f"them — so it costs local outcome data, not merely a sweep. And the "
        f"normalised utility score has a cohort-specific, outcome-informed "
        f"denominator, so part of the movement between units is the metric "
        f"responding to a septic rate of {home['septic_admissions_pct']:.1f}% against "
        f"{sicu['septic_admissions_pct']:.1f}%, not the model behaving differently. "
        f"What the score distribution actually does across the boundary is directly "
        f"visible: mean predicted risk {home['mean_score']:.4f} in the MICU against "
        f"{sicu['mean_score']:.4f} in the SICU, alerting on "
        f"{home['alerted_admissions_pct']:.0f}% against "
        f"{sicu['alerted_admissions_pct']:.0f}% of admissions at the same threshold.\n\n"
        f"The third bucket is the one it would be convenient to omit: "
        f"{unknown['n_admissions']:,} admissions where neither unit indicator was "
        f"recorded — more than either named unit — scoring AUROC "
        f"{unknown['auroc']:.4f} at {unknown['septic_admissions_pct']:.1f}% "
        f"prevalence. Dropping them would have made the transfer claim cleaner and "
        f"the cohort unrepresentative of the hospital it came from."
    )
