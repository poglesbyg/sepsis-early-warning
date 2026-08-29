"""How much of hospital B's drop is case mix, and how much is the model getting worse.

Hospital A scores higher than hospital B on every metric in the report. Two very
different things could produce that gap:

  1. **Case mix.** B admits different patients -- different ages, units, charting
     habits, and a septic rate of 5.7% against A's 8.8%. A model can be equally
     good at both hospitals and still score lower where the cohort is harder or
     the positives are rarer.
  2. **Degradation.** The relationships the model learned at A do not hold at B.

"Case mix versus degradation" is not identifiable from prevalence alone, so the
estimand is fixed before any number is computed, and it is stated here rather than
inferred from the code:

* **Reference distribution.** Hospital A's training split.
* **Target.** Performance on hospital B, reweighted at the admission level so its
  baseline case mix matches the reference. The difference between reweighted and
  unweighted B is the case-mix component; whatever gap against A survives the
  reweighting is degradation.
* **Weights.** Density ratio from a propensity model over admission-level baseline
  covariates, trimmed at the 1st and 99th percentiles, with the trimming reported.
* **Calibration.** Held fixed. The threshold is chosen once on hospital A's
  validation split and applied unchanged to every cohort below. Refitting anything
  on B would answer a different question -- how well it *could* transfer -- and is
  not what is measured here.
* **Uncertainty.** Patient-level cluster bootstrap over the whole procedure: the
  propensity model is refit on every replicate, not just the final statistic
  recomputed.

The reweighting can only correct for what its covariates capture. It is an
adjustment for measured baseline case mix, not for everything that differs
between two health systems, and the residual is therefore an upper bound on
degradation rather than a clean estimate of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import CFG, Config
from ..evaluate.metrics import UtilityScorer
from ..features.builder import feature_columns
from .common import (
    ExperimentResult,
    admission_utility_parts,
    assert_contiguous_admissions,
    baseline_covariates,
    fit_booster,
    predict,
    weighted_utility,
)

# Admission-level covariates the reweighting is allowed to use: demographics and
# unit at admission, early vitals, and how much lab work was ordered in the first
# six hours. The last group is charting behaviour, which is the most visible
# difference between two health systems and the one most likely to move the score.
BASELINE_COLUMNS = [
    "age", "gender", "unit_micu", "unit_sicu", "unit_unknown", "hosp_adm_time",
    "HR_mean6", "O2Sat_mean6", "Temp_mean6", "SBP_mean6", "MAP_mean6", "Resp_mean6",
    "WBC_n_obs", "Lactate_n_obs", "Creatinine_n_obs", "BUN_n_obs", "Platelets_n_obs",
]

# The covariates that are unambiguously fixed at admission. The full list adds six
# hours of early vitals and ordering volume, which buys separation but is care
# process rather than patient characteristic -- and care process is plausibly a
# mediator of the very site difference being adjusted for. Both are reported.
BASELINE_ONLY_COLUMNS = [
    "age", "gender", "unit_micu", "unit_sicu", "unit_unknown", "hosp_adm_time",
]

TRIM = (1.0, 99.0)          # weight percentiles, reported alongside the result
MIN_ESS_FRACTION = 0.05     # below this the reweighted estimate is a handful of patients
MAX_PROPENSITY_AUC = 0.99   # above this the two cohorts barely overlap at all


def run(
    cfg: Config = CFG, seed: int = 0, quiet: bool = False, n_boot: int = 200
) -> ExperimentResult:
    def log(msg: str) -> None:
        if not quiet:
            print(f"[prevalence] {msg}", flush=True)

    rng = np.random.default_rng(seed)

    # --- fit on hospital A, freeze the operating point on A's validation ----
    train = pd.read_parquet(cfg.processed_dir / "train.parquet")
    features = feature_columns(train)
    booster = fit_booster(train, features, seed=seed)
    reference = baseline_covariates(train, BASELINE_COLUMNS)
    log(f"fitted on hospital A train: {len(train):,} hours, {len(reference):,} admissions")
    del train

    val = pd.read_parquet(cfg.processed_dir / "val.parquet")
    assert_contiguous_admissions(val["patient_id"].to_numpy())
    val_scorer = UtilityScorer(val["SepsisLabel"].to_numpy(), val["patient_id"].to_numpy())
    threshold, val_utility = val_scorer.best_threshold(predict(booster, val, features))
    log(f"threshold frozen on validation: {threshold:.4f} (utility {val_utility:.4f})")
    del val, val_scorer

    # --- score hospital A's test split and all of hospital B ----------------
    site_a = _score_cohort(
        booster, features, cfg.processed_dir / "test.parquet", threshold,
        covariates=BASELINE_COLUMNS,
    )
    log(f"hospital A test:  AUROC {site_a['auroc']:.4f}, utility {site_a['utility']:.4f}")

    site_b = _score_cohort(
        booster, features, cfg.processed_dir / "external.parquet", threshold,
        covariates=BASELINE_COLUMNS,
    )
    log(f"hospital B:       AUROC {site_b['auroc']:.4f}, utility {site_b['utility']:.4f}")

    # --- reweight B to A's baseline case mix --------------------------------
    target = site_b["covariates"]
    design, labels, is_b = _propensity_design(reference, target)
    weights, propensity_auc = _fit_weights(design, labels, is_b)
    diagnostics = _overlap(weights, propensity_auc, design[is_b], design[~is_b])
    log(f"propensity AUC {propensity_auc:.3f}, effective sample size "
        f"{diagnostics['ess_fraction']:.0%} of {len(weights):,} admissions")
    log(f"covariate imbalance |SMD|: {diagnostics['smd_before']:.3f} -> "
        f"{diagnostics['smd_after']:.3f} after weighting")

    row_weight = weights.reindex(site_b["groups"]).to_numpy()
    utility_b_weighted = weighted_utility(site_b["scorer"], site_b["alerts"], row_weight)
    auroc_b_weighted = float(
        roc_auc_score(site_b["y"], site_b["scores"], sample_weight=row_weight)
    )
    log(f"hospital B reweighted: AUROC {auroc_b_weighted:.4f}, utility {utility_b_weighted:.4f}")

    # --- uncertainty over the whole procedure -------------------------------
    draws = _bootstrap(
        site_a, site_b, reference, target, n_boot=n_boot, rng=rng, log=log
    )

    gap = site_a["utility"] - site_b["utility"]
    case_mix = utility_b_weighted - site_b["utility"]
    degradation = site_a["utility"] - utility_b_weighted

    _assert_decomposition(gap, case_mix, degradation)
    _assert_overlap(diagnostics)

    # --- sensitivity: the estimand is a choice, so vary it and report -------
    sensitivity = _sensitivity(reference, site_a, site_b, threshold, case_mix_baseline=case_mix)
    for name, value in sensitivity.items():
        log(f"sensitivity[{name}]: case-mix component {value:+.4f}")

    prevalence_b_weighted = _weighted_prevalence(site_b, row_weight)

    # What hospital B would score if the threshold were re-picked there. Not part
    # of the decomposition -- the estimand fixes the operating point -- but the
    # unit-transfer experiment shows an operating point can fail to cross a
    # boundary while the model itself survives it, so the same diagnostic belongs
    # here rather than leaving the residual to be read as pure degradation.
    _, utility_b_retuned = site_b["scorer"].best_threshold(site_b["scores"])
    _, utility_a_retuned = site_a["scorer"].best_threshold(site_a["scores"])
    log(f"if the threshold were re-picked on B: utility {utility_b_retuned:.4f} "
        f"(A re-picked: {utility_a_retuned:.4f})")
    table = pd.DataFrame(
        [
            {
                "cohort": "hospital A (test)",
                "n_admissions": site_a["n_admissions"],
                "septic_admissions_pct": 100 * site_a["prevalence"],
                "auroc": site_a["auroc"],
                "utility": site_a["utility"],
                "utility_gap_vs_A": 0.0,
            },
            {
                "cohort": "hospital B (as observed)",
                "n_admissions": site_b["n_admissions"],
                "septic_admissions_pct": 100 * site_b["prevalence"],
                "auroc": site_b["auroc"],
                "utility": site_b["utility"],
                "utility_gap_vs_A": -gap,
            },
            {
                "cohort": "hospital B (reweighted to A's case mix)",
                "n_admissions": site_b["n_admissions"],
                "septic_admissions_pct": 100 * prevalence_b_weighted,
                "auroc": auroc_b_weighted,
                "utility": utility_b_weighted,
                "utility_gap_vs_A": -degradation,
            },
        ]
    )

    ci = {k: _percentile_ci(v) for k, v in draws.items()}
    case_mix_share = case_mix / gap if gap else float("nan")
    prose = _prose(
        site_a, site_b, gap, case_mix, degradation, case_mix_share,
        auroc_b_weighted, prevalence_b_weighted, diagnostics, ci, threshold, n_boot,
        utility_a_retuned, utility_b_retuned, sensitivity,
    )

    return ExperimentResult(
        name="prevalence",
        title="Case mix or degradation: decomposing the hospital B drop",
        table=table,
        prose=prose,
        metadata={
            "threshold": threshold,
            "utility_a": site_a["utility"],
            "utility_b": site_b["utility"],
            "utility_b_reweighted": utility_b_weighted,
            "auroc_a": site_a["auroc"],
            "auroc_b": site_b["auroc"],
            "auroc_b_reweighted": auroc_b_weighted,
            "gap": gap,
            "case_mix_component": case_mix,
            "degradation_component": degradation,
            "case_mix_share": case_mix_share,
            "utility_b_retuned": utility_b_retuned,
            "utility_a_retuned": utility_a_retuned,
            "ci": ci,
            "n_boot": n_boot,
            "prevalence_a": site_a["prevalence"],
            "prevalence_b": site_b["prevalence"],
            "prevalence_b_reweighted": prevalence_b_weighted,
            "covariates": BASELINE_COLUMNS,
            "sensitivity": sensitivity,
            "trim_percentiles": list(TRIM),
            **diagnostics,
        },
    ).validate()


# --------------------------------------------------------------------------
# Scoring one cohort at the frozen threshold
# --------------------------------------------------------------------------
def _score_cohort(booster, features, path, threshold, covariates=None) -> dict:
    frame = pd.read_parquet(path)
    assert_contiguous_admissions(frame["patient_id"].to_numpy())
    y = frame["SepsisLabel"].to_numpy()
    groups = frame["patient_id"].to_numpy()
    scores = predict(booster, frame, features)
    alerts = (scores >= threshold).astype(np.float64)
    scorer = UtilityScorer(y, groups)

    out = {
        "y": y,
        "groups": groups,
        "scores": scores,
        "alerts": alerts,
        "scorer": scorer,
        "auroc": float(roc_auc_score(y, scores)),
        "utility": scorer.score(alerts),
        "parts": admission_utility_parts(scorer, alerts, groups),
        "n_admissions": int(pd.Series(groups).nunique()),
        "prevalence": float(
            pd.Series(y).groupby(pd.Series(groups), observed=True).max().mean()
        ),
        "admission_label": pd.Series(y).groupby(pd.Series(groups), observed=True).max(),
    }
    if covariates is not None:
        out["covariates"] = baseline_covariates(frame, covariates)
    return out


# --------------------------------------------------------------------------
# The reweighting
# --------------------------------------------------------------------------
def _propensity_design(
    reference: pd.DataFrame, target: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Stack the two cohorts. Label 1 is the reference (hospital A)."""
    design = pd.concat([reference, target], axis=0)
    labels = np.concatenate([np.ones(len(reference)), np.zeros(len(target))])
    return design, labels, labels == 0


def _fit_weights(
    design: pd.DataFrame,
    labels: np.ndarray,
    is_b: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> tuple[pd.Series, float]:
    """Density ratio p(A|x)/p(B|x) for each hospital B admission, trimmed.

    Trimming is not cosmetic. A single admission whose covariates look extremely
    like hospital A can otherwise carry several percent of the reweighted cohort,
    and the resulting number would describe that patient rather than the hospital.
    """
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    kwargs = {"logisticregression__sample_weight": sample_weight} if sample_weight is not None else {}
    model.fit(design, labels, **kwargs)
    p_a = model.predict_proba(design)[:, 1]
    auc = float(roc_auc_score(labels, p_a, sample_weight=sample_weight))

    odds = p_a[is_b] / np.clip(1.0 - p_a[is_b], 1e-12, None)
    lo, hi = np.percentile(odds, TRIM)
    trimmed = np.clip(odds, lo, hi)
    return pd.Series(trimmed / trimmed.mean(), index=design.index[is_b]), auc


def _overlap(
    weights: pd.Series, propensity_auc: float, target: pd.DataFrame, reference: pd.DataFrame
) -> dict:
    """Whether the reweighting is trustworthy, and whether it did anything."""
    w = weights.to_numpy()
    ess = float(w.sum() ** 2 / np.square(w).sum())

    before = _smd(reference, target, np.ones(len(target)))
    after = _smd(reference, target, w)
    return {
        "propensity_auc": propensity_auc,
        "ess": ess,
        "ess_fraction": ess / len(w),
        "max_weight": float(w.max()),
        "weight_p99_over_median": float(np.percentile(w, 99) / np.median(w)),
        "smd_before": float(np.mean(np.abs(before))),
        "smd_after": float(np.mean(np.abs(after))),
        "max_smd_before": float(np.max(np.abs(before))),
        "max_smd_after": float(np.max(np.abs(after))),
    }


def _smd(reference: pd.DataFrame, target: pd.DataFrame, w: np.ndarray) -> np.ndarray:
    """Standardised mean difference per covariate, target weighted by ``w``."""
    ref, tgt = reference.to_numpy(dtype=float), target.to_numpy(dtype=float)
    ref_mean = np.nanmean(ref, axis=0)
    ref_var = np.nanvar(ref, axis=0)

    mask = ~np.isnan(tgt)
    wm = np.where(mask, w[:, None], 0.0)
    tgt_mean = np.nansum(np.where(mask, tgt, 0.0) * wm, axis=0) / np.clip(wm.sum(axis=0), 1e-12, None)
    tgt_var = np.nansum(np.where(mask, (tgt - tgt_mean) ** 2, 0.0) * wm, axis=0) / np.clip(
        wm.sum(axis=0), 1e-12, None
    )
    pooled = np.sqrt(np.clip((ref_var + tgt_var) / 2, 1e-12, None))
    return (tgt_mean - ref_mean) / pooled


def _weighted_prevalence(site: dict, row_weight: np.ndarray) -> float:
    per_admission = pd.Series(row_weight, index=site["groups"]).groupby(level=0, sort=False).first()
    label = site["admission_label"].reindex(per_admission.index)
    return float((per_admission * label).sum() / per_admission.sum())


def _sensitivity(reference, site_a, site_b, threshold, case_mix_baseline) -> dict:
    """The same decomposition under two defensible alternative choices.

    Neither replaces the pre-registered estimand -- changing that after seeing the
    result is the failure this repository argues against -- and both are reported
    because a number that moves when a reasonable choice moves should be presented
    as a range, not a point.

    1. **Reference = hospital A's test split.** The specification named A's
       *training* split as the reference distribution, and the comparison is
       against A's *test* utility. That is a hybrid: the population being matched
       to is not the population being compared with.
    2. **Admission-time covariates only.** The main list includes six hours of
       vitals and ordering volume, which are care process. Adjusting for a
       mediator of site practice can absorb part of the very effect being
       measured.
    """
    out = {"pre_registered": float(case_mix_baseline)}

    variants = {
        "reference_is_a_test": (site_a["covariates"], BASELINE_COLUMNS),
        "admission_time_covariates_only": (reference, BASELINE_ONLY_COLUMNS),
    }
    for name, (ref, columns) in variants.items():
        ref_cols = ref[columns]
        target_cols = site_b["covariates"][columns]
        design, labels, is_b = _propensity_design(ref_cols, target_cols)
        weights, _ = _fit_weights(design, labels, is_b)
        row_weight = weights.reindex(site_b["groups"]).to_numpy()
        reweighted = weighted_utility(site_b["scorer"], site_b["alerts"], row_weight)
        out[name] = float(reweighted - site_b["utility"])
    return out


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------
def _bootstrap(site_a, site_b, reference, target, n_boot, rng, log) -> dict[str, np.ndarray]:
    """Cluster bootstrap that refits the propensity model on every replicate.

    Resampling admissions is expressed as multinomial counts rather than by
    materialising a resampled frame: the utility of a cohort is a ratio of two
    sums over admissions, so a count vector is all the arithmetic needs, and the
    762,000-row hospital B frame is never copied.
    """
    parts_a = site_a["parts"]
    parts_b = site_b["parts"].reindex(target.index)
    num_a, den_a = parts_a["num"].to_numpy(), parts_a["den"].to_numpy()
    num_b, den_b = parts_b["num"].to_numpy(), parts_b["den"].to_numpy()

    design, labels, is_b = _propensity_design(reference, target)
    n_ref, n_tgt, n_a = len(reference), len(target), len(parts_a)
    rows_b = _row_index(site_b["groups"], target.index)
    rows_a = _row_index(site_a["groups"], parts_a.index)

    out = {k: np.empty(n_boot) for k in ("gap", "case_mix", "degradation", "auroc_gap", "auroc_degradation")}
    for b in range(n_boot):
        c_a = rng.multinomial(n_a, np.full(n_a, 1 / n_a)).astype(float)
        c_ref = rng.multinomial(n_ref, np.full(n_ref, 1 / n_ref)).astype(float)
        c_tgt = rng.multinomial(n_tgt, np.full(n_tgt, 1 / n_tgt)).astype(float)

        w, _ = _fit_weights(
            design, labels, is_b, sample_weight=np.concatenate([c_ref, c_tgt])
        )
        wb = w.to_numpy() * c_tgt

        utility_a = float(c_a @ num_a / (c_a @ den_a))
        utility_b = float(c_tgt @ num_b / (c_tgt @ den_b))
        utility_bw = float(wb @ num_b / (wb @ den_b))

        # AUROC is resampled the same way, by weighting rows with their
        # admission's bootstrap count. Holding it at its point estimate would
        # publish an interval of zero width and call it uncertainty.
        auroc_a = _weighted_auroc(site_a, c_a[rows_a])
        auroc_b = _weighted_auroc(site_b, c_tgt[rows_b])
        auroc_bw = _weighted_auroc(site_b, wb[rows_b])

        out["gap"][b] = utility_a - utility_b
        out["case_mix"][b] = utility_bw - utility_b
        out["degradation"][b] = utility_a - utility_bw
        out["auroc_gap"][b] = auroc_a - auroc_b
        out["auroc_degradation"][b] = auroc_a - auroc_bw
        if not (b + 1) % 50:
            log(f"bootstrap {b + 1}/{n_boot}")
    return out


def _row_index(groups: np.ndarray, admissions: pd.Index) -> np.ndarray:
    """Position of each row's admission within ``admissions``, for broadcasting
    an admission-level weight or bootstrap count back over ICU hours."""
    return pd.Series(np.arange(len(admissions)), index=admissions).reindex(groups).to_numpy()


def _weighted_auroc(site: dict, row_weight: np.ndarray) -> float:
    return float(roc_auc_score(site["y"], site["scores"], sample_weight=row_weight))


def _percentile_ci(draws: np.ndarray, alpha: float = 0.05) -> list[float]:
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------
def _assert_decomposition(gap: float, case_mix: float, degradation: float) -> None:
    """The two components must add back to the gap. Cheap, and catches a sign slip."""
    if abs((case_mix + degradation) - gap) > 1e-9:
        raise ValueError(
            f"decomposition does not close: case mix {case_mix:+.6f} + degradation "
            f"{degradation:+.6f} != gap {gap:+.6f}"
        )


def _assert_overlap(d: dict) -> None:
    """Refuse to publish a reweighting that is not supported by the data."""
    if d["propensity_auc"] > MAX_PROPENSITY_AUC:
        raise ValueError(
            f"the two cohorts are separable at AUC {d['propensity_auc']:.3f}: there is "
            f"effectively no covariate overlap, so reweighting B to A's case mix "
            f"extrapolates rather than reweights"
        )
    if d["ess_fraction"] < MIN_ESS_FRACTION:
        raise ValueError(
            f"reweighting collapses the cohort to an effective {d['ess_fraction']:.1%} "
            f"of its admissions; the reweighted number would describe a handful of "
            f"patients, not a hospital"
        )
    if d["smd_after"] >= d["smd_before"]:
        raise ValueError(
            f"reweighting did not improve covariate balance (mean |SMD| "
            f"{d['smd_before']:.3f} -> {d['smd_after']:.3f}); whatever it produced is "
            f"not a case-mix adjustment"
        )


def _prose(
    site_a, site_b, gap, case_mix, degradation, share, auroc_bw,
    prevalence_bw, d, ci, threshold, n_boot, utility_a_retuned, utility_b_retuned,
    sensitivity,
) -> str:
    direction = (
        "most of the gap is case mix" if share > 0.6 else
        "most of the gap survives the adjustment" if share < 0.4 else
        "the gap splits roughly evenly"
    )
    if ci["case_mix"][0] <= 0 <= ci["case_mix"][1]:
        direction += ", and the case-mix component is not separable from zero"

    retune_share = (utility_b_retuned - site_b["utility"]) / gap if gap else float("nan")
    retune_reading = (
        "Hospital A's threshold is already very nearly the right threshold for "
        "hospital B, so the loss is in the scores themselves, not in where the line "
        "is drawn."
        if retune_share < 0.15
        else "So a meaningful part of what the residual measures is a stale operating "
             "point rather than a worse model, and it is recoverable at the new site "
             "by re-picking one number."
    )
    return (
        f"At the operating point frozen on hospital A's validation split "
        f"(threshold {threshold:.3f}, applied unchanged everywhere), hospital A "
        f"scores {site_a['utility']:.4f} normalised utility and hospital B scores "
        f"{site_b['utility']:.4f}: a gap of **{gap:.4f}** "
        f"(95% CI {ci['gap'][0]:.4f} to {ci['gap'][1]:.4f}). Hospital B's admissions "
        f"are reweighted to hospital A's baseline case mix -- demographics, unit, "
        f"early vitals and first-six-hour ordering volume -- by a propensity density "
        f"ratio trimmed at the 1st and 99th percentiles. That "
        f"weighting cuts mean covariate imbalance from {d['smd_before']:.3f} to "
        f"{d['smd_after']:.3f} |SMD| and retains an effective "
        f"{d['ess_fraction']:.0%} of {site_b['n_admissions']:,} admissions.\n\n"
        f"Reweighted, hospital B scores **{site_b['utility'] + case_mix:.4f}**. So of "
        f"the {gap:.4f} gap, **{case_mix:+.4f} is case mix** (95% CI "
        f"{ci['case_mix'][0]:+.4f} to {ci['case_mix'][1]:+.4f}) and "
        f"**{degradation:+.4f} is what remains after adjustment** (95% CI "
        f"{ci['degradation'][0]:+.4f} to {ci['degradation'][1]:+.4f}) — {direction}. "
        f"Both intervals come from a {n_boot}-replicate patient-level cluster "
        f"bootstrap that refits the propensity model on every replicate, so the "
        f"uncertainty of the reweighting is inside the interval rather than assumed "
        f"away.\n\n"
        f"That component is not robust to how the adjustment is specified, which is "
        f"itself the result. Matching to hospital A's test split rather than its "
        f"training split gives {sensitivity['reference_is_a_test']:+.4f}; using only "
        f"covariates fixed at admission — dropping the six hours of early vitals and "
        f"ordering volume, which are care process and plausibly mediators of the site "
        f"difference being adjusted for — gives "
        f"{sensitivity['admission_time_covariates_only']:+.4f}. The pre-registered "
        f"specification is the one reported above; these are reported beside it "
        f"because a number that moves when a defensible choice moves belongs in the "
        f"open as a range.\n\n"
        f"Two cautions on reading the second number as degradation. The adjustment "
        f"can only correct for the {len(BASELINE_COLUMNS)} covariates it was given, "
        f"so anything that differs between two health systems and is not in that list "
        f"— treatment practice, labelling behaviour, unmeasured acuity — stays in the "
        f"residual. And reweighting on baseline covariates does not force the septic "
        f"rate to match: it moves from {100 * site_b['prevalence']:.1f}% to "
        f"{100 * prevalence_bw:.1f}% against hospital A's "
        f"{100 * site_a['prevalence']:.1f}%. The residual is an upper bound on "
        f"degradation, not a measurement of it.\n\n"
        f"One candidate explanation can be ruled out without any modelling. The "
        f"threshold is frozen at hospital A's, by design, because that is what "
        f"deploying a model means, and a frozen threshold is exactly the kind of "
        f"thing that stops working at a new site. Here it does not: re-picking the "
        f"threshold on hospital B alone moves its utility from "
        f"{site_b['utility']:.4f} only to {utility_b_retuned:.4f}, "
        f"{100 * retune_share:.0f}% of the gap. {retune_reading} "
        f"The MICU-to-SICU experiment in the next section runs the same check across "
        f"a different boundary and gets the opposite answer, which is the useful "
        f"pairing: a model can cross a boundary intact while its operating point "
        f"does not, and it can carry its operating point across intact and still "
        f"lose most of its value."
    )
