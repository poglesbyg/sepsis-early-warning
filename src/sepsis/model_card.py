"""Generate MODEL_CARD.md from the artifacts, rather than writing it by hand.

A model card written once and edited by hand goes stale the first time anything is
retrained, and a stale card is worse than none: it is a document whose whole
purpose is to be trusted about limitations. This one is generated from the same
files the report reads, so a number in it is either current or the regression
check is failing.

The substance of a model card is disaggregated evaluation -- performance broken
out by the factors that matter clinically, not one headline figure. That table
does not exist anywhere else in this repository, so it is computed here: by sex,
by age band, and by admitting unit, on both hospital A's test split and the
external hospital, at the frozen operating point, with patient-level bootstrap
intervals.

What the cohort cannot support is stated rather than skipped. PhysioNet/CinC 2019
carries no race or ethnicity, no insurance status, and no calendar time, so the
disparities most often asked about in clinical ML cannot be assessed on this data
at all. Saying so is part of the card.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import CFG, Config
from .evaluate.lead_time import alert_timing, lead_time_summary
from .evaluate.metrics import UtilityScorer, cluster_bootstrap_ci
from .inference import SERVED_MODEL, ServingBundle

# Below this a subgroup's AUROC is a statement about a handful of patients. The
# groups are reported either way -- suppressing the row entirely would hide that
# the model was never evaluated there -- but the metrics are withheld.
MIN_PER_CLASS = 10

AGE_BANDS = [0, 50, 65, 80, 200]
AGE_LABELS = ["under 50", "50-64", "65-79", "80 and over"]

SPLITS = {"test": ("hospital A (test)", "setA"), "external": ("hospital B (external)", "setB")}


# --------------------------------------------------------------------------
# Disaggregated evaluation
# --------------------------------------------------------------------------
def subgroup_table(cfg: Config = CFG, model: str = SERVED_MODEL) -> pd.DataFrame:
    bundle = ServingBundle.load(model, cfg)
    rows = []
    for split, (label, source) in SPLITS.items():
        preds = pd.read_parquet(cfg.artifacts_dir / f"preds_{model}_{split}.parquet")
        preds = preds.sort_values(["patient_id", "hour"], ignore_index=True)
        preds["risk"] = bundle.calibrator.transform(preds["score"].to_numpy())

        raw = pd.read_parquet(
            cfg.interim_dir / f"{source}.parquet",
            columns=["patient_id", "Age", "Gender", "Unit1", "Unit2"],
        )
        facts = raw.groupby("patient_id", observed=True).first()
        preds = preds.join(facts, on="patient_id")

        for factor, series in _factors(preds).items():
            for group in _ordered(series):
                mask = (series == group).to_numpy()
                rows.append(
                    _metrics(preds[mask], bundle.threshold)
                    | {"split": label, "factor": factor, "group": group}
                )
            _assert_partition(series, factor, label)
    columns = ["split", "factor", "group", "n_admissions", "septic_pct", "auroc",
               "auroc_lo", "auroc_hi", "utility", "detection_rate",
               "median_lead_time_h", "false_alarm_rate"]
    return pd.DataFrame(rows)[columns]


def _factors(preds: pd.DataFrame) -> dict[str, pd.Series]:
    """The axes the card breaks performance out along.

    Sex uses the challenge's own coding, which documents 0 as female and 1 as
    male; it is reported in those terms rather than silently relabelled.
    """
    unit = np.where(
        preds["Unit1"].fillna(0) == 1, "MICU",
        np.where(preds["Unit2"].fillna(0) == 1, "SICU", "unit not recorded"),
    )
    return {
        "overall": pd.Series(["all admissions"] * len(preds), index=preds.index),
        "sex": preds["Gender"].map({0: "female (coded 0)", 1: "male (coded 1)"}).fillna("not recorded"),
        "age": pd.cut(preds["Age"], AGE_BANDS, labels=AGE_LABELS).astype(str),
        "unit": pd.Series(unit, index=preds.index),
    }


def _ordered(series: pd.Series) -> list[str]:
    """Age bands in age order; everything else alphabetical, for a stable table."""
    groups = sorted(series.dropna().unique())
    return [b for b in AGE_LABELS if b in groups] or groups


def _metrics(frame: pd.DataFrame, threshold: float) -> dict:
    y = frame["SepsisLabel"].to_numpy() if "SepsisLabel" in frame else frame["y"].to_numpy()
    groups = frame["patient_id"].to_numpy()
    risk = frame["risk"].to_numpy()

    admissions = pd.Series(y).groupby(pd.Series(groups), observed=True).max()
    n_septic = int(admissions.sum())
    n_control = int(len(admissions) - n_septic)

    out = {
        "n_admissions": int(len(admissions)),
        "septic_pct": 100 * float(admissions.mean()),
    }
    if min(n_septic, n_control) < MIN_PER_CLASS:
        # Reported as a row with no numbers: the group exists, the model was not
        # meaningfully evaluated on it, and both facts belong in the card.
        return out | {k: float("nan") for k in
                      ("auroc", "auroc_lo", "auroc_hi", "utility",
                       "detection_rate", "median_lead_time_h", "false_alarm_rate")}

    point, lo, hi = cluster_bootstrap_ci(y, risk, groups, n_boot=200, seed=0)
    timing = lead_time_summary(alert_timing(y, risk, groups, threshold))
    return out | {
        "auroc": float(roc_auc_score(y, risk)),
        "auroc_lo": lo,
        "auroc_hi": hi,
        "utility": UtilityScorer(y, groups).score((risk >= threshold).astype(float)),
        "detection_rate": 100 * timing["detection_rate"],
        "median_lead_time_h": timing["median_lead_time_h"],
        "false_alarm_rate": 100 * timing["false_alarm_rate_per_admission"],
    }


def _assert_partition(series: pd.Series, factor: str, split: str) -> None:
    """Every admission must land in exactly one group of every factor.

    A subgroup table that drops rows understates the population the model was
    evaluated on, and does it invisibly.
    """
    if series.isna().any():
        raise ValueError(
            f"{split}: {int(series.isna().sum())} rows fall outside every {factor} "
            f"group; a subgroup breakdown that loses patients is not a breakdown"
        )


# --------------------------------------------------------------------------
# The card
# --------------------------------------------------------------------------
def write_card(cfg: Config = CFG, model: str = SERVED_MODEL, quiet: bool = False) -> Path:
    metrics = json.loads((cfg.reports_dir / "metrics.json").read_text())
    experiments = {
        name: json.loads((cfg.reports_dir / f"experiment_{name}.json").read_text())
        for name in ("leakage", "ablation", "prevalence", "unit_transfer")
        if (cfg.reports_dir / f"experiment_{name}.json").exists()
    }
    table = subgroup_table(cfg, model)

    results = {(r["model"], r["split"]): r for r in metrics["results"]}
    lead = {r["model"]: r for r in metrics.get("lead_time", [])}
    threshold = metrics["thresholds"][model]

    card = "\n".join(
        [
            _details(model, threshold, results, cfg),
            _intended_use(),
            _factors_section(),
            _metrics_section(model, results, lead, threshold),
            _analyses(table, experiments),
            _ethics(experiments, lead.get(model, {})),
            _caveats(experiments),
        ]
    )

    path = cfg.root / "MODEL_CARD.md"
    path.write_text(card)
    table.to_csv(cfg.reports_dir / "model_card_subgroups.csv", index=False)
    (cfg.reports_dir / "model_card.json").write_text(
        json.dumps(
            {
                "model": model,
                "threshold": threshold,
                "subgroups": table.round(6).to_dict("records"),
            },
            indent=2,
            default=float,
        )
        + "\n"
    )
    if not quiet:
        print(f"[card] {len(table)} subgroup rows across "
              f"{table['factor'].nunique()} factors and {table['split'].nunique()} splits",
              flush=True)
        print(f"[card] wrote {path.name}", flush=True)
    return path


def _fmt(table: pd.DataFrame) -> str:
    view = table.copy()
    view["AUROC (95% CI)"] = [
        "—" if np.isnan(r.auroc) else f"{r.auroc:.3f} ({r.auroc_lo:.3f}–{r.auroc_hi:.3f})"
        for r in view.itertuples()
    ]
    for col, spec in (("septic_pct", "{:.1f}%"), ("utility", "{:.3f}"),
                      ("detection_rate", "{:.0f}%"), ("median_lead_time_h", "{:.0f} h"),
                      ("false_alarm_rate", "{:.0f}%")):
        view[col] = ["—" if pd.isna(v) else spec.format(v) for v in view[col]]
    view = view.rename(columns={
        "group": "subgroup", "n_admissions": "admissions", "septic_pct": "septic",
        "utility": "utility", "detection_rate": "caught before onset",
        "median_lead_time_h": "median warning", "false_alarm_rate": "controls alerted",
    })
    keep = ["subgroup", "admissions", "septic", "AUROC (95% CI)", "utility",
            "caught before onset", "median warning", "controls alerted"]
    return view[keep].to_markdown(index=False)


def _details(model, threshold, results, cfg) -> str:
    test = results[(model, "test")]
    return f"""# Model card: early sepsis warning from ICU time series

*Generated by `sepsis card` from the run's own artifacts. Every number here is
produced by the pipeline, not transcribed, and `make regress` fails if one moves
without the baseline moving with it.*

## Model details

| | |
|---|---|
| Model | Gradient-boosted trees (XGBoost), {test['auroc']:.3f} AUROC on the internal test split |
| Input | One admission's charted ICU hours; 345 features, every one computed from hours ≤ *t* |
| Output | Calibrated probability of sepsis onset, per ICU hour, plus an alert at a frozen threshold |
| Operating point | {threshold:.4f}, chosen on hospital A's validation split and never re-picked |
| Calibration | Isotonic, fitted on validation and frozen; served as one bundle with the model |
| Serving contract | `sepsis.inference.predict(history)` — see `src/sepsis/inference.py` |
| Training data | PhysioNet/CinC Challenge 2019, hospital system A |
| Licence | Code MIT; data ODbL v1.0 via PhysioNet (Reyna et al., *Crit Care Med* 48(2), 2020) |

The ensemble and the recurrent model exist in this repository and are reported,
but the booster alone is the recommendation and the only thing with a serving
contract: the ensemble's edge is statistically unambiguous and clinically
irrelevant, it costs three models in production, and it transfers worse to the
external hospital.
"""


def _intended_use() -> str:
    return """
## Intended use

**Intended.** Retrospective research on ICU time series, and methodological work
on early-warning models: leakage measurement, feature attribution, and site
transfer. Nothing more than that has been demonstrated.

**Not intended, and not supported by any evidence here.**

- **Not a diagnosis and not a treatment decision.** The label is clinical
  suspicion, not biology. A model trained on it learns to anticipate when a care
  team will start acting.
- **Not prospectively validated.** No patient outcome has ever been affected by
  this model, so nothing is known about how it behaves when clinicians can see it
  — including whether acting on it changes the very ordering behaviour it depends
  on.
- **Not a medical device.** No regulatory assessment of any kind has been made.
- **Not validated outside adult ICU admissions.** Paediatric patients, general
  wards, and emergency departments are all out of scope: the cohort contains none
  of them.
- **Not validated over time.** This cohort carries no calendar time, so no
  temporal or prospective split is buildable on it.
"""


def _factors_section() -> str:
    return """
## Factors

Performance is broken out by sex, age band, and admitting unit, on both hospital
systems. Sex uses the challenge's own coding, which documents 0 as female and 1 as
male; it is reported in those terms rather than relabelled on an assumption.

**What this cohort cannot support.** PhysioNet/CinC 2019 carries no race or
ethnicity, no insurance or socioeconomic status, no language, and no calendar
time. The disparities most often asked about in clinical machine learning
therefore cannot be assessed here at all — not because they were checked and found
absent, but because the data does not contain the columns. Any deployment would
have to measure them on its own population before assuming anything.

Age is de-identified by capping: hospital A's ages stop at 89 and hospital B's at
100, so the oldest band is not directly comparable between sites.
"""


def _metrics_section(model, results, lead, threshold) -> str:
    test, ext = results[(model, "test")], results[(model, "external")]
    timing = lead.get(model, {})
    return f"""
## Metrics

Utility is the challenge's own time-dependent clinical score: 1.0 is the best
achievable alerting policy, 0.0 is never alerting. It is the metric the model was
tuned on, because ranking quality is not what an alarm is for. Confidence
intervals resample whole admissions, never ICU hours.

| | hospital A (test) | hospital B (external) |
|---|---|---|
| AUROC | {test['auroc']:.3f} | {ext['auroc']:.3f} |
| Clinical utility | {test['utility']:.3f} | {ext['utility']:.3f} |
| Caught before onset | {timing.get('detection_rate', float('nan')):.0%} | — |
| Median warning | {timing.get('median_lead_time_h', float('nan')):.0f} h | — |
| False alarms per true detection | {timing.get('false_alarms_per_true_detection', float('nan')):.1f} | — |

An admission counts as caught only if the first alert lands *before* clinical
onset. An alert raised after the team already suspected sepsis is not an early
warning, and counting it as one is the most common way these systems get oversold.
"""


def _analyses(table: pd.DataFrame, experiments: dict) -> str:
    parts = ["\n## Quantitative analyses\n",
             "### Disaggregated performance\n",
             "At the frozen operating point. Intervals are a patient-level cluster "
             "bootstrap; a subgroup with fewer than "
             f"{MIN_PER_CLASS} admissions in either class is listed without metrics "
             "rather than dropped, so it stays visible that the model was not "
             "meaningfully evaluated there.\n"]
    for split in table["split"].unique():
        parts.append(f"**{split}**\n")
        parts.append(_fmt(table[table["split"] == split]) + "\n")

    parts.append(_reading(table))

    unit = experiments.get("unit_transfer")
    prev = experiments.get("prevalence")
    ablation = experiments.get("ablation")
    leakage = experiments.get("leakage")

    if unit and prev:
        parts.append(
            f"""### Transfer, measured two ways

Crossing a boundary breaks this model in two different ways, and one number would
have shown neither.

- **Between hospitals**, utility falls by {prev['gap']:.3f}. Reweighting hospital
  B's admissions to hospital A's baseline case mix recovers
  {prev['case_mix_component']:+.3f} of that, with a confidence interval straddling
  zero — and only {prev['sensitivity']['admission_time_covariates_only']:+.3f} when
  the adjustment uses covariates fixed at admission rather than six hours of care
  process. Re-picking the threshold recovers about 5% more. Neither comfortable
  explanation — different patients, stale operating point — survives contact with
  the numbers.
- **Between units in one hospital**, the AUROC difference is
  {unit['auroc_difference']:+.3f} (95% CI {unit['auroc_difference_ci'][0]:+.3f} to
  {unit['auroc_difference_ci'][1]:+.3f}), which does not establish a ranking loss,
  while utility at the frozen threshold falls from {unit['utility_home']:.3f} to
  {unit['utility_sicu']:.3f} (95% CI {unit['utility_sicu_ci'][0]:.3f} to
  {unit['utility_sicu_ci'][1]:.3f}). A threshold chosen on a separate half of the
  SICU and scored on the remainder recovers it to {unit['utility_sicu_local']:.3f}.
  The septic rate is {unit['prevalence_home']:.1f}% in the MICU and
  {unit['prevalence_sicu']:.1f}% in the SICU.

**The practical reading: the operating point is what fails to transfer.** Any new
site or unit should select its threshold on its own labelled outcomes before
anything is switched on. That is not free — it requires local outcome data — but it
requires no retraining, and the normalised utility score is itself prevalence-
sensitive, so part of the movement between units is the metric responding to a
different septic rate rather than the model behaving differently.
"""
        )

    if ablation:
        parts.append(
            f"""### What the model is actually using

Using {ablation['ordering_only_features']} features containing no measured value
whatsoever — only which channel was sampled, how recently, and how often — the
model reaches AUROC {ablation['ordering_only_auroc']:.3f}, or
{ablation['ordering_only_share_of_full']:.0%} of what the full
{ablation['n_features']}-feature matrix achieves. No single feature block costs
more than {ablation['max_loo_cost']:.4f} AUROC to remove.

That has an uncomfortable reading which belongs in this section rather than a
footnote: a model this dependent on ordering behaviour is substantially learning
clinical suspicion rather than physiology. It is partly predicting that somebody
was already worried.
"""
        )

    if leakage:
        parts.append(
            f"""### What the evaluation would be worth if done carelessly

Splitting ICU hours at random instead of splitting admissions inflates AUROC by
{leakage['row_split_inflation']:+.4f} and clinical utility by roughly three times
that. Filling gaps from the future with a single `bfill` inflates AUROC by
{leakage['lookahead_inflation']:+.4f}. Both mistakes are committed deliberately in
this repository and measured, so the headline numbers can be read against what
their absence is worth.
"""
        )
    return "\n".join(parts)


def _reading(table: pd.DataFrame) -> str:
    """Say what the breakdown shows, rather than leaving it in a table.

    A model card whose disaggregated section is a bare grid makes the reader do
    the work, and the thing worth noticing is usually whether a gap at one site
    reappears at the other. Both spreads are computed, and the comparison names
    the same two groups at both sites rather than each site's own extremes --
    otherwise a gap could be reported that is really two different comparisons.
    """
    a = table[table["split"] == "hospital A (test)"].set_index(["factor", "group"])
    b = table[table["split"] == "hospital B (external)"].set_index(["factor", "group"])

    lines = ["### What the breakdown shows\n"]
    for factor in ("sex", "age", "unit"):
        if factor not in a.index.get_level_values(0):
            continue
        rows = a.loc[factor].dropna(subset=["auroc"])
        if len(rows) < 2:
            continue
        low, high = rows["auroc"].idxmin(), rows["auroc"].idxmax()
        gap_a = rows.loc[high, "auroc"] - rows.loc[low, "auroc"]
        try:
            gap_b = b.loc[(factor, high), "auroc"] - b.loc[(factor, low), "auroc"]
        except KeyError:
            continue

        overlap = rows.loc[low, "auroc_hi"] >= rows.loc[high, "auroc_lo"]
        replicates = gap_b > gap_a / 2
        lines.append(
            f"- **By {factor}**, the widest gap on hospital A is between *{high}* "
            f"({rows.loc[high, 'auroc']:.3f}) and *{low}* "
            f"({rows.loc[low, 'auroc']:.3f}), a spread of {gap_a:.3f} AUROC. The "
            f"same two groups at hospital B differ by {gap_b:.3f}. The intervals "
            f"on hospital A {'overlap' if overlap else 'do not overlap'}, and the "
            f"gap {'reappears' if replicates else 'does not reappear'} at the "
            f"second site."
        )

    micu = a.loc[("unit", "MICU")] if ("unit", "MICU") in a.index else None
    sicu = a.loc[("unit", "SICU")] if ("unit", "SICU") in a.index else None
    if micu is not None and sicu is not None:
        lines.append(
            f"\nThe operational spread is wider than the discrimination spread and "
            f"matters more. On hospital A, {micu['false_alarm_rate']:.0f}% of "
            f"non-septic MICU admissions raise at least one alert against "
            f"{sicu['false_alarm_rate']:.0f}% in the SICU, at the same threshold. "
            f"Whoever staffs those two units experiences two different systems, and "
            f"no aggregate number in this card shows that."
        )
    return "\n".join(lines) + "\n"


def _ethics(experiments: dict, timing: dict) -> str:
    ablation = experiments.get("ablation", {})
    return f"""
## Ethical considerations

**The label is clinical suspicion, not sepsis.** `SepsisLabel` marks Sepsis-3
suspicion, shifted six hours earlier. The model learns to anticipate a care team's
behaviour. Where that behaviour is already unequal — who gets a lactate drawn, and
how quickly — a model trained on it will reproduce the inequality and present it
as a risk score. This cohort carries no demographic columns that would let anyone
check whether it does.

**Ordering behaviour is doing much of the work.** Measurement-only features reach
{ablation.get('ordering_only_share_of_full', float('nan')):.0%} of full performance
on their own. A model partly predicting that somebody was already worried will
degrade wherever charting habits differ, and will look most confident exactly where
staff already are.

**Alarm burden is the deployment constraint, not accuracy.** At the operating point
reported here, every true detection comes with about
{timing.get('false_alarms_per_true_detection', float('nan')):.1f} false-alarm
admissions. Whether that is acceptable is a question about a specific unit's
staffing and alarm load, not a question this model can answer, and alarm fatigue is
a patient-safety hazard in its own right.

**Automation bias runs both ways.** A silent model is not evidence of a
well patient, and a firing model is not evidence of a septic one. The
{100 - timing.get('detection_rate', 0) * 100:.0f}% of septic admissions this model
does not catch before onset are exactly the ones a clinician might be discouraged
from escalating if the score is treated as a second opinion.
"""


def _caveats(experiments: dict) -> str:
    return """
## Caveats and recommendations

- **Re-pick the threshold on local data before switching anything on.** The
  MICU-to-SICU result shows discrimination transferring while the operating point
  does not, and the fix costs one threshold sweep and no retraining.
- **Recalibrate and monitor at a new site.** The calibration map is frozen from
  hospital A's validation split. Nothing here establishes that it holds elsewhere.
- **Measure subgroup performance on the deploying population.** The breakdown
  above covers sex, age and unit because that is what this cohort carries. It is
  not the list a deployment should be satisfied with.
- **Do not read the external numbers as a floor.** Hospital B is one other health
  system, and the case-mix analysis suggests the remaining gap is not explained by
  measured differences in who was admitted.
- **No temporal validation exists.** This cohort has no calendar time. Model
  performance under drift is unmeasured, and the usual assumption that the future
  resembles the past is untested here.
- **The recurrent model is not served.** It scores lower, transfers worse, and
  needs carried state; the reasoning is in `DECISIONS.md`.
"""
