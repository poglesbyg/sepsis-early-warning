# Decisions

Why this pipeline is built the way it is. Each entry states the choice, the
alternative that was rejected, and what it would have cost to get it wrong.

Entries marked **PLANNED** describe work that is specified but not yet built. They
are here because the specification has to exist before the code, not after.

---

## Evaluation

### Splits are at the level of an admission, never an ICU hour

An ICU stay contributes roughly 40 strongly autocorrelated hours. Splitting rows
at random puts hour 12 of a patient in training and hour 13 in test, and the model
is then graded on how well it interpolates within stays it has already seen.

Applies everywhere without exception: train/validation/test, the cross-validation
folds (`StratifiedGroupKFold` on `patient_id`), and the bootstrap. `GroupKFold`
alone would keep stays intact but let positive rate drift between folds, which
makes utility fold-dependent for the wrong reason, so the stratified variant is
used.

**Rejected:** row-level splitting. Cost of getting it wrong: an inflated held-out
score and a model that fails the moment it sees a patient it has not met.

### Hospital B is never used for anything except the final table

Not for fitting, tuning, calibration, threshold selection, or blend weights. It is
a different health system with different case mix, charting practice and
prevalence (5.7% septic admissions against 8.8%).

**Rejected:** pooling both hospitals and splitting at random. That would have
produced better-looking numbers and destroyed the only external estimate in the
project.

### Hyperparameters are tuned on clinical utility, not AUROC

The challenge's utility function is time-dependent and asymmetric: a false alarm
costs 0.05, a missed septic hour costs up to 2.0, and a correct alert is worth the
most exactly six hours before onset. Ranking quality is not what the model is for.

Tuning on utility only became affordable after the scorer was vectorised (below).
Before that, the reference implementation's per-hour Python loop made a 60-trial
search with an in-fold threshold sweep impractical, which is the usual reason
people tune on AUROC and hope.

### Thresholds and blend weights are chosen on validation and then frozen

The test split is scored once, at the operating point already fixed. Choosing a
threshold on the test set and then reporting performance at that threshold is
optimisation against the test set with extra steps.

### Statistical tests use one observation per admission

790,000 ICU hours are not 790,000 independent observations. Testing them as if
they were produces p-values with far too many zeros to carry meaning. The
univariate screen collapses each admission to a single summary value first, so *n*
is the number of patients.

With 345 features screened, Benjamini-Hochberg controls the false discovery rate.
Effect sizes (Hedges' *g*, univariate AUC) are reported alongside significance
because at *n* = 20,000 nearly everything is significant and the size of the
separation is the only interesting part.

### Confidence intervals resample admissions, not hours

An hour-level bootstrap treats 40 correlated rows of one stay as 40 independent
draws and produces intervals several times too narrow. The cluster bootstrap
resamples whole admissions.

### Model comparison uses DeLong, not an unpaired test

Every model scores the identical rows, so the comparison is paired. An unpaired
test would be needlessly conservative and would miss real differences.

---

## Features

### Missingness is a feature, not a nuisance

Roughly 90% of lab cells are empty, and the emptiness is not random: a clinician
orders a lactate *because* they are worried. The feature matrix therefore carries,
for every channel, hours since it was last measured and how often it has been
sampled so far.

This is testable rather than assumed, and it was tested: ordering rate alone,
ignoring the value entirely, separates septic from non-septic admissions
(`Lactate` order-rate AUC 0.666, all 8 sparse labs significant after FDR
correction). Imputing missingness away would discard that.

### Two different design matrices, deliberately

The gradient booster receives raw NaNs and all 345 features. XGBoost learns a
default direction per split, which is a learned, per-split treatment of
missingness; imputing first would erase exactly the signal above.

The logistic regression receives median-imputed, standardised, collinearity-pruned
features. Median imputation is safe there *because* the recency and intensity
blocks already encode missingness explicitly, so the model can still distinguish a
stale carry-forward from a fresh reading.

**Rejected:** one matrix for both. It would handicap whichever model it was not
designed for, and the comparison would then be about preprocessing rather than
about model class.

### Domain features exist so the linear model gets a fair fight

SIRS, qSOFA, partial SOFA, shock index, BUN/creatinine ratio. A booster can in
principle discover "heart rate over systolic pressure" through axis-aligned
splits. Logistic regression cannot discover it at all. Encoding the ratios
clinicians already use puts the physiology in the design matrix instead of hoping
the model reinvents it.

### The personal baseline is expanding, not whole-stay

A heart rate of 105 is unremarkable in someone who arrived at 100 and alarming in
someone who arrived at 62. Centring on the patient's own running mean captures
that. The mean is expanding rather than computed over the full stay, because a
whole-stay statistic at hour 3 would encode hour 40.

---

## Models

### Class weighting buys ranking and destroys the probability scale

`scale_pos_weight` and `class_weight="balanced"` are the right answers to a 1.8%
positive rate. Both push mean predicted risk far above the base rate, so the raw
output is a good ranking and a meaningless probability.

Isotonic recalibration is fitted on validation and frozen. Platt scaling is
strictly increasing and leaves AUROC exactly unchanged; isotonic is
non-*decreasing*, so it merges neighbouring scores and moves AUROC by a few
thousandths through ties alone. Both are reported.

Calibration quality is measured on the **test** split, not validation. In-sample
isotonic ECE is 0 by construction, so reporting it on the split it was fitted to
would be circular.

### The recurrent network is unidirectional, and that is not an oversight

A bidirectional GRU reading hour 40 while scoring hour 20 would produce a
beautiful offline number and be undeployable. Convolutions use
`padding="causal"` for the same reason.

Batches are bucketed by stay length rather than padded to a global maximum. Stays
run from 8 to 336 hours; padding everything to 336 would make over 90% of the
compute mask, and truncating to a fixed window would discard about a quarter of
the positive hours, which arrive late in long stays.

### The blend operates in rank space

The three models live on incompatible scales, so averaging raw outputs would
weight by scale rather than by quality. Utility depends only on the ordering
induced by a threshold, so nothing is lost by working in ranks. The blend then
gets its own calibration map, because a rank average is not a probability.

### The ensemble is reported but not recommended

It beats XGBoost alone by ΔAUROC +0.0059 (DeLong *p* = 6×10⁻⁷): statistically
unambiguous, clinically irrelevant, and it costs three models in production
instead of one. The booster alone also transfers better to hospital B (0.328
against 0.257 utility). The honest recommendation is the single model.

---

## Implementation

### The utility scorer is vectorised exactly

For a fixed patient, utility is linear in the binary prediction vector:

```
u(t) = pred[t] · u_pos[t] + (1 − pred[t]) · u_neg[t]
```

so the cohort total is `u_neg.sum() + pred @ (u_pos − u_neg)`, and the normalising
constants fall out of the same two arrays. A 200-threshold sweep over 119,000
hours becomes one matrix multiply, about 4 ms.

This is not a micro-optimisation. It is what made tuning on the metric that
matters affordable rather than tuning on AUROC and hoping. Verified against the
published definition in `tests/test_utility.py` and against the official PhysioNet
implementation to 3×10⁻¹⁶.

### Preprocessing lives inside the sklearn Pipeline

The imputer and scaler are refit on each cross-validation training fold. Fitting
them once on the full training set before cross-validating leaks fold-level
information into every fold.

---

## What this project does *not* claim

### There is no temporal validation, because this cohort cannot support it

PhysioNet/CinC 2019 carries no calendar time. The only time-like columns are
`HospAdmTime` (hours between hospital and ICU admission, per-patient relative) and
`ICULOS` (hours since ICU admission). There is no admission date, so an
admission-ordered split is not buildable here.

The tempting workaround is to treat the sequential `patient_id` ordering as a
proxy for admission order. **Rejected.** The cohort is de-identified, the ordering
is undocumented, and if the identifiers were randomised the result is a random
split wearing a temporal label. An unverifiable claim is worse than no claim.

### The labels are clinical suspicion, not biological ground truth

`SepsisLabel` marks Sepsis-3 clinical suspicion, shifted six hours earlier. A model
trained on it learns to anticipate when a care team will start acting. That is the
right target for an early-warning tool and a real limitation of what the numbers
mean.

### Reproducibility is not the same as a regression check

Seeds are fixed and the pipeline is deterministic on one machine, but nothing here
guarantees a third party reproduces these numbers across package versions,
hardware, or solver behaviour. The planned check (below) verifies that code changes
do not silently move published results, which is a narrower and honest claim.

---

## Planned specifications

Written before the corresponding code exists, because a shift experiment without a
stated estimand produces a number nobody can interpret.

### PLANNED — prevalence-shift decomposition

**Question.** Hospital B utility is 0.328 against hospital A's 0.394, and septic
prevalence is 5.7% against 8.8%. How much of the gap is case mix, and how much is
genuine model degradation?

"Case mix versus degradation" is not identifiable from prevalence alone, so the
decomposition must fix the following before any code:

- **Estimand.** Performance on hospital B reweighted so its covariate distribution
  matches hospital A's, compared against hospital B unweighted. The difference
  between those two is the case-mix component; the residual gap against hospital A
  is degradation.
- **Reference distribution.** Hospital A's training split.
- **Weights.** Estimated by a propensity model over the static covariates and
  admission-level summaries, with weights trimmed at the 1st and 99th percentiles
  and the trimming reported.
- **Calibration.** Held fixed, not refit on hospital B. Refitting would answer a
  different question (how well *could* it transfer) and must be reported
  separately if done at all.
- **Uncertainty.** Patient-level cluster bootstrap over the whole reweighting
  procedure, not over the final statistic alone.

### PLANNED — MICU to SICU transfer

**Claim under test.** A model trained on medical ICU admissions transfers to
surgical ICU admissions within the same health system.

- **Direction.** Train on `Unit1` (MICU), test on `Unit2` (SICU). One direction
  only, stated in advance, so the better-looking direction cannot be chosen after
  the fact.
- **Missing units.** Admissions where both indicators are absent form an explicit
  third bucket, reported separately. They are never silently dropped.
- **Additive.** The existing random-split numbers stay published unchanged. This
  is an additional result, not a replacement.
- **What it is not.** Not a temporal split and not external validation. Both units
  sit in the same hospital system, so this measures care-pathway shift and may be
  dominated by prevalence differences between units, which will be reported
  alongside it.

### PLANNED — illustrative cases come from validation

The replay demo and the failure gallery both display individual admissions. Those
are selected from the **validation** split, which was already spent on tuning and
threshold selection.

Selecting them from test would convert a held-out set into a presentation set. The
reported metrics would remain arithmetically correct, but a human would have
inspected test cases to decide what to show, and the split would no longer be
untouched in the sense the rest of this document claims.
