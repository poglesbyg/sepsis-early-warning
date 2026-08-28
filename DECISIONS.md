# Decisions

Why this pipeline is built the way it is. Each entry states the choice, the
alternative that was rejected, and what it would have cost to get it wrong.

Several entries were written before the code they describe, because a shift
experiment without a stated estimand produces a number nobody can interpret. They
are kept in the order they were decided rather than rewritten to look inevitable.

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
hardware, or solver behaviour. `make regress` verifies something narrower and
checkable: that a change to the code does not silently move a number this
repository has already published.

Every number the report and README quote already lands in a JSON artifact, so the
check flattens those artifacts to dotted keys and compares 362 of them against
`configs/regression_baseline.json`. Three properties make it worth having:

- **The baseline is committed**, so moving a published number is a reviewable diff
  in version control rather than an edit nobody sees. `make regress-update` is the
  only way to move it.
- **A number that stops being published fails the check.** Deleting a metric is
  otherwise the easiest way to make a regression disappear.
- **It compares artifacts rather than recomputing them.** It costs seconds, not the
  35 minutes a rerun costs, and it verifies whatever the last pipeline run wrote.
  That is the trade: it cannot tell you a stage was never re-run, so it belongs
  after the stage whose numbers you expect to have left alone.

**Rejected:** exact equality across machines. The tolerance is 1e-6 absolute — tight
enough that any real change trips it, loose enough not to fail on floating-point
summation order — and the claim stays scoped to one machine, because that is the
only claim the evidence supports.

### The data integrity check is a self-check, and says so

A file count detects a truncated download and nothing else. A file truncated
mid-write, a byte flipped in transit, or an upstream revision under a stable
filename all produce a complete-looking dataset and a model trained on something
other than what the published numbers describe.

PhysioNet publishes no per-file checksums for this release: its `SHA256SUMS.txt`
is three lines covering `LICENSE.txt` and two SVG diagrams, none of the 40,336
patient files. So there is nothing upstream to verify against, and
`configs/data_checksums.json` pins what *this* repository downloaded rather than
what PhysioNet says it should have.

**Rejected:** 40,336 per-file hashes. That is a 2 MB committed artifact whose diff
nobody reads. The digest is rolled up per hospital over `name:digest` lines in
sorted filename order — sensitive to renamed, added and removed files as well as
to changed bytes — and on a mismatch the check re-walks the files to say whether
the file list, the total size, or only the contents moved.

**A missing manifest is not a failure.** A clone whose data predates the check has
nothing to compare against, and refusing to run would push people to skip the
check entirely.

---

## Planned specifications

Written before the corresponding code exists, because a shift experiment without a
stated estimand produces a number nobody can interpret.

### Prevalence-shift decomposition — built

**Question.** Hospital B utility is lower than hospital A's, and septic prevalence
is 5.7% against 8.8%. How much of the gap is case mix, and how much is genuine
model degradation?

"Case mix versus degradation" is not identifiable from prevalence alone, so the
following was fixed before any code was written, and the implementation in
`src/sepsis/experiments/prevalence.py` follows it:

- **Estimand.** Performance on hospital B reweighted so its covariate
  distribution matches hospital A's, compared against hospital B unweighted. The
  difference is the case-mix component; the residual gap against hospital A is
  degradation.
- **Reference distribution.** Hospital A's training split.
- **Weights.** A propensity model over 17 admission-level baseline covariates,
  weights trimmed at the 1st and 99th percentiles, trimming reported.
- **Calibration.** Held fixed, not refit on hospital B. The threshold is frozen on
  hospital A's validation split and applied unchanged everywhere.
- **Uncertainty.** Patient-level cluster bootstrap that refits the propensity
  model on every replicate, not one that recomputes the final statistic alone.

Three implementation choices the specification did not settle:

**The baseline window is six hours, not one.** Case mix has to be characterised by
something, and the admission's first hour alone barely separates two hospitals.
Six hours buys early vitals and, more importantly, ordering volume — the most
visible difference between two health systems. These values weight a population
and never reach a prediction, so the no-lookahead invariant is not in play; the
weights are admission-level by construction.

**The reweighting asserts that it did something.** Mean |SMD| must fall, the
propensity AUC must stay below 0.99, and the effective sample size must stay above
5% of the cohort. A reweighting that extrapolates instead of reweighting, or that
collapses the cohort onto a handful of patients, produces a number that describes
those patients rather than a hospital, and nothing downstream would notice.

**The retuned-threshold diagnostic is reported alongside.** It is not part of the
estimand — the estimand freezes the operating point deliberately, because that is
what deploying a model means — but the unit-transfer experiment below shows an
operating point can fail to cross a boundary while the model survives it. Without
the diagnostic, a reader would have no way to tell which of the two was happening
here.

**Result.** Of a 0.130 utility gap, measured case mix accounts for +0.017 with a
confidence interval straddling zero, and re-picking the threshold on hospital B
recovers about 5%. The residual is an upper bound on degradation, not a
measurement of it: the adjustment can only correct for the covariates it was
given.

### MICU to SICU transfer — built

**Claim under test.** A model trained on medical ICU admissions transfers to
surgical ICU admissions within the same health system.

- **Direction.** Train on `Unit1` (MICU), test on `Unit2` (SICU). One direction
  only, stated in advance, so the better-looking direction could not be chosen
  after the fact.
- **Missing units.** Admissions where both indicators are absent form an explicit
  third bucket, reported separately. They are never silently dropped — and they
  turned out to be the largest of the three.
- **Additive.** The random-split numbers stay published unchanged.
- **What it is not.** Not a temporal split and not external validation. Both units
  sit in the same hospital system, so this measures care-pathway shift, and the
  units differ sharply in septic rate, which is reported alongside.

Two implementation choices:

**MICU is split three ways, not two.** Fit, freeze the threshold, and a held-out
within-unit slice. Without the within-unit number the SICU result is unreadable: a
drop could just as easily be the cost of training on a third of the pool as the
cost of crossing the boundary.

**The unit indicator columns are dropped from the feature matrix.** They are
constant across the training cohort and take an unseen value in every evaluation
bucket, so leaving them in would measure the model's reaction to a dead column
rather than the transfer.

**Result.** Discrimination crosses the boundary intact — AUROC 0.805 to 0.781,
overlapping intervals — while utility at the frozen threshold falls from 0.458 to
0.042, and retuning the threshold alone recovers most of that. The failure is the
operating point, not the model. Paired with the hospital result, which fails the
other way round, that is the argument against reporting one external number and
calling it generalisation.

### Illustrative cases come from validation, and the headline case is the median

The replay displays individual admissions. They are selected from the
**validation** split, which was already spent on tuning, calibration and
threshold selection, so nothing is lost by looking at it again.

Selecting them from test would convert a held-out set into a presentation set.
The reported metrics would stay arithmetically correct, but a human would have
inspected test admissions to decide what to show, and the split would no longer be
untouched in the sense the rest of this document claims. The check runs against
the split files themselves rather than trusting the filename the payload came
from.

**The case shown first is the one whose lead time is closest to the cohort
median**, not the largest. Picking the best trace in 225 caught admissions would
have been one line of code, would have looked considerably better, and would have
described nothing. The other three are a marginal catch, a septic admission the
model alerted on only after the care team was already acting, and a control that
sat above the threshold for 56 of its 58 hours. Two of the four cases are
failures because a demo that only shows the model working is an advertisement.

**Rejected:** hand-picking. Every case is a deterministic argmin or argmax with an
explicit tie-break, so the selection is reproducible and none of it is a judgement
about which trace looks best. Each case is then asserted against the role it is
published under: a case captioned as caught must have alerted before onset, and a
case captioned as a false alarm must not have been septic. A caption that
contradicts its own chart is the failure that matters here, because the chart is
what a reader believes.

**The probabilities on screen are in-sample.** The isotonic map was fitted on
validation, which is where these admissions come from, so the replay is evidence
about timing rather than about calibration quality, and the report says so where
it is displayed rather than in a footnote.

---

## Serving

### The contract takes a history and returns one hour, and batch falls out of it

The open question, recorded in `TODOS.md` before anything was written, was whether
inference should accept a batch of hours or one hour at a time with carried state.
The booster is stateless and the GRU is not, and that difference changes the API
shape.

**Settled: a history in, the risk for its most recent hour out.** Streaming is the
native call and batch scoring is a convenience that returns the same values,
because every feature at hour *t* depends only on hours <= *t*. The equivalence is
asserted at every hour of a stay rather than assumed, so if the feature builder
ever starts reading ahead, the contract's own tests fail before a deployment
behaves unlike the report.

**Rejected: carried state.** Caching derived features between hours would turn an
O(t) call into an O(1) one and buy nothing at ICU scale -- tens of hours, one call
an hour, per patient -- while adding state that needs invalidation rules, restart
recovery, and consistency across replicas. Cost of getting that wrong: a served
score that silently disagrees with the published one, which is the failure this
project exists to argue against.

**Rejected: serving the ensemble or the GRU.** The recurrent model is the reason
the shape question was open, and it is not served. Replaying a whole stay through
the network every hour or persisting a hidden state per admission is a different
system with different failure modes, in exchange for a model that scores lower on
this data and transfers worse to hospital B. The booster alone was already the
honest recommendation, so the booster alone has a contract.

### A model without its calibration map is not a deployable object

The isotonic map was fitted inside the evaluate stage and never persisted, so
nothing outside that process could reproduce a published prediction. The model,
its calibration map and its frozen threshold are now written together as one
bundle, and loading refuses a bundle whose feature list disagrees with the fitted
model rather than serving numbers nobody could trace.

### Input that would be misread is refused, not repaired

Two admissions in one call, duplicate hours, missing channel columns, and a
skipped hour are all rejected. The last one matters most and is the least
obvious: recency and intensity features count *rows*, not the `hour` column, so an
omitted hour reads as though no time passed and every staleness feature shifts.
Gaps must be materialised as all-NaN rows.

**Rejected:** filling gaps automatically. Silently inventing rows on the way into
a model is how the leakage this repository measures gets introduced in the first
place, and a caller who is dropping hours has a data problem worth knowing about.

---

## The model card

### It is generated, not written

A card edited by hand goes stale the first time anything is retrained, and a stale
card is worse than none: it is a document whose whole purpose is to be trusted
about limitations. `MODEL_CARD.md` is produced from the same artifacts the report
reads, and its numbers are pinned by `make regress` like every other published
figure.

**Rejected:** a hand-written card. Cost of getting it wrong: a limitations section
describing a model that no longer exists, with nothing to catch the drift.

### Subgroups are shown even when they cannot be scored

A group with fewer than 10 admissions in either class is listed with its size and
no metrics, rather than dropped. Dropping it would hide that the model was never
meaningfully evaluated there, which is exactly the fact a reader of a model card is
looking for.

### The written reading names the same two groups at both sites

The card does not only tabulate subgroups, it says what the table shows. The
comparison is computed by finding the widest gap on hospital A and then reporting
**those same two groups** at hospital B, rather than each site's own extremes:
comparing each site's best-against-worst would be two different comparisons
reported as one, and would manufacture a disparity out of noise.

That check earns its keep immediately. The sex gap on hospital A is 0.052 AUROC
with overlapping intervals, and the same two groups differ by 0.005 at hospital B.
The card says so, rather than reporting the first number alone.

### What the data cannot answer is stated as such

PhysioNet/CinC 2019 carries no race, ethnicity, insurance status, language or
calendar time. The card says these cannot be assessed rather than omitting the
subject, because silence on a fairness axis reads as a clean bill of health.
