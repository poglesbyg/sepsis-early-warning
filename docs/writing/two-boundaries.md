# Your model transferred. Your alerts didn't.

I trained a sepsis early-warning model on medical ICU admissions and tested it on
surgical ICU admissions in the same hospital. Discrimination barely moved: AUROC
0.805 at home, 0.781 across the boundary, confidence intervals overlapping.

The deployed configuration lost 91% of its clinical value.

| | admissions | septic | AUROC (95% CI) | value at the frozen threshold |
|---|---|---|---|---|
| MICU, held out | 1,137 | 10.8% | 0.805 (0.779–0.835) | 0.458 |
| SICU | 4,650 | 4.0% | 0.781 (0.748–0.818) | **0.042** |

Both columns are true at the same time, and only one of them would have appeared in
a paper.

## What the second column measures

"Value" here is the PhysioNet/CinC 2019 clinical utility score, which is worth
explaining because it is the whole point. It is time-dependent and asymmetric: a
false alarm costs a little, a missed septic hour costs a lot, and a correct alert is
worth the most about six hours before onset. It is normalised so 1.0 is the best
achievable alerting policy and 0.0 is never alerting at all.

So 0.042 is not "somewhat worse." It is *almost exactly as useful as switching the
system off.*

## Why it happens, and how cheap the fix is

The septic rate is 10.8% in the MICU and 4.0% in the SICU. A decision threshold
tuned where sepsis is common fires far too often where it is rare, and the utility
score charges for every one of those alarms.

Re-picking the threshold on SICU data alone — no retraining, no new labels, no
architecture change — recovers it from 0.042 to **0.299**.

That is the entire finding. The model crossed the boundary intact. The operating
point did not, and the operating point is the part everyone ships and nobody
revalidates.

It is not subtle in deployment either. Scoring the shipped model on the held-out
test split and breaking it out by unit: **60% of non-septic MICU admissions raise at
least one alert, against 12% of non-septic SICU admissions, at the same threshold.**
Two units, one number, two completely different experiences of the same system. The
staff in one of them would turn it off within a week.

## Now the other boundary, which fails the opposite way

The same project has an external hospital — a different health system, held back
from everything: fitting, tuning, calibration, threshold selection. Transferring
there costs 0.130 utility (95% CI 0.066–0.182).

Same diagnostic, opposite answer:

- **Re-picking the threshold on the new hospital buys back about 5% of the gap.**
  Hospital A's threshold was already very nearly the right threshold for hospital B.
- I reweighted hospital B's admissions to match hospital A's baseline case mix — a
  trimmed propensity density ratio over 17 admission-level covariates, which cut mean
  covariate imbalance from 0.220 to 0.075 |SMD| while retaining an effective 53% of
  20,000 admissions. That explains **+0.017 of the 0.130 gap, with a confidence
  interval straddling zero.**

Neither comfortable explanation survives. Not "different patients." Not "stale
operating point." Across hospitals, the scores themselves got worse.

Put the two boundaries side by side:

| | discrimination | operating point |
|---|---|---|
| MICU → SICU | survives | catastrophic |
| Hospital A → B | degrades | fine |

**A single "external AUROC" column would have shown neither.** It is the wrong
summary statistic for the question people actually ask, which is not "does it rank"
but "does it still work."

## A candidate mechanism for the hospital gap

Why would the scores themselves degrade between health systems? The same repository
has an ablation that suggests an uncomfortable answer.

Of 345 features, 109 contain no measured value whatsoever — only which channel was
sampled, how recently, and how often. No physiology at all. On that subset alone the
model reaches **AUROC 0.794, which is 97% of what the full matrix achieves.**

The model is substantially reading what the care team chose to measure. Ordering a
lactate is an act of suspicion, and suspicion is predictive. That works beautifully
until you move somewhere with different charting habits, which is exactly what a
different health system is.

I do not think this is a flaw peculiar to my model. Every clinical time-series model
trained on real charting data has this exposure. Most do not measure it.

## Three things worth doing

1. **Report discrimination and the operating point separately.** They fail
   independently, which is the whole finding. One AUROC number hides both failure
   modes because it is insensitive to the one that costs you everything.
2. **Re-pick the threshold at every site, unit, and population you deploy into.** It
   is one threshold sweep on local data. In my case it was worth more than any
   modelling decision in the entire project.
3. **Publish the alarm burden by subgroup.** "60% versus 12% of non-septic
   admissions" is a fact about whether anyone will keep the system switched on. It
   never appears in a results table, and it is the number that decides adoption.

## How the numbers were produced

Every one of them is generated by the pipeline, not transcribed, and pinned by a
regression check that fails if a published value moves without its baseline moving
too.

- **Splits are by admission, never by ICU hour.** Splitting hours at random instead
  inflates AUROC by +0.0275 and clinical utility by +0.0796 on this cohort. I know
  that because the repository commits the mistake deliberately and measures it.
- **Every feature at hour *t* depends only on hours ≤ *t*,** enforced by rebuilding
  features on truncated stays and requiring bit-identical rows.
- **Confidence intervals resample whole admissions,** not hours. Forty correlated
  rows from one stay are not forty observations.
- **The transfer direction was fixed before the numbers existed** (train MICU, test
  SICU), so the better-looking direction could not be chosen afterwards. Admissions
  with no recorded unit are reported as a third bucket rather than dropped — in this
  cohort they are the largest group of the three.
- **The reweighting asserts that it worked** before publishing: balance must improve,
  propensity AUC must stay below 0.99, effective sample size above 5%.

## What this is not

One retrospective cohort, PhysioNet/CinC 2019. No prospective validation: no patient
outcome has ever been affected by this model, so nothing here says what happens when
clinicians can see it. The labels are Sepsis-3 clinical suspicion shifted six hours
earlier, not biology — the model learns to anticipate when a care team will act. The
dataset carries no calendar time, so temporal validation is not buildable on it, and
no race, ethnicity, insurance status or language, so the disparities most often asked
about cannot be assessed at all.

The two-boundary result is a finding about this cohort. Whether it generalises is
exactly the kind of claim this post is arguing against making from one number.

---

Code, data pipeline, the experiments that produced these numbers, and a model card:
**https://github.com/poglesbyg/sepsis-early-warning**

The full report, including an interactive replay of individual admissions hour by
hour: **https://poglesbyg.github.io/sepsis-early-warning/**
