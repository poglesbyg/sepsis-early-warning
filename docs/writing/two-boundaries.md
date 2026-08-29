# The model transferred. The alert threshold didn't.

I trained a sepsis early-warning model on medical ICU admissions and tested it on
surgical ICU admissions in the same hospital.

The model's behaviour barely changed. Mean predicted risk was 0.342 in the MICU and
0.335 in the SICU. It raised an alert on 76% of MICU admissions and 75% of SICU
admissions, at the same threshold. The AUROC difference was +0.024, with a 95%
interval from −0.013 to +0.063 — this experiment does not establish that ranking got
worse.

Clinical utility fell from **0.469 to 0.088**, with a confidence interval on the
SICU number running from −0.059 to 0.188.

Zero on that scale means *never alerting at all*. The interval includes it.

## What that number is, and what it is not

The metric is the PhysioNet/CinC 2019 clinical utility score. It is time-dependent
and asymmetric: a false alarm costs a little, a missed septic hour costs a lot, and
a correct alert is worth most about six hours before onset. It is normalised so 1.0
is a label-informed oracle policy — alert exactly when alerting is worth more than
silence — and 0.0 is never alerting.

**That denominator is cohort-specific and computed from the outcomes.** So part of
what happens between these two units is the metric responding to a different septic
rate, not the model behaving differently. I want to be exact about this, because it
is the first thing a statistician will say: this is not a prevalence-invariant
measure of deployable benefit, and "lost 91% of its value" would be the wrong way to
describe it.

Here is what the score-distribution numbers rule out, though. The model is not
scoring SICU patients differently. It is not alerting more often. It is doing almost
exactly what it did at home, and the value of doing that is completely different when
the septic rate is 4.0% instead of 10.8%. The same alerting behaviour buys far less
and costs the same.

## The threshold is the part that didn't transfer

Choosing a new threshold on one half of the SICU and scoring it on the other half
takes utility from 0.088 to **0.408** — a gain of +0.319, 95% CI +0.243 to +0.444.

Two things about that number, both of which I got wrong in the first draft of this
post.

**It is held out.** An earlier version chose the threshold on the SICU and reported
the resulting utility on those same admissions, which is the maximum of a sweep
dressed up as a measurement. Splitting the bucket in half fixes it. The check that
this is now honest: run the identical procedure on the MICU bucket, where the
threshold was already tuned, and the gain is +0.000 with an interval spanning zero.
It only finds a recovery where one exists.

**It is not free.** Utility is optimised against labelled outcomes, so re-picking a
threshold requires the new unit's own labelled sepsis cases. It needs no retraining
and no new features, which is genuinely cheap. It is not "just a threshold sweep,"
and I said that it was.

## The other boundary fails the other way

The same project has an external hospital — a different health system, held back from
fitting, tuning, calibration and threshold selection. Transferring there costs 0.130
utility (95% CI 0.066–0.182).

Same two diagnostics, opposite answers:

- **Re-picking the threshold at hospital B buys back about 5% of the gap.** Hospital
  A's threshold was already nearly right for hospital B.
- **Reweighting hospital B's admissions to match hospital A's case mix explains
  +0.017 of the gap, with an interval straddling zero** — and only +0.006 if the
  adjustment uses covariates fixed at admission rather than six hours of early care.

So across units, the model transfers and the threshold does not. Across hospitals,
the threshold transfers and something in the scores does not. One "external AUROC"
column would have shown neither.

I want to flag the limit of that framing: these are two separately trained models on
two different cohorts with two different analyses. They are two observations, not an
identified pair of orthogonal failure modes. What they jointly establish is narrower
and still useful — **discrimination and the operating point can fail independently,
so reporting one number that is insensitive to the second is not enough.**

## Is it ordering behaviour? I tested it rather than asserting it

There is a tempting explanation for the hospital gap. In this feature matrix, 109 of
345 features contain no measured value at all — only which channel was sampled, how
recently, how often. On that subset alone the model reaches AUROC 0.794, 97% of what
the full matrix achieves. Ordering a lactate is an act of suspicion, and suspicion is
predictive. Charting habits differ between health systems. The story writes itself.

Stories that fit the numbers are the dangerous kind, so I ran the test: two models on
hospital A, identical except one never sees those 109 features, both scored at both
hospitals.

| | AUROC, hospital A | AUROC, hospital B | transfer gap |
|---|---|---|---|
| everything (345 features) | 0.823 | 0.787 | 0.036 |
| no ordering behaviour (236) | 0.817 | 0.792 | 0.025 |

Withholding ordering behaviour **shrinks the transfer gap by 0.011, 95% CI +0.0004 to
+0.021.** The interval excludes zero and sits right against it: the direction holds,
the size is not established by this.

The pattern is the interesting part. The model that cannot see ordering behaviour is
*worse at home and better away*. Those features are not noise to regularise out.
They are real signal about a real thing — what the care team was worried about — and
that signal is partly what does not travel.

## What I'd take from this

1. **Report discrimination and the operating point separately.** They can fail
   independently. A single AUROC is insensitive to the failure that costs you the
   deployment.
2. **Budget for local threshold selection at every site and unit,** including the
   labelled outcomes it requires. In this project it was worth more than any
   modelling decision.
3. **Test your mechanism story.** Mine survived, barely, and I would not have known
   which way it went. The ablation result was true and I was still one experiment
   away from knowing whether it explained anything.

## How the numbers were produced

Generated by the pipeline, not transcribed, and pinned by a regression check that
fails if any published value moves without its baseline moving too.

- **Splits are by admission, never by ICU hour.** Splitting hours at random inflates
  AUROC by +0.0275 and utility by +0.0796 on this cohort — measured by committing the
  mistake deliberately.
- **Every feature at hour *t* depends only on hours ≤ *t*,** enforced by rebuilding
  features on truncated stays and requiring bit-identical rows.
- **Intervals resample whole admissions.** The AUROC difference across units uses an
  independent two-sample bootstrap, because the cohorts are different patients and
  DeLong assumes they are not. The utility comparison is paired, since both
  thresholds score the same held-out admissions.
- **The transfer direction was fixed in advance** (train MICU, test SICU). Admissions
  with no recorded unit are reported as a third bucket — the largest of the three.

## What this is not

One retrospective cohort, PhysioNet/CinC 2019. No prospective validation: no patient
outcome has ever been affected by this model, so nothing here says what happens when
clinicians can see it — including whether acting on it changes the ordering behaviour
it partly depends on. The labels are Sepsis-3 clinical suspicion shifted six hours
earlier, not biology. The dataset carries no calendar time, so temporal validation is
not buildable, and no race, ethnicity, insurance status or language, so the
disparities most often asked about cannot be assessed at all.

An earlier draft of this post claimed more than the analyses supported in four
separate places. I found that out by having a second model review it against the
repository before publishing, which I recommend more than anything else here.

---

Code, experiments, and a model card: **https://github.com/poglesbyg/sepsis-early-warning**

The full report, with an interactive replay of individual admissions hour by hour:
**https://poglesbyg.github.io/sepsis-early-warning/**
