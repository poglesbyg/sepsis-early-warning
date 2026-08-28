# TODOS

Deferred work, with enough context to pick up cold.

---

## Generalize the leakage experiment into a reusable harness

**Status:** deferred.

**What:** the leakage ablation in this repository measures two specific mistakes
on one dataset. Generalized, it becomes a harness anyone can point at their own
longitudinal dataset to estimate how much their split strategy inflates their
reported score.

**Why deferred:** it is a separate tool with a separate audience, and building it
now would pull focus before the in-repo version has proven useful.

**Depends on:** the leakage ablation existing first.

**Revisit when:** the ablation has landed and the measured inflation is known. If
the number is large and stable, the harness is worth building; if it is small,
the case for a general tool is weaker.

**Now known.** Splitting rows instead of admissions inflates AUROC by +0.0275 and
clinical utility by +0.0796 on this cohort — the utility movement is roughly three
times the AUROC movement, and that ratio is the part a general tool would exist to
show. It clears the revisit bar. What it does not settle is whether the number
holds past one dataset; a harness that reported one cohort's inflation as though
it were a constant would be its own failure mode.

---

## Interactive cost-ratio explorer in the published report

**Status:** deferred.

**What:** a control for the false-alarm-to-missed-case cost ratio. Dragging it
moves the operating point and updates alerts per day and catches per week, making
the alarm-fatigue trade-off tangible rather than tabular.

**Already built:** `src/sepsis/models/calibration.py::expected_cost_threshold`
computes exactly this sweep. The remaining work is a pre-computed sweep plus a
slider. No server required.

**Why deferred:** the replay demo is already the report's interactive centrepiece,
and two interactives on one page compete for the same attention.

---

## FastAPI serving layer and container

**Status:** cut, recorded so the reasoning is not lost.

**Why cut:** a service wrapper is the most generatable artifact in an ML
repository and carries little information about the work. What survived instead is
a minimal documented inference contract.

**If it returns, one design decision is unresolved:** what is the inference
contract — a batch of hours, or one hour at a time with carried state? The
gradient booster is stateless. The GRU is not. That difference changes the API
shape and should be settled before anything is written.

---

## Temporal validation on a cohort that supports it

**Status:** not possible here; recorded as a known limitation.

PhysioNet/CinC 2019 carries no calendar time, so admission-ordered splitting is
not buildable on this data. A cohort with real admission timestamps (MIMIC-IV,
eICU) would support it, but both require credentialed access and amount to a
separate project. See `docs/designs/experiment-layer.md` for why the
`patient_id`-as-time workaround was rejected.
