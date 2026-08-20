# Design: experiment layer and evidence for sepsis-early-warning

Records the decisions behind the next phase of work, the alternatives that were
rejected, and one accepted item that turned out to be infeasible. Written after a
structured review plus an independent cross-model review (Codex, codex-cli
0.148.0).

## Problem

The repository already produces a defensible model and an honest report. What it
does not do is *demonstrate* its own claims. "No leakage" is asserted and tested
in code, but never quantified. "Each feature block earns its place" is implied by
345 features existing, never measured. The external-validation drop from
hospital A to hospital B is reported but not decomposed. This phase converts
claims into numbers.

## Approaches considered

| | Summary | Effort (CC) | Completeness |
|---|---|---|---|
| A | Repackage documentation only; no new results | ~20 min | 4/10 |
| B | A + quantify the leakage claim and the feature-block contributions | ~45 min | 8/10 |
| C | B + a pre-computed replay demo, an inference contract, a model card | ~8-10 h | 10/10 |

**Chosen: C.** B is the highest value per unit of effort and that was recorded at
decision time; C was chosen because the replay makes 33 hours of lead time legible
in about ten seconds, which no table does.

## Decisions

### Feature-block ablation and leakage ablation

Two distinct mistakes are measured **separately and never conflated**:

1. splitting rows instead of admissions
2. non-causal (lookahead) features

Mistake 1 is the common one in the wild, so it carries the point. Each experiment
asserts its own invariants and fails loudly rather than publishing a plausible
wrong number: the leaky variant must score at least as high as the honest one,
and the feature blocks must partition all 345 columns exactly, with no overlap
and no orphans. A non-partitioning block set produces meaningless ablation
numbers and raises nothing, which is the failure class this repository exists to
argue against.

### Temporal validation: accepted, then found infeasible

An earlier version of this plan called for splitting hospital A by admission time
rather than at random, on the grounds that random patient splits assume the future
resembles the past.

**PhysioNet/CinC 2019 carries no calendar time.** The only time-like columns are
`HospAdmTime` (hours between hospital and ICU admission, per-patient relative) and
`ICULOS` (hours since ICU admission). Verified directly against the raw parquet
columns. There is no admission date, so a temporal split is not buildable on this
cohort.

The tempting workaround is to treat the sequential `patient_id` ordering as a
proxy for admission order. **Rejected.** The cohort is de-identified, the ordering
is undocumented, and if the identifiers were randomized the result is a random
split wearing a temporal label. Publishing an unverifiable rigor claim is worse
than publishing no claim.

**Substituted:** a MICU-to-SICU transfer split (`Unit1` to `Unit2`). A real
covariate shift the data supports, with an honest name. Admissions where the unit
indicators are missing get an explicit third bucket rather than silent exclusion.
Additive, not a replacement: the random-split numbers stay published.

### Shift experiments must state their estimand first

Hospital B's utility drop is not attributable to model degradation from prevalence
alone, and "case mix versus degradation" is not identifiable without saying what
is held fixed. Both shift experiments write their estimand into `DECISIONS.md`
before any code is written: the reference distribution, whether calibration is
refit, how uncertainty is carried, and for the unit split, the exact transfer
claim being made.

### Illustrative cases come from validation, never test

The replay demo and the failure gallery both display individual admissions. Those
are selected from the **validation** split, which was already spent on tuning and
threshold selection. Selecting them from the test split would convert a held-out
set into a presentation set: the reported metrics would remain arithmetically
correct, but a human would have inspected test cases to choose what to show.

### Naming

Two names in an earlier draft overclaimed and were corrected:

- The truncation-rebuild test proves **no lookahead**, not causality. It verifies
  implementation timing, not a causal relationship.
- `make regress` is a **regression check**, not reproducibility. It cannot
  guarantee a third party reproduces these numbers across package versions,
  hardware, or solver behaviour. It catches code changes silently moving published
  results, which is a narrower and honest claim.

### Data integrity

A file count detects a truncated download but not corruption or a silent upstream
revision. PhysioNet publishes no per-file checksums for this release: its
`SHA256SUMS.txt` is three lines covering `LICENSE.txt` and two SVG diagrams. The
integrity check therefore computes a rolled-up sha256 per hospital over the sorted
per-file digests, commits it, and verifies against it on rebuild.

### Experiment integration

Five experiments would each touch the pipeline, the report generator, the plotting
module and the HTML builder: twenty edit points, repeated for every future
experiment. A full self-describing registry was considered and **rejected as
premature** — it is an abstraction and a schema built for experiments that may
never exist. Settled on a shared fit-and-score helper plus an explicit
registration list, which removes the real duplication without the indirection.

### Serving

A FastAPI service, Dockerfile and container were considered and cut: a service
wrapper is the most generatable artifact in an ML repository and signals little.
What survived is a **minimal inference contract** — one documented module with a
typed `predict()` and an explicit statement of the batch-versus-streaming
semantics, including the fact that the booster is stateless while the GRU is not.
That distinction is a real design constraint and worth stating; the container is
not.

## Sequencing

Documentation and evidence land before the experiment layer, and the experiment
layer is gated on the documentation actually being read. Acceptance criteria for
the first phase:

- the published report is reachable without credentials
- a clean clone reproduces `metrics.json`
- the report builds from a clean checkout
- the core claim is readable in under a minute

## Not in scope

| Item | Reason |
|---|---|
| Temporal / prospective validation | Infeasible; no calendar time in this cohort |
| `patient_id` order as a time proxy | Unverifiable; a rigor claim that cannot be defended |
| FastAPI service, Dockerfile, container | Generatable, low signal; a documented inference contract covers the real content |
| Reusable leakage harness for other datasets | Deferred; a separate project with a separate audience. See `TODOS.md` |
| Interactive cost-ratio explorer | Deferred; the replay is already the report's interactive centrepiece |
| Any new model family | The gap is evidence, not modelling |

## Open risk

The premise behind this phase is that the repository is limited by how its
existing evidence is presented rather than by what it contains. That premise is
plausible and untested. If the documentation phase lands and nothing changes, the
constraint is elsewhere and the remaining work is the wrong lever.
