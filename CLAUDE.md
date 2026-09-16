# sepsis-early-warning

Early sepsis prediction from ICU time series (PhysioNet/CinC Challenge 2019).

## Invariants — do not break these

- **No lookahead.** Every feature at hour `t` depends only on hours `<= t`.
  No `bfill`, no whole-stay statistics, no centred windows, no bidirectional layers.
  `tests/test_features.py` enforces this with [nopeek](https://github.com/poglesbyg/nopeek),
  which hides the future, rebuilds from scratch and requires the surviving rows
  to be unchanged -- and separately rebuilds each admission in isolation, so a
  window that spans two stays fails too. ("Causal" elsewhere in the codebase
  carries its signal-processing sense: depending only on past inputs.)

  One known exception is recorded as a strict `xfail` there: `rolling_summaries`
  rolls the whole patient-sorted array in a single pass, so pandas' running
  accumulator carries floating-point state across admission boundaries. The
  blanking removes every contaminated *window* but not that state. Real-data
  magnitude is around 1e-7 -- clinically irrelevant, and not a reason to slow the
  builder down -- but it is a genuine dependence and is written down rather than
  tolerated silently.
- **Splits are by admission, never by hour.** Rows within a stay are strongly
  autocorrelated. This applies to train/val/test, CV folds, and the bootstrap.
- **Hospital B is external.** It is never used for fitting, tuning, calibration,
  threshold selection, or blend weights.
- **Statistics use one observation per admission**, not one per ICU hour.

## Layout

`src/sepsis/{data,features,stats,models,evaluate}` — pipeline stages orchestrated
by `pipeline.py`, driven by `cli.py`. `make all` runs everything end to end.

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Design system/plan review → invoke /design-consultation or /plan-design-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke /investigate
- QA/testing site behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Visual polish → invoke /design-review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore
- Author a backlog-ready spec/issue → invoke /spec
