# sepsis-early-warning

Early sepsis prediction from ICU time series (PhysioNet/CinC Challenge 2019).

## Invariants — do not break these

- **No lookahead.** Every feature at hour `t` depends only on hours `<= t`.
  No `bfill`, no whole-stay statistics, no centred windows, no bidirectional layers.
  `tests/test_features.py` enforces this by rebuilding on truncated stays and
  requiring bit-identical rows. ("Causal" elsewhere in the codebase carries its
  signal-processing sense: depending only on past inputs.)
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
