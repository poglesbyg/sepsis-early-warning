"""Catch code changes that silently move a published number.

This is **not** a reproducibility guarantee, and calling it one would be the kind
of overclaim the rest of this repository argues against. Nothing here says a third
party will reproduce these numbers across package versions, hardware or solver
behaviour. What it says is narrower and checkable: on one machine, a change to the
code either leaves every published number where it was, or it does not, and if it
does not you find out from a failing check rather than from a reader noticing that
the README and the report disagree.

The mechanism is deliberately dull. Every number this repository publishes already
lands in a JSON file under ``reports/``. Those files are flattened to
``file.path.to.value`` keys and compared against a committed baseline. A number
that moves shows up as a failure; a number that stops being published shows up as
a failure; a number that is new shows up as a note, because adding an experiment
is not a regression.

Two consequences worth stating:

* **The baseline is committed**, so moving a published number is a reviewable diff
  in version control rather than a silent edit. ``--update`` is the only way to
  move it, and it is meant to be run deliberately and read in the pull request.
* **The check reads artifacts, it does not recompute them.** It is worth seconds,
  not hours, and it verifies whatever the last pipeline run wrote. Run it after
  the stage whose numbers you expect to have left alone.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import CFG, Config

BASELINE = "regression_baseline.json"

# Absolute tolerance. XGBoost, sklearn and numpy are deterministic here given a
# fixed seed on fixed data, so a number that moves at all moved because the code
# changed. The tolerance exists for the last bits of floating-point summation
# order, not to wave through a real difference.
TOLERANCE = 1e-6


def baseline_path(cfg: Config = CFG) -> Path:
    return cfg.root / "configs" / BASELINE


def published_files(cfg: Config = CFG) -> list[Path]:
    """Every JSON artifact the report and README quote from, in a stable order."""
    from .experiments import REGISTRY

    candidates = [cfg.reports_dir / "metrics.json", cfg.reports_dir / "replay_summary.json"]
    candidates += [cfg.reports_dir / f"experiment_{name}.json" for name in REGISTRY]
    return [p for p in candidates if p.exists()]


def flatten(value, prefix: str = "") -> dict[str, object]:
    """Every scalar leaf, keyed by its dotted path.

    Strings come along with the numbers. ``most_costly_to_drop: "recency"`` is a
    published claim in the README exactly as much as the AUROC beside it is, and it
    can change for the same reasons.
    """
    out: dict[str, object] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            out |= flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            out |= flatten(v, f"{prefix}[{i}]")
    elif isinstance(value, (int, float, str, bool)):
        out[prefix] = value
    return out


def collect(cfg: Config = CFG) -> dict[str, object]:
    """Flatten every published artifact into one comparable mapping."""
    files = published_files(cfg)
    if not files:
        raise FileNotFoundError(
            f"no published artifacts in {cfg.reports_dir}; run `make all` (or at "
            f"least `make evaluate`) before checking for regressions"
        )
    numbers: dict[str, object] = {}
    for path in files:
        numbers |= flatten(json.loads(path.read_text()), path.stem)
    return numbers


def compare(
    baseline: dict[str, object], current: dict[str, object], tolerance: float = TOLERANCE
) -> dict[str, list]:
    """Split the comparison into moved, missing and added."""
    moved, missing = [], []
    for key, was in baseline.items():
        if key not in current:
            missing.append(key)
            continue
        now = current[key]
        if isinstance(was, bool) or isinstance(now, bool) or not isinstance(was, (int, float)):
            if was != now:
                moved.append((key, was, now, None))
        elif not isinstance(now, (int, float)) or abs(now - was) > tolerance:
            delta = (now - was) if isinstance(now, (int, float)) else None
            moved.append((key, was, now, delta))

    added = [k for k in current if k not in baseline]
    return {"moved": moved, "missing": missing, "added": added}


def write_baseline(cfg: Config = CFG, quiet: bool = False) -> Path:
    path = baseline_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    numbers = collect(cfg)
    path.write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n")
    if not quiet:
        print(f"[regress] pinned {len(numbers):,} published values from "
              f"{len(published_files(cfg))} artifacts to {path.name}", flush=True)
    return path


def check(cfg: Config = CFG, tolerance: float = TOLERANCE, quiet: bool = False) -> dict:
    """Compare the current artifacts against the committed baseline."""
    path = baseline_path(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"no baseline at {path}. Create one with `make regress-update` once the "
            f"current numbers are the ones you intend to publish."
        )

    baseline = json.loads(path.read_text())
    result = compare(baseline, collect(cfg), tolerance)

    if not quiet:
        print(f"[regress] checked {len(baseline):,} published values "
              f"(tolerance {tolerance:g})", flush=True)
        for key in result["added"]:
            print(f"[regress] new, not yet pinned: {key}", flush=True)

    if result["moved"] or result["missing"]:
        raise ValueError(_describe(result, tolerance))
    if not quiet:
        print("[regress] every published number is where the baseline left it", flush=True)
    return result


def _describe(result: dict, tolerance: float) -> str:
    lines = []
    if result["moved"]:
        lines.append(f"{len(result['moved'])} published value(s) moved by more than {tolerance:g}:")
        for key, was, now, delta in sorted(result["moved"])[:20]:
            shift = f" ({delta:+.3g})" if delta is not None else ""
            lines.append(f"  {key}: {was!r} -> {now!r}{shift}")
        if len(result["moved"]) > 20:
            lines.append(f"  ... and {len(result['moved']) - 20} more")
    if result["missing"]:
        lines.append(f"{len(result['missing'])} value(s) are no longer published at all:")
        lines += [f"  {k}" for k in sorted(result["missing"])[:20]]
    lines.append(
        "\nIf the change was intended, re-run the affected stage and then "
        "`make regress-update`, so the new numbers land in the baseline as a "
        "reviewable diff rather than silently."
    )
    return "\n".join(lines)
