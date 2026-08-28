"""The regression check's own guardrails.

The check exists so that a code change cannot move a published number quietly. The
failure that matters is therefore the one where a number moves and the check still
passes, so most of what is asserted below is that it does *not* do that.
"""

from __future__ import annotations

import json

import pytest

from sepsis.config import Config
from sepsis.regress import (
    TOLERANCE,
    check,
    collect,
    compare,
    flatten,
    write_baseline,
)


def _cfg(tmp_path, payload: dict) -> Config:
    cfg = Config(root=tmp_path)
    cfg.ensure_dirs()
    (cfg.reports_dir / "metrics.json").write_text(json.dumps(payload))
    return cfg


# --------------------------------------------------------------------------
# Flattening: everything published must end up comparable
# --------------------------------------------------------------------------
def test_flatten_reaches_nested_values_and_list_elements():
    flat = flatten({"a": {"b": 1.0}, "ci": [0.1, 0.2]}, "m")
    assert flat == {"m.a.b": 1.0, "m.ci[0]": 0.1, "m.ci[1]": 0.2}


def test_flatten_keeps_strings_because_they_are_published_claims_too():
    """`most_costly_to_drop: "recency"` is quoted in the README exactly as much as
    the AUROC beside it, and it can change for the same reasons."""
    assert flatten({"most_costly_to_drop": "recency"}) == {"most_costly_to_drop": "recency"}


def test_flatten_drops_nulls_rather_than_comparing_them():
    assert flatten({"a": None, "b": 2}) == {"b": 2}


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
def test_a_moved_number_is_caught():
    result = compare({"x": 0.8206}, {"x": 0.8246})
    assert result["moved"] and result["moved"][0][0] == "x"
    assert result["moved"][0][3] == pytest.approx(0.004)


def test_movement_inside_the_tolerance_is_not_a_regression():
    assert not compare({"x": 0.8206}, {"x": 0.8206 + TOLERANCE / 10})["moved"]


def test_a_number_that_stopped_being_published_is_caught():
    """Deleting a metric is not a way to pass the check."""
    assert compare({"x": 1.0, "y": 2.0}, {"x": 1.0})["missing"] == ["y"]


def test_a_changed_string_is_caught():
    moved = compare({"k": "recency"}, {"k": "clinical"})["moved"]
    assert moved and moved[0][:3] == ("k", "recency", "clinical")


def test_a_number_that_became_a_string_is_caught():
    assert compare({"x": 1.0}, {"x": "n/a"})["moved"]


def test_new_values_are_reported_but_do_not_fail():
    """Adding an experiment is not a regression."""
    result = compare({"x": 1.0}, {"x": 1.0, "new": 2.0})
    assert result["added"] == ["new"] and not result["moved"] and not result["missing"]


# --------------------------------------------------------------------------
# End to end against real artifact files
# --------------------------------------------------------------------------
def test_baseline_round_trip_passes(tmp_path):
    cfg = _cfg(tmp_path, {"auroc": 0.8, "nested": {"utility": 0.4}})
    write_baseline(cfg, quiet=True)
    assert check(cfg, quiet=True)["moved"] == []


def test_check_fails_after_an_artifact_moves(tmp_path):
    cfg = _cfg(tmp_path, {"auroc": 0.8})
    write_baseline(cfg, quiet=True)
    (cfg.reports_dir / "metrics.json").write_text(json.dumps({"auroc": 0.9}))

    with pytest.raises(ValueError, match="moved by more than"):
        check(cfg, quiet=True)


def test_check_without_a_baseline_says_how_to_make_one(tmp_path):
    cfg = _cfg(tmp_path, {"auroc": 0.8})
    with pytest.raises(FileNotFoundError, match="regress-update"):
        check(cfg, quiet=True)


def test_collect_refuses_to_verify_nothing(tmp_path):
    """An empty reports directory must not read as 'no regressions'."""
    cfg = Config(root=tmp_path)
    cfg.ensure_dirs()
    with pytest.raises(FileNotFoundError, match="no published artifacts"):
        collect(cfg)
