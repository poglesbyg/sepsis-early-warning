"""The replay's own guardrails.

The replay is the one place in this project where a reader sees individual
patients rather than an aggregate, which makes it the easiest place to mislead:
show the best case, caption a miss as a catch, or quietly illustrate the argument
with admissions from the test split. Every assertion here exists to make one of
those loud.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sepsis.replay import ROLES, _assert_roles, assert_not_from_test, select_cases


def _timing(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("patient_id")


def _cohort() -> pd.DataFrame:
    """A validation cohort with a spread of lead times, a miss and a false alarm."""
    rows = [
        {"patient_id": f"p{i:03d}", "septic": True, "stay_hours": 40 + i,
         "onset_hour": 30.0, "first_alert_hour": 30.0 - lead,
         "alerted": True, "n_alert_hours": 5, "lead_time_hours": float(lead)}
        for i, lead in enumerate([1, 12, 24, 48, 96])
    ]
    rows.append(
        {"patient_id": "p900", "septic": True, "stay_hours": 200, "onset_hour": 30.0,
         "first_alert_hour": 34.0, "alerted": True, "n_alert_hours": 1,
         "lead_time_hours": -4.0}
    )
    rows.append(
        {"patient_id": "p901", "septic": False, "stay_hours": 60, "onset_hour": np.nan,
         "first_alert_hour": 2.0, "alerted": True, "n_alert_hours": 55,
         "lead_time_hours": np.nan}
    )
    rows.append(
        {"patient_id": "p902", "septic": False, "stay_hours": 60, "onset_hour": np.nan,
         "first_alert_hour": 9.0, "alerted": True, "n_alert_hours": 3,
         "lead_time_hours": np.nan}
    )
    return _timing(rows)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------
def test_headline_case_is_the_median_catch_not_the_best_one():
    """The whole credibility of the section rests on this one line."""
    chosen = select_cases(_cohort())
    timing = _cohort()

    caught = timing[timing["lead_time_hours"] > 0]
    best = caught["lead_time_hours"].idxmax()
    median = caught["lead_time_hours"].median()

    assert chosen["median_catch"] != best, "the demo must not show the best case"
    assert timing.loc[chosen["median_catch"], "lead_time_hours"] == median


def test_selection_includes_two_failures():
    chosen = select_cases(_cohort())
    timing = _cohort()

    assert timing.loc[chosen["missed"], "septic"]
    assert not timing.loc[chosen["missed"], "lead_time_hours"] > 0
    assert not timing.loc[chosen["false_alarm"], "septic"]


def test_false_alarm_is_the_worst_one_available():
    chosen = select_cases(_cohort())
    assert chosen["false_alarm"] == "p901", "should pick the most alert-hours control"


def test_selection_is_deterministic():
    assert select_cases(_cohort()) == select_cases(_cohort())


def test_selection_covers_every_published_role():
    assert set(select_cases(_cohort())) == set(ROLES)


def test_selection_refuses_a_cohort_with_nothing_caught():
    timing = _timing([
        {"patient_id": "p0", "septic": True, "stay_hours": 40, "onset_hour": 10.0,
         "first_alert_hour": np.nan, "alerted": False, "n_alert_hours": 0,
         "lead_time_hours": np.nan}
    ])
    with pytest.raises(ValueError, match="caught before onset"):
        select_cases(timing)


def test_selection_refuses_a_cohort_with_no_failure_to_show():
    """A gallery of nothing but successes is an advertisement, not evidence."""
    timing = _timing([
        {"patient_id": f"p{i}", "septic": True, "stay_hours": 40, "onset_hour": 30.0,
         "first_alert_hour": 10.0, "alerted": True, "n_alert_hours": 5,
         "lead_time_hours": 20.0}
        for i in range(3)
    ])
    with pytest.raises(ValueError, match="missed septic admission"):
        select_cases(timing)


# --------------------------------------------------------------------------
# Captions must match the data
# --------------------------------------------------------------------------
def _case(role, **over):
    base = {
        "role": role, "patient_id": "p1", "septic": True, "hours": [0, 1, 2],
        "risk": [0.01, 0.02, 0.09], "onset_hour": 2.0, "first_alert_hour": 1.0,
        "lead_time_hours": 1.0, "n_alert_hours": 2, "stay_hours": 3,
    }
    base.update(over)
    return base


def _gallery(**over):
    cases = [
        _case("median_catch"),
        _case("near_miss"),
        _case("missed", lead_time_hours=-3.0, first_alert_hour=5.0),
        _case("false_alarm", septic=False, onset_hour=None, lead_time_hours=None),
    ]
    for role, changes in over.items():
        for c in cases:
            if c["role"] == role:
                c.update(changes)
    return cases


def test_a_consistent_gallery_passes():
    _assert_roles(_gallery())


def test_a_catch_that_did_not_catch_is_rejected():
    with pytest.raises(ValueError, match="published as caught"):
        _assert_roles(_gallery(median_catch={"lead_time_hours": -2.0}))


def test_a_miss_that_was_actually_caught_is_rejected():
    with pytest.raises(ValueError, match="published as a miss"):
        _assert_roles(_gallery(missed={"lead_time_hours": 9.0}))


def test_a_false_alarm_that_was_actually_septic_is_rejected():
    with pytest.raises(ValueError, match="published as a false alarm"):
        _assert_roles(_gallery(false_alarm={"septic": True}))


def test_a_missing_role_is_rejected():
    with pytest.raises(ValueError, match="missing cases"):
        _assert_roles(_gallery()[:3])


def test_a_ragged_series_is_rejected():
    with pytest.raises(ValueError, match="disagree"):
        _assert_roles(_gallery(near_miss={"risk": [0.1, 0.2]}))


def test_a_hole_in_the_risk_series_is_rejected():
    with pytest.raises(ValueError, match="holes"):
        _assert_roles(_gallery(near_miss={"risk": [0.1, float("nan"), 0.2]}))


# --------------------------------------------------------------------------
# The split invariant
# --------------------------------------------------------------------------
def test_illustrative_cases_may_not_come_from_test(tmp_path):
    from sepsis.config import Config

    cfg = Config(root=tmp_path)
    cfg.ensure_dirs()
    for split, ids in (("test", ["p1", "p2"]), ("external", ["p9"])):
        pd.DataFrame({"patient_id": ids}).to_parquet(cfg.processed_dir / f"{split}.parquet")

    assert_not_from_test({"cases": [{"patient_id": "p7"}]}, cfg)
    with pytest.raises(ValueError, match="from the test split"):
        assert_not_from_test({"cases": [{"patient_id": "p1"}]}, cfg)
    with pytest.raises(ValueError, match="from the external split"):
        assert_not_from_test({"cases": [{"patient_id": "p9"}]}, cfg)


def test_the_shipped_payload_obeys_its_own_invariants():
    """The committed payload, checked rather than assumed."""
    import json
    from sepsis.config import CFG

    path = CFG.reports_dir / "replay.json"
    if not path.exists():
        pytest.skip("replay payload not built in this checkout")

    payload = json.loads(path.read_text())
    assert payload["split"] == "val"
    _assert_roles(payload["cases"])
    assert_not_from_test(payload)
