"""Model-side behaviour: fold hygiene, calibration, blending, alert timing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from sepsis.evaluate.lead_time import alert_timing, lead_time_summary
from sepsis.models.calibration import Calibrator, reliability_curve
from sepsis.models.common import downsample_negatives, grouped_folds, scale_pos_weight
from sepsis.models.ensemble import blend, optimise_weights, rank_normalise


def _cohort(n_patients=300, seed=0):
    rng = np.random.default_rng(seed)
    y, groups = [], []
    for i in range(n_patients):
        n = int(rng.integers(10, 60))
        yy = np.zeros(n, dtype=np.int8)
        if rng.random() < 0.25:
            yy[int(rng.integers(n // 2, n)) :] = 1
        y.append(yy)
        groups.append(np.full(n, f"p{i:05d}"))
    return np.concatenate(y), np.concatenate(groups)


# --------------------------------------------------------------------------
# Cross-validation hygiene
# --------------------------------------------------------------------------
def test_no_admission_spans_a_fold_boundary():
    y, g = _cohort()
    for train_idx, test_idx in grouped_folds(y, g, n_splits=5, seed=0):
        assert not (set(g[train_idx]) & set(g[test_idx]))


def test_folds_cover_every_row_exactly_once():
    y, g = _cohort()
    seen = np.zeros(len(y), dtype=int)
    for _, test_idx in grouped_folds(y, g, n_splits=5, seed=0):
        seen[test_idx] += 1
    assert (seen == 1).all()


def test_folds_keep_admission_level_prevalence_comparable():
    """StratifiedGroupKFold, not GroupKFold: fold prevalence must not drift."""
    y, g = _cohort(n_patients=600, seed=2)
    rates = []
    for _, test_idx in grouped_folds(y, g, n_splits=5, seed=1):
        by_patient = pd.Series(y[test_idx]).groupby(pd.Series(g[test_idx])).max()
        rates.append(by_patient.mean())
    assert np.ptp(rates) < 0.05


def test_scale_pos_weight_is_the_negative_to_positive_ratio():
    y = np.array([0] * 90 + [1] * 10)
    assert scale_pos_weight(y) == pytest.approx(9.0)
    assert scale_pos_weight(np.zeros(10)) == 1.0


def test_downsample_keeps_every_positive():
    y, g = _cohort(seed=4)
    idx = downsample_negatives(y, g, keep=0.2, seed=0)
    assert (y[idx] == 1).sum() == (y == 1).sum()
    assert (y[idx] == 0).sum() == pytest.approx((y == 0).sum() * 0.2, rel=0.01)


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def _miscalibrated(seed=0, n=40_000):
    """Well-ranked but badly scaled scores, as a class-weighted model produces."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(latent * 1.5 - 3.0)))).astype(int)
    raw = 1 / (1 + np.exp(-(latent * 1.5)))  # same ordering, wrong scale
    return y, raw


def test_platt_scaling_preserves_the_ranking_exactly():
    """A strictly increasing map cannot reorder anything, so AUROC is untouched."""
    y, raw = _miscalibrated()
    cal = Calibrator("platt").fit(raw, y).transform(raw)
    assert brier_score_loss(y, cal) < brier_score_loss(y, raw)
    assert roc_auc_score(y, cal) == pytest.approx(roc_auc_score(y, raw), abs=1e-9)


def test_isotonic_improves_the_scale_and_only_perturbs_auroc_via_ties():
    """Isotonic regression is non-*decreasing*, not strictly increasing.

    It merges neighbouring scores into flat segments, and tied scores count half
    a concordant pair each -- so AUROC moves slightly rather than not at all. The
    distinction matters when reporting "calibration cannot change discrimination".
    """
    y, raw = _miscalibrated()
    cal = Calibrator("isotonic").fit(raw, y).transform(raw)
    assert brier_score_loss(y, cal) < brier_score_loss(y, raw)
    assert len(np.unique(cal)) < len(np.unique(raw))
    assert roc_auc_score(y, cal) == pytest.approx(roc_auc_score(y, raw), abs=5e-3)


def test_calibrated_mean_matches_the_base_rate():
    y, raw = _miscalibrated(seed=1)
    cal = Calibrator("isotonic").fit(raw, y).transform(raw)
    assert abs(raw.mean() - y.mean()) > 0.1
    assert cal.mean() == pytest.approx(y.mean(), abs=0.01)


def test_reliability_curve_uses_equal_count_bins():
    y, raw = _miscalibrated(seed=2)
    curve = reliability_curve(y, raw, n_bins=10)
    assert len(curve) == 10
    assert curve["count"].std() / curve["count"].mean() < 0.1


# --------------------------------------------------------------------------
# Blending
# --------------------------------------------------------------------------
def test_rank_normalise_is_monotone_and_bounded():
    rng = np.random.default_rng(0)
    s = rng.normal(size=1000) * 100 + 5
    r = rank_normalise(s)
    assert 0 < r.min() and r.max() < 1
    assert np.array_equal(np.argsort(s), np.argsort(r))


def test_blend_is_scale_invariant():
    rng = np.random.default_rng(1)
    a, b = rng.random(500), rng.random(500)
    w = {"a": 0.6, "b": 0.4}
    plain = blend({"a": a, "b": b}, w)
    rescaled = blend({"a": a * 1000, "b": b / 7 + 20}, w)
    np.testing.assert_allclose(plain, rescaled)


def test_weight_search_favours_the_informative_model():
    y, g = _cohort(n_patients=500, seed=6)
    rng = np.random.default_rng(6)
    good = np.clip(0.5 * y + rng.normal(0, 0.25, len(y)), 0, 1)
    noise = rng.random(len(y))
    w = optimise_weights({"good": good, "noise": noise}, y, g, seed=0)
    assert w["good"] > w["noise"]
    assert sum(w.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Lead time
# --------------------------------------------------------------------------
def test_alert_timing_recovers_onset_and_lead_time():
    """Labels turn on 6 h before onset, so onset = first positive hour + 6."""
    n = 40
    y = np.zeros(n, dtype=np.int8)
    y[20:] = 1  # first positive hour 20 -> onset at hour 26
    scores = np.zeros(n)
    scores[14:] = 1.0  # first alert at hour 14 -> 12 h of warning
    groups = np.full(n, "p1")

    timing = alert_timing(y, scores, groups, threshold=0.5)
    row = timing.iloc[0]
    assert row["onset_hour"] == 26
    assert row["first_alert_hour"] == 14
    assert row["lead_time_hours"] == 12


def test_alert_after_onset_is_not_counted_as_a_detection():
    n = 40
    y = np.zeros(n, dtype=np.int8)
    y[20:] = 1                       # onset at hour 26
    scores = np.zeros(n)
    scores[30:] = 1.0                # alert only at hour 30, four hours too late
    timing = alert_timing(y, scores, np.full(n, "p1"), threshold=0.5)
    assert timing.iloc[0]["lead_time_hours"] == -4
    assert lead_time_summary(timing)["detection_rate"] == 0.0


def test_summary_counts_false_alarms_on_control_admissions():
    y = np.zeros(60, dtype=np.int8)
    groups = np.repeat(["a", "b", "c"], 20)
    scores = np.zeros(60)
    scores[25] = 1.0  # one alert, on patient b only
    summary = lead_time_summary(alert_timing(y, scores, groups, threshold=0.5))
    assert summary["control_admissions"] == 3
    assert summary["control_admissions_with_any_alert"] == 1
    assert summary["false_alarm_rate_per_admission"] == pytest.approx(1 / 3)
