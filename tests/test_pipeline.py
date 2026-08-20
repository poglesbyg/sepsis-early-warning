"""End-to-end smoke test on synthetic admissions.

Runs the real code path -- feature build, grouped CV, XGBoost, utility scoring,
calibration, blending -- on a small cohort with a *planted* signal, so the test
can assert the pipeline actually recovers it rather than merely running without
raising. Takes a couple of seconds and needs no downloaded data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from sepsis.config import CHANNELS
from sepsis.evaluate.lead_time import alert_timing, lead_time_summary
from sepsis.evaluate.metrics import UtilityScorer, evaluate
from sepsis.features import build_features
from sepsis.features.builder import feature_columns, matrices
from sepsis.models.calibration import Calibrator
from sepsis.models.common import grouped_folds, scale_pos_weight
from sepsis.models.ensemble import blend, optimise_weights
from sepsis.models.xgb import BASE_PARAMS


def synthetic_cohort(n_patients=400, seed=0) -> pd.DataFrame:
    """Admissions where septic patients drift on HR, Resp and Lactate before onset.

    The drift starts 10 hours before the labelled onset, so a causal model has
    something real to find and a leaky one has a much easier time -- which is what
    makes this a useful integration check rather than a tautology.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_patients):
        n = int(rng.integers(20, 70))
        septic = rng.random() < 0.3
        onset = int(rng.integers(15, n)) if septic else n + 100

        drift = np.clip(np.arange(n) - (onset - 10), 0, 12) / 12.0
        row = {}
        for ch in CHANNELS:
            base = rng.normal(60, 10)
            vals = rng.normal(base, 6, n).astype("float32")
            if ch == "HR":
                vals += 25 * drift
            elif ch == "Resp":
                vals += 12 * drift
            elif ch == "Lactate":
                vals += 4 * drift
            dense = ch in ("HR", "Resp", "SBP", "MAP", "O2Sat", "Temp")
            vals[rng.random(n) > (0.9 if dense else 0.1)] = np.nan
            row[ch] = vals

        row["Age"] = np.full(n, rng.uniform(20, 90), dtype="float32")
        row["Gender"] = np.full(n, float(rng.integers(0, 2)), dtype="float32")
        row["Unit1"] = np.full(n, float(rng.integers(0, 2)), dtype="float32")
        row["Unit2"] = np.full(n, float(rng.integers(0, 2)), dtype="float32")
        row["HospAdmTime"] = np.full(n, rng.uniform(-40, 0), dtype="float32")
        row["ICULOS"] = np.arange(1, n + 1)
        y = np.zeros(n, dtype=np.int8)
        if septic:
            y[max(onset - 6, 0) :] = 1  # challenge labels lead onset by 6 h
        row["SepsisLabel"] = y

        f = pd.DataFrame(row)
        f["patient_id"] = f"p{i:05d}"
        f["hospital"] = "S"
        f["hour"] = np.arange(n)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def cohort():
    return build_features(synthetic_cohort())


def test_pipeline_recovers_a_planted_signal(cohort):
    X, y, g = matrices(cohort)
    assert X.shape[1] > 300
    assert 0 < y.mean() < 0.25

    train_idx, test_idx = next(grouped_folds(y, g, n_splits=4, seed=0))
    train_idx, test_idx = np.sort(train_idx), np.sort(test_idx)

    booster = xgb.train(
        {**BASE_PARAMS, "max_depth": 4, "learning_rate": 0.1,
         "scale_pos_weight": scale_pos_weight(y[train_idx])},
        xgb.DMatrix(X.iloc[train_idx], label=y[train_idx]),
        num_boost_round=60,
    )
    scores = booster.predict(xgb.DMatrix(X.iloc[test_idx]))

    scorer = UtilityScorer(y[test_idx], g[test_idx])
    metrics = evaluate(y[test_idx], scores, g[test_idx], scorer)
    assert metrics["auroc"] > 0.75, "planted signal was not recovered"
    assert metrics["utility"] > 0.0, "the model should beat never alerting"
    assert 0 <= metrics["alert_rate"] <= 1


def test_calibration_and_blending_compose(cohort):
    X, y, g = matrices(cohort)
    train_idx, test_idx = next(grouped_folds(y, g, n_splits=4, seed=1))
    train_idx, test_idx = np.sort(train_idx), np.sort(test_idx)

    booster = xgb.train(
        {**BASE_PARAMS, "max_depth": 3, "learning_rate": 0.15,
         "scale_pos_weight": scale_pos_weight(y[train_idx])},
        xgb.DMatrix(X.iloc[train_idx], label=y[train_idx]),
        num_boost_round=40,
    )
    raw = booster.predict(xgb.DMatrix(X.iloc[test_idx]))
    rule = X.iloc[test_idx]["sirs_score"].fillna(0).to_numpy() / 4.0

    # Weighted training inflates mean predicted risk above the base rate;
    # recalibration should pull it back onto it.
    base = y[test_idx].mean()
    assert raw.mean() > base
    calibrated = Calibrator("isotonic").fit(raw, y[test_idx]).transform(raw)
    assert abs(calibrated.mean() - base) < abs(raw.mean() - base)
    assert calibrated.mean() == pytest.approx(base, abs=0.02)

    weights = optimise_weights({"xgb": calibrated, "rule": rule}, y[test_idx], g[test_idx])
    combined = blend({"xgb": calibrated, "rule": rule}, weights)
    assert len(combined) == len(test_idx)
    assert weights["xgb"] > weights["rule"]


def test_lead_time_is_reported_for_the_planted_onsets(cohort):
    X, y, g = matrices(cohort)
    train_idx, test_idx = next(grouped_folds(y, g, n_splits=4, seed=2))
    train_idx, test_idx = np.sort(train_idx), np.sort(test_idx)

    booster = xgb.train(
        {**BASE_PARAMS, "max_depth": 4, "learning_rate": 0.1,
         "scale_pos_weight": scale_pos_weight(y[train_idx])},
        xgb.DMatrix(X.iloc[train_idx], label=y[train_idx]),
        num_boost_round=60,
    )
    scores = booster.predict(xgb.DMatrix(X.iloc[test_idx]))
    scorer = UtilityScorer(y[test_idx], g[test_idx])
    threshold, _ = scorer.best_threshold(scores)

    timing = alert_timing(y[test_idx], scores, g[test_idx], threshold)
    summary = lead_time_summary(timing)
    assert summary["septic_admissions"] > 0
    assert 0 < summary["detection_rate"] <= 1
    assert summary["median_lead_time_h"] > 0
