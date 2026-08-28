"""The inference contract's guarantees, asserted rather than described.

The central claim is that the served model cannot tell whether it was handed a
whole stay or replayed hour by hour. If that ever stops being true, either the
feature builder has started reading ahead or streaming and batch have quietly
become different models -- and a deployment would then behave unlike everything
in the report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from sepsis.config import CFG
from sepsis.features import build_features
from sepsis.features.builder import feature_columns
from sepsis.inference import (
    REQUIRED_COLUMNS,
    SepsisRisk,
    ServingBundle,
    validate_stay,
)
from sepsis.models import calibration as calib
from sepsis.models.common import ModelArtifact
from tests.test_features import _toy_stays


@pytest.fixture(scope="module")
def toy_model():
    """A real booster on toy data, so the contract is exercised end to end."""
    raw = _toy_stays(n_patients=12, n_hours=30)
    features = build_features(raw)
    cols = feature_columns(features)
    y = features["SepsisLabel"].to_numpy()

    booster = xgb.train(
        {"objective": "binary:logistic", "max_depth": 3, "eta": 0.3, "seed": 0},
        xgb.DMatrix(features[cols], label=y, feature_names=cols),
        num_boost_round=8,
    )
    artifact = ModelArtifact(name="toy", estimator=booster, features=cols)
    raw_scores = booster.predict(xgb.DMatrix(features[cols], feature_names=cols))
    bundle = ServingBundle(
        model="toy",
        features=cols,
        calibrator=calib.Calibrator("isotonic").fit(raw_scores, y),
        threshold=float(np.quantile(raw_scores, 0.9)),
    )
    return SepsisRisk(artifact, bundle), raw


# --------------------------------------------------------------------------
# The guarantee
# --------------------------------------------------------------------------
def test_streaming_equals_batch_at_every_hour(toy_model):
    """The contract's whole shape rests on this equivalence."""
    model, raw = toy_model
    stay = raw[raw["patient_id"] == raw["patient_id"].iloc[0]].reset_index(drop=True)

    batch = model.score_stay(stay)
    for t in range(len(stay)):
        streamed = model.score_latest(stay.iloc[: t + 1])
        assert streamed.hour == int(stay["hour"].iloc[t])
        assert streamed.risk == pytest.approx(batch["risk"].iloc[t], abs=0, rel=0)


def test_a_later_hour_cannot_change_an_earlier_score(toy_model):
    """The no-lookahead invariant, stated as an API property.

    Appending hours must leave every previously returned risk untouched. If it
    does not, the model is reading the future and every lead time in the report
    is inflated.
    """
    model, raw = toy_model
    stay = raw[raw["patient_id"] == raw["patient_id"].iloc[1]].reset_index(drop=True)

    early = model.score_stay(stay.iloc[:10])
    full = model.score_stay(stay)
    assert np.array_equal(early["risk"].to_numpy(), full["risk"].to_numpy()[:10])


def test_prediction_carries_the_threshold_it_was_judged_against(toy_model):
    model, raw = toy_model
    stay = raw[raw["patient_id"] == raw["patient_id"].iloc[0]].reset_index(drop=True)

    p = model.score_latest(stay)
    assert p.threshold == model.threshold
    assert p.alert == (p.risk >= p.threshold)


# --------------------------------------------------------------------------
# Input validation: every one of these otherwise returns a plausible number
# --------------------------------------------------------------------------
def _valid_stay():
    raw = _toy_stays(n_patients=1, n_hours=12)
    return raw[REQUIRED_COLUMNS].copy()


def test_a_well_formed_stay_is_accepted():
    validate_stay(_valid_stay())


def test_empty_history_is_rejected():
    with pytest.raises(ValueError, match="no hour to score"):
        validate_stay(_valid_stay().iloc[:0])


def test_missing_channel_columns_are_rejected():
    stay = _valid_stay().drop(columns=["Lactate", "WBC"])
    with pytest.raises(ValueError, match="required column"):
        validate_stay(stay)


def test_two_admissions_in_one_call_are_rejected():
    stay = pd.concat([_valid_stay(), _valid_stay().assign(patient_id="other")], ignore_index=True)
    with pytest.raises(ValueError, match="scores one at a time"):
        validate_stay(stay)


def test_duplicate_hours_are_rejected():
    stay = _valid_stay()
    stay.loc[3, "hour"] = 2
    with pytest.raises(ValueError, match="duplicate hours"):
        validate_stay(stay)


def test_a_skipped_hour_is_rejected_because_recency_counts_rows():
    """Recency and intensity use row position, not the hour column, so a dropped
    hour reads as though no time passed."""
    stay = _valid_stay().drop(index=5).reset_index(drop=True)
    with pytest.raises(ValueError, match="skips hours"):
        validate_stay(stay)


def test_non_numeric_hours_are_rejected():
    stay = _valid_stay()
    stay["hour"] = stay["hour"].astype(float)
    stay.loc[2, "hour"] = np.nan
    with pytest.raises(ValueError, match="integer count of hours"):
        validate_stay(stay)


def test_scoring_validates_before_it_predicts(toy_model):
    model, raw = toy_model
    stay = raw[raw["patient_id"] == raw["patient_id"].iloc[0]].reset_index(drop=True)
    with pytest.raises(ValueError, match="skips hours"):
        model.score_stay(stay.drop(index=4))


# --------------------------------------------------------------------------
# The bundle
# --------------------------------------------------------------------------
def test_a_stale_bundle_is_refused(toy_model):
    """A bundle written for a different feature matrix would produce numbers no
    one could trace back to a published one."""
    model, _ = toy_model
    stale = ServingBundle(
        model="toy",
        features=model.bundle.features[:-1],
        calibrator=model.bundle.calibrator,
        threshold=model.bundle.threshold,
    )
    with pytest.raises(ValueError, match="stale"):
        SepsisRisk(model.artifact, stale)


def test_a_missing_bundle_says_which_stage_writes_it(tmp_path):
    from sepsis.config import Config

    with pytest.raises(FileNotFoundError, match="make evaluate"):
        ServingBundle.load("xgboost", Config(root=tmp_path))


def test_bundle_round_trips(tmp_path, toy_model):
    from sepsis.config import Config

    model, _ = toy_model
    cfg = Config(root=tmp_path)
    model.bundle.save(cfg)
    loaded = ServingBundle.load("toy", cfg)
    assert loaded.threshold == model.bundle.threshold
    assert loaded.features == model.bundle.features


# --------------------------------------------------------------------------
# Against the shipped artifacts
# --------------------------------------------------------------------------
def test_the_served_model_reproduces_its_published_predictions():
    """The contract must return the same number the report published, not merely
    a similar one."""
    if not ServingBundle.path("xgboost", CFG).exists():
        pytest.skip("no serving bundle in this checkout; run the evaluate stage")

    stored = pd.read_parquet(CFG.artifacts_dir / "preds_xgboost_val.parquet")
    raw = pd.read_parquet(CFG.interim_dir / "setA.parquet")
    pid = stored["patient_id"].iloc[0]

    model = SepsisRisk.load()
    stay = raw[raw["patient_id"] == pid].sort_values("hour", ignore_index=True)
    served = model.score_stay(stay)

    mine = stored[stored["patient_id"] == pid].sort_values("hour")
    published = np.asarray(model.bundle.calibrator.transform(mine["score"].to_numpy()), dtype=float)
    assert np.abs(published - served["risk"].to_numpy()).max() == 0.0
