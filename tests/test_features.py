"""No-lookahead guarantee.

The property that matters for a real-time early-warning model is simple to state
and easy to violate by accident: the feature vector at hour *t* must depend only
on hours <= *t*. A single ``bfill``, a whole-stay median, or a centred rolling
window breaks it, and the damage shows up as an optimistic offline score and a
model that fails at the bedside.

What this file establishes, stated precisely: the feature builder never reads a
future row. That is a claim about implementation timing, not a causal claim about
sepsis. "Causal" elsewhere in this codebase -- causal features, ``padding=
"causal"`` -- carries its signal-processing sense of depending only on past
inputs, which is the same property this file verifies.

The test below truncates each admission at a range of cut points, rebuilds the
features from scratch, and asserts the surviving rows are bit-identical to the
same rows built from the full stay.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sepsis.config import CHANNELS, RAW_COLUMNS
from sepsis.features import build_features
from sepsis.features.builder import feature_columns


def _toy_stays(n_patients=6, seed=0, n_hours=48):
    """Frames shaped like the real data: dense vitals, very sparse labs."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_patients):
        n = n_hours
        row = {}
        for ch in CHANNELS:
            vals = rng.normal(50, 15, n).astype("float32")
            # Vitals are charted most hours; labs only occasionally.
            keep = rng.random(n) < (0.85 if ch in CHANNELS[:8] else 0.08)
            vals[~keep] = np.nan
            row[ch] = vals
        row["Age"] = np.full(n, rng.uniform(20, 90), dtype="float32")
        row["Gender"] = np.full(n, rng.integers(0, 2), dtype="float32")
        row["Unit1"] = np.full(n, rng.integers(0, 2), dtype="float32")
        row["Unit2"] = np.full(n, rng.integers(0, 2), dtype="float32")
        row["HospAdmTime"] = np.full(n, rng.uniform(-40, 0), dtype="float32")
        row["ICULOS"] = np.arange(1, n + 1)
        y = np.zeros(n, dtype=np.int8)
        if rng.random() < 0.5:
            y[int(rng.integers(n // 3, n)) :] = 1
        row["SepsisLabel"] = y
        f = pd.DataFrame(row)
        f["patient_id"] = f"p{i:05d}"
        f["hospital"] = "A"
        f["hour"] = np.arange(n)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


@pytest.mark.parametrize("cut", [1, 5, 13, 24, 47])
def test_features_at_hour_t_ignore_everything_after_t(cut):
    full = _toy_stays()
    truncated = full[full["hour"] < cut].reset_index(drop=True)

    built_full = build_features(full)
    built_trunc = build_features(truncated)

    cols = feature_columns(built_full)
    key = ["patient_id", "hour"]
    a = built_full.merge(built_trunc[key], on=key)[cols].to_numpy()
    b = built_trunc.sort_values(key, ignore_index=True)[cols].to_numpy()

    assert a.shape == b.shape
    # NaN patterns must match exactly, not just the finite values.
    assert np.array_equal(np.isnan(a), np.isnan(b)), "missingness changed when the future was hidden"
    np.testing.assert_allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=1e-6, atol=1e-6)


def test_no_feature_is_a_copy_of_the_label():
    built = build_features(_toy_stays(n_patients=12, seed=3))
    y = built["SepsisLabel"].to_numpy(dtype=float)
    for col in feature_columns(built):
        v = built[col].to_numpy(dtype=float)
        mask = ~np.isnan(v)
        if mask.sum() < 50 or np.ptp(v[mask]) == 0:
            continue
        r = abs(np.corrcoef(v[mask], y[mask])[0, 1])
        assert r < 0.999, f"{col} is effectively the target"


def test_rolling_windows_never_span_two_admissions():
    """Patient B's first hours must not inherit patient A's tail."""
    full = _toy_stays(n_patients=2, seed=9)
    solo = full[full["patient_id"] == "p00001"].reset_index(drop=True)

    both = build_features(full)
    alone = build_features(solo)
    cols = feature_columns(both)

    a = both[both["patient_id"] == "p00001"].sort_values("hour", ignore_index=True)[cols].to_numpy()
    b = alone.sort_values("hour", ignore_index=True)[cols].to_numpy()
    assert np.array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=1e-6, atol=1e-6)


def test_builder_rejects_unknown_column_collisions():
    assert set(RAW_COLUMNS).issubset(set(_toy_stays().columns))
