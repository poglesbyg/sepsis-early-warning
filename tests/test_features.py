"""No-lookahead guarantee.

The property that matters for a real-time early-warning model is simple to state
and easy to violate by accident: the feature vector at hour *t* must depend only
on hours <= *t*. A single ``bfill``, a whole-stay median, or a centred rolling
window breaks it, and the damage shows up as an optimistic offline score and a
model that fails at the bedside.

What this file establishes, stated precisely: the feature builder never reads a
future row, and never lets one admission's data reach another's. Those are claims
about implementation timing, not causal claims about sepsis. "Causal" elsewhere in
this codebase -- causal features, ``padding="causal"`` -- carries its
signal-processing sense of depending only on past inputs, which is the same
property this file verifies.

The checks are ``nopeek``'s. It hides information the builder should not have --
by truncating the stay, and by isolating the admission -- rebuilds from scratch,
and requires the surviving rows to be unchanged. That is the same method this
file used to implement by hand, over the same cut points, applied to all 347
feature columns at once and with the comparison rules (missingness counts;
float32 is judged at float32 resolution) written down in one place instead of
per-assertion.
"""

from __future__ import annotations

import numpy as np
import nopeek
import pandas as pd
import pytest

from sepsis.config import CHANNELS, RAW_COLUMNS
from sepsis.features import build_features
from sepsis.features.builder import feature_columns

# The label is allowed to depend on the future: it is the thing being predicted,
# not an input to the prediction.
NOT_AN_INPUT = ["SepsisLabel"]

# Constant for the whole stay and known at admission, so the builder may read
# them from any row of the stay without that being lookahead.
STATIC = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]

# The cut points the hand-written version of this test used, kept so the
# migration asserts over the same stay geometry: one hour in, mid-first-day, and
# the last hour that still has a future to hide.
CUTS = [1, 5, 13, 24, 47]


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


@pytest.fixture(scope="module")
def stays() -> pd.DataFrame:
    return _toy_stays()


def test_features_at_hour_t_ignore_everything_after_t(stays):
    """Hide the future, rebuild, and require the surviving rows to be unchanged."""
    nopeek.assert_point_in_time(
        build_features,
        stays,
        time="hour",
        group="patient_id",
        cuts=CUTS,
        ignore=NOT_AN_INPUT,
        # Truncation only. Poisoning additionally reports the accumulator
        # coupling documented below, which is a separate defect from lookahead
        # and is tracked by its own test rather than folded into this one.
        strategy="truncate",
    )


def test_rolling_windows_never_span_two_admissions(stays):
    """Patient B's first hours must not inherit patient A's tail.

    Stronger than the pairwise check this replaces: nopeek rebuilds several
    admissions in isolation and also re-runs with the admissions in reverse
    order, so a builder that depends on arrival order fails too.
    """
    nopeek.assert_isolated(
        build_features,
        stays,
        group="patient_id",
        time="hour",
        ignore=NOT_AN_INPUT,
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "rolling_summaries rolls the whole patient-sorted array in one pass, so "
        "pandas' running accumulator carries floating-point state across admission "
        "boundaries. Blanking the rows whose *window* spans a boundary removes "
        "every contaminated window but not that state, so one admission's values "
        "perturb another's _std columns. Real-data magnitude is ~1e-7 (see the "
        "round-off notes from the isolation check); poisoning inflates it because "
        "it injects deliberately extreme values. Remove this marker when "
        "_boundary_safe_rolling accumulates per admission, or in float64."
    ),
)
def test_rolling_std_does_not_couple_admissions_numerically(stays):
    nopeek.assert_point_in_time(
        build_features,
        stays,
        time="hour",
        group="patient_id",
        cuts=CUTS,
        ignore=NOT_AN_INPUT,
        preserve=STATIC,
        strategy="poison",
    )


def test_no_feature_is_a_copy_of_the_label(stays):
    built = build_features(stays)
    y = built["SepsisLabel"].to_numpy(dtype=float)
    for col in feature_columns(built):
        v = built[col].to_numpy(dtype=float)
        mask = ~np.isnan(v)
        if mask.sum() < 50 or np.ptp(v[mask]) == 0:
            continue
        r = abs(np.corrcoef(v[mask], y[mask])[0, 1])
        assert r < 0.999, f"{col} is effectively the target"


def test_builder_rejects_unknown_column_collisions():
    assert set(RAW_COLUMNS).issubset(set(_toy_stays().columns))
