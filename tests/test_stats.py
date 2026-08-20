"""Statistical helpers: known-answer checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from sepsis.stats.drift import population_stability_index
from sepsis.stats.multicollinearity import (
    condition_number,
    greedy_decorrelate,
    variance_inflation_factors,
)
from sepsis.stats.univariate import benjamini_hochberg, hedges_g, rank_biserial_auc


def test_bh_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.stats.multitest")
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(0, 1e-4, 20), rng.uniform(0, 1, 180)])
    expected = sm.multipletests(p, method="fdr_bh")[1]
    np.testing.assert_allclose(benjamini_hochberg(p), expected, rtol=1e-12)


def test_bh_is_monotone_and_never_below_raw_p():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 500)
    q = benjamini_hochberg(p)
    assert np.all(q >= p - 1e-12)
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)


def test_hedges_g_recovers_a_planted_effect():
    rng = np.random.default_rng(2)
    a = rng.normal(1.0, 1.0, 20_000)
    b = rng.normal(0.0, 1.0, 20_000)
    assert hedges_g(a, b) == pytest.approx(1.0, abs=0.05)
    assert hedges_g(b, b.copy()) == pytest.approx(0.0, abs=0.05)


def test_rank_biserial_auc_equals_roc_auc():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(3)
    a, b = rng.normal(0.7, 1, 900), rng.normal(0, 1, 1100)
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    assert rank_biserial_auc(a, b) == pytest.approx(roc_auc_score(y, np.r_[a, b]), abs=1e-9)


def test_vif_matches_the_regression_definition():
    rng = np.random.default_rng(4)
    x1 = rng.normal(size=4000)
    x2 = rng.normal(size=4000)
    x3 = 0.9 * x1 + 0.1 * rng.normal(size=4000)  # deliberately collinear
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

    vif = variance_inflation_factors(X)
    # Compare against 1 / (1 - R^2) from an explicit OLS fit of x3 on x1, x2.
    others = X[["x1", "x2"]].to_numpy()
    A = np.c_[np.ones(len(others)), others]
    target = X["x3"].to_numpy()
    beta, *_ = np.linalg.lstsq(A, target, rcond=None)
    resid = target - A @ beta
    r2 = 1 - resid.var(ddof=1) / target.var(ddof=1)
    np.testing.assert_allclose(vif["x3"], 1 / (1 - r2), rtol=1e-6)
    assert vif["x2"] < 1.1


def test_greedy_decorrelate_keeps_the_higher_priority_twin():
    rng = np.random.default_rng(5)
    base = rng.normal(size=2000)
    X = pd.DataFrame({"a": base, "a_copy": base + 1e-9 * rng.normal(size=2000), "b": rng.normal(size=2000)})
    kept = greedy_decorrelate(X, threshold=0.95, priority=pd.Series({"a": 0.1, "a_copy": 0.9, "b": 0.5}))
    assert "a_copy" in kept and "a" not in kept and "b" in kept


def test_condition_number_flags_collinearity():
    rng = np.random.default_rng(6)
    x = rng.normal(size=3000)
    clean = pd.DataFrame({"a": x, "b": rng.normal(size=3000)})
    dirty = pd.DataFrame({"a": x, "b": x + 1e-6 * rng.normal(size=3000)})
    assert condition_number(clean) < 5
    assert condition_number(dirty) > 100


def test_psi_is_zero_for_identical_and_grows_with_shift():
    rng = np.random.default_rng(7)
    ref = rng.normal(size=50_000)
    assert population_stability_index(ref, rng.normal(size=50_000)) < 0.01
    small = population_stability_index(ref, rng.normal(0.2, 1, 50_000))
    large = population_stability_index(ref, rng.normal(1.0, 1, 50_000))
    assert 0 < small < large
    assert large > 0.25


def test_psi_reacts_to_a_change_in_missingness_alone():
    rng = np.random.default_rng(8)
    ref = rng.normal(size=20_000)
    cmp_ = rng.normal(size=20_000)
    cmp_[rng.random(20_000) < 0.5] = np.nan
    assert population_stability_index(ref, cmp_) > 0.25


def test_bh_isolates_nan_pvalues():
    """A constant feature yields a NaN p-value; it must not poison the table.

    ``np.minimum.accumulate`` propagates NaN through every subsequent entry, so a
    single degenerate feature silently turns the whole q-value column to NaN --
    which reads as "nothing is significant" rather than as an error.
    """
    p = np.array([1e-40, np.nan, 0.02, 0.9, np.nan])
    q = benjamini_hochberg(p)
    assert np.isnan(q[[1, 4]]).all()
    assert np.isfinite(q[[0, 2, 3]]).all()
    assert q[0] < 1e-30
    # The correction is computed over the three valid entries only.
    np.testing.assert_allclose(q[[0, 2, 3]], benjamini_hochberg(np.array([1e-40, 0.02, 0.9])))
