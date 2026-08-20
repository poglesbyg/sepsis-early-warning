"""Collinearity diagnostics.

345 features derived from 40 channels are heavily redundant by construction:
``MAP_locf``, ``MAP_mean6`` and ``MAP_mean24`` are near-duplicates for a stable
patient. Trees shrug this off -- they just split on whichever copy they see
first, at the cost of scattered importance. Logistic regression does not: with
near-collinear columns the coefficients become unstable and huge in magnitude
with opposing signs, and any interpretation of them is meaningless.

So the linear model gets a pruned, decorrelated design matrix and the booster
gets the full one. Both are legitimate; using the same matrix for both would
handicap one of them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def variance_inflation_factors(X: pd.DataFrame) -> pd.Series:
    """VIF for every column, computed from the correlation matrix in one shot.

    The textbook route regresses each column on all the others -- 345 OLS fits
    over 550,000 rows. The identity ``VIF_i = [R^-1]_ii`` gives exactly the same
    numbers from a single inversion of the correlation matrix, which is why this
    returns in under a second instead of several minutes.

    Degenerate columns (zero variance, or all-missing) have no VIF and are
    reported as NaN rather than silently poisoning the inversion.
    """
    usable = [c for c in X.columns if X[c].notna().any() and X[c].std(skipna=True) > 0]
    vif = pd.Series(np.nan, index=X.columns, name="vif")
    if len(usable) < 2:
        return vif

    corr = np.nan_to_num(X[usable].corr().to_numpy(), nan=0.0)
    np.fill_diagonal(corr, 1.0)
    try:
        inv = np.linalg.inv(corr)
    except np.linalg.LinAlgError:  # exactly duplicated columns
        inv = np.linalg.pinv(corr)
    vif.loc[usable] = np.diag(inv)
    return vif.sort_values(ascending=False, na_position="last")


def greedy_decorrelate(
    X: pd.DataFrame,
    threshold: float = 0.85,
    priority: pd.Series | None = None,
    max_features: int | None = None,
) -> list[str]:
    """Keep one column from each near-duplicate cluster.

    Walks features best-first (by ``priority``, e.g. univariate effect size) and
    drops any later feature correlated above ``threshold`` with one already kept,
    so the survivor of each cluster is its most informative member rather than
    whichever happened to be alphabetically first.
    """
    corr = X.corr().abs()
    corr = corr.fillna(0.0)
    order = (
        list(priority.reindex(X.columns).sort_values(ascending=False).index)
        if priority is not None
        else list(X.columns)
    )

    kept: list[str] = []
    for col in order:
        if col not in corr.index:
            continue
        if not kept or corr.loc[col, kept].max() < threshold:
            kept.append(col)
        if max_features and len(kept) >= max_features:
            break
    return kept


def condition_number(X: pd.DataFrame) -> float:
    """Ratio of largest to smallest singular value of the standardised matrix.

    A rule of thumb from Belsley et al.: above ~30 indicates collinearity worth
    worrying about, above ~100 indicates severe collinearity.
    """
    usable = [c for c in X.columns if X[c].notna().any() and X[c].std(skipna=True) > 0]
    if len(usable) < 2:
        return float("nan")
    Z = X[usable].to_numpy(dtype=float)
    Z = np.nan_to_num(Z - np.nanmean(Z, axis=0), nan=0.0)
    scale = Z.std(axis=0)
    scale[scale == 0] = 1.0
    sv = np.linalg.svd(Z / scale, compute_uv=False)
    return float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")


def collinearity_report(
    X: pd.DataFrame,
    priority: pd.Series | None = None,
    threshold: float = 0.85,
    max_features: int | None = 80,
) -> dict:
    """VIF diagnostics on the full matrix, plus the pruned set for the linear model.

    ``max_features`` caps how many survive: the inference fit that consumes this
    is an unpenalised MLE, which needs many more observations per parameter than a
    regularised fit to produce standard errors worth reading.
    """
    vif = variance_inflation_factors(X)
    kept = greedy_decorrelate(X, threshold=threshold, priority=priority, max_features=max_features)
    return {
        "n_features": X.shape[1],
        "n_after_pruning": len(kept),
        "max_vif": float(vif.max()),
        "median_vif": float(vif.median()),
        "n_vif_over_10": int((vif > 10).sum()),
        "condition_number": condition_number(X[kept]),
        "worst_offenders": vif.head(10).round(1).to_dict(),
        "kept": kept,
    }
