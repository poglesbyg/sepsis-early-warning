"""Univariate screening with honest units of analysis and multiplicity control.

Two mistakes are easy to make here and both inflate significance dramatically.

The first is pseudo-replication: an ICU stay contributes ~40 strongly
autocorrelated hours, so testing 790,000 rows as if they were 790,000
independent observations produces p-values with far too many zeros to be
meaningful. Everything below reduces each admission to a single summary value
first, giving one observation per patient.

The second is multiplicity: 345 features screened at alpha = 0.05 yields ~17
false positives by construction. Raw p-values are therefore reported alongside
Benjamini-Hochberg adjusted values, and "significant" means the adjusted one.

Effect size is reported next to significance because with n = 20,000 admissions
almost everything is significant; the question worth asking is how large the
separation is, not whether it is nonzero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Step-up FDR adjustment. Returns q-values in the input order.

    A NaN p-value -- a constant feature, or a group with no variance -- is
    excluded from the correction and returned as NaN. Leaving it in would poison
    the running minimum and silently turn every q-value in the table to NaN.
    """
    p = np.asarray(pvalues, dtype=float)
    q = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if valid.size == 0:
        return q

    pv = p[valid]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    # Enforce monotonicity from the largest p downwards.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n)
    adjusted[order] = np.clip(ranked, 0, 1)
    q[valid] = adjusted
    return q


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference with the small-sample correction."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if pooled_var <= 0:
        return 0.0
    d = (a.mean() - b.mean()) / np.sqrt(pooled_var)
    correction = 1 - 3 / (4 * (na + nb) - 9)
    return float(d * correction)


def rank_biserial_auc(a: np.ndarray, b: np.ndarray) -> float:
    """P(random septic value > random non-septic value); 0.5 means no separation.

    This is the Mann-Whitney U statistic rescaled, i.e. the univariate AUROC of
    the feature, which makes it directly comparable to the model AUROCs later.
    """
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan")
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(u / (na * nb))


def admission_summaries(
    frame: pd.DataFrame, features: list[str], how: str = "mean"
) -> tuple[pd.DataFrame, pd.Series]:
    """Collapse each admission to one value per feature, plus its sepsis label."""
    g = frame.groupby("patient_id", observed=True, sort=True)
    summary = getattr(g[features], how)()
    label = g["SepsisLabel"].max()
    return summary, label


def screen(
    frame: pd.DataFrame,
    features: list[str],
    min_admissions: int = 30,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Rank features by admission-level separation between septic and non-septic stays."""
    summary, label = admission_summaries(frame, features)
    is_septic = label.to_numpy().astype(bool)

    rows = []
    for col in features:
        v = summary[col].to_numpy(dtype=float)
        a, b = v[is_septic], v[~is_septic]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < min_admissions or len(b) < min_admissions:
            continue
        # Welch rather than Student: septic and non-septic groups differ in both
        # size (1:10) and variance, which is exactly when pooled-variance t fails.
        t_stat, p_t = stats.ttest_ind(a, b, equal_var=False)
        p_u = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        rows.append(
            {
                "feature": col,
                "n_septic": len(a),
                "n_control": len(b),
                "mean_septic": float(a.mean()),
                "mean_control": float(b.mean()),
                "hedges_g": hedges_g(a, b),
                "univariate_auc": rank_biserial_auc(a, b),
                "welch_t": float(t_stat),
                "p_welch": float(p_t),
                "p_mannwhitney": float(p_u),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_welch"] = benjamini_hochberg(out["p_welch"].to_numpy())
    out["q_mannwhitney"] = benjamini_hochberg(out["p_mannwhitney"].to_numpy())
    out["significant"] = out["q_welch"].lt(alpha).fillna(False)
    out["abs_g"] = out["hedges_g"].abs()
    return out.sort_values("abs_g", ascending=False, ignore_index=True)


def missingness_is_informative(
    frame: pd.DataFrame, channels: list[str], alpha: float = 0.05
) -> pd.DataFrame:
    """Test whether *ordering* a lab -- ignoring its value -- separates the groups.

    If it does, imputation alone throws away signal, and the recency/intensity
    feature blocks are earning their place rather than padding the matrix.
    """
    g = frame.groupby("patient_id", observed=True, sort=True)
    label = g["SepsisLabel"].max().to_numpy().astype(bool)

    rows = []
    for ch in channels:
        col = f"{ch}_obs_rate"
        if col not in frame.columns:
            continue
        rate = g[col].last().to_numpy(dtype=float)
        a, b = rate[label], rate[~label]
        p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        rows.append(
            {
                "channel": ch,
                "order_rate_septic": float(np.nanmean(a)),
                "order_rate_control": float(np.nanmean(b)),
                "rate_ratio": float(np.nanmean(a) / np.nanmean(b)) if np.nanmean(b) else np.nan,
                "auc_of_ordering": rank_biserial_auc(a, b),
                "p_value": float(p),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value"] = benjamini_hochberg(out["p_value"].to_numpy())
    out["significant"] = out["q_value"].lt(alpha).fillna(False)
    return out.sort_values("auc_of_ordering", ascending=False, ignore_index=True)
