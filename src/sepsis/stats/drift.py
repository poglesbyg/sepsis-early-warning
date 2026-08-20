"""Distribution shift between the two hospital systems.

Hospital A and hospital B are different health systems with different case mix,
charting habits and sepsis prevalence (8.8% vs 5.7% of admissions). That makes
B a genuine external validation set -- and it makes it worth measuring *which*
features move, because a feature that shifts hard between sites is one whose
learned coefficient will not transfer.

Population Stability Index is the standard summary in credit risk and reads the
same way here: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 material shift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def population_stability_index(
    reference: np.ndarray, comparison: np.ndarray, n_bins: int = 10, eps: float = 1e-6
) -> float:
    """PSI = sum (p_comp - p_ref) * ln(p_comp / p_ref) over reference quantile bins.

    Bin edges come from the reference distribution so the reference is uniform by
    construction and every bit of the statistic reflects movement in ``comparison``.
    Missingness is carried as its own bin: a lab that stops being ordered at the
    new site is exactly the kind of shift this is meant to catch.
    """
    ref = np.asarray(reference, dtype=float)
    cmp_ = np.asarray(comparison, dtype=float)

    ref_missing = float(np.isnan(ref).mean())
    cmp_missing = float(np.isnan(cmp_).mean())
    ref, cmp_ = ref[~np.isnan(ref)], cmp_[~np.isnan(cmp_)]
    if len(ref) < n_bins or len(cmp_) == 0:
        return float("nan")

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return abs(cmp_missing - ref_missing)
    edges[0], edges[-1] = -np.inf, np.inf

    p_ref = np.histogram(ref, bins=edges)[0] / len(ref) * (1 - ref_missing)
    p_cmp = np.histogram(cmp_, bins=edges)[0] / len(cmp_) * (1 - cmp_missing)
    p_ref = np.append(p_ref, ref_missing) + eps
    p_cmp = np.append(p_cmp, cmp_missing) + eps
    return float(np.sum((p_cmp - p_ref) * np.log(p_cmp / p_ref)))


def drift_report(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    features: list[str],
    sample: int | None = 60_000,
    seed: int = 0,
) -> pd.DataFrame:
    """PSI and a two-sample KS test per feature, ranked by PSI."""
    rng = np.random.default_rng(seed)

    def draw(df):
        if sample and len(df) > sample:
            return df.iloc[rng.choice(len(df), sample, replace=False)]
        return df

    ref, cmp_ = draw(reference), draw(comparison)

    rows = []
    for col in features:
        a = ref[col].to_numpy(dtype=float)
        b = cmp_[col].to_numpy(dtype=float)
        a_f, b_f = a[np.isfinite(a)], b[np.isfinite(b)]
        ks = stats.ks_2samp(a_f, b_f) if len(a_f) > 10 and len(b_f) > 10 else None
        rows.append(
            {
                "feature": col,
                "psi": population_stability_index(a, b),
                "missing_ref": float(np.isnan(a).mean()),
                "missing_cmp": float(np.isnan(b).mean()),
                "mean_ref": float(np.nanmean(a)) if len(a_f) else np.nan,
                "mean_cmp": float(np.nanmean(b)) if len(b_f) else np.nan,
                "ks_stat": float(ks.statistic) if ks else np.nan,
                "ks_p": float(ks.pvalue) if ks else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    out["severity"] = pd.cut(
        out["psi"],
        [-np.inf, 0.1, 0.25, np.inf],
        labels=["stable", "moderate", "material"],
    )
    return out.sort_values("psi", ascending=False, ignore_index=True)
