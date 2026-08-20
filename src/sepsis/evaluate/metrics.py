"""Evaluation metrics, including a vectorised form of the official 2019 utility score.

The challenge's clinical utility score is time-dependent: at every ICU hour, a
positive alert is worth something different depending on how far that hour sits
from the labelled onset of sepsis. The reference implementation loops over hours
in Python, which is fine to score one submission but far too slow to sit inside a
threshold sweep or an Optuna objective.

The trick used here is that, for a fixed patient, utility is *linear* in the
binary prediction vector::

    u(t) = pred[t] * u_pos[t] + (1 - pred[t]) * u_neg[t]

so the total is ``u_neg.sum() + pred @ (u_pos - u_neg)``. Precomputing the two
per-hour weight vectors once turns scoring into a single dot product, and the
normalising constants fall out of the same arrays. ``tests/test_utility.py``
asserts exact agreement with the reference implementation on random cohorts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from ..config import CFG, UtilityParams


# --------------------------------------------------------------------------
# Utility score
# --------------------------------------------------------------------------
def utility_weights(
    labels: np.ndarray,
    groups: np.ndarray,
    params: UtilityParams | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-hour utility of alerting (``u_pos``) versus staying silent (``u_neg``).

    Parameters
    ----------
    labels : (n_hours,) 0/1 challenge labels, ordered by patient then hour.
    groups : (n_hours,) patient identifiers, contiguous within a patient.

    Returns
    -------
    (u_pos, u_neg) : each (n_hours,) float arrays.
    """
    p = params or CFG.utility
    labels = np.asarray(labels, dtype=np.int8)
    u_pos = np.zeros(len(labels), dtype=np.float64)
    u_neg = np.zeros(len(labels), dtype=np.float64)

    m_1 = p.max_u_tp / (p.dt_optimal - p.dt_early)
    b_1 = -m_1 * p.dt_early
    m_2 = -p.max_u_tp / (p.dt_late - p.dt_optimal)
    b_2 = -m_2 * p.dt_late
    m_3 = p.min_u_fn / (p.dt_late - p.dt_optimal)
    b_3 = -m_3 * p.dt_optimal

    for start, stop in _group_slices(groups):
        y = labels[start:stop]
        n = stop - start
        t = np.arange(n, dtype=np.float64)

        if y.any():
            # The label flips on at t_sepsis + dt_optimal, so invert to recover onset.
            t_sepsis = float(np.argmax(y) - p.dt_optimal)
            dt = t - t_sepsis
            scored = dt <= p.dt_late  # after this the patient contributes nothing

            pos = np.where(
                dt <= p.dt_optimal,
                np.maximum(m_1 * dt + b_1, p.u_fp),
                m_2 * dt + b_2,
            )
            neg = np.where(dt <= p.dt_optimal, 0.0, m_3 * dt + b_3)
            u_pos[start:stop] = np.where(scored, pos, 0.0)
            u_neg[start:stop] = np.where(scored, neg, 0.0)
        else:
            u_pos[start:stop] = p.u_fp
            u_neg[start:stop] = p.u_tn

    return u_pos, u_neg


def _group_slices(groups: np.ndarray) -> list[tuple[int, int]]:
    """Start/stop indices of each contiguous run in ``groups``."""
    groups = np.asarray(groups)
    if len(groups) == 0:
        return []
    edges = np.flatnonzero(groups[1:] != groups[:-1]) + 1
    bounds = np.concatenate(([0], edges, [len(groups)]))
    return list(zip(bounds[:-1], bounds[1:]))


@dataclass
class UtilityScorer:
    """Reusable scorer bound to one cohort's labels.

    Build it once per evaluation set; scoring any prediction vector afterwards
    costs a single dot product, which is what makes the threshold sweep and the
    Optuna objective affordable.
    """

    labels: np.ndarray
    groups: np.ndarray
    params: UtilityParams | None = None

    def __post_init__(self) -> None:
        self.u_pos, self.u_neg = utility_weights(self.labels, self.groups, self.params)
        self.delta = self.u_pos - self.u_neg
        self.inaction = float(self.u_neg.sum())
        # The best achievable policy alerts exactly when alerting is worth more
        # than silence, i.e. wherever delta > 0.
        self.best = self.inaction + float(np.maximum(self.delta, 0.0).sum())

    def raw(self, predictions: np.ndarray) -> float:
        return self.inaction + float(np.asarray(predictions, dtype=np.float64) @ self.delta)

    def score(self, predictions: np.ndarray) -> float:
        """Normalised utility: 1.0 is the best possible policy, 0.0 is never alerting."""
        denom = self.best - self.inaction
        if denom == 0:
            return 0.0
        return float(np.asarray(predictions, dtype=np.float64) @ self.delta) / denom

    def sweep(self, scores: np.ndarray, n_thresholds: int = 200) -> pd.DataFrame:
        """Normalised utility across a grid of decision thresholds.

        Thresholds are score quantiles rather than a uniform grid, so the grid
        stays informative even when predicted risks are heavily skewed.
        """
        scores = np.asarray(scores, dtype=np.float64)
        qs = np.linspace(0.0, 1.0, n_thresholds)
        thresholds = np.unique(np.quantile(scores, qs))
        # Vectorised over thresholds: (n_thresholds, n_hours) @ (n_hours,)
        alerts = (scores[None, :] >= thresholds[:, None]).astype(np.float64)
        denom = self.best - self.inaction
        utility = (alerts @ self.delta) / denom
        return pd.DataFrame(
            {
                "threshold": thresholds,
                "utility": utility,
                "alert_rate": alerts.mean(axis=1),
            }
        )

    def best_threshold(self, scores: np.ndarray, n_thresholds: int = 200) -> tuple[float, float]:
        sweep = self.sweep(scores, n_thresholds)
        row = sweep.loc[sweep["utility"].idxmax()]
        return float(row["threshold"]), float(row["utility"])


# --------------------------------------------------------------------------
# Discrimination, calibration, and uncertainty
# --------------------------------------------------------------------------
def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 20) -> float:
    """Binned |confidence - accuracy| gap, weighted by bin population."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return float(abs(p.mean() - y.mean()))
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    err = 0.0
    for b in range(len(edges) - 1):
        mask = idx == b
        if mask.sum() == 0:
            continue
        err += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(err)


def evaluate(
    y: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    scorer: UtilityScorer | None = None,
    threshold: float | None = None,
) -> dict[str, float]:
    """Full metric bundle for one cohort at one operating point."""
    scorer = scorer or UtilityScorer(y, groups)
    if threshold is None:
        threshold, _ = scorer.best_threshold(scores)
    alerts = (scores >= threshold).astype(np.int8)

    tp = int(((alerts == 1) & (y == 1)).sum())
    fp = int(((alerts == 1) & (y == 0)).sum())
    fn = int(((alerts == 0) & (y == 1)).sum())

    return {
        "auroc": float(roc_auc_score(y, scores)),
        "auprc": float(average_precision_score(y, scores)),
        "utility": scorer.score(alerts),
        "threshold": float(threshold),
        "alert_rate": float(alerts.mean()),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "precision": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "brier": float(brier_score_loss(y, np.clip(scores, 0, 1))),
        "ece": expected_calibration_error(y, np.clip(scores, 0, 1)),
    }


def cluster_bootstrap_ci(
    y: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    metric: str = "auroc",
    n_boot: int = 400,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile CI from a *patient-level* bootstrap.

    Resampling hours independently would treat the ~40 correlated rows of one
    admission as 40 independent observations and produce intervals several times
    too narrow. Resampling whole admissions preserves that correlation.
    """
    rng = np.random.default_rng(seed)
    y, scores, groups = np.asarray(y), np.asarray(scores), np.asarray(groups)
    slices = _group_slices(groups)
    fn = {"auroc": roc_auc_score, "auprc": average_precision_score}[metric]

    point = float(fn(y, scores))
    draws = np.empty(n_boot)
    n = len(slices)
    for b in range(n_boot):
        picks = rng.integers(0, n, size=n)
        idx = np.concatenate([np.arange(*slices[i]) for i in picks])
        yb = y[idx]
        draws[b] = fn(yb, scores[idx]) if 0 < yb.sum() < len(yb) else np.nan
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def delong_test(y: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> dict[str, float]:
    """DeLong test for two correlated ROC curves on the same sample.

    Answers "is model A's AUC really higher than model B's, or is the gap within
    sampling noise?" -- the right question when both models score the identical
    rows, where an unpaired comparison would be needlessly conservative.
    """
    y = np.asarray(y)
    pos = np.asarray(scores_a)[y == 1], np.asarray(scores_b)[y == 1]
    neg = np.asarray(scores_a)[y == 0], np.asarray(scores_b)[y == 0]
    m, n = len(pos[0]), len(neg[0])

    v10, v01, aucs = [], [], []
    for p, q in zip(pos, neg):
        # Midrank-based structural components (DeLong et al. 1988).
        tx = _midrank(p)
        ty = _midrank(q)
        tz = _midrank(np.concatenate([p, q]))
        auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        aucs.append(auc)
        v10.append((tz[:m] - tx) / n)
        v01.append(1.0 - (tz[m:] - ty) / m)

    v10, v01 = np.array(v10), np.array(v01)
    s10 = np.cov(v10, ddof=1) if v10.shape[0] > 1 else np.array([[v10.var(ddof=1)]])
    s01 = np.cov(v01, ddof=1) if v01.shape[0] > 1 else np.array([[v01.var(ddof=1)]])
    cov = s10 / m + s01 / n

    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return {"auc_a": aucs[0], "auc_b": aucs[1], "diff": diff, "z": 0.0, "p_value": 1.0}
    z = diff / np.sqrt(var)
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "diff": float(diff),
        "z": float(z),
        "p_value": float(2 * stats.norm.sf(abs(z))),
    }


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranked = np.empty(len(x))
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j < len(x) - 1 and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranked[i : j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(len(x))
    out[order] = ranked
    return out
