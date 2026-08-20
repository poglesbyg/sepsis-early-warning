"""Blending the model families.

The three models fail differently: the linear model is smooth and stable but
cannot express interactions; the booster is sharp but jumpy across neighbouring
hours; the recurrent model sees trajectory shape but is noisier per hour. That
difference is what a blend monetises -- averaging two models that make the same
mistakes buys nothing.

Weights are searched on the validation split against normalised utility and then
frozen. The test split and the external hospital never inform the weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata

from ..evaluate.metrics import UtilityScorer


def rank_normalise(scores: np.ndarray) -> np.ndarray:
    """Map scores to (0, 1) by rank.

    The three models live on incompatible scales -- a weighted booster's 0.5 and
    a GRU's 0.5 are different risks -- so averaging raw outputs would silently
    weight by scale rather than by quality. Ranks are scale-free, and because the
    utility score depends only on the ordering induced by a threshold, nothing is
    lost by working in rank space.
    """
    return (rankdata(scores) - 0.5) / len(scores)


def blend(scores: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total = sum(weights.values())
    out = np.zeros(len(next(iter(scores.values()))))
    for name, s in scores.items():
        out += weights.get(name, 0.0) * rank_normalise(s)
    return out / total if total else out


def optimise_weights(
    scores: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    seed: int = 0,
) -> dict[str, float]:
    """Non-negative weights maximising validation utility.

    Utility as a function of the weights is piecewise constant -- it only changes
    when the blend reorders two hours across the threshold -- so gradient methods
    stall immediately. Nelder-Mead from several starts is the pragmatic choice at
    this dimensionality; the softmax parameterisation keeps weights non-negative
    and summing to one without constrained optimisation.
    """
    names = list(scores)
    ranked = {k: rank_normalise(v) for k, v in scores.items()}
    scorer = UtilityScorer(y, groups)

    def utility_of(w: np.ndarray) -> float:
        w = np.exp(w - w.max())
        w = w / w.sum()
        combined = sum(wi * ranked[n] for wi, n in zip(w, names))
        _, u = scorer.best_threshold(combined, n_thresholds=120)
        return -u

    rng = np.random.default_rng(seed)
    best, best_val = None, np.inf
    starts = [np.zeros(len(names))] + [rng.normal(0, 1, len(names)) for _ in range(4)]
    for x0 in starts:
        res = minimize(utility_of, x0, method="Nelder-Mead",
                       options={"maxiter": 300, "xatol": 1e-3, "fatol": 1e-5})
        if res.fun < best_val:
            best, best_val = res.x, res.fun

    w = np.exp(best - best.max())
    w = w / w.sum()
    return {n: float(wi) for n, wi in zip(names, w)}


def contribution_table(
    scores: dict[str, np.ndarray],
    weights: dict[str, float],
    y: np.ndarray,
    groups: np.ndarray,
) -> pd.DataFrame:
    """Leave-one-out ablation: what the blend loses without each member.

    A member with a large weight but no ablation cost is redundant -- it agrees
    with the others and could be dropped, which is worth knowing before shipping
    three models into production instead of one.
    """
    scorer = UtilityScorer(y, groups)
    full = scorer.best_threshold(blend(scores, weights))[1]
    rows = []
    for name in scores:
        rest = {k: v for k, v in scores.items() if k != name}
        if not rest:
            continue
        without = scorer.best_threshold(blend(rest, {k: weights[k] for k in rest}))[1]
        solo = scorer.best_threshold(scores[name])[1]
        rows.append(
            {
                "model": name,
                "weight": weights[name],
                "utility_alone": solo,
                "utility_without": without,
                "marginal_contribution": full - without,
            }
        )
    return pd.DataFrame(rows).sort_values("marginal_contribution", ascending=False, ignore_index=True)
