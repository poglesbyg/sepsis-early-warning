"""Calibration and operating-point selection.

Class weighting and ``scale_pos_weight`` are the standard answers to a 1.8%
positive rate, and both work -- but they buy ranking quality by destroying the
probability scale. A weighted booster on this data emits scores near 0.5 for
hours whose true risk is a few percent, so the raw output is a good *ranking* and
a meaningless *probability*.

That matters as soon as anyone acts on the number. "Alert if risk > 0.4" is not
interpretable unless 0.4 means 40%, and any expected-cost calculation --
including the challenge's own utility function -- is arithmetic on probabilities.

The fix is a monotone post-hoc map fitted on held-out data. Platt scaling is
strictly increasing and leaves AUROC exactly unchanged; isotonic regression is
non-*decreasing*, so it merges neighbouring scores into flat segments and moves
AUROC by a few thousandths through ties alone. Either way discrimination is
essentially preserved while Brier score and calibration error improve a long
way -- and the chosen threshold starts meaning something.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ..evaluate.metrics import UtilityScorer, expected_calibration_error


@dataclass
class Calibrator:
    """Monotone recalibration map fitted on a held-out split.

    ``isotonic`` is nonparametric and fits any monotone shape given enough data;
    ``platt`` fits a single sigmoid and is the safer choice when positives are
    scarce. With ~2,500 positive hours in validation, isotonic is well supported
    here, but both are fitted so the report can show the difference.
    """

    method: str = "isotonic"
    model: object = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "Calibrator":
        scores = np.clip(np.asarray(scores, dtype=float), 1e-7, 1 - 1e-7)
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.model.fit(scores, y)
        elif self.method == "platt":
            # Platt scaling is a logistic regression on the log-odds of the raw
            # score, not on the score itself -- fitting on the raw probability
            # would double up the sigmoid.
            self.model = LogisticRegression(C=1e6, solver="lbfgs")
            self.model.fit(_logit(scores).reshape(-1, 1), y)
        else:
            raise ValueError(f"unknown calibration method: {self.method}")
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        scores = np.clip(np.asarray(scores, dtype=float), 1e-7, 1 - 1e-7)
        if self.method == "isotonic":
            return self.model.predict(scores)
        return self.model.predict_proba(_logit(scores).reshape(-1, 1))[:, 1]


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1 - p))


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    """Observed frequency against predicted risk, in equal-count bins.

    Equal-count rather than equal-width bins: with scores piled up near zero, a
    uniform grid puts almost every hour in the first bucket and the plot says
    nothing about the region where alerts actually fire.
    """
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return pd.DataFrame(columns=["bin", "predicted", "observed", "count"])
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append(
            {"bin": b, "predicted": float(p[m].mean()), "observed": float(y[m].mean()),
             "count": int(m.sum())}
        )
    return pd.DataFrame(rows)


def calibration_report(
    y: np.ndarray, raw: np.ndarray, calibrated: np.ndarray
) -> dict[str, float]:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    return {
        "brier_raw": float(brier_score_loss(y, np.clip(raw, 0, 1))),
        "brier_calibrated": float(brier_score_loss(y, np.clip(calibrated, 0, 1))),
        "ece_raw": expected_calibration_error(y, np.clip(raw, 0, 1)),
        "ece_calibrated": expected_calibration_error(y, np.clip(calibrated, 0, 1)),
        # Platt leaves this identical; isotonic moves it only through the ties
        # it introduces, so a change beyond ~1e-2 means something is wrong.
        "auroc_raw": float(roc_auc_score(y, raw)),
        "auroc_calibrated": float(roc_auc_score(y, calibrated)),
    }


def expected_cost_threshold(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    cost_false_alarm: float = 1.0,
    cost_missed_sepsis: float = 40.0,
) -> pd.DataFrame:
    """Threshold sweep under an explicit, editable cost ratio.

    The challenge utility encodes one particular trade-off. A hospital deploying
    this has its own -- how many false alarms per averted missed case it will
    tolerate before alarm fatigue does more harm than the model prevents. This
    exposes that ratio as a parameter rather than burying it in a metric, and
    reports both the cost-optimal and the utility-optimal operating point so the
    two can be compared.
    """
    scorer = UtilityScorer(y, groups)
    thresholds = np.unique(np.quantile(p, np.linspace(0, 1, 200)))
    rows = []
    for t in thresholds:
        alert = (p >= t).astype(np.int8)
        fp = int(((alert == 1) & (y == 0)).sum())
        fn = int(((alert == 0) & (y == 1)).sum())
        rows.append(
            {
                "threshold": float(t),
                "alerts_per_1000_hours": float(alert.mean() * 1000),
                "false_alarms": fp,
                "missed_positive_hours": fn,
                "expected_cost": cost_false_alarm * fp + cost_missed_sepsis * fn,
                "utility": scorer.score(alert),
            }
        )
    return pd.DataFrame(rows)
