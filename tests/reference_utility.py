"""A deliberately naive, hour-by-hour utility implementation.

Written straight from the piecewise definition in the challenge paper (Reyna et
al., Crit Care Med 2020) and used only as a test oracle: if the vectorised
scorer in ``sepsis.evaluate.metrics`` ever drifts from the specification, the
comparison in ``test_utility.py`` fails.
"""

from __future__ import annotations

import numpy as np

from sepsis.config import UtilityParams


def naive_patient_utility(labels, predictions, p: UtilityParams | None = None) -> float:
    p = p or UtilityParams()
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    m_1 = p.max_u_tp / (p.dt_optimal - p.dt_early)
    b_1 = -m_1 * p.dt_early
    m_2 = -p.max_u_tp / (p.dt_late - p.dt_optimal)
    b_2 = -m_2 * p.dt_late
    m_3 = p.min_u_fn / (p.dt_late - p.dt_optimal)
    b_3 = -m_3 * p.dt_optimal

    septic = bool(labels.any())
    t_sepsis = (np.argmax(labels) - p.dt_optimal) if septic else float("inf")

    total = 0.0
    for t in range(len(labels)):
        if t > t_sepsis + p.dt_late:
            continue
        dt = t - t_sepsis
        if septic and predictions[t]:
            total += max(m_1 * dt + b_1, p.u_fp) if dt <= p.dt_optimal else m_2 * dt + b_2
        elif septic and not predictions[t]:
            total += 0.0 if dt <= p.dt_optimal else m_3 * dt + b_3
        elif predictions[t]:
            total += p.u_fp
        else:
            total += p.u_tn
    return float(total)


def naive_normalized_utility(cohort_labels, cohort_predictions, p: UtilityParams | None = None) -> float:
    """Cohort score normalised so 1.0 = the best possible alerting policy, 0.0 = never alert."""
    p = p or UtilityParams()
    observed = best = inaction = 0.0
    for labels, preds in zip(cohort_labels, cohort_predictions):
        labels = np.asarray(labels)
        n = len(labels)
        best_preds = np.zeros(n)
        if labels.any():
            t_sepsis = int(np.argmax(labels) - p.dt_optimal)
            lo = max(0, t_sepsis + p.dt_early)
            hi = min(int(t_sepsis + p.dt_late) + 1, n)
            best_preds[lo:hi] = 1
        observed += naive_patient_utility(labels, preds, p)
        best += naive_patient_utility(labels, best_preds, p)
        inaction += naive_patient_utility(labels, np.zeros(n), p)
    return (observed - inaction) / (best - inaction)
