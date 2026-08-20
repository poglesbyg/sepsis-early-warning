"""How early does the alert actually fire?

AUROC and utility are both aggregates over hours. Neither answers the question a
clinician asks first: *if this thing had been running, how much warning would I
have had?* This module converts hourly scores into per-admission alert timing --
lead time before onset for the cases it caught, and the false-alarm burden on
everyone else.

Lead time is measured against t_sepsis, the labelled onset of clinical
suspicion, recovered from the label edge (the challenge shifts labels 6 hours
earlier than onset, so the first positive hour is t_sepsis - 6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CFG, UtilityParams


def alert_timing(
    y: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    threshold: float,
    params: UtilityParams | None = None,
) -> pd.DataFrame:
    """One row per admission: when sepsis began, when the model first alerted."""
    p = params or CFG.utility
    frame = pd.DataFrame({"patient_id": groups, "y": y, "score": scores})
    frame["hour"] = frame.groupby("patient_id", observed=True, sort=False).cumcount()
    frame["alert"] = frame["score"] >= threshold

    rows = []
    for pid, g in frame.groupby("patient_id", observed=True, sort=False):
        septic = bool(g["y"].any())
        onset = float(g.loc[g["y"] == 1, "hour"].min() - p.dt_optimal) if septic else np.nan
        alerts = g.loc[g["alert"], "hour"]
        first_alert = float(alerts.min()) if len(alerts) else np.nan
        rows.append(
            {
                "patient_id": pid,
                "septic": septic,
                "stay_hours": int(len(g)),
                "onset_hour": onset,
                "first_alert_hour": first_alert,
                "alerted": bool(len(alerts)),
                "n_alert_hours": int(len(alerts)),
                "lead_time_hours": onset - first_alert if septic and len(alerts) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def lead_time_summary(timing: pd.DataFrame, horizon_hours: int = 48) -> dict[str, float]:
    """Headline clinical numbers derived from the per-admission timing table.

    ``detected`` counts a septic admission as caught only if the first alert
    lands *before* onset. An alert raised after the clinician already suspected
    sepsis is not an early warning, and counting it as one is the most common way
    these systems get oversold.
    """
    septic = timing[timing["septic"]]
    control = timing[~timing["septic"]]

    caught = septic[septic["lead_time_hours"] > 0]
    useful = caught[caught["lead_time_hours"] <= horizon_hours]

    return {
        "septic_admissions": int(len(septic)),
        "control_admissions": int(len(control)),
        "detected_before_onset": int(len(caught)),
        "detection_rate": float(len(caught) / len(septic)) if len(septic) else np.nan,
        "median_lead_time_h": float(caught["lead_time_hours"].median()) if len(caught) else np.nan,
        "iqr_lead_time_h": (
            float(caught["lead_time_hours"].quantile(0.75) - caught["lead_time_hours"].quantile(0.25))
            if len(caught)
            else np.nan
        ),
        f"detected_within_{horizon_hours}h": int(len(useful)),
        "control_admissions_with_any_alert": int(control["alerted"].sum()),
        "false_alarm_rate_per_admission": float(control["alerted"].mean()) if len(control) else np.nan,
        "alert_hours_per_control_admission": float(control["n_alert_hours"].mean())
        if len(control)
        else np.nan,
        # The number that decides whether anyone keeps the alarm switched on.
        "false_alarms_per_true_detection": (
            float(control["alerted"].sum() / len(caught)) if len(caught) else np.inf
        ),
    }


def lead_time_histogram(timing: pd.DataFrame, bins: int = 24, cap: int = 48) -> pd.DataFrame:
    caught = timing.loc[timing["lead_time_hours"] > 0, "lead_time_hours"].clip(upper=cap)
    counts, edges = np.histogram(caught, bins=bins, range=(0, cap))
    return pd.DataFrame(
        {"lead_time_low": edges[:-1], "lead_time_high": edges[1:], "admissions": counts}
    )
