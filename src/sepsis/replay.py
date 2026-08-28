"""Pre-computed hour-by-hour replay of individual admissions.

Every other number in this repository is an aggregate. A median lead time of 38
hours is a fact about a cohort, and it does not answer the question a clinician
asks, which is what the thing would actually have done in front of one patient.
This module picks four admissions, replays the risk score hour by hour, and marks
where the alert fired against where the care team started acting.

**The cases come from validation, never from test.** Validation was already spent
on tuning, calibration and threshold selection, so nothing is lost by looking at
it again. Selecting illustrative cases from the test split would convert a
held-out set into a presentation set: the reported metrics would stay
arithmetically correct, but a human would have inspected test admissions to decide
what to show, and the split would no longer be untouched in the sense the rest of
this project claims.

**The headline case is the median, not the best.** Among admissions caught before
onset, the one displayed is the one whose lead time sits closest to the median.
Picking the largest lead time in the cohort would be trivially easy, would look
far better, and would describe nothing. Two of the four cases are failures for the
same reason: a demo that only shows the model working is an advertisement.

The payload is written once to ``reports/replay.json`` and rendered as a static
table in Markdown and an interactive replay in the HTML report. Nothing is
computed at view time and there is no server.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import CFG, SPARSE_LABS, Config
from .evaluate.lead_time import alert_timing
from .models import calibration as calib

# Shown under the risk curve, because these are the channels a bedside chart
# leads with. Displayed as charted -- gaps included -- rather than carried
# forward, since the gaps are what the recency and intensity blocks are made of.
DISPLAY_VITALS = ["HR", "MAP", "Temp", "Resp"]

MODEL = "xgboost"

# What each case is published as. The role is asserted against the data before
# the payload is written: a case captioned as caught must actually have alerted
# before onset, and a case captioned as missed must not have.
ROLES = ("median_catch", "near_miss", "missed", "false_alarm")


def build_payload(cfg: Config = CFG, model: str = MODEL) -> dict:
    preds = pd.read_parquet(cfg.artifacts_dir / f"preds_{model}_val.parquet")
    preds = preds.sort_values(["patient_id", "hour"], ignore_index=True)

    y = preds["y"].to_numpy()
    groups = preds["patient_id"].to_numpy()

    # The same isotonic map and the same frozen threshold the report publishes.
    # Fitted on validation, which is the split these admissions come from, so the
    # probabilities on screen are in-sample -- the replay illustrates timing, not
    # calibration quality, and the report says so.
    calibrator = calib.Calibrator("isotonic").fit(preds["score"].to_numpy(), y)
    risk = calibrator.transform(preds["score"].to_numpy())
    threshold = float(json.loads((cfg.reports_dir / "metrics.json").read_text())["thresholds"][model])

    timing = alert_timing(y, risk, groups, threshold).set_index("patient_id")
    chosen = select_cases(timing)

    raw = pd.read_parquet(cfg.interim_dir / "setA.parquet")
    raw = raw[raw["patient_id"].isin(set(chosen.values()))]

    preds = preds.assign(risk=risk)
    cases = [
        _case(role, pid, preds, raw, timing.loc[pid], threshold)
        for role, pid in chosen.items()
    ]
    _assert_roles(cases)

    caught = timing[timing["lead_time_hours"] > 0]
    return {
        "model": model,
        "split": "val",
        "threshold": threshold,
        "cohort": {
            "septic_admissions": int(timing["septic"].sum()),
            "admissions": int(len(timing)),
            "detected_before_onset": int(len(caught)),
            "median_lead_time_h": float(caught["lead_time_hours"].median()),
        },
        "vitals": DISPLAY_VITALS,
        "cases": cases,
    }


def select_cases(timing: pd.DataFrame) -> dict[str, str]:
    """Four admissions, chosen by stated rule rather than by eye.

    Every rule is a deterministic argmin or argmax with an explicit tie-break on
    patient id, so the selection is reproducible and none of it is a judgement
    about which trace looks best.
    """
    septic = timing[timing["septic"]]
    caught = septic[septic["lead_time_hours"] > 0]
    if caught.empty:
        raise ValueError("no septic admission was caught before onset; nothing to replay")

    # 1. Representative catch: lead time closest to the cohort median.
    median = caught["lead_time_hours"].median()
    typical = (caught["lead_time_hours"] - median).abs().sort_values(kind="stable")

    # 2. The narrowest catch that still counts as one.
    narrow = caught["lead_time_hours"].sort_values(kind="stable")

    # 3. A septic admission the model did not catch: never alerted, or alerted
    #    only after the care team was already acting. Longest stay, so there is
    #    the most to look at while it fails to fire.
    missed = septic[~(septic["lead_time_hours"] > 0)]
    if missed.empty:
        raise ValueError("no missed septic admission in validation; the gallery would be one-sided")

    # 4. The worst false alarm: a non-septic admission alerting on the most hours.
    controls = timing[~timing["septic"] & timing["alerted"]]
    if controls.empty:
        raise ValueError("no false alarm in validation; the gallery would be one-sided")

    return {
        "median_catch": str(typical.index[0]),
        "near_miss": str(narrow.index[0]),
        "missed": str(missed["stay_hours"].sort_values(kind="stable").index[-1]),
        "false_alarm": str(controls["n_alert_hours"].sort_values(kind="stable").index[-1]),
    }


def _case(role, pid, preds, raw, timing_row, threshold) -> dict:
    """One admission's full trace, in true ICU-hour coordinates.

    ``alert_timing`` works in per-stay row positions; the hours here are the
    stay's own ``hour`` values, which do not always start at zero. Lead time is a
    difference, so it is identical in both coordinate systems, and it is asserted
    against ``alert_timing`` rather than trusted.
    """
    trace = preds[preds["patient_id"] == pid].sort_values("hour")
    hours = trace["hour"].to_numpy()
    risk = trace["risk"].to_numpy()
    label = trace["y"].to_numpy()

    septic = bool(label.any())
    onset = float(hours[label.argmax()] - CFG.utility.dt_optimal) if septic else None
    alerts = hours[risk >= threshold]
    first_alert = float(alerts[0]) if len(alerts) else None
    lead = onset - first_alert if septic and first_alert is not None else None

    if lead is not None and not np.isnan(timing_row["lead_time_hours"]):
        if abs(lead - float(timing_row["lead_time_hours"])) > 1e-9:
            raise ValueError(
                f"{pid}: replay lead time {lead} disagrees with alert_timing "
                f"{timing_row['lead_time_hours']}; the two are reading different rows"
            )

    bedside = raw[raw["patient_id"] == pid].sort_values("hour").set_index("hour")
    bedside = bedside.reindex(hours)

    return {
        "role": role,
        "patient_id": pid,
        "septic": septic,
        "hours": [int(h) for h in hours],
        "risk": [round(float(r), 5) for r in risk],
        "onset_hour": onset,
        "first_alert_hour": first_alert,
        "lead_time_hours": lead,
        "n_alert_hours": int((risk >= threshold).sum()),
        "stay_hours": int(len(hours)),
        "age": _static(bedside, "Age"),
        "unit": _unit(bedside),
        "vitals": {
            channel: [_null(v) for v in bedside[channel].to_numpy()]
            for channel in DISPLAY_VITALS
        },
        # Which hours a sparse lab was drawn on. The ablation found that ordering
        # behaviour alone reaches 97% of the full matrix's AUROC, so what the team
        # chose to measure belongs on the same timeline as the risk it produced.
        "lab_draws": [
            int(h) for h, drawn in zip(hours, bedside[SPARSE_LABS].notna().any(axis=1)) if drawn
        ],
    }


def _null(value) -> float | None:
    return None if pd.isna(value) else round(float(value), 2)


def _static(frame: pd.DataFrame, column: str) -> float | None:
    values = frame[column].dropna()
    return round(float(values.iloc[0]), 1) if len(values) else None


def _unit(frame: pd.DataFrame) -> str:
    if frame["Unit1"].fillna(0).max() == 1:
        return "MICU"
    if frame["Unit2"].fillna(0).max() == 1:
        return "SICU"
    return "unit not recorded"


def _assert_roles(cases: list[dict]) -> None:
    """Every case must be what its caption says it is.

    A case published as a catch that did not alert before onset, or a false alarm
    that was actually septic, would be a caption contradicting its own chart --
    and the chart is what a reader believes.
    """
    by_role = {c["role"]: c for c in cases}
    missing = set(ROLES) - set(by_role)
    if missing:
        raise ValueError(f"replay is missing cases for {sorted(missing)}")

    for role in ("median_catch", "near_miss"):
        case = by_role[role]
        if not case["septic"] or not (case["lead_time_hours"] or 0) > 0:
            raise ValueError(
                f"{role} ({case['patient_id']}) is published as caught before onset "
                f"but has lead time {case['lead_time_hours']!r}"
            )

    missed = by_role["missed"]
    if not missed["septic"] or (missed["lead_time_hours"] or 0) > 0:
        raise ValueError(
            f"missed case ({missed['patient_id']}) is published as a miss but was "
            f"caught {missed['lead_time_hours']} hours before onset"
        )

    alarm = by_role["false_alarm"]
    if alarm["septic"] or alarm["n_alert_hours"] < 1:
        raise ValueError(
            f"false alarm case ({alarm['patient_id']}) is published as a false alarm "
            f"but septic={alarm['septic']} with {alarm['n_alert_hours']} alert hours"
        )

    for case in cases:
        lengths = {len(case["risk"]), len(case["hours"]), case["stay_hours"]}
        if len(lengths) != 1:
            raise ValueError(
                f"{case['patient_id']}: risk, hours and stay length disagree {sorted(lengths)}"
            )
        if any(r is None or not np.isfinite(r) for r in case["risk"]):
            raise ValueError(f"{case['patient_id']}: risk series has holes in it")


def assert_not_from_test(payload: dict, cfg: Config = CFG) -> None:
    """The replayed admissions must not appear in the test or external splits.

    This is the invariant the whole section rests on, so it is checked against the
    split files themselves rather than trusted from the filename the payload was
    read out of.
    """
    shown = {c["patient_id"] for c in payload["cases"]}
    for split in ("test", "external"):
        ids = set(pd.read_parquet(cfg.processed_dir / f"{split}.parquet", columns=["patient_id"])["patient_id"])
        overlap = shown & ids
        if overlap:
            raise ValueError(
                f"replay would display {len(overlap)} admission(s) from the {split} "
                f"split ({sorted(overlap)}); illustrative cases come from validation"
            )


def write(cfg: Config = CFG, model: str = MODEL, quiet: bool = False) -> dict:
    payload = build_payload(cfg, model)
    assert_not_from_test(payload, cfg)

    path = cfg.reports_dir / "replay.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")

    # The traces themselves are deliberately not pinned by `make regress`: several
    # hundred per-hour floats would make every baseline diff unreadable, and a
    # change in them moves the metrics the baseline already tracks. What is pinned
    # is which admissions got selected and what they are captioned as, because
    # that is the part a reader takes on trust.
    (cfg.reports_dir / "replay_summary.json").write_text(
        json.dumps(
            {
                "model": payload["model"],
                "split": payload["split"],
                "threshold": payload["threshold"],
                "cohort": payload["cohort"],
                "cases": [
                    {k: case[k] for k in (
                        "role", "patient_id", "septic", "stay_hours", "onset_hour",
                        "first_alert_hour", "lead_time_hours", "n_alert_hours",
                    )}
                    for case in payload["cases"]
                ],
            },
            indent=2,
        )
        + "\n"
    )
    if not quiet:
        for case in payload["cases"]:
            lead = case["lead_time_hours"]
            when = f"{lead:+.0f} h vs onset" if lead is not None else "no alert before onset"
            print(f"[replay] {case['role']:<13} {case['patient_id']}  "
                  f"{case['stay_hours']:>3} h stay  {when}", flush=True)
        print(f"[replay] wrote {path.name} ({path.stat().st_size / 1024:.0f} KB)", flush=True)
    return payload
