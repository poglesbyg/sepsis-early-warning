"""Assemble the model matrix from the raw hourly frame.

``build_features`` is the single entry point. It is deliberately a pure function
of one hospital's frame plus config -- no fitted state, no reference to other
splits -- so it can be applied to train, validation, test and the external
hospital identically, and so the leakage tests can call it on a toy frame.

Feature blocks, and roughly what each is for:

    locf        carried-forward channel values (the clinician's current view)
    recency     hours since each channel was last measured
    intensity   how often each channel has been sampled so far
    deviation   each channel against the patient's own running mean
    rolling     level / spread / trend over 6h and 24h windows
    missing     panel-level ordering activity
    clinical    SIRS, qSOFA, partial SOFA, shock index and friends
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CFG, CHANNELS, Config
from .clinical import build_clinical
from .temporal import (
    baseline_deviation,
    last_observation_carried_forward,
    measurement_intensity,
    measurement_recency,
    missingness_profile,
    rolling_summaries,
)

FEATURE_BLOCKS = (
    "locf",
    "recency",
    "intensity",
    "deviation",
    "rolling",
    "missing",
    "clinical",
)

ID_COLUMNS = ["patient_id", "hospital", "hour", "ICULOS"]
TARGET = "SepsisLabel"


def build_features(
    df: pd.DataFrame,
    cfg: Config = CFG,
    blocks: tuple[str, ...] = FEATURE_BLOCKS,
) -> pd.DataFrame:
    """Return a frame of ID columns + features + target, one row per ICU hour."""
    df = df.sort_values(["patient_id", "hour"], ignore_index=True)
    locf = last_observation_carried_forward(df, CHANNELS)

    parts: list[pd.DataFrame] = []
    if "locf" in blocks:
        parts.append(locf.add_suffix("_locf"))
    if "recency" in blocks:
        parts.append(measurement_recency(df, CHANNELS))
    if "intensity" in blocks:
        parts.append(measurement_intensity(df, CHANNELS))
    if "deviation" in blocks:
        parts.append(baseline_deviation(df, locf, CHANNELS))
    if "rolling" in blocks:
        parts.append(rolling_summaries(df, locf, cfg.windows))
    if "missing" in blocks:
        parts.append(missingness_profile(df))
    if "clinical" in blocks:
        # Clinical scores read carried-forward values, not the sparse raw ones,
        # so a criterion stays evaluable between lab draws.
        clinical_input = locf.join(df[["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]])
        parts.append(build_clinical(clinical_input))

    features = pd.concat(parts, axis=1)
    features = features.replace([np.inf, -np.inf], np.nan).astype("float32")

    ids = df[[c for c in ID_COLUMNS if c in df.columns]]
    out = pd.concat([ids, features, df[[TARGET]]], axis=1)
    if out.columns.duplicated().any():
        dupes = out.columns[out.columns.duplicated()].tolist()
        raise ValueError(f"duplicate feature names: {dupes}")
    return out


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in ID_COLUMNS and c != TARGET]


def matrices(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Split a built frame into (X, y, patient groups)."""
    cols = feature_columns(frame)
    return (
        frame[cols],
        frame[TARGET].to_numpy(dtype=np.int8),
        frame["patient_id"].to_numpy(),
    )
