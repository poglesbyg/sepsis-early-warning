"""Temporal feature blocks.

Two things dominate this dataset. The first is that it is mostly empty: vitals
are charted hourly but labs are drawn a handful of times per stay, so ~90% of lab
cells are missing. The second is that the missingness is not random -- a
clinician orders a lactate *because* they are worried. That makes "when was this
last measured, and how often" a feature in its own right rather than a nuisance
to impute away.

Every transform here is causal: a value at hour *t* is a function of hours
<= *t* only. There is no back-fill, no whole-stay statistic, and no centring on a
quantity computed over the full admission.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import LABS, SPARSE_LABS, VITALS, WindowParams

# Rolling variability is informative for continuously charted signals; running it
# over a lab that is drawn twice a stay just re-describes the carried-forward value.
ROLLING_CHANNELS = VITALS
ROLLING_LABS = ["Lactate", "WBC", "Creatinine", "BUN", "Platelets", "Hct", "Glucose", "pH"]


def _grouped(df: pd.DataFrame):
    return df.groupby("patient_id", observed=True, sort=False)


def _boundary_safe_rolling(
    values: pd.DataFrame, hour_idx: np.ndarray, window: int, how: str
) -> pd.DataFrame:
    """Rolling statistic over a patient-sorted frame, without a groupby.

    Rolling the whole array at once is several times faster than
    ``groupby(...).rolling(...)``, but windows near a patient boundary would mix
    two admissions. Because the frame is sorted and hours are contiguous, exactly
    the rows whose within-stay index is < ``window - 1`` can span a boundary, so
    blanking those rows removes every contaminated value and no valid one.
    """
    rolled = getattr(values.rolling(window, min_periods=1), how)()
    rolled[hour_idx < window - 1] = np.nan
    return rolled


def last_observation_carried_forward(df: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    """Forward-filled channel values -- the clinician's "most recent known" view."""
    out = _grouped(df)[channels].ffill()
    return out.astype("float32")


def measurement_recency(df: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    """Hours since each channel was last actually measured.

    Large values mean the carried-forward number is stale and should be trusted
    less; small values on a rarely drawn lab mean somebody just ordered it.
    """
    hour_idx = _grouped(df).cumcount()
    out = {}
    for ch in channels:
        seen_at = hour_idx.where(df[ch].notna())
        last_seen = seen_at.groupby(df["patient_id"], observed=True, sort=False).ffill()
        out[f"{ch}_hours_since"] = (hour_idx - last_seen).astype("float32")
    return pd.DataFrame(out, index=df.index)


def measurement_intensity(df: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    """How often each channel has been sampled so far, in absolute and rate terms.

    Ordering intensity is a proxy for clinical concern: the count of lactates
    drawn in the first day is a record of what the care team was worried about.
    """
    g = _grouped(df)
    hours_elapsed = (g.cumcount() + 1).astype("float32")
    out = {}
    for ch in channels:
        n = df[ch].notna().groupby(df["patient_id"], observed=True, sort=False).cumsum()
        out[f"{ch}_n_obs"] = n.astype("float32")
        out[f"{ch}_obs_rate"] = (n / hours_elapsed).astype("float32")
    return pd.DataFrame(out, index=df.index)


def baseline_deviation(df: pd.DataFrame, locf: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    """Each channel relative to the patient's own running mean.

    A heart rate of 105 is unremarkable in someone who arrived at 100 and alarming
    in someone who arrived at 62. The running mean is expanding, not whole-stay,
    so hour *t* is centred only on hours <= *t*.
    """
    pid = df["patient_id"]
    out = {}
    for ch in channels:
        s = locf[ch]
        g = s.groupby(pid, observed=True, sort=False)
        running_mean = g.cumsum() / g.cumcount().add(1)
        out[f"{ch}_dev"] = (s - running_mean).astype("float32")
    return pd.DataFrame(out, index=df.index)


def rolling_summaries(
    df: pd.DataFrame, locf: pd.DataFrame, windows: WindowParams
) -> pd.DataFrame:
    """Level, spread and trend of the continuously monitored signals."""
    hour_idx = _grouped(df).cumcount().to_numpy()
    channels = ROLLING_CHANNELS + ROLLING_LABS
    values = locf[channels]

    blocks: list[pd.DataFrame] = []
    for window, stats in (
        (windows.short, ("mean", "min", "max", "std")),
        (windows.long, ("mean", "std")),
    ):
        for how in stats:
            block = _boundary_safe_rolling(values, hour_idx, window, how)
            block.columns = [f"{c}_{how}{window}" for c in channels]
            blocks.append(block.astype("float32"))

    # Trend: change over the window, per hour. Sign matters as much as magnitude --
    # a falling blood pressure is a different story from a low but stable one.
    pid = df["patient_id"]
    for window in (windows.short, windows.medium):
        slope = values.groupby(pid, observed=True, sort=False).diff(window) / window
        slope.columns = [f"{c}_slope{window}" for c in channels]
        blocks.append(slope.astype("float32"))

    return pd.concat(blocks, axis=1)


def missingness_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Whole-panel measurement activity, independent of any single channel."""
    vitals_seen = df[VITALS].notna().sum(axis=1)
    labs_seen = df[LABS].notna().sum(axis=1)
    sparse_seen = df[SPARSE_LABS].notna().sum(axis=1)
    pid = df["patient_id"]
    hours_elapsed = (_grouped(df).cumcount() + 1).astype("float32")

    out = pd.DataFrame(index=df.index)
    out["n_vitals_this_hour"] = vitals_seen.astype("float32")
    out["n_labs_this_hour"] = labs_seen.astype("float32")
    out["any_lab_this_hour"] = (labs_seen > 0).astype("float32")
    out["n_sparse_labs_this_hour"] = sparse_seen.astype("float32")
    out["cum_labs"] = labs_seen.groupby(pid, observed=True, sort=False).cumsum().astype("float32")
    out["lab_order_rate"] = (out["cum_labs"] / hours_elapsed).astype("float32")
    out["hours_since_any_lab"] = _hours_since_flag(labs_seen > 0, pid)
    return out


def _hours_since_flag(flag: pd.Series, pid: pd.Series) -> pd.Series:
    hour_idx = flag.groupby(pid, observed=True, sort=False).cumcount()
    seen_at = hour_idx.where(flag.to_numpy())
    last = seen_at.groupby(pid, observed=True, sort=False).ffill()
    return (hour_idx - last).astype("float32")
