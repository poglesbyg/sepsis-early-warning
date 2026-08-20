"""Loading and splitting.

Every split in this project is made at the level of an *admission*, never a row.
An ICU stay contributes ~40 highly autocorrelated hours; splitting rows at random
puts hour 12 of a patient in train and hour 13 in test, which inflates held-out
AUROC by a wide margin. ``tests/test_splits.py`` pins that invariant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import CFG, Config
from .download import ensure_data


@dataclass
class Split:
    """One modelling split: the hour-level frame plus its patient roster."""

    name: str
    frame: pd.DataFrame

    @property
    def patients(self) -> np.ndarray:
        return self.frame["patient_id"].unique()

    def __len__(self) -> int:
        return len(self.frame)

    def describe(self) -> dict[str, float]:
        by_patient = self.frame.groupby("patient_id", observed=True)["SepsisLabel"].max()
        return {
            "hours": len(self.frame),
            "admissions": int(by_patient.size),
            "septic_admissions": int(by_patient.sum()),
            "septic_admission_rate": float(by_patient.mean()),
            "positive_hour_rate": float(self.frame["SepsisLabel"].mean()),
        }


def load_hospital(hospital: str, cfg: Config = CFG) -> pd.DataFrame:
    ensure_data([hospital], cfg, quiet=True)
    return pd.read_parquet(cfg.interim_dir / f"set{hospital}.parquet")


def patient_labels(df: pd.DataFrame) -> pd.Series:
    """Admission-level sepsis flag, used to stratify the split."""
    return df.groupby("patient_id", observed=True)["SepsisLabel"].max()


def make_splits(cfg: Config = CFG) -> dict[str, Split]:
    """Development splits from hospital A; hospital B is held back entirely.

    Hospital B is a different health system with different case mix and
    measurement practice. Keeping it untouched until the very end gives a true
    external-validation estimate rather than another internal one.
    """
    a = load_hospital("A", cfg)
    labels = patient_labels(a)
    ids = labels.index.to_numpy()
    y = labels.to_numpy()

    dev_ids, test_ids = train_test_split(
        ids, test_size=cfg.test_size, stratify=y, random_state=cfg.seed
    )
    dev_y = labels.loc[dev_ids].to_numpy()
    train_ids, val_ids = train_test_split(
        dev_ids,
        test_size=cfg.val_size / (1 - cfg.test_size),
        stratify=dev_y,
        random_state=cfg.seed,
    )

    def take(ids_):
        return a[a["patient_id"].isin(set(ids_))].reset_index(drop=True)

    splits = {
        "train": Split("train", take(train_ids)),
        "val": Split("val", take(val_ids)),
        "test": Split("test", take(test_ids)),
    }
    splits["external"] = Split("external", load_hospital("B", cfg))
    return splits


def split_summary(splits: dict[str, Split]) -> pd.DataFrame:
    return pd.DataFrame({name: s.describe() for name, s in splits.items()}).T
