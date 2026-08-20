"""Shared model-side plumbing: grouped CV, imputation policy, persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ..config import CFG, Config


def grouped_folds(
    y: np.ndarray, groups: np.ndarray, n_splits: int = 5, seed: int = 0
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Cross-validation folds that keep an admission whole and balance prevalence.

    ``StratifiedGroupKFold`` is the right primitive here: plain ``KFold`` leaks
    hours of the same stay across the fold boundary, and plain ``GroupKFold``
    keeps stays intact but lets the positive rate drift between folds, which
    makes utility scores fold-dependent for the wrong reason.
    """
    admission_label = (
        pd.Series(y).groupby(pd.Series(groups), observed=True).transform("max").to_numpy()
    )
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from splitter.split(np.zeros(len(y)), admission_label, groups)


@dataclass
class Predictions:
    """Risk scores for one model on one split, kept alongside their identifiers."""

    model: str
    split: str
    scores: np.ndarray
    y: np.ndarray
    groups: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"patient_id": self.groups, "y": self.y, "score": self.scores}
        )


@dataclass
class ModelArtifact:
    """A fitted estimator plus everything needed to reproduce and audit it."""

    name: str
    estimator: object
    features: list[str]
    params: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    def save(self, cfg: Config = CFG) -> Path:
        cfg.ensure_dirs()
        path = cfg.artifacts_dir / f"{self.name}.joblib"
        joblib.dump(self, path)
        (cfg.artifacts_dir / f"{self.name}.json").write_text(
            json.dumps({"name": self.name, "params": self.params, "metrics": self.metrics,
                        "n_features": len(self.features)}, indent=2, default=float)
        )
        return path

    @staticmethod
    def load(name: str, cfg: Config = CFG) -> "ModelArtifact":
        return joblib.load(cfg.artifacts_dir / f"{name}.joblib")


def downsample_negatives(
    y: np.ndarray, groups: np.ndarray, keep: float, seed: int = 0
) -> np.ndarray:
    """Row indices keeping all positives and a fraction of negative *hours*.

    Used only to make the slower estimators (statsmodels inference, the deep
    models) tractable. Negatives are thinned at the hour level rather than by
    dropping admissions, so every patient still contributes their trajectory.
    """
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n_keep = int(len(neg) * keep)
    return np.sort(np.concatenate([pos, rng.choice(neg, n_keep, replace=False)]))


def scale_pos_weight(y: np.ndarray) -> float:
    """Ratio of negative to positive hours -- XGBoost's imbalance knob."""
    pos = float((y == 1).sum())
    return float((y == 0).sum() / pos) if pos else 1.0
