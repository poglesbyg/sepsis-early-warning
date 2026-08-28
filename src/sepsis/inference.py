"""The inference contract: what this model expects, what it returns, and when.

A FastAPI wrapper and a Dockerfile were considered for this repository and cut.
They are the most generatable artifacts in an ML project and they carry almost no
information about the work. What could not be skipped is the part a service
wrapper usually leaves implicit, which is the contract itself.

## The shape question, settled

The open decision was whether the contract takes a batch of hours or one hour at
a time with carried state. **It takes an admission's history and returns the risk
for its most recent hour.** Both other shapes fall out of that one:

* Streaming is the native call. ``score_latest(history_up_to_t)`` is what a live
  system does every hour.
* Batch is a convenience. ``score_stay(whole_stay)`` returns a risk for every
  hour, and it is exactly equal, value for value, to calling ``score_latest`` on
  each prefix of the stay.

That equivalence is not a coincidence and it is not an implementation detail. It
holds because every feature at hour *t* depends only on hours <= *t*, which
``tests/test_features.py`` enforces by rebuilding on truncated stays. The
no-lookahead invariant, stated as an API guarantee: **the model cannot tell
whether you handed it the whole stay or replayed it hour by hour.**
``tests/test_inference.py`` asserts it directly.

The cost is worth knowing. ``score_latest`` rebuilds features from the whole
prefix, so it is O(t) at hour *t*, and streaming an entire stay hour by hour is
O(n^2) where scoring it in one call is O(n). At ICU scale -- tens of hours, one
call an hour, per patient -- that is microseconds against microseconds, and the
alternative is caching derived state that would then need its own invalidation
rules and its own way of going quietly wrong.

## The recurrent model is not served, and that is the reason

The gradient booster is stateless: hand it a history, get a number. The GRU is
not. Serving it either replays the whole stay through the network every hour, or
carries a hidden state per admission -- which means state that must be persisted,
recovered after a restart, invalidated on a correction to a past hour, and kept
consistent across replicas. That is a different system with different failure
modes, and it buys a model that scores lower on this data and transfers worse.
The booster alone is the honest recommendation, so the booster alone is what has
a contract.

## What this does not do

It does not calibrate to a new site, choose a threshold, or claim the numbers in
the report transfer. The calibration map and the operating point are frozen
artifacts of hospital A's validation split. What crossing a site boundary does to
them is measured in the two shift experiments, and the answer is different for
each boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import CFG, CHANNELS, DEMOGRAPHICS, Config
from .features import build_features
from .models.xgb import predict as xgb_predict

# Everything ``build_features`` reads. Stated here rather than derived, because a
# caller needs to know the schema before it has a frame to inspect.
REQUIRED_COLUMNS = ["patient_id", "hour", *CHANNELS, *DEMOGRAPHICS]

SERVED_MODEL = "xgboost"


@dataclass(frozen=True)
class Prediction:
    """One hour's answer, with the operating point that produced it.

    The threshold travels with the prediction deliberately. A risk without the
    threshold it is compared against invites a caller to invent their own, which
    is the point at which the reported false-alarm burden stops describing what
    the system does.
    """

    hour: int
    risk: float
    alert: bool
    threshold: float


@dataclass(frozen=True)
class ServingBundle:
    """Everything needed to reproduce a published prediction, in one file.

    The fitted calibration map was previously only ever in memory during the
    evaluate stage, which meant nothing outside that process could reproduce a
    served number. A model without its calibration map and its threshold is not a
    deployable object, so the three travel together.
    """

    model: str
    features: list[str]
    calibrator: object
    threshold: float

    @staticmethod
    def path(model: str, cfg: Config = CFG) -> Path:
        return cfg.artifacts_dir / f"serving_{model}.joblib"

    def save(self, cfg: Config = CFG) -> Path:
        cfg.ensure_dirs()
        path = self.path(self.model, cfg)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, model: str = SERVED_MODEL, cfg: Config = CFG) -> "ServingBundle":
        path = cls.path(model, cfg)
        if not path.exists():
            raise FileNotFoundError(
                f"no serving bundle at {path}. It is written by the evaluate stage, "
                f"which is where the calibration map and the threshold are frozen: "
                f"run `make evaluate`."
            )
        return joblib.load(path)


class SepsisRisk:
    """The served model. Load once, score many times."""

    def __init__(self, artifact, bundle: ServingBundle, cfg: Config = CFG) -> None:
        if list(bundle.features) != list(artifact.features):
            raise ValueError(
                f"the serving bundle expects {len(bundle.features)} features and the "
                f"fitted model was trained on {len(artifact.features)}. The bundle is "
                f"stale: re-run the evaluate stage so the calibration map and the "
                f"model describe the same matrix."
            )
        self.artifact = artifact
        self.bundle = bundle
        self.cfg = cfg

    @classmethod
    def load(cls, model: str = SERVED_MODEL, cfg: Config = CFG) -> "SepsisRisk":
        from .models.common import ModelArtifact

        return cls(ModelArtifact.load(model, cfg), ServingBundle.load(model, cfg), cfg)

    @property
    def threshold(self) -> float:
        return self.bundle.threshold

    def score_stay(self, stay: pd.DataFrame) -> pd.DataFrame:
        """Risk for every hour of one admission.

        Equal, value for value, to calling ``score_latest`` on each prefix. See
        the module docstring for why that is guaranteed rather than hoped for.
        """
        validate_stay(stay)
        frame = stay.sort_values("hour", ignore_index=True)
        features = build_features(frame, self.cfg)
        # Routed through the model's own predict helper rather than calling the
        # booster directly: it honours ``best_iteration``, and a served score that
        # used a different number of rounds from the published one would be a
        # divergence nothing downstream could see.
        raw = xgb_predict(self.artifact, features)
        risk = np.asarray(self.bundle.calibrator.transform(raw), dtype=float)
        return pd.DataFrame(
            {
                "hour": frame["hour"].to_numpy(),
                "risk": risk,
                "alert": risk >= self.bundle.threshold,
            }
        )

    def score_latest(self, history: pd.DataFrame) -> Prediction:
        """Risk for the most recent hour in ``history``. The streaming call.

        ``history`` is everything charted so far for one admission, including the
        hour being scored. Passing a stay that continues past the hour of interest
        would be asking the model to read the future, so the contract is that the
        last row *is* the present.
        """
        scored = self.score_stay(history)
        last = scored.iloc[-1]
        return Prediction(
            hour=int(last["hour"]),
            risk=float(last["risk"]),
            alert=bool(last["alert"]),
            threshold=self.bundle.threshold,
        )


def validate_stay(stay: pd.DataFrame) -> None:
    """Refuse a frame the feature builder would silently misread.

    Each of these is a way to get a plausible number out of the wrong input, which
    is worse than an exception: a caller sees a risk, not a warning.
    """
    if stay is None or len(stay) == 0:
        raise ValueError("empty history: there is no hour to score")

    missing = [c for c in REQUIRED_COLUMNS if c not in stay.columns]
    if missing:
        raise ValueError(
            f"history is missing {len(missing)} required column(s): {missing[:8]}"
            f"{' ...' if len(missing) > 8 else ''}. Unmeasured channels belong in the "
            f"frame as NaN -- their absence is a feature, and dropping the column "
            f"loses it."
        )

    if stay["patient_id"].nunique() != 1:
        raise ValueError(
            f"history covers {stay['patient_id'].nunique()} admissions; the contract "
            f"scores one at a time, because every feature is computed within a stay"
        )

    hours = stay["hour"].to_numpy()
    if not np.issubdtype(hours.dtype, np.number) or np.isnan(hours).any():
        raise ValueError("hour must be an integer count of hours since ICU admission")

    order = np.argsort(hours, kind="stable")
    ordered = hours[order]
    if len(np.unique(ordered)) != len(ordered):
        raise ValueError("history contains duplicate hours; each hour must appear once")

    gaps = np.diff(ordered)
    if len(gaps) and (gaps != 1).any():
        bad = int(ordered[np.argmax(gaps != 1)])
        raise ValueError(
            f"history skips hours (first gap after hour {bad}). Recency and intensity "
            f"features count rows, not clock time, so an omitted hour silently reads "
            f"as though no time passed. Materialise missing hours as all-NaN rows."
        )


_CACHE: dict[str, SepsisRisk] = {}


def predict(history: pd.DataFrame, model: str = SERVED_MODEL, cfg: Config = CFG) -> Prediction:
    """Score the most recent hour of one admission's history.

    The one-call entry point, with the model loaded once per process. For
    scoring many admissions, load ``SepsisRisk`` yourself and keep it.
    """
    if model not in _CACHE:
        _CACHE[model] = SepsisRisk.load(model, cfg)
    return _CACHE[model].score_latest(history)
