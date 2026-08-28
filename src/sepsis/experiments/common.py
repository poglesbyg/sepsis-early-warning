"""Shared machinery for the experiment layer.

The experiments here all have the same shape: take the cohort, change exactly one
thing, refit, and report what that change was worth. The helper below is what
stops each of them reimplementing fit-and-score, and the ``ExperimentResult``
container is what lets the report generator consume them uniformly without a
discovery mechanism or a schema.

Every experiment is expected to assert its own invariants. An experiment that
returns a plausible but wrong number is worse than one that crashes, because the
wrong number gets published.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb

from ..config import CFG, Config
from ..evaluate.metrics import UtilityScorer
from ..features import build_features
from ..features.builder import FEATURE_BLOCKS, feature_columns
from ..models.common import scale_pos_weight
from ..models.xgb import BASE_PARAMS

# The tuned champion configuration, frozen. Ablations must differ from each other
# in the thing being ablated and in nothing else, so re-searching per variant
# would confound the comparison with search noise.
CHAMPION_PARAMS = {
    "max_depth": 3,
    "min_child_weight": 110.32376430909859,
    "learning_rate": 0.04046561115098005,
    "subsample": 0.9427464125126266,
    "colsample_bytree": 0.41909631445313117,
    "colsample_bylevel": 0.8843433163697136,
    "reg_lambda": 0.6205152894669264,
    "reg_alpha": 0.005769803734856525,
    "gamma": 0.16811735708496622,
    "max_delta_step": 6,
}
CHAMPION_ROUNDS = 155


@dataclass
class ExperimentResult:
    """What every experiment returns.

    ``table`` is the publishable result. ``prose`` is the one-paragraph reading of
    it that goes into the report, written by the experiment because the experiment
    is the only thing that knows what its own numbers mean.
    """

    name: str
    title: str
    table: pd.DataFrame
    prose: str
    figures: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def validate(self) -> "ExperimentResult":
        """Refuse to publish a table with holes in it."""
        if self.table.empty:
            raise ValueError(f"{self.name}: produced an empty table")
        numeric = self.table.select_dtypes(include=[np.number])
        if numeric.isna().to_numpy().any():
            bad = numeric.columns[numeric.isna().any()].tolist()
            raise ValueError(f"{self.name}: NaN in published columns {bad}")
        return self


def fit_and_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    rounds: int = CHAMPION_ROUNDS,
    seed: int = 0,
) -> dict[str, float]:
    """Fit the champion configuration on ``train`` and score it on ``test``.

    Returns AUROC and normalised utility. Utility is scored at the threshold that
    maximises it on the test split itself, which is optimistic in absolute terms
    but identical across variants -- and these experiments are read as differences,
    never as absolute performance claims.
    """
    from sklearn.metrics import roc_auc_score

    booster = fit_booster(train, features, rounds=rounds, seed=seed)
    scores = predict(booster, test, features)
    y = test["SepsisLabel"].to_numpy()

    scorer = UtilityScorer(y, test["patient_id"].to_numpy())
    _, utility = scorer.best_threshold(scores)
    return {
        "auroc": float(roc_auc_score(y, scores)),
        "utility": float(utility),
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "n_features": int(len(features)),
    }


def block_columns(sample: pd.DataFrame, cfg: Config = CFG) -> dict[str, list[str]]:
    """Column membership per feature block, derived from the builder itself.

    Deliberately not a regex over column names. The builder is the source of truth
    for what a block contains, so each block is built in isolation on a small
    sample and asked what it produced. A regex would drift silently the first time
    someone adds a feature with an unexpected suffix, and a silent drift here
    makes every ablation number meaningless.
    """
    membership: dict[str, list[str]] = {}
    for block in FEATURE_BLOCKS:
        built = build_features(sample, cfg, blocks=(block,))
        membership[block] = feature_columns(built)
    return membership


def assert_partition(membership: dict[str, list[str]], all_features: list[str]) -> None:
    """The blocks must tile the feature space exactly: no overlap, no orphans.

    If they do not, an ablation that drops a block is not measuring what it claims
    to measure, and nothing else in the pipeline would notice.
    """
    seen: dict[str, str] = {}
    for block, cols in membership.items():
        for c in cols:
            if c in seen:
                raise ValueError(
                    f"feature blocks overlap: {c!r} is in both {seen[c]!r} and {block!r}"
                )
            seen[c] = block

    covered, expected = set(seen), set(all_features)
    if covered != expected:
        orphans = sorted(expected - covered)
        strays = sorted(covered - expected)
        raise ValueError(
            f"feature blocks do not partition the matrix: "
            f"{len(orphans)} uncovered {orphans[:5]}, {len(strays)} unexpected {strays[:5]}"
        )


def admission_split(
    frame: pd.DataFrame, test_size: float = 0.25, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Correct split: whole admissions, stratified on admission-level sepsis."""
    from sklearn.model_selection import train_test_split

    labels = frame.groupby("patient_id", observed=True)["SepsisLabel"].max()
    train_ids, test_ids = train_test_split(
        labels.index.to_numpy(), test_size=test_size,
        stratify=labels.to_numpy(), random_state=seed,
    )
    train_ids, test_ids = set(train_ids), set(test_ids)
    mask = frame["patient_id"].isin(train_ids)
    return frame[mask].reset_index(drop=True), frame[~mask].reset_index(drop=True)


def row_split(
    frame: pd.DataFrame, test_size: float = 0.25, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The mistake: split ICU hours at random, ignoring which admission they belong to.

    Hour 12 of a patient ends up in training and hour 13 of the same patient in
    test. The two rows are nearly identical, so the model is graded on
    interpolation within stays it has already memorised.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(frame))
    cut = int(len(frame) * (1 - test_size))
    return (
        frame.iloc[np.sort(idx[:cut])].reset_index(drop=True),
        frame.iloc[np.sort(idx[cut:])].reset_index(drop=True),
    )


# --------------------------------------------------------------------------
# Shift experiments: fit once, score several cohorts, at a frozen threshold
# --------------------------------------------------------------------------
def fit_booster(
    train: pd.DataFrame,
    features: list[str],
    rounds: int = CHAMPION_ROUNDS,
    seed: int = 0,
) -> xgb.Booster:
    """The champion configuration, fitted. Separated from scoring because the
    shift experiments fit once and then score three or four different cohorts."""
    dtrain = xgb.DMatrix(train[features], label=train["SepsisLabel"], feature_names=features)
    return xgb.train(
        {**BASE_PARAMS, **CHAMPION_PARAMS, "seed": seed,
         "scale_pos_weight": scale_pos_weight(train["SepsisLabel"].to_numpy())},
        dtrain,
        num_boost_round=rounds,
        verbose_eval=False,
    )


def predict(booster: xgb.Booster, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return booster.predict(
        xgb.DMatrix(frame[features], feature_names=features)
    ).astype(np.float64)


def assert_contiguous_admissions(groups: np.ndarray) -> None:
    """``UtilityScorer`` reads each admission as one contiguous run of rows.

    If a frame is not sorted that way the scorer silently treats one stay as
    several, invents extra onsets, and returns a number that looks fine.
    """
    groups = pd.Series(np.asarray(groups))
    runs = int((groups != groups.shift()).sum())
    distinct = int(groups.nunique())
    if runs != distinct:
        raise ValueError(
            f"{runs - distinct} admission(s) split across non-adjacent rows; "
            f"sort by patient then hour before scoring utility"
        )


def weighted_utility(
    scorer: UtilityScorer, alerts: np.ndarray, row_weights: np.ndarray
) -> float:
    """Normalised utility on a population where each hour counts ``row_weights``.

    Identical to ``UtilityScorer.score`` when every weight is 1: the numerator and
    the normalising constant are both linear in the per-hour weight, so this is the
    utility of the pseudo-population in which each admission appears ``w`` times.
    """
    alerts = np.asarray(alerts, dtype=np.float64)
    row_weights = np.asarray(row_weights, dtype=np.float64)
    denom = float(row_weights @ np.maximum(scorer.delta, 0.0))
    if denom <= 0:
        raise ValueError("weighted utility has no achievable range; the cohort has no scored positive hour")
    return float((row_weights * alerts) @ scorer.delta / denom)


def admission_utility_parts(
    scorer: UtilityScorer, alerts: np.ndarray, groups: np.ndarray
) -> pd.DataFrame:
    """Per-admission numerator and denominator of the normalised utility.

    Utility is a ratio of two sums over hours, and both sums decompose by
    admission. Precomputing the two per-admission terms turns every later
    reweighting or bootstrap replicate into two dot products over ~20,000
    admissions instead of a pass over ~760,000 hours.
    """
    delta = scorer.delta
    frame = pd.DataFrame(
        {
            "patient_id": np.asarray(groups),
            "num": np.asarray(alerts, dtype=np.float64) * delta,
            "den": np.maximum(delta, 0.0),
        }
    )
    return frame.groupby("patient_id", observed=True, sort=False)[["num", "den"]].sum()


def baseline_covariates(
    frame: pd.DataFrame, columns: list[str], window: int = 6
) -> pd.DataFrame:
    """One row per admission, taken from the last hour of its first ``window``.

    Case mix has to be characterised by something, and the alternative -- the very
    first hour alone -- barely separates two hospitals. The six-hour window buys
    ordering behaviour and early vitals. It is *not* a model input: these values
    weight a population, they never reach a prediction, so the no-lookahead
    invariant is not in play. Stays shorter than the window contribute their last
    available hour.
    """
    hour = frame["hour"] - frame.groupby("patient_id", observed=True)["hour"].transform("min")
    early = frame.loc[hour < window, ["patient_id", *columns]]
    return early.groupby("patient_id", observed=True, sort=False).tail(1).set_index("patient_id")
