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

    dtrain = xgb.DMatrix(train[features], label=train["SepsisLabel"], feature_names=features)
    dtest = xgb.DMatrix(test[features], label=test["SepsisLabel"], feature_names=features)

    booster = xgb.train(
        {**BASE_PARAMS, **CHAMPION_PARAMS, "seed": seed,
         "scale_pos_weight": scale_pos_weight(train["SepsisLabel"].to_numpy())},
        dtrain,
        num_boost_round=rounds,
        verbose_eval=False,
    )
    scores = booster.predict(dtest)
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
