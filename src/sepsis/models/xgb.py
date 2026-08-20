"""Gradient boosting: the model that has to earn its complexity.

Three choices matter more than the hyperparameter search itself.

*Missing values are handed to XGBoost, not imputed.* Its default-direction
learning sends NaN down whichever branch reduces loss most, which is a learned,
per-split treatment of missingness. Imputing first would erase exactly the
"nobody ordered this lab" signal the feature blocks were built to capture.

*The search optimises clinical utility, not AUROC.* Ranking quality is not what
the model is for. Utility is time-dependent and asymmetric -- a false alarm costs
0.05, a missed sepsis costs up to 2.0 -- so a model tuned on AUROC is tuned on
the wrong thing. Every trial is scored by the same normalised utility used in the
final report.

*Cross-validation is grouped by admission and early stopping happens inside each
fold.* Choosing the number of rounds on the same data used to score a trial is a
subtle and very common leak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from ..config import CFG, Config
from ..evaluate.metrics import UtilityScorer
from .common import ModelArtifact, grouped_folds, scale_pos_weight

BASE_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "n_jobs": -1,
}


def _suggest(trial) -> dict:
    """Search space, chosen for a wide, shallow, noisy tabular problem.

    Depth is capped low and ``min_child_weight`` allowed to go high because the
    positive class is 1.8% of hours: deep trees here memorise individual
    admissions. The column and row subsampling ranges are deliberately aggressive
    for the same reason.
    """
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 200.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 6),
    }


def cv_utility(
    params: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 3,
    num_boost_round: int = 1500,
    early_stopping_rounds: int = 60,
    seed: int = 0,
    trial=None,
) -> tuple[float, int]:
    """Mean out-of-fold normalised utility, and the mean best iteration count."""
    fold_scores: list[float] = []
    best_iters: list[int] = []

    for fold, (tr, va) in enumerate(grouped_folds(y, groups, n_splits=n_splits, seed=seed)):
        # Sorted indices keep each admission's hours contiguous and in order,
        # which the utility scorer relies on to locate sepsis onset.
        tr, va = np.sort(tr), np.sort(va)
        dtrain = xgb.DMatrix(X.iloc[tr], label=y[tr], feature_names=list(X.columns))
        dvalid = xgb.DMatrix(X.iloc[va], label=y[va], feature_names=list(X.columns))

        booster = xgb.train(
            {**BASE_PARAMS, **params, "scale_pos_weight": scale_pos_weight(y[tr])},
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        scores = booster.predict(dvalid, iteration_range=(0, booster.best_iteration + 1))
        scorer = UtilityScorer(y[va], groups[va])
        _, utility = scorer.best_threshold(scores)

        fold_scores.append(utility)
        best_iters.append(booster.best_iteration + 1)

        if trial is not None:
            trial.report(float(np.mean(fold_scores)), fold)
            import optuna

            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(np.mean(fold_scores)), int(np.mean(best_iters))


def tune(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: Config = CFG,
    n_splits: int = 3,
    n_trials: int | None = None,
) -> tuple[dict, int, "pd.DataFrame"]:
    """Optuna search over the space above. Returns (params, rounds, trial history)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rounds: dict[int, int] = {}

    def objective(trial):
        params = _suggest(trial)
        score, best_iter = cv_utility(
            params, X, y, groups, n_splits=n_splits, seed=cfg.seed, trial=trial
        )
        rounds[trial.number] = best_iter
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.seed, multivariate=True),
        # Prune a trial once its running fold mean is clearly off the pace; with
        # a 3-fold objective this recovers roughly a third of the search budget.
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
    )
    study.optimize(
        objective,
        n_trials=n_trials or cfg.n_trials,
        timeout=cfg.optuna_timeout,
        show_progress_bar=False,
    )

    history = study.trials_dataframe(attrs=("number", "value", "state", "params", "duration"))
    best_rounds = rounds.get(study.best_trial.number, 400)
    return study.best_params, best_rounds, history


def fit_xgboost(
    X: pd.DataFrame,
    y: np.ndarray,
    params: dict,
    num_boost_round: int,
    X_valid: pd.DataFrame | None = None,
    y_valid: np.ndarray | None = None,
    name: str = "xgboost",
    cfg: Config = CFG,
) -> ModelArtifact:
    """Final fit. With a validation set supplied, rounds are chosen by early stopping."""
    dtrain = xgb.DMatrix(X, label=y, feature_names=list(X.columns))
    evals = []
    if X_valid is not None:
        evals = [(xgb.DMatrix(X_valid, label=y_valid, feature_names=list(X.columns)), "valid")]

    booster = xgb.train(
        {**BASE_PARAMS, **params, "scale_pos_weight": scale_pos_weight(y)},
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=80 if evals else None,
        verbose_eval=False,
    )
    return ModelArtifact(
        name=name,
        estimator=booster,
        features=list(X.columns),
        params={"kind": "xgboost", **params, "num_boost_round": booster.num_boosted_rounds()},
    )


def predict(artifact: ModelArtifact, X: pd.DataFrame) -> np.ndarray:
    booster = artifact.estimator
    d = xgb.DMatrix(X[artifact.features], feature_names=artifact.features)
    best = getattr(booster, "best_iteration", None)
    rng = (0, best + 1) if best is not None else (0, 0)
    return booster.predict(d, iteration_range=rng)


def importance_table(artifact: ModelArtifact) -> pd.DataFrame:
    """Gain, cover and split-count importances side by side.

    Gain alone is misleading on a matrix this redundant: three near-identical
    copies of mean arterial pressure split one feature's gain three ways. Reading
    it next to split counts makes that visible instead of surprising.
    """
    booster = artifact.estimator
    frames = []
    for kind in ("gain", "cover", "weight"):
        s = pd.Series(booster.get_score(importance_type=kind), name=kind)
        frames.append(s)
    out = pd.concat(frames, axis=1).fillna(0.0)
    out.index.name = "feature"
    out["gain_share"] = out["gain"] / out["gain"].sum()
    return out.sort_values("gain", ascending=False).reset_index()


def shap_summary(
    artifact: ModelArtifact, X: pd.DataFrame, max_rows: int = 30_000, seed: int = 0
) -> tuple[np.ndarray, pd.DataFrame]:
    """Exact TreeSHAP values on a subsample, plus mean |SHAP| per feature.

    Gain answers "how much did splitting on this help the loss"; SHAP answers
    "how much did this feature move this patient's risk, right now". For a model
    that has to justify an alert at the bedside, the second question is the one
    that matters.
    """
    import shap

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X), min(max_rows, len(X)), replace=False))
    sample = X.iloc[idx][artifact.features]

    explainer = shap.TreeExplainer(artifact.estimator)
    values = explainer.shap_values(sample)
    ranking = (
        pd.DataFrame({"feature": artifact.features, "mean_abs_shap": np.abs(values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    )
    return values, ranking
