"""Regularised logistic regression, plus the inference that makes it worth having.

A booster will beat a linear model on this task. The linear model is still the
one worth building first, for three reasons:

1. It sets an honest floor. A gradient-boosting AUROC only means something
   against a competently tuned linear baseline on the same features.
2. It produces coefficients that can be read as odds ratios with confidence
   intervals, so a clinician can interrogate what the model learned.
3. Sparse elastic-net solutions collapse 345 features to a short list, which is
   a usable answer to "what would you actually chart?"

Two design matrices are used deliberately. The dense pipeline gets everything;
the inference model gets the collinearity-pruned subset, because coefficient
interpretation on a matrix with VIFs in the hundreds is not interpretation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import CFG, Config
from .common import ModelArtifact


def build_pipeline(
    C: float = 1.0,
    penalty: str = "l2",
    l1_ratio: float | None = None,
    class_weight: str | None = "balanced",
    max_iter: int = 400,
    seed: int = 0,
) -> Pipeline:
    """Impute -> standardise -> penalised logistic regression.

    Median imputation is safe here precisely because missingness is already
    encoded explicitly by the recency and intensity feature blocks: the model
    can tell "this value is a stale carry-forward" from "this value was measured
    this hour" without needing the NaN itself.

    The imputer and scaler live inside the ``Pipeline`` so their statistics are
    refit on each CV training fold. Fitting them once on the whole training set
    before cross-validating would leak fold-level information into the folds.
    """
    solver = "saga" if penalty in {"l1", "elasticnet"} else "lbfgs"
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=C,
                    penalty=penalty,
                    l1_ratio=l1_ratio,
                    solver=solver,
                    class_weight=class_weight,
                    max_iter=max_iter,
                    n_jobs=-1 if solver == "saga" else None,
                    random_state=seed,
                ),
            ),
        ]
    )


def fit_logistic(
    X: pd.DataFrame,
    y: np.ndarray,
    name: str = "logistic",
    cfg: Config = CFG,
    **kwargs,
) -> ModelArtifact:
    pipe = build_pipeline(seed=cfg.seed, **kwargs)
    pipe.fit(X, y)
    return ModelArtifact(
        name=name, estimator=pipe, features=list(X.columns), params={"kind": "logistic", **kwargs}
    )


def sparse_signature(
    X: pd.DataFrame,
    y: np.ndarray,
    target_features: int = 25,
    l1_ratio: float = 0.95,
    seed: int = 0,
    max_rows: int = 200_000,
    path: tuple[float, ...] = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0),
) -> pd.DataFrame:
    """Walk an elastic-net regularisation path and return a compact signature.

    Picking a single ``C`` and reporting whatever survives is arbitrary -- at
    C = 0.02 on this matrix, 285 of 345 coefficients are still non-zero, which is
    not a signature. Instead the path is walked from strong to weak penalty and
    the sparsest solution that still retains at least ``target_features`` is
    returned, so the output is a genuinely short list: the measurements that
    carry the signal once the redundant copies have been shrunk away.

    Fitted on a stratified row subsample -- ``saga`` is the only solver
    supporting an elastic-net penalty and it is coordinate-wise, so the full
    550,000 x 345 matrix costs far more time than the extra rows buy in a
    variable-selection step whose output is a ranked list.
    """
    from .common import downsample_negatives

    if len(X) > max_rows:
        idx = downsample_negatives(y, np.arange(len(y)), keep=float(max_rows) / len(X), seed=seed)
        X, y = X.iloc[idx], y[idx]

    chosen, coefs = None, None
    for C in path:
        pipe = build_pipeline(
            C=C, penalty="elasticnet", l1_ratio=l1_ratio, seed=seed, max_iter=400
        )
        pipe.fit(X, y)
        w = pipe.named_steps["lr"].coef_.ravel()
        n_nonzero = int((w != 0).sum())
        chosen, coefs = C, w
        if n_nonzero >= target_features:
            break

    out = pd.DataFrame({"feature": X.columns, "coefficient": coefs})
    out = out[out["coefficient"] != 0].copy()
    out["odds_ratio_per_sd"] = np.exp(out["coefficient"])
    out["abs_coefficient"] = out["coefficient"].abs()
    out["C"] = chosen
    out["l1_ratio"] = l1_ratio
    return out.sort_values("abs_coefficient", ascending=False, ignore_index=True)


def inference_table(
    X: pd.DataFrame,
    y: np.ndarray,
    max_rows: int = 120_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Unpenalised MLE fit with standard errors, p-values and odds-ratio CIs.

    Penalised coefficients are biased by design, so the usual Wald machinery does
    not apply to them. This refits without a penalty on the pruned matrix -- the
    only setting in which the resulting p-values mean what they normally mean --
    and reports odds ratios per standard deviation of each feature.

    Standard errors here treat hours as independent, which they are not; the
    intervals are therefore optimistic. They are reported as a ranking and sanity
    aid, with the honest uncertainty coming from the patient-level bootstrap in
    ``evaluate.metrics``.
    """
    import statsmodels.api as sm

    rng = np.random.default_rng(seed)
    idx = (
        np.sort(rng.choice(len(X), max_rows, replace=False))
        if len(X) > max_rows
        else np.arange(len(X))
    )
    Xs = X.iloc[idx]
    ys = y[idx]

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    Z = scaler.fit_transform(imputer.fit_transform(Xs))
    Z = sm.add_constant(pd.DataFrame(Z, columns=Xs.columns), has_constant="add")

    model = sm.Logit(ys, Z).fit(disp=0, maxiter=200, method="lbfgs")
    conf = model.conf_int()
    out = pd.DataFrame(
        {
            "feature": model.params.index,
            "coefficient": model.params.to_numpy(),
            "std_err": model.bse.to_numpy(),
            "z": model.tvalues.to_numpy(),
            "p_value": model.pvalues.to_numpy(),
            "odds_ratio_per_sd": np.exp(model.params.to_numpy()),
            "or_ci_low": np.exp(conf.iloc[:, 0].to_numpy()),
            "or_ci_high": np.exp(conf.iloc[:, 1].to_numpy()),
        }
    )
    out = out[out["feature"] != "const"].reset_index(drop=True)
    from ..stats.univariate import benjamini_hochberg

    out["q_value"] = benjamini_hochberg(out["p_value"].to_numpy())
    out["pseudo_r2"] = model.prsquared
    return out.sort_values("p_value", ignore_index=True)


def clinical_rule_baseline(frame: pd.DataFrame) -> np.ndarray:
    """Score from bedside criteria alone -- the bar any model must clear.

    qSOFA >= 2 is the screen a clinician can run without a computer. Blending it
    with the SIRS count gives a slightly stronger non-learned reference. If a
    345-feature model cannot beat this, the model is not worth deploying.
    """
    qsofa = frame["qsofa_score"].fillna(0).to_numpy(dtype=float)
    sirs = frame["sirs_score"].fillna(0).to_numpy(dtype=float)
    lactate = frame["lactate_high"].fillna(0).to_numpy(dtype=float)
    return (qsofa / 2.0) * 0.5 + (sirs / 4.0) * 0.35 + lactate * 0.15
