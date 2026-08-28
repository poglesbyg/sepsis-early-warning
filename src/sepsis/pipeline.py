"""End-to-end pipeline: data -> features -> statistics -> models -> report.

Each stage caches to disk and can be run on its own, so a change to the report
does not mean retraining, and a change to the search space does not mean
re-downloading 40,000 files.

Stage order and what each guarantees:

    data      raw .psv files and per-hospital Parquet
    features  345 causal features per ICU hour, cached per split
    explore   univariate screen, collinearity, cross-site drift
    train     clinical rule, logistic regression, XGBoost, causal GRU
    evaluate  calibration, blending, bootstrap CIs, lead time, figures, REPORT.md

The split contract is fixed everywhere: hyperparameters and blend weights are
chosen on validation, the internal test set is scored once, and hospital B is
never touched until the final evaluation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CFG, SPARSE_LABS, Config
from .data.loader import make_splits, split_summary
from .evaluate import plots
from .evaluate.lead_time import alert_timing, lead_time_histogram, lead_time_summary
from .evaluate.metrics import UtilityScorer, cluster_bootstrap_ci, delong_test, evaluate
from .features import build_features
from .features.builder import feature_columns, matrices
from .models import calibration as calib
from .models import ensemble as ens
from .models import logistic as lr_mod
from .models import xgb as xgb_mod
from .models.common import ModelArtifact
from .stats.drift import drift_report
from .stats.multicollinearity import collinearity_report
from .stats.univariate import missingness_is_informative, screen

SPLITS = ("train", "val", "test", "external")
MODELS = ("clinical_rule", "logistic", "xgboost", "gru")


def _log(stage: str, msg: str) -> None:
    print(f"[{stage}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Stage 1-2: data and features
# --------------------------------------------------------------------------
def stage_features(cfg: Config = CFG, force: bool = False) -> dict[str, pd.DataFrame]:
    cfg.ensure_dirs()
    paths = {s: cfg.processed_dir / f"{s}.parquet" for s in SPLITS}
    if not force and all(p.exists() for p in paths.values()):
        _log("features", "using cached feature tables")
        frames = {s: pd.read_parquet(p) for s, p in paths.items()}
        _write_split_summary(frames, cfg)
        return frames

    t = time.time()
    splits = make_splits(cfg)
    _log("features", "split sizes:\n" + split_summary(splits).to_string())

    built = {}
    for name, split in splits.items():
        frame = build_features(split.frame, cfg)
        frame.to_parquet(paths[name], index=False)
        built[name] = frame
    _log("features", f"built {built['train'].shape[1] - 5} features in {time.time() - t:.0f}s")
    _write_split_summary(built, cfg)
    return built


def _write_split_summary(frames: dict[str, pd.DataFrame], cfg: Config) -> None:
    """Cohort table for the report, derived from whatever is on disk."""
    rows = {}
    for name, frame in frames.items():
        by_patient = frame.groupby("patient_id", observed=True)["SepsisLabel"].max()
        rows[name] = {
            "hours": len(frame),
            "admissions": int(by_patient.size),
            "septic_admissions": int(by_patient.sum()),
            "septic_admission_rate": float(by_patient.mean()),
            "positive_hour_rate": float(frame["SepsisLabel"].mean()),
        }
    pd.DataFrame(rows).T.to_csv(cfg.reports_dir / "split_summary.csv", index_label="split")


# --------------------------------------------------------------------------
# Stage 3: statistical analysis
# --------------------------------------------------------------------------
def stage_explore(frames: dict[str, pd.DataFrame], cfg: Config = CFG) -> dict:
    train = frames["train"]
    features = feature_columns(train)

    t = time.time()
    scr = screen(train, features)
    scr.to_csv(cfg.reports_dir / "univariate_screen.csv", index=False)
    n_sig = int(scr["significant"].sum())
    _log(
        "explore",
        f"{n_sig}/{len(scr)} features separate septic from non-septic admissions "
        f"at FDR<0.05 ({time.time() - t:.0f}s)",
    )

    informative = missingness_is_informative(train, SPARSE_LABS)
    informative.to_csv(cfg.reports_dir / "missingness_tests.csv", index=False)
    if len(informative):
        top = informative.iloc[0]
        _log(
            "explore",
            f"ordering alone is informative: {top['channel']} order-rate AUC="
            f"{top['auc_of_ordering']:.3f}, {int(informative['significant'].sum())}"
            f"/{len(informative)} labs significant",
        )

    priority = scr.set_index("feature")["abs_g"]
    collin = collinearity_report(train[features].sample(min(60_000, len(train)), random_state=cfg.seed), priority)
    kept = collin.pop("kept")
    (cfg.reports_dir / "pruned_features.json").write_text(json.dumps(kept, indent=2))
    _log(
        "explore",
        f"collinearity: max VIF {collin['max_vif']:.0f}, "
        f"{collin['n_vif_over_10']} features over 10; pruned {len(features)} -> {len(kept)}",
    )

    dr = drift_report(train, frames["external"], features)
    dr.to_csv(cfg.reports_dir / "drift_report.csv", index=False)
    material = int((dr["psi"] > 0.25).sum())
    _log("explore", f"cross-site drift: {material}/{len(dr)} features with PSI > 0.25")

    plots.effect_sizes(scr, cfg)
    plots.drift(dr, cfg)

    return {
        "univariate": scr,
        "missingness": informative,
        "collinearity": collin,
        "pruned_features": kept,
        "drift": dr,
    }


# --------------------------------------------------------------------------
# Stage 4: training
# --------------------------------------------------------------------------
def _store_predictions(name: str, split: str, frame: pd.DataFrame, scores: np.ndarray, cfg: Config) -> None:
    pd.DataFrame(
        {
            "patient_id": frame["patient_id"].to_numpy(),
            "hour": frame["hour"].to_numpy(),
            "y": frame["SepsisLabel"].to_numpy(),
            "score": np.asarray(scores, dtype=np.float32),
        }
    ).to_parquet(cfg.artifacts_dir / f"preds_{name}_{split}.parquet", index=False)


def stage_train(
    frames: dict[str, pd.DataFrame],
    explore: dict,
    cfg: Config = CFG,
    skip: tuple[str, ...] = (),
) -> dict:
    cfg.ensure_dirs()
    train, val = frames["train"], frames["val"]
    features = feature_columns(train)
    pruned = explore["pruned_features"]
    Xtr, ytr, gtr = matrices(train)
    info: dict = {}

    # -- baseline: bedside criteria, no learning at all --------------------
    if "clinical_rule" not in skip:
        for split, frame in frames.items():
            _store_predictions("clinical_rule", split, frame, lr_mod.clinical_rule_baseline(frame), cfg)
        _log("train", "clinical rule baseline scored")

    # -- logistic regression ----------------------------------------------
    if "logistic" not in skip:
        t = time.time()
        art = lr_mod.fit_logistic(train[features], ytr, name="logistic", cfg=cfg)
        art.save(cfg)
        for split, frame in frames.items():
            _store_predictions("logistic", split, frame, art.estimator.predict_proba(frame[features])[:, 1], cfg)
        _log("train", f"logistic regression fitted on {len(features)} features in {time.time() - t:.0f}s")

        signature = lr_mod.sparse_signature(train[features], ytr, seed=cfg.seed)
        signature.to_csv(cfg.reports_dir / "sparse_signature.csv", index=False)
        _log("train", f"elastic-net keeps {len(signature)}/{len(features)} features")

        inference = lr_mod.inference_table(train[pruned], ytr, seed=cfg.seed)
        inference.to_csv(cfg.reports_dir / "logistic_inference.csv", index=False)
        _log(
            "train",
            f"unpenalised inference on {len(pruned)} pruned features: "
            f"pseudo-R2={inference['pseudo_r2'].iloc[0]:.3f}, "
            f"{int((inference['q_value'] < 0.05).sum())} with q<0.05",
        )
        info["signature"] = signature
        info["inference"] = inference

    # -- gradient boosting -------------------------------------------------
    if "xgboost" not in skip:
        t = time.time()
        params, rounds, history = xgb_mod.tune(Xtr, ytr, gtr, cfg=cfg)
        history.to_csv(cfg.reports_dir / "optuna_trials.csv", index=False)
        plots.optuna_history(history, cfg)
        _log(
            "train",
            f"Optuna: {len(history)} trials "
            f"({int((history['state'] == 'PRUNED').sum())} pruned) in {time.time() - t:.0f}s; "
            f"best CV utility {history['value'].max():.4f}",
        )

        art = xgb_mod.fit_xgboost(
            train[features], ytr, params, max(rounds * 2, 200),
            X_valid=val[features], y_valid=val["SepsisLabel"].to_numpy(),
            name="xgboost", cfg=cfg,
        )
        art.save(cfg)
        for split, frame in frames.items():
            _store_predictions("xgboost", split, frame, xgb_mod.predict(art, frame[features]), cfg)
        _log("train", f"XGBoost final fit: {art.params['num_boost_round']} rounds")

        gain = xgb_mod.importance_table(art)
        gain.to_csv(cfg.reports_dir / "xgb_importance.csv", index=False)
        _, shap_rank = xgb_mod.shap_summary(art, val[features], seed=cfg.seed)
        shap_rank.to_csv(cfg.reports_dir / "shap_ranking.csv", index=False)
        plots.importance(gain, shap_rank, cfg)
        info.update({"xgb_params": params, "gain": gain, "shap": shap_rank})

    # -- recurrent network -------------------------------------------------
    if "gru" not in skip:
        t = time.time()
        from .models.deep import predict_sequences, train_sequence_model

        model, encoder, hist = train_sequence_model(train, val, cfg=cfg, epochs=25, verbose=0)
        model.save(cfg.artifacts_dir / "gru.keras")
        for split, frame in frames.items():
            ordered = frame.sort_values(["patient_id", "hour"], ignore_index=True)
            _store_predictions("gru", split, ordered, predict_sequences(model, encoder, ordered), cfg)
        _log(
            "train",
            f"causal GRU: {model.count_params():,} params, "
            f"{len(hist.history['loss'])} epochs in {time.time() - t:.0f}s",
        )
        info["gru_history"] = pd.DataFrame(hist.history)
        info["gru_history"].to_csv(cfg.reports_dir / "gru_history.csv", index=False)

    return info


# --------------------------------------------------------------------------
# Stage 5: evaluation
# --------------------------------------------------------------------------
def _load_predictions(name: str, split: str, cfg: Config) -> pd.DataFrame | None:
    path = cfg.artifacts_dir / f"preds_{name}_{split}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path).sort_values(["patient_id", "hour"], ignore_index=True)


def stage_evaluate(cfg: Config = CFG, models: tuple[str, ...] = MODELS) -> dict:
    available = [m for m in models if _load_predictions(m, "test", cfg) is not None]
    if not available:
        raise RuntimeError("no cached predictions found -- run the train stage first")

    preds = {
        split: {m: _load_predictions(m, split, cfg) for m in available} for split in SPLITS
    }
    ref = preds["val"][available[0]]
    y_val, g_val = ref["y"].to_numpy(), ref["patient_id"].to_numpy()
    scorer_val = UtilityScorer(y_val, g_val)

    # -- recalibrate on validation, then freeze ----------------------------
    calibrators = {
        m: calib.Calibrator("isotonic").fit(preds["val"][m]["score"].to_numpy(), y_val)
        for m in available
    }

    def calibrated(split: str, m: str) -> np.ndarray:
        return calibrators[m].transform(preds[split][m]["score"].to_numpy())

    # -- blend weights, chosen on validation only --------------------------
    learned = [m for m in available if m != "clinical_rule"]
    val_members = {m: calibrated("val", m) for m in learned}
    weights = ens.optimise_weights(val_members, y_val, g_val, cfg.seed)
    _log("evaluate", "blend weights: " + ", ".join(f"{k}={v:.2f}" for k, v in weights.items()))
    contributions = ens.contribution_table(val_members, weights, y_val, g_val)
    contributions.to_csv(cfg.reports_dir / "ensemble_contributions.csv", index=False)

    # The blend lives in rank space, which orders correctly but is not a
    # probability -- so it gets its own calibration map, fitted on validation
    # like every other one. Without this the ensemble's Brier score and ECE are
    # meaningless even though its AUROC and utility are fine.
    val_blend_raw = ens.blend(val_members, weights)
    blend_calibrator = calib.Calibrator("isotonic").fit(val_blend_raw, y_val)

    def blended(split: str) -> np.ndarray:
        members = {m: calibrated(split, m) for m in learned}
        return blend_calibrator.transform(ens.blend(members, weights))

    # -- operating points fixed on validation ------------------------------
    thresholds = {m: scorer_val.best_threshold(calibrated("val", m))[0] for m in available}
    thresholds["ensemble"] = scorer_val.best_threshold(blended("val"))[0]

    # -- score every split at the frozen thresholds ------------------------
    rows, sweeps, curve_data, timings = [], {}, {}, {}
    for split in ("test", "external"):
        ref_s = preds[split][available[0]]
        y, g = ref_s["y"].to_numpy(), ref_s["patient_id"].to_numpy()
        scorer = UtilityScorer(y, g)

        split_scores = {m: calibrated(split, m) for m in available}
        split_scores["ensemble"] = blended(split)

        for m, s_ in split_scores.items():
            met = evaluate(y, s_, g, scorer, threshold=thresholds[m])
            met.update({"model": m, "split": split})
            if split == "test":
                point, lo, hi = cluster_bootstrap_ci(y, s_, g, "auroc", n_boot=300, seed=cfg.seed)
                met.update({"auroc_ci_low": lo, "auroc_ci_high": hi})
                sweeps[m] = scorer.sweep(s_)
                curve_data[m] = (y, s_)
                timing = alert_timing(y, s_, g, thresholds[m])
                timings[m] = (timing, lead_time_summary(timing))
            rows.append(met)

    # -- calibration quality, measured out of sample -----------------------
    # Reporting these on validation would be circular: isotonic is fitted there,
    # so its in-sample ECE is 0 by construction. The test split is the first data
    # the calibration map has not seen.
    y_test = preds["test"][available[0]]["y"].to_numpy()
    calibration_rows, curves = [], {}
    for m in available:
        raw_test = preds["test"][m]["score"].to_numpy()
        row = calib.calibration_report(y_test, raw_test, calibrated("test", m))
        row["model"] = m
        calibration_rows.append(row)
        if m in ("xgboost", "logistic"):
            curves[f"{m} raw"] = calib.reliability_curve(y_test, raw_test)
            curves[f"{m} calibrated"] = calib.reliability_curve(y_test, calibrated("test", m))
    calibration_table = pd.DataFrame(calibration_rows).set_index("model")
    calibration_table.to_csv(cfg.reports_dir / "calibration.csv")
    plots.calibration(curves, cfg, tag="test")
    _log("evaluate", "calibration (held-out test):\n" + calibration_table.round(4).to_string())

    results = pd.DataFrame(rows)
    results.to_csv(cfg.reports_dir / "results.csv", index=False)

    # -- figures -----------------------------------------------------------
    plots.discrimination(curve_data, cfg, tag="test")
    plots.utility_curves(sweeps, cfg, tag="test")
    best = max(timings, key=lambda m: timings[m][1]["detection_rate"])
    plots.lead_time(lead_time_histogram(timings[best][0]), timings[best][1], cfg, tag="test", model=best)

    wide = results.pivot(index="model", columns="split", values=["auroc", "auprc", "utility"])
    wide.columns = [f"{split}_{metric}" for metric, split in wide.columns]
    wide = wide.reset_index()
    plots.external_validation(wide, cfg)

    # -- is the best model's edge real? ------------------------------------
    test_rows = results[results["split"] == "test"].set_index("model")
    ranked = test_rows["auroc"].sort_values(ascending=False)
    comparisons = []
    all_test = {m: calibrated("test", m) for m in available}
    all_test["ensemble"] = blended("test")
    champion = ranked.index[0]
    for m in ranked.index[1:]:
        d = delong_test(y_test, all_test[champion], all_test[m])
        d.update({"champion": champion, "challenger": m})
        comparisons.append(d)
    comparison_table = pd.DataFrame(comparisons)
    comparison_table.to_csv(cfg.reports_dir / "delong_comparisons.csv", index=False)

    lead_summary = pd.DataFrame({m: s for m, (_, s) in timings.items()}).T
    lead_summary.to_csv(cfg.reports_dir / "lead_time.csv")

    _log("evaluate", "results:\n" + results.round(4).to_string(index=False))
    _log("evaluate", "lead time (test):\n" + lead_summary.round(3).to_string())

    payload = {
        "results": results.to_dict("records"),
        "weights": weights,
        "thresholds": thresholds,
        "calibration": calibration_table.reset_index().to_dict("records"),
        "delong": comparisons,
        "lead_time": lead_summary.reset_index(names="model").to_dict("records"),
        "contributions": contributions.to_dict("records"),
    }
    (cfg.reports_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=float))
    return payload


# --------------------------------------------------------------------------
# Stage 4b: replay
# --------------------------------------------------------------------------
def stage_replay(cfg: Config = CFG) -> dict:
    """Pick and pre-compute the admissions the report replays hour by hour."""
    from . import replay

    cfg.ensure_dirs()
    t = time.time()
    payload = replay.write(cfg)
    _log("replay", f"{len(payload['cases'])} admissions in {time.time() - t:.0f}s")
    return payload


# --------------------------------------------------------------------------
# Stage 4c: experiments
# --------------------------------------------------------------------------
def stage_experiments(cfg: Config = CFG, only: tuple[str, ...] = ()) -> dict:
    """Run the registered experiments and persist each one's table, prose and metadata.

    Each experiment validates its own result before returning, so anything that
    reaches disk here has already asserted that it measured what it claims to
    measure. The report generator reads these files; it never re-runs anything.
    """
    from .experiments import REGISTRY

    cfg.ensure_dirs()
    results = {}
    for name, fn in REGISTRY.items():
        if only and name not in only:
            continue
        t = time.time()
        result = fn(cfg=cfg)
        result.table.to_csv(cfg.reports_dir / f"experiment_{name}.csv", index=False)
        (cfg.reports_dir / f"experiment_{name}.md").write_text(
            f"### {result.title}\n\n{result.prose}\n"
        )
        (cfg.reports_dir / f"experiment_{name}.json").write_text(
            json.dumps(result.metadata, indent=2, default=float)
        )
        _log("experiments", f"{name}: {len(result.table)} rows in {time.time() - t:.0f}s")
        results[name] = result
    return results


def run_all(cfg: Config = CFG, skip: tuple[str, ...] = ()) -> dict:
    frames = stage_features(cfg)
    explore = stage_explore(frames, cfg)
    stage_train(frames, explore, cfg, skip=skip)
    payload = stage_evaluate(cfg)
    stage_replay(cfg)
    stage_experiments(cfg)
    return payload
