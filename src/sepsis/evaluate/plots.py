"""Report figures. One concern per figure, no decoration that isn't information."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from ..config import CFG, Config

PALETTE = {
    "clinical_rule": "#9aa0a6",
    "logistic": "#3b7dd8",
    "xgboost": "#e2703a",
    "gru": "#3f9e6a",
    "ensemble": "#8e5bd0",
}
# Every piece of chrome -- text, spines, ticks, grid, legend frame -- is drawn in
# this one colour, which exists only to be found and replaced. ``_save`` rewrites it
# to ``currentColor`` in the SVG, so the figure takes its ink from whatever the page
# is using and follows the reader's theme. Nothing else in the palette is near it.
INK_SENTINEL = "#123456"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "text.color": INK_SENTINEL,
        "axes.labelcolor": INK_SENTINEL,
        "axes.titlecolor": INK_SENTINEL,
        "axes.edgecolor": INK_SENTINEL,
        "xtick.color": INK_SENTINEL,
        "ytick.color": INK_SENTINEL,
        "grid.color": INK_SENTINEL,
        "legend.edgecolor": INK_SENTINEL,
        "legend.facecolor": "none",
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
    }
)

# Curves are drawn from every distinct threshold, which on 119,000 scored hours can
# be tens of thousands of vertices -- invisible on screen and enormous in SVG.
CURVE_POINTS = 1500


def thin(*arrays, n: int = CURVE_POINTS):
    """Evenly subsample parallel curve arrays, always keeping both endpoints."""
    length = len(arrays[0])
    if length <= n:
        return arrays
    idx = np.unique(np.linspace(0, length - 1, n).astype(int))
    return tuple(np.asarray(a)[idx] for a in arrays)


def _colour(name: str) -> str:
    return PALETTE.get(name, "#555555")


def _save(fig, path: Path) -> Path:
    """Write a transparent SVG whose ink follows the page rather than fighting it.

    A PNG cannot do this: it carries its own background and its own black text, so
    on a dark page it arrives as a lit slab and the only remedy is to dim it. An SVG
    with ``currentColor`` ink and no background is theme-native by construction --
    one asset, correct in both themes, and sharp at any zoom.

    Text is left as paths (matplotlib's default) so the layout does not depend on
    the reader having the font matplotlib measured with; paths carry a fill and are
    recoloured by the same substitution as everything else.
    """
    path = path.with_suffix(".svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, transparent=True)
    plt.close(fig)

    markup = path.read_text()
    for form in (INK_SENTINEL, INK_SENTINEL.upper()):
        markup = markup.replace(form, "currentColor")
    _assert_recoloured(markup, path)
    path.write_text(markup)
    return path


def _assert_recoloured(markup: str, path: Path) -> None:
    """A figure that kept a hardcoded ink colour is unreadable in one of the themes.

    The failure is silent and theme-dependent: it looks perfect to whoever generated
    it and invisible to half the readers, so it is checked rather than trusted.
    """
    if "currentColor" not in markup:
        raise ValueError(f"{path.name}: no ink was recoloured; the sentinel never reached the SVG")
    for stray in ("#000000", "#000)", "black"):
        if stray in markup:
            raise ValueError(
                f"{path.name}: contains hardcoded {stray!r}, which will be invisible "
                f"on a dark ground. Every mark must be either currentColor or a "
                f"palette colour chosen to work on both."
            )


def discrimination(
    results: dict[str, tuple[np.ndarray, np.ndarray]], cfg: Config = CFG, tag: str = "test"
) -> Path:
    """ROC and precision-recall side by side.

    Both are shown because they answer different questions at a 1.8% positive
    rate: ROC is dominated by the enormous negative class and looks flattering,
    while PR reflects what a clinician experiences when an alert fires.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for name, (y, s) in results.items():
        fpr, tpr = thin(*roc_curve(y, s)[:2])
        axes[0].plot(fpr, tpr, label=name, color=_colour(name), lw=1.6)
        prec, rec = thin(*precision_recall_curve(y, s)[:2])
        axes[1].plot(rec, prec, label=name, color=_colour(name), lw=1.6)

    axes[0].plot([0, 1], [0, 1], ls=":", c=INK_SENTINEL, alpha=.35, lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    base = float(np.mean(next(iter(results.values()))[0]))
    axes[1].axhline(base, ls=":", c=INK_SENTINEL, alpha=.35, lw=1)
    axes[1].set(
        xlabel="Recall", ylabel="Precision",
        title=f"Precision-recall (base rate {base:.1%})",
    )
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Hour-level discrimination — {tag}", y=1.02, fontsize=11)
    return _save(fig, cfg.figures_dir / f"discrimination_{tag}.svg")


def utility_curves(
    sweeps: dict[str, pd.DataFrame], cfg: Config = CFG, tag: str = "test"
) -> Path:
    """Normalised clinical utility against decision threshold and alert rate."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for name, sweep in sweeps.items():
        c = _colour(name)
        axes[0].plot(sweep["threshold"], sweep["utility"], color=c, lw=1.6, label=name)
        axes[1].plot(sweep["alert_rate"], sweep["utility"], color=c, lw=1.6, label=name)
        peak = sweep.loc[sweep["utility"].idxmax()]
        axes[0].plot(peak["threshold"], peak["utility"], "o", color=c, ms=4)
        axes[1].plot(peak["alert_rate"], peak["utility"], "o", color=c, ms=4)

    axes[0].axhline(0, ls=":", c=INK_SENTINEL, alpha=.35, lw=1)
    axes[0].set(xlabel="Decision threshold", ylabel="Normalised utility", title="Utility vs threshold")
    axes[1].axhline(0, ls=":", c=INK_SENTINEL, alpha=.35, lw=1)
    axes[1].set(
        xscale="log", xlabel="Fraction of ICU hours alerting (log)",
        ylabel="Normalised utility", title="Utility vs alert burden",
    )
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Operating points — {tag}", y=1.02, fontsize=11)
    return _save(fig, cfg.figures_dir / f"utility_{tag}.svg")


def calibration(
    curves: dict[str, pd.DataFrame], cfg: Config = CFG, tag: str = "test"
) -> Path:
    """Reliability diagram: predicted risk against observed frequency."""
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    lim = 0.0
    for name, curve in curves.items():
        ax.plot(
            curve["predicted"], curve["observed"],
            marker="o", color=_colour(name.split(" ")[0]), lw=1.4, ms=4, label=name,
            alpha=1.0 if "calibrated" in name else 0.55,
            ls="-" if "calibrated" in name else "--",
        )
        lim = max(lim, curve["predicted"].max(), curve["observed"].max())
    ax.plot([0, lim], [0, lim], ls=":", c=INK_SENTINEL, alpha=.45, lw=1, label="perfect")
    ax.set(
        xlabel="Predicted risk", ylabel="Observed frequency",
        title=f"Reliability — {tag}", xlim=(0, lim * 1.02), ylim=(0, lim * 1.02),
    )
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, cfg.figures_dir / f"calibration_{tag}.svg")


def lead_time(
    hist: pd.DataFrame, summary: dict, cfg: Config = CFG, tag: str = "test", model: str = ""
) -> Path:
    """How much warning the model gives on the admissions it catches."""
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    centres = (hist["lead_time_low"] + hist["lead_time_high"]) / 2
    ax.bar(centres, hist["admissions"], width=np.diff(hist["lead_time_low"]).mean() * 0.9,
           color="#3f9e6a", alpha=0.85)
    median = summary.get("median_lead_time_h", np.nan)
    if np.isfinite(median):
        ax.axvline(median, color="#e2703a", lw=1.6,
                   label=f"median {median:.0f} h before onset")
        ax.legend(frameon=False, fontsize=8)
    ax.set(
        xlabel="Hours of warning before clinical onset",
        ylabel="Septic admissions",
        title=f"Alert lead time — {model or 'model'} on {tag} "
              f"({summary.get('detection_rate', float('nan')):.0%} detected before onset)",
    )
    return _save(fig, cfg.figures_dir / f"lead_time_{tag}.svg")


def effect_sizes(screen: pd.DataFrame, cfg: Config = CFG, top: int = 25) -> Path:
    """Univariate separation, FDR-adjusted, ranked by effect size not p-value."""
    top_rows = screen.head(top).iloc[::-1]
    worst_q = np.nanmax(screen["q_welch"].head(top).to_numpy())
    fig, ax = plt.subplots(figsize=(6.4, 0.26 * len(top_rows) + 1.2))
    colours = ["#3b7dd8" if g > 0 else "#e2703a" for g in top_rows["hedges_g"]]
    ax.barh(top_rows["feature"], top_rows["hedges_g"], color=colours, alpha=0.9)
    ax.axvline(0, color="#666", lw=0.8)
    subtitle = f"all q < {worst_q:.1e}" if np.isfinite(worst_q) else "FDR-adjusted"
    ax.set(
        xlabel="Hedges' g  (septic vs non-septic admissions)",
        title=f"Strongest univariate separation ({subtitle})",
    )
    ax.tick_params(axis="y", labelsize=7)
    return _save(fig, cfg.figures_dir / "univariate_effects.svg")


def drift(report: pd.DataFrame, cfg: Config = CFG, top: int = 25) -> Path:
    """Which features move most between the two hospital systems."""
    top_rows = report.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.26 * len(top_rows) + 1.2))
    colours = [
        "#c0392b" if v > 0.25 else "#e0a030" if v > 0.1 else "#3f9e6a" for v in top_rows["psi"]
    ]
    ax.barh(top_rows["feature"], top_rows["psi"], color=colours, alpha=0.9)
    for x, label in ((0.1, "moderate"), (0.25, "material")):
        ax.axvline(x, ls=":", c=INK_SENTINEL, alpha=.45, lw=1)
        ax.text(x, len(top_rows) - 0.5, f" {label}", fontsize=7, color="#777", va="top")
    ax.set(xlabel="Population Stability Index  (hospital A → hospital B)",
           title="Largest cross-site distribution shift")
    ax.tick_params(axis="y", labelsize=7)
    return _save(fig, cfg.figures_dir / "drift_psi.svg")


def importance(
    gain: pd.DataFrame, shap_rank: pd.DataFrame, cfg: Config = CFG, top: int = 20
) -> Path:
    """Split gain against mean |SHAP| -- the two disagree, and the gap is the point."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 0.28 * top + 1.4))
    g = gain.head(top).iloc[::-1]
    axes[0].barh(g["feature"], g["gain_share"], color="#e2703a", alpha=0.9)
    axes[0].set(xlabel="Share of total split gain", title="XGBoost gain")

    s = shap_rank.head(top).iloc[::-1]
    axes[1].barh(s["feature"], s["mean_abs_shap"], color="#3b7dd8", alpha=0.9)
    axes[1].set(xlabel="Mean |SHAP| (log-odds)", title="TreeSHAP attribution")
    for ax in axes:
        ax.tick_params(axis="y", labelsize=7)
    return _save(fig, cfg.figures_dir / "importance.svg")


def optuna_history(history: pd.DataFrame, cfg: Config = CFG) -> Path:
    """Search progress: every trial, and the running best."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    complete = history[history["state"] == "COMPLETE"]
    pruned = history[history["state"] == "PRUNED"]
    ax.scatter(complete["number"], complete["value"], s=16, color="#3b7dd8",
               alpha=0.7, label="completed")
    if len(pruned):
        ax.scatter(pruned["number"], [complete["value"].min()] * len(pruned), s=10,
                   marker="x", color=INK_SENTINEL, alpha=.5, label="pruned")
    ax.plot(complete["number"], complete["value"].cummax(), color="#e2703a", lw=1.8,
            label="running best")
    ax.set(xlabel="Trial", ylabel="Mean out-of-fold utility",
           title="Hyperparameter search (objective = clinical utility)")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, cfg.figures_dir / "optuna_history.svg")


def external_validation(table: pd.DataFrame, cfg: Config = CFG) -> Path:
    """Internal test versus the held-out second hospital, per model."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    models = table["model"].tolist()
    x = np.arange(len(models))
    for ax, metric, title in ((axes[0], "auroc", "AUROC"), (axes[1], "utility", "Normalised utility")):
        ax.bar(x - 0.19, table[f"test_{metric}"], 0.36, label="hospital A (test)",
               color="#3b7dd8", alpha=0.9)
        ax.bar(x + 0.19, table[f"external_{metric}"], 0.36, label="hospital B (external)",
               color="#e2703a", alpha=0.9)
        ax.set_xticks(x, models, rotation=20, ha="right", fontsize=8)
        ax.set_title(title)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Generalisation to an unseen health system", y=1.03, fontsize=11)
    return _save(fig, cfg.figures_dir / "external_validation.svg")
