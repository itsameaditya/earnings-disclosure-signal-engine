"""Figures for the evaluation report.

Palette and chart rules follow a validated categorical palette: slots are
assigned in fixed order and never cycled, magnitude uses one axis per panel
(no dual-axis comparisons - the two metrics below live in separate panels
precisely because AUC and Brier skill are not on a common scale), and chrome is
recessive.

Slot 3 (aqua) sits below 3:1 contrast on the light surface, so every figure ships
visible direct labels and writes a companion CSV table - the relief the contrast
warning requires, and useful on its own for anyone who wants the numbers.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless: figures are written to disk, never displayed
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

# --- Validated palette (light mode) ------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e8e7e4"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # blue, orange, aqua - fixed order
REFERENCE = "#a8a7a2"  # recessive rule for the perfect-calibration diagonal


def _style_axes(ax, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    # Solid hairlines only; dashed grids read as thresholds.
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-", alpha=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)


def _save(fig, path: Path, table: pd.DataFrame | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    if table is not None:
        table.to_csv(path.with_suffix(".csv"), index=False)
    log.info("wrote %s", path)
    return path


def plot_reliability(
    y_true: np.ndarray, probs: np.ndarray, out_path: Path, n_bins: int = 10, title: str = ""
) -> Path:
    """Reliability curve with a predicted-probability histogram beneath it.

    The histogram is not decoration: a calibration curve over a narrow band of
    predictions can look excellent while the model is barely differentiating
    anything, and the histogram is what makes that visible.
    """
    from ..model import reliability_curve

    pred, obs, counts = reliability_curve(y_true, probs, n_bins)

    fig, (ax, ax_hist) = plt.subplots(
        2, 1, figsize=(7, 7), height_ratios=[3, 1], sharex=True,
        gridspec_kw={"hspace": 0.12}, facecolor=SURFACE,
    )

    ax.plot([0, 1], [0, 1], color=REFERENCE, linewidth=1.5, zorder=1)
    ax.annotate(
        "perfect calibration", xy=(0.72, 0.72), xytext=(0.74, 0.62),
        color=REFERENCE, fontsize=9,
    )
    ax.plot(pred, obs, color=SERIES[0], linewidth=2.0, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)

    # Selective direct labels: the endpoints carry the story; the rest is the axis.
    for i in (0, len(pred) - 1):
        ax.annotate(
            f"{obs[i]:.2f}", xy=(pred[i], obs[i]), xytext=(6, -12),
            textcoords="offset points", color=INK_SOFT, fontsize=9,
        )

    lo = min(pred.min(), obs.min(), 0.0)
    hi = max(pred.max(), obs.max(), 1.0)
    pad = 0.04 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    _style_axes(ax, ylabel="Observed frequency",
                title=title or "Calibration: predicted vs. observed vol expansion")

    ax_hist.bar(pred, counts, width=0.035, color=SERIES[0], alpha=0.85)
    _style_axes(ax_hist, xlabel="Predicted probability", ylabel="Events")

    table = pd.DataFrame(
        {"mean_predicted": pred, "observed_frequency": obs, "n_events": counts}
    )
    return _save(fig, out_path, table)


def plot_ablation(results: list, out_path: Path, claim_source: str = "LLM") -> Path:
    """Ablation comparison as small multiples - one panel per metric.

    AUC and Brier skill live on different scales, so they get separate panels
    rather than a second y-axis.
    """
    frame = pd.DataFrame([r.row() for r in results])
    labels = [n.replace("_", " ") for n in frame["model"]]
    y = np.arange(len(frame))

    fig, axes = plt.subplots(1, 2, figsize=(11, 0.9 * len(frame) + 2.6), facecolor=SURFACE)

    for ax, (col, label) in zip(
        axes, [("auc", "ROC AUC"), ("brier_skill", "Brier skill vs. base rate")]
    ):
        values = frame[col].to_numpy()
        # One series per panel -> one color. A value-ramp here would double-encode
        # bar length as hue.
        ax.barh(y, values, height=0.54, color=SERIES[0])
        ax.set_yticks(y, labels, color=INK, fontsize=10)
        ax.invert_yaxis()
        for i, v in enumerate(values):
            ax.annotate(
                f"{v:.3f}",
                xy=(v, i),
                xytext=(6 if v >= 0 else -6, 0),
                textcoords="offset points",
                va="center",
                ha="left" if v >= 0 else "right",
                color=INK,
                fontsize=10,
            )
        if col == "auc":
            ax.axvline(0.5, color=REFERENCE, linewidth=1.5)
            # Axes-fraction y keeps the annotation inside the view regardless of
            # bar count and the inverted y-axis.
            # Sits just above the frame so it never overlaps the top bar.
            ax.annotate(
                "coin flip", xy=(0.5, 1.0), xycoords=("data", "axes fraction"),
                xytext=(4, 5), textcoords="offset points",
                color=REFERENCE, fontsize=9, va="bottom",
            )
            ax.set_xlim(0.4, max(0.75, values.max() * 1.18))
        else:
            ax.axvline(0.0, color=REFERENCE, linewidth=1.5)
            span = max(abs(values.min()), abs(values.max()), 0.02)
            ax.set_xlim(-span * 1.35, span * 1.35)
        _style_axes(ax, xlabel=label)

    fig.suptitle(
        f"Does the {claim_source} claim block add signal over market-state controls?",
        color=INK, fontsize=13, x=0.01, ha="left", y=1.02,
    )
    return _save(fig, out_path, frame)


def plot_extraction_quality(
    llm_table: pd.DataFrame, baseline_table: pd.DataFrame | None, out_path: Path
) -> Path:
    """Per-field extraction accuracy, LLM vs. rule-based baseline."""
    merged = llm_table[["field", "accuracy"]].rename(columns={"accuracy": "LLM"})
    if baseline_table is not None and not baseline_table.empty:
        merged = merged.merge(
            baseline_table[["field", "accuracy"]].rename(columns={"accuracy": "Baseline"}),
            on="field", how="left",
        )
    merged = merged.sort_values("LLM")

    series_cols = [c for c in ("LLM", "Baseline") if c in merged.columns]
    y = np.arange(len(merged))
    # Gap between grouped bars so fills never touch.
    height = 0.36 if len(series_cols) == 2 else 0.62
    offsets = [-height / 2 - 0.02, height / 2 + 0.02] if len(series_cols) == 2 else [0.0]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(merged) + 2.4), facecolor=SURFACE)
    for (col, colour, offset) in zip(series_cols, SERIES, offsets):
        values = merged[col].to_numpy(dtype=float)
        ax.barh(y + offset, values, height=height, color=colour, label=col)
        for i, v in enumerate(values):
            if np.isfinite(v):
                ax.annotate(f"{v:.2f}", xy=(v, y[i] + offset), xytext=(5, 0),
                            textcoords="offset points", va="center",
                            color=INK_SOFT, fontsize=8)

    ax.set_yticks(y, merged["field"], fontsize=9, color=INK)
    ax.set_xlim(0, 1.12)
    _style_axes(ax, xlabel="Accuracy vs. hand-labeled gold",
                title="Extraction accuracy by field")
    if len(series_cols) > 1:
        legend = ax.legend(frameon=False, loc="lower right", fontsize=10)
        for text in legend.get_texts():
            text.set_color(INK_SOFT)
    return _save(fig, out_path, merged)


def plot_consistency(table: pd.DataFrame, out_path: Path) -> Path:
    """Self-consistency by field across repeated extractions."""
    frame = table.sort_values("mean_agreement")
    y = np.arange(len(frame))
    values = frame["mean_agreement"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 0.4 * len(frame) + 2.4), facecolor=SURFACE)
    ax.barh(y, values, height=0.62, color=SERIES[2])
    for i, v in enumerate(values):
        ax.annotate(f"{v:.2f}", xy=(v, i), xytext=(5, 0), textcoords="offset points",
                    va="center", color=INK_SOFT, fontsize=9)
    ax.set_yticks(y, frame["field"], fontsize=9, color=INK)
    ax.set_xlim(0, 1.1)
    _style_axes(ax, xlabel="Mean modal agreement across runs",
                title="Extraction stability (no gold labels needed)")
    return _save(fig, out_path, frame)
