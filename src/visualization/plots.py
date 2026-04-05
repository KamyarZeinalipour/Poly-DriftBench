"""
Visualization Module
====================
Publication-ready plots for the Token Squeeze Hypothesis.

Generates:
    1. Decay curves (DDM score vs. turn number per language)
    2. Fertility bar charts (TFR across languages and models)
    3. SPAR heatmaps (attention to system prompt per layer vs. turn)
    4. Regression scatter plots (TFR vs. Drift Onset Point)
    5. Multi-level constraint breakdown charts
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Style Configuration
# ──────────────────────────────────────────────────────────

# Publication-quality defaults
STYLE_CONFIG = {
    "figure.figsize": (10, 6),
    "figure.dpi": 300,
    "font.size": 12,
    "font.family": "serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Color palette for languages
LANG_COLORS = {
    "en": "#2E86AB",    # Steel blue
    "it": "#A23B72",    # Magenta
    "es": "#F18F01",    # Orange
    "fr": "#C73E1D",    # Red
    "de": "#3B1F2B",    # Dark purple
}

LANG_LABELS = {
    "en": "English",
    "it": "Italian",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
}

EXPANSION_STYLES = {
    "bte": {"linestyle": "--", "marker": "s", "label": "EN (BTE)"},
    "cpi": {"linestyle": "-.", "marker": "^", "label": "EN (CPI)"},
    "cri": {"linestyle": ":", "marker": "D", "label": "EN (CRI)"},
}


def apply_style():
    """Apply publication-quality matplotlib style."""
    mpl.rcParams.update(STYLE_CONFIG)
    sns.set_palette("colorblind")


# ──────────────────────────────────────────────────────────
# Plot 1: Decay Curves
# ──────────────────────────────────────────────────────────

def plot_decay_curves(
    drift_df: pd.DataFrame,
    output_path: str | Path = "results/figures/decay_curves.pdf",
    title: str = "Instruction Drift Decay Curves",
    show_confidence: bool = True,
) -> Path:
    """
    Plot DDM score vs. turn number for each language.

    Args:
        drift_df: DataFrame with columns: [turn, ddm_score, language, model].
        output_path: Where to save the figure.
        title: Plot title.
        show_confidence: Whether to show confidence bands.
    """
    apply_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = drift_df["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)

    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        model_data = drift_df[drift_df["model"] == model]

        for lang in sorted(model_data["language"].unique()):
            lang_data = model_data[model_data["language"] == lang]
            color = LANG_COLORS.get(lang, "#888888")
            label = LANG_LABELS.get(lang, lang.upper())

            # Mean and confidence band across conversations
            grouped = lang_data.groupby("turn")["ddm_score"]
            means = grouped.mean()
            stds = grouped.std()

            ax.plot(
                means.index, means.values,
                color=color, linewidth=2, label=label,
                marker="o", markersize=3,
            )

            if show_confidence and not stds.isna().all():
                ax.fill_between(
                    means.index,
                    means.values - stds.values,
                    means.values + stds.values,
                    color=color, alpha=0.15,
                )

        ax.set_xlabel("Conversational Turn")
        ax.set_ylabel("DDM Score")
        ax.set_title(f"{model}")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.75, color="gray", linestyle="--", alpha=0.5, label="Threshold")
        ax.legend(loc="lower left")

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved decay curves: {output_path}")
    return output_path


# ──────────────────────────────────────────────────────────
# Plot 2: Fertility Bar Chart
# ──────────────────────────────────────────────────────────

def plot_fertility_bars(
    fertility_df: pd.DataFrame,
    output_path: str | Path = "results/figures/fertility_bars.pdf",
) -> Path:
    """
    Bar chart of Token Fertility Ratios across languages and models.
    """
    apply_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    languages = fertility_df["language"].unique()
    models = fertility_df["model"].unique()
    x = np.arange(len(languages))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        model_data = fertility_df[fertility_df["model"] == model]
        tfrs = [
            model_data[model_data["language"] == lang]["tfr"].values[0]
            if lang in model_data["language"].values else 0
            for lang in languages
        ]
        bars = ax.bar(
            x + i * width, tfrs, width,
            label=model, alpha=0.85,
        )
        # Add value labels
        for bar, tfr in zip(bars, tfrs):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{tfr:.3f}", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xlabel("Language")
    ax.set_ylabel("Token Fertility Ratio (TFR)")
    ax.set_title("Token Fertility Ratios Across Languages and Models", fontweight="bold")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([LANG_LABELS.get(l, l) for l in languages])
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Baseline (EN)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved fertility bars: {output_path}")
    return output_path


# ──────────────────────────────────────────────────────────
# Plot 3: SPAR Heatmaps
# ──────────────────────────────────────────────────────────

def plot_spar_heatmap(
    per_layer_curves: list[list[float]],
    turn_numbers: list[int],
    language: str,
    model_name: str,
    output_path: str | Path = "results/figures/spar_heatmap.pdf",
) -> Path:
    """
    Heatmap of SPAR per layer vs. conversational turn.

    Args:
        per_layer_curves: Shape [num_layers, num_turns].
        turn_numbers: Turn indices.
        language: Language code.
        model_name: Model name.
    """
    apply_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.array(per_layer_curves)

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(
        data, aspect="auto", cmap="YlOrRd_r",
        interpolation="nearest",
    )

    ax.set_xlabel("Conversational Turn")
    ax.set_ylabel("Transformer Layer")
    ax.set_title(
        f"System Prompt Attention (SPAR) — {LANG_LABELS.get(language, language)} / {model_name}",
        fontweight="bold",
    )

    # Tick labels
    ax.set_xticks(range(len(turn_numbers)))
    ax.set_xticklabels(turn_numbers, rotation=45)
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels([f"L{i}" for i in range(data.shape[0])])

    plt.colorbar(im, ax=ax, label="SPAR Score", shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved SPAR heatmap: {output_path}")
    return output_path


# ──────────────────────────────────────────────────────────
# Plot 4: Regression Scatter
# ──────────────────────────────────────────────────────────

def plot_regression(
    fertility_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    output_path: str | Path = "results/figures/regression.pdf",
) -> Path:
    """
    Scatter plot: TFR vs. Drift Onset Point with OLS regression line.
    """
    apply_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged = pd.merge(fertility_df, drift_df, on=["model", "language"], how="inner")

    fig, ax = plt.subplots(figsize=(8, 6))

    for lang in merged["language"].unique():
        lang_data = merged[merged["language"] == lang]
        color = LANG_COLORS.get(lang, "#888888")
        label = LANG_LABELS.get(lang, lang)
        ax.scatter(
            lang_data["tfr"], lang_data["drift_onset"],
            color=color, s=80, label=label, edgecolors="white", linewidth=0.5,
            zorder=5,
        )

    # Fit regression line
    from numpy.polynomial import polynomial as P
    x = merged["tfr"].values
    y = merged["drift_onset"].values
    coeffs = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min() - 0.05, x.max() + 0.05, 100)
    y_line = np.polyval(coeffs, x_line)

    r_squared = 1 - np.sum((y - np.polyval(coeffs, x))**2) / np.sum((y - y.mean())**2)

    ax.plot(
        x_line, y_line, color="black", linewidth=2, linestyle="--",
        label=f"OLS (R²={r_squared:.3f})",
    )

    ax.set_xlabel("Token Fertility Ratio (TFR)")
    ax.set_ylabel("Drift Onset Point (Turn #)")
    ax.set_title("Fertility → Drift Regression", fontweight="bold")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved regression plot: {output_path}")
    return output_path


# ──────────────────────────────────────────────────────────
# Plot 5: Multi-Level Constraint Breakdown
# ──────────────────────────────────────────────────────────

def plot_constraint_breakdown(
    drift_df: pd.DataFrame,
    model_name: str,
    output_path: str | Path = "results/figures/constraint_breakdown.pdf",
) -> Path:
    """
    Stacked area chart showing which constraint levels fail first.
    """
    apply_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_data = drift_df[drift_df["model"] == model_name]
    languages = sorted(model_data["language"].unique())

    fig, axes = plt.subplots(1, len(languages), figsize=(5 * len(languages), 4), sharey=True)
    if len(languages) == 1:
        axes = [axes]

    constraint_cols = ["l1_format", "l2_structure", "l3_lexical", "l4_citation"]
    constraint_labels = ["L1: Format Tag", "L2: Bullet Points", "L3: No 'however'", "L4: Citation"]
    colors = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A"]

    for ax, lang in zip(axes, languages):
        lang_data = model_data[model_data["language"] == lang]
        grouped = lang_data.groupby("turn")[constraint_cols].mean()

        for col, label, color in zip(constraint_cols, constraint_labels, colors):
            ax.plot(
                grouped.index, grouped[col],
                label=label, linewidth=2, color=color,
            )

        ax.set_xlabel("Turn")
        ax.set_ylabel("Pass Rate")
        ax.set_title(f"{LANG_LABELS.get(lang, lang)}", fontweight="bold")
        ax.set_ylim(-0.05, 1.05)

    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(
        f"Constraint-Level Decay Breakdown — {model_name}",
        fontsize=14, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved constraint breakdown: {output_path}")
    return output_path
