#!/usr/bin/env python3
"""
evaluate_cv0.py
===============
Post-training evaluation for CV0 (leave-one-year-out) experiments.

This script reads the per-fold test-prediction CSVs produced by train_cv0.py
and generates a full metrics report and publication-quality figures.

Single-experiment mode
----------------------
  python scripts/evaluate_cv0.py --experiment-dir experiments/eva02_base_cv0

Outputs written to results/<experiment_dir.name>/evaluation/:
  metrics_per_fold.csv       Per-fold R², Pearson r, Spearman ρ, RMSE, MAE
  metrics_summary.csv        Same with mean and std rows appended
  metrics_aggregate.json     Aggregate mean ± std (machine-readable)
  figures/scatter_panel.pdf  2×2 actual-vs-predicted panel (PDF + PNG)
  figures/scatter_panel.png
  figures/signed_error_panel.pdf 2×2 signed-error-vs-actual panel (PDF + PNG)
  figures/signed_error_panel.png

Cross-model comparison mode (--compare-dirs)
---------------------------------------------
  python scripts/evaluate_cv0.py \\
      --compare-dirs experiments/eva02_base_cv0 experiments/convnextv2_base_cv0 \\
      --model-labels "EVA-02-B" "ConvNeXt V2-B" \\
      --output-dir results/model_comparison

Outputs written to <output_dir>/ (default results/model_comparison):
  model_comparison.csv  One row per model with all aggregate metrics
  model_comparison.tex  LaTeX booktabs table for direct manuscript inclusion

If an experiment has already been evaluated
(results/<model>_cv0/evaluation/metrics_aggregate.json exists), its
pre-computed aggregate metrics are reused.  Pass --force-recompute to
re-evaluate all experiments from raw prediction CSVs.

Aggregate results table mode (--aggregate-table)
--------------------------------------------------
  python scripts/evaluate_cv0.py --aggregate-table

Reads <experiments-root>/<model>_cv0/cv_results.csv for each of the 9 models
in the manuscript (no training or raw predictions required) and writes a
single long-format CSV with one row per model x fold (9 x 3 = 27 rows),
covering R^2, MAE, RMSE, and the train/val/test sample counts.  Averaging
the per-model rows reproduces the Figure 3 bar-chart values; the sample
counts reproduce Table 1.  Default output: results/cv0_aggregate_results.csv

Expected directory layout (produced by train_cv0.py)
-----------------------------------------------------
  experiments/<model>_cv0/
    predictions/
      fold_2023_test_predictions.csv
      fold_2024_test_predictions.csv
      fold_2025_test_predictions.csv

Prediction CSV columns (written by train_cv0.py)
-------------------------------------------------
  image_filename, actual, predicted, signed_error, abs_error
  where signed_error = predicted − actual
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe on headless HPC nodes
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Project path bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# All evaluation outputs are written under <project_root>/results/, keeping
# the experiments/ tree (training inputs/checkpoints) read-only.
RESULTS_ROOT = PROJECT_ROOT / "results"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate_cv0")


# ===========================================================================
# Publication style
# ===========================================================================

# pdf.fonttype=42 embeds TrueType (Type 2) fonts in the PDF output so that
# text remains selectable and editable in Adobe Illustrator / Inkscape.
# ps.fonttype=42 applies the same setting to PostScript output.
matplotlib.rcParams.update({
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "font.family":       "sans-serif",
    "font.size":         8,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   7,
    "figure.dpi":        150,
    "savefig.dpi":       150,
    # Suppress the top and right spines globally (seaborn despine is also
    # called per-axes, but this ensures the default is clean)
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# tab10 colors per test year — consistent with the year/field coloring
# convention used across the entire uav_for_slb analysis pipeline
FOLD_COLORS: Dict[str, str] = {
    "fold_2023": "#1f77b4",   # tab10 blue
    "fold_2024": "#ff7f0e",   # tab10 orange
    "fold_2025": "#2ca02c",   # tab10 green
}
# Fallback for unexpected fold names (future years, smoke-tests, etc.)
_FALLBACK_PALETTE = sns.color_palette("tab10", 10)

# SLB severity axis bounds (ordinal 1–9 scale; padding for aesthetics)
SLB_MIN = 0.5
SLB_MAX = 9.5


# ===========================================================================
# Manuscript model metadata (Figure 3 / Table 1)
# ===========================================================================
# Keys match the "<model>_cv0" experiment directory names under
# experiments/.  Kept in sync with figures/fig03_model_comparison/
# plot_model_comparison.py so the aggregate table, Figure 3, and Table 1
# all agree on model identity, display name, and architecture family.

MODEL_FAMILY: Dict[str, str] = {
    "efficientnetv2_s":  "CNN",
    "convnextv2_base":   "CNN",
    "convnextv2_large":  "CNN",
    "dinov2_vits14":     "ViT",
    "dinov2_vitb14":     "ViT",
    "eva02_base":        "ViT",
    "maxvit_small":      "Hybrid",
    "swinv2_base":       "Hybrid",
    "coatnet2":          "Hybrid",
}

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "efficientnetv2_s":  "EfficientNet V2-S",
    "convnextv2_base":   "ConvNeXt V2-B",
    "convnextv2_large":  "ConvNeXt V2-L",
    "dinov2_vits14":     "DINOv2 ViT-S/14",
    "dinov2_vitb14":     "DINOv2 ViT-B/14",
    "eva02_base":        "EVA-02-B",
    "maxvit_small":      "MaxViT-S",
    "swinv2_base":       "SwinV2-B",
    "coatnet2":          "CoAtNet-2",
}

# Preferred display order: grouped by family, then ascending parameter count
MODEL_SORT_ORDER: List[str] = [
    # CNN
    "efficientnetv2_s",
    "convnextv2_base",
    "convnextv2_large",
    # ViT
    "dinov2_vits14",
    "dinov2_vitb14",
    "eva02_base",
    # Hybrid
    "maxvit_small",
    "swinv2_base",
    "coatnet2",
]

# CV0 fold identifiers (as written in cv_results.csv) and the held-out test
# year each one corresponds to ("Leave Out <year>" in Table 1).
FOLD_TEST_YEAR: Dict[str, str] = {
    "fold_23": "2023",
    "fold_24": "2024",
    "fold_25": "2025",
}


def _fold_color(fold_name: str, idx: int) -> str:
    """Return the canonical tab10 color for a fold, with fallback."""
    return FOLD_COLORS.get(fold_name, _FALLBACK_PALETTE[idx % len(_FALLBACK_PALETTE)])


# ===========================================================================
# Metrics computation
# ===========================================================================

def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict:
    """
    Compute the full suite of regression metrics for one fold's predictions.

    Metrics
    -------
    r2           Coefficient of determination (sklearn; can be negative)
    pearson_r    Pearson product-moment correlation coefficient
    pearson_p    Two-tailed p-value for pearson_r
    spearman_rho Spearman rank correlation coefficient
    spearman_p   Two-tailed p-value for spearman_rho
    rmse         Root mean squared error
    mae          Mean absolute error
    mean_bias    Mean(actual − predicted); positive = systematic underprediction
    n            Number of observations
    """
    n = len(actual)
    mse                    = float(mean_squared_error(actual, predicted))
    r2                     = float(r2_score(actual, predicted))
    mae                    = float(mean_absolute_error(actual, predicted))
    pearson_r, pearson_p   = stats.pearsonr(actual, predicted)
    spearman_rho, spear_p  = stats.spearmanr(actual, predicted)

    return {
        "n":            n,
        "r2":           r2,
        "pearson_r":    float(pearson_r),
        "pearson_p":    float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p":   float(spear_p),
        "rmse":         float(np.sqrt(mse)),
        "mae":          mae,
        "mean_bias":    float(np.mean(actual - predicted)),
    }


def aggregate_fold_metrics(
    per_fold: Dict[str, Dict]
) -> Dict:
    """
    Compute mean and std (ddof=1) across folds for each scalar metric.

    Returns a dict with keys 'mean', 'std', and 'n_folds'.
    """
    # Metrics to aggregate — excludes p-values and sample count
    scalar_keys = ["r2", "pearson_r", "spearman_rho", "rmse", "mae", "mean_bias"]
    arrays = {k: [m[k] for m in per_fold.values()] for k in scalar_keys}

    return {
        "mean":    {k: float(np.mean(v)) for k, v in arrays.items()},
        "std":     {k: float(np.std(v, ddof=1)) for k, v in arrays.items()},
        "n_folds": len(per_fold),
    }


# ===========================================================================
# Data loading
# ===========================================================================

def load_fold_predictions(experiment_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Discover and load all per-fold prediction CSVs from the predictions/ subdir.

    Prediction files are expected to follow the naming convention written by
    train_cv0.py:  fold_YYYY_test_predictions.csv

    Returns
    -------
    dict mapping fold_name → DataFrame(image_filename, actual, predicted,
                                        signed_error, abs_error)
    """
    preds_dir = experiment_dir / "predictions"
    if not preds_dir.exists():
        raise FileNotFoundError(
            f"Predictions directory not found: {preds_dir}\n"
            "Run train_cv0.py first to generate test predictions."
        )

    csv_files = sorted(preds_dir.glob("*_test_predictions.csv"))
    if not csv_files:
        raise ValueError(f"No *_test_predictions.csv files found in {preds_dir}")

    fold_dfs: Dict[str, pd.DataFrame] = {}
    for csv_path in csv_files:
        # Derive fold name by stripping the "_test_predictions" suffix
        fold_name = csv_path.stem.replace("_test_predictions", "")
        df = pd.read_csv(csv_path)

        # Validate that the expected columns are present before proceeding
        required = {"actual", "predicted", "signed_error"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{csv_path.name} is missing columns: {missing}.  "
                "Re-run train_cv0.py to regenerate predictions."
            )

        fold_dfs[fold_name] = df
        logger.info(f"  Loaded {fold_name}: {len(df)} test samples")

    return fold_dfs


# ===========================================================================
# Figure helpers
# ===========================================================================

def _metrics_textbox(ax, m: Dict, *, fontsize: int = 7) -> None:
    """
    Place a metrics annotation box in the upper-left corner of an axes.

    Displays R², Pearson r, Spearman ρ, RMSE, MAE, and sample count N.
    Uses LaTeX math mode for the metric symbols (rendered via matplotlib's
    mathtext engine, no external LaTeX installation required).
    """
    text = (
        f"$R^2$ = {m['r2']:.3f}\n"
        f"$r$  = {m['pearson_r']:.3f}\n"
        f"$\\rho$ = {m['spearman_rho']:.3f}\n"
        f"RMSE = {m['rmse']:.3f}\n"
        f"MAE  = {m['mae']:.3f}\n"
        f"$n$ = {m['n']}"
    )
    ax.text(
        0.04, 0.97, text,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#cccccc", alpha=0.90),
        linespacing=1.5,
    )


def _scatter_cell(
    ax,
    actual:    np.ndarray,
    predicted: np.ndarray,
    metrics:   Dict,
    title:     str,
    color:     str,
) -> None:
    """
    Draw a single actual-vs-predicted panel cell.

    Components
    ----------
    - Semi-transparent scatter points (rasterized to reduce PDF file size)
    - Identity line (y = x) as a dashed gray reference
    - Metrics text box (R², r, ρ, RMSE, MAE, N)
    - Square axis limits matching the SLB 1–9 ordinal scale
    """
    ax.scatter(
        actual, predicted,
        color=color, alpha=0.45, s=12,
        edgecolors="none", linewidths=0,
        rasterized=True,    # rasterize dense point clouds to keep PDF small
    )
    # Identity line: where prediction = truth
    ax.plot(
        [SLB_MIN, SLB_MAX], [SLB_MIN, SLB_MAX],
        color="#888888", linestyle="--", linewidth=0.8, zorder=0,
    )
    ax.set_xlim(SLB_MIN, SLB_MAX)
    ax.set_ylim(SLB_MIN, SLB_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.set_title(title, pad=4)
    sns.despine(ax=ax)
    _metrics_textbox(ax, metrics)


def _signed_error_cell(
    ax,
    actual:        np.ndarray,
    signed_errors: np.ndarray,
    title:         str,
    color:         str,
) -> None:
    """
    Draw a single signed-error-vs-actual panel cell.

    Signed errors are predicted − actual.

    A binned mean line (grouping predictions by nearest integer SLB score)
    reveals systematic conditional bias without requiring a statistical
    smoother.  The expected pattern is inverted-U-shaped: models trained with
    MSE loss on an imbalanced severity distribution tend to over-predict
    high-severity plots and under-predict low-severity plots because the
    loss gradient is dominated by the majority mid-range scores.
    """
    ax.scatter(
        actual, signed_errors,
        color=color, alpha=0.45, s=12,
        edgecolors="none", linewidths=0,
        rasterized=True,
    )
    # Zero signed-error reference line (unbiased predictor benchmark)
    ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.8, zorder=0)

    # Binned mean: compute the mean signed error within each integer SLB bin (1–9).
    # Only bins with at least three observations are plotted to avoid visual
    # artefacts from single-point bins at extreme severity values.
    bins = np.arange(1, 10)
    bin_x, bin_means = [], []
    for b in bins:
        mask = (actual >= b - 0.5) & (actual < b + 0.5)
        if mask.sum() >= 3:
            bin_x.append(b)
            bin_means.append(float(np.mean(signed_errors[mask])))
    if len(bin_x) >= 2:
        ax.plot(
            bin_x, bin_means,
            color=color, linewidth=1.5, alpha=0.85,
            marker="o", markersize=3, zorder=3,
        )

    ax.set_xlim(SLB_MIN, SLB_MAX)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.set_title(title, pad=4)
    sns.despine(ax=ax)


# ===========================================================================
# Panel figure generation
# ===========================================================================

def make_scatter_panel(
    fold_dfs:         Dict[str, pd.DataFrame],
    per_fold_metrics: Dict[str, Dict],
    save_stem:        Path,
) -> None:
    """
    Generate and save the 2×2 actual-vs-predicted scatter panel.

    Layout
    ------
      [0,0] fold_2023   [0,1] fold_2024
      [1,0] fold_2025   [1,1] All Folds (pooled, colour-coded by year)

    The "All Folds" cell pools predictions from all test years; points are
    coloured by fold so the temporal separation is still visible.

    Files saved
    -----------
    <save_stem>.pdf  (vector, fonts embedded as TrueType for Illustrator)
    <save_stem>.png  (raster at 150 DPI for quick review)
    """
    fold_names = list(fold_dfs.keys())

    fig, axes = plt.subplots(2, 2, figsize=(6.0, 5.5), constrained_layout=True)

    # ---- Per-fold cells (positions [0,0], [0,1], [1,0]) ---------------------
    positions = [(0, 0), (0, 1), (1, 0)]
    for i, fold_name in enumerate(fold_names[:3]):
        row, col = positions[i]
        df       = fold_dfs[fold_name]
        color    = _fold_color(fold_name, i)
        year     = fold_name.replace("fold_", "")
        _scatter_cell(
            ax=axes[row, col],
            actual=df["actual"].values,
            predicted=df["predicted"].values,
            metrics=per_fold_metrics[fold_name],
            title=f"Test year {year}",
            color=color,
        )

    # ---- "All Folds" cell (position [1,1]) ----------------------------------
    ax_all = axes[1, 1]
    pooled_actual, pooled_pred = [], []

    for i, (fold_name, df) in enumerate(fold_dfs.items()):
        color = _fold_color(fold_name, i)
        ax_all.scatter(
            df["actual"].values, df["predicted"].values,
            color=color, alpha=0.40, s=10,
            edgecolors="none", linewidths=0,
            label=fold_name.replace("fold_", ""),   # year label for legend
            rasterized=True,
        )
        pooled_actual.extend(df["actual"].values)
        pooled_pred.extend(df["predicted"].values)

    pooled_actual = np.array(pooled_actual)
    pooled_pred   = np.array(pooled_pred)
    all_metrics   = compute_metrics(pooled_actual, pooled_pred)

    # Identity line and formatting for the All Folds cell
    ax_all.plot(
        [SLB_MIN, SLB_MAX], [SLB_MIN, SLB_MAX],
        color="#888888", linestyle="--", linewidth=0.8, zorder=0,
    )
    ax_all.set_xlim(SLB_MIN, SLB_MAX)
    ax_all.set_ylim(SLB_MIN, SLB_MAX)
    ax_all.set_aspect("equal", adjustable="box")
    ax_all.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax_all.yaxis.set_major_locator(mticker.MultipleLocator(2))
    ax_all.set_title("All Folds", pad=4)
    ax_all.legend(
        title="Test year", title_fontsize=6,
        fontsize=6, loc="lower right",
        handlelength=0.8, handletextpad=0.4,
        frameon=True, framealpha=0.85, edgecolor="#cccccc",
    )
    sns.despine(ax=ax_all)
    _metrics_textbox(ax_all, all_metrics)

    # ---- Shared axis labels (outer edges only) --------------------------------
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted SLB severity")
    for ax in axes[1, :]:
        ax.set_xlabel("Actual SLB severity")

    _save_figure(fig, save_stem)


def make_signed_error_panel(
    fold_dfs:  Dict[str, pd.DataFrame],
    save_stem: Path,
) -> None:
    """
    Generate and save the 2×2 signed-error diagnostic panel.

    Layout mirrors the scatter panel:
      [0,0] fold_2023   [0,1] fold_2024
      [1,0] fold_2025   [1,1] All Folds (pooled, colour-coded by year)

    A binned mean line in each cell exposes conditional bias: the inverted-U
    pattern (over-prediction at high severity, under-prediction at low
    severity) is a known consequence of MSE loss minimisation combined with
    an imbalanced SLB severity distribution.

    Files saved
    -----------
    <save_stem>.pdf
    <save_stem>.png
    """
    fold_names = list(fold_dfs.keys())

    fig, axes = plt.subplots(2, 2, figsize=(6.0, 5.5), constrained_layout=True)

    # ---- Per-fold cells ------------------------------------------------------
    positions = [(0, 0), (0, 1), (1, 0)]
    for i, fold_name in enumerate(fold_names[:3]):
        row, col = positions[i]
        df       = fold_dfs[fold_name]
        color    = _fold_color(fold_name, i)
        year     = fold_name.replace("fold_", "")
        ax       = axes[row, col]

        _signed_error_cell(
            ax=ax,
            actual=df["actual"].values,
            signed_errors=df["signed_error"].values,
            title=f"Test year {year}",
            color=color,
        )
        # Only label axes on the outer edges to avoid clutter
        ax.set_ylabel("Signed error (predicted − actual)" if col == 0 else "")
        ax.set_xlabel("Actual SLB severity" if row == 1 else "")

    # ---- "All Folds" cell ---------------------------------------------------
    ax_all = axes[1, 1]
    pooled_actual, pooled_signed_error = [], []

    for i, (fold_name, df) in enumerate(fold_dfs.items()):
        color = _fold_color(fold_name, i)
        ax_all.scatter(
            df["actual"].values, df["signed_error"].values,
            color=color, alpha=0.35, s=10,
            edgecolors="none", linewidths=0,
            label=fold_name.replace("fold_", ""),
            rasterized=True,
        )
        pooled_actual.extend(df["actual"].values)
        pooled_signed_error.extend(df["signed_error"].values)

    pooled_actual       = np.array(pooled_actual)
    pooled_signed_error = np.array(pooled_signed_error)

    # Zero signed-error reference
    ax_all.axhline(0.0, color="#888888", linestyle="--", linewidth=0.8, zorder=0)

    # Pooled binned mean line in black to show the overall conditional bias
    bins = np.arange(1, 10)
    bin_x, bin_means = [], []
    for b in bins:
        mask = (pooled_actual >= b - 0.5) & (pooled_actual < b + 0.5)
        if mask.sum() >= 3:
            bin_x.append(b)
            bin_means.append(float(np.mean(pooled_signed_error[mask])))
    if len(bin_x) >= 2:
        ax_all.plot(
            bin_x, bin_means,
            color="black", linewidth=1.5, alpha=0.80,
            marker="o", markersize=3, zorder=3, label="bin mean",
        )

    ax_all.set_xlim(SLB_MIN, SLB_MAX)
    ax_all.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax_all.set_title("All Folds", pad=4)
    ax_all.set_xlabel("Actual SLB severity")
    ax_all.legend(
        title="Test year", title_fontsize=6,
        fontsize=6, loc="upper right",
        handlelength=0.8, handletextpad=0.4,
        frameon=True, framealpha=0.85, edgecolor="#cccccc",
    )
    sns.despine(ax=ax_all)

    _save_figure(fig, save_stem)


def _save_figure(fig, save_stem: Path) -> None:
    """Save a figure to both PDF (vector) and PNG (raster) at the same stem."""
    for ext in (".pdf", ".png"):
        out = save_stem.with_suffix(ext)
        fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
        logger.info(f"  Saved → {out}")
    plt.close(fig)


# ===========================================================================
# Core evaluation function
# ===========================================================================

def evaluate_experiment(
    experiment_dir: Path,
    output_dir:     Optional[Path] = None,
) -> Dict:
    """
    Full evaluation pipeline for one completed CV0 experiment.

    Steps
    -----
    1. Load per-fold prediction CSVs from <experiment_dir>/predictions/
    2. Compute metrics for each fold
    3. Compute aggregate mean ± std across folds
    4. Save metrics_per_fold.csv, metrics_summary.csv, metrics_aggregate.json
    5. Generate scatter_panel.{pdf,png}
    6. Generate signed_error_panel.{pdf,png}

    Returns
    -------
    dict with keys 'per_fold' (Dict[fold_name → metrics]) and
    'aggregate' (Dict with 'mean', 'std', 'n_folds')
    """
    if output_dir is None:
        output_dir = RESULTS_ROOT / experiment_dir.name / "evaluation"
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Evaluating: {experiment_dir.name}")

    # ---- Load predictions ---------------------------------------------------
    fold_dfs   = load_fold_predictions(experiment_dir)
    fold_names = list(fold_dfs.keys())

    # ---- Per-fold metrics ---------------------------------------------------
    per_fold_metrics: Dict[str, Dict] = {}
    for fold_name, df in fold_dfs.items():
        m = compute_metrics(
            actual=df["actual"].values,
            predicted=df["predicted"].values,
        )
        per_fold_metrics[fold_name] = m
        logger.info(
            f"  {fold_name}: "
            f"R²={m['r2']:.4f}  r={m['pearson_r']:.4f}  "
            f"ρ={m['spearman_rho']:.4f}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}"
        )

    # ---- Aggregate ----------------------------------------------------------
    aggregate = aggregate_fold_metrics(per_fold_metrics)
    m_mean = aggregate["mean"]
    m_std  = aggregate["std"]
    logger.info(
        f"\n  Aggregate ({aggregate['n_folds']} folds):\n"
        f"  R²   = {m_mean['r2']:.4f} ± {m_std['r2']:.4f}\n"
        f"  r    = {m_mean['pearson_r']:.4f} ± {m_std['pearson_r']:.4f}\n"
        f"  ρ    = {m_mean['spearman_rho']:.4f} ± {m_std['spearman_rho']:.4f}\n"
        f"  RMSE = {m_mean['rmse']:.4f} ± {m_std['rmse']:.4f}\n"
        f"  MAE  = {m_mean['mae']:.4f} ± {m_std['mae']:.4f}"
    )

    # ---- Save metrics tables ------------------------------------------------
    per_fold_rows = [{"fold": fn, **m} for fn, m in per_fold_metrics.items()]
    per_fold_df   = pd.DataFrame(per_fold_rows)
    per_fold_df.to_csv(output_dir / "metrics_per_fold.csv", index=False)

    # Summary CSV: per-fold rows + mean + std rows appended for easy inspection
    summary_df = pd.concat([
        per_fold_df,
        pd.DataFrame([{"fold": "mean", **m_mean}, {"fold": "std", **m_std}]),
    ], ignore_index=True)
    summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    logger.info(f"  Saved → {output_dir / 'metrics_summary.csv'}")

    # Aggregate JSON for programmatic access by build_comparison_table
    with open(output_dir / "metrics_aggregate.json", "w") as f:
        json.dump(aggregate, f, indent=2)
    logger.info(f"  Saved → {output_dir / 'metrics_aggregate.json'}")

    # ---- Figures ------------------------------------------------------------
    logger.info("  Generating scatter panel ...")
    make_scatter_panel(fold_dfs, per_fold_metrics, fig_dir / "scatter_panel")

    logger.info("  Generating signed error panel ...")
    make_signed_error_panel(fold_dfs, fig_dir / "signed_error_panel")

    return {"per_fold": per_fold_metrics, "aggregate": aggregate}


# ===========================================================================
# Cross-model comparison table
# ===========================================================================

def build_comparison_table(
    experiment_dirs:  List[Path],
    model_labels:     Optional[List[str]],
    output_dir:       Path,
    force_recompute:  bool = False,
) -> pd.DataFrame:
    """
    Build a cross-model comparison table from multiple CV0 experiment directories.

    For each directory, the function first checks for a pre-computed
    metrics_aggregate.json (written by evaluate_experiment).  If it is absent
    or --force-recompute is set, evaluate_experiment is called to compute
    metrics from the raw prediction CSVs.

    The resulting table is sorted by mean test R² (descending) to match the
    presentation order used in the manuscript's main results table.

    Outputs
    -------
    model_comparison.csv — CSV table, one row per model
    model_comparison.tex — LaTeX booktabs table for direct manuscript inclusion

    Parameters
    ----------
    experiment_dirs   List of completed train_cv0.py output directories
    model_labels      Human-readable model names (same order as experiment_dirs)
    output_dir        Where to write comparison outputs
    force_recompute   Re-evaluate all experiments even if JSON cache exists
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_labels is None:
        model_labels = [d.name for d in experiment_dirs]

    if len(model_labels) != len(experiment_dirs):
        raise ValueError(
            f"--model-labels ({len(model_labels)}) must match "
            f"--compare-dirs ({len(experiment_dirs)})"
        )

    rows = []
    for exp_dir, label in zip(experiment_dirs, model_labels):
        json_path = RESULTS_ROOT / exp_dir.name / "evaluation" / "metrics_aggregate.json"

        if json_path.exists() and not force_recompute:
            # Re-use pre-computed aggregate to avoid re-running all evaluations
            logger.info(f"  Loading cached metrics for {label} ← {json_path}")
            with open(json_path) as f:
                agg = json.load(f)
        else:
            logger.info(f"  Computing metrics for {label} from predictions ...")
            results = evaluate_experiment(exp_dir)
            agg     = results["aggregate"]

        m_mean = agg["mean"]
        m_std  = agg["std"]
        rows.append({
            "model":          label,
            "n_folds":        agg.get("n_folds", "—"),
            "mean_r2":        m_mean["r2"],
            "std_r2":         m_std["r2"],
            "mean_pearson_r": m_mean["pearson_r"],
            "std_pearson_r":  m_std["pearson_r"],
            "mean_spearman":  m_mean["spearman_rho"],
            "std_spearman":   m_std["spearman_rho"],
            "mean_rmse":      m_mean["rmse"],
            "std_rmse":       m_std["rmse"],
            "mean_mae":       m_mean["mae"],
            "std_mae":        m_std["mae"],
        })

    # Sort by mean test R² descending — matches manuscript presentation order
    comp_df = pd.DataFrame(rows).sort_values("mean_r2", ascending=False).reset_index(drop=True)

    comp_df.to_csv(output_dir / "model_comparison.csv", index=False)
    logger.info(f"  Saved → {output_dir / 'model_comparison.csv'}")

    _write_latex_table(comp_df, output_dir / "model_comparison.tex")

    return comp_df


def _write_latex_table(df: pd.DataFrame, path: Path) -> None:
    """
    Write a LaTeX booktabs table for direct inclusion in the manuscript.

    Columns: Model | R² (mean±std) | r | ρ | RMSE | MAE
    Each metric is formatted as "mean±std" with three decimal places.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{CV0 leave-one-year-out test performance (mean $\pm$ std "
        r"across three test years). Best value per column in \textbf{bold}.}",
        r"\label{tab:cv0_results}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & $R^2$ & $r$ & $\rho$ & RMSE & MAE \\",
        r"\midrule",
    ]

    # Identify best (highest for R²/r/ρ, lowest for RMSE/MAE) for bold formatting
    best_r2   = df["mean_r2"].max()
    best_r    = df["mean_pearson_r"].max()
    best_rho  = df["mean_spearman"].max()
    best_rmse = df["mean_rmse"].min()
    best_mae  = df["mean_mae"].min()

    def _fmt(val, std, best, higher_is_better=True):
        """Format 'mean±std', bolding if this row achieves the best value."""
        cell = f"{val:.3f}$\\pm${std:.3f}"
        is_best = (val == best) if higher_is_better else (val == best)
        return f"\\textbf{{{cell}}}" if is_best else cell

    for _, row in df.iterrows():
        lines.append(
            f"{row['model']} & "
            f"{_fmt(row['mean_r2'],        row['std_r2'],        best_r2)} & "
            f"{_fmt(row['mean_pearson_r'],  row['std_pearson_r'], best_r)} & "
            f"{_fmt(row['mean_spearman'],   row['std_spearman'],  best_rho)} & "
            f"{_fmt(row['mean_rmse'],       row['std_rmse'],      best_rmse, False)} & "
            f"{_fmt(row['mean_mae'],        row['std_mae'],       best_mae, False)} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"  Saved → {path}")


# ===========================================================================
# Top-level aggregate results table (Figure 3 / Table 1 reproduction)
# ===========================================================================

def build_aggregate_results_table(
    experiments_root: Path,
    output_path:      Path,
) -> pd.DataFrame:
    """
    Build a single top-level CSV covering all 9 models x 3 folds x 3 metrics,
    reproducing the numbers behind Figure 3 (mean test R^2/MAE/RMSE per model)
    and Table 1 (per-fold train/val/test sample counts).

    Reads <experiments_root>/<model>_cv0/cv_results.csv directly -- these
    files already contain the per-fold sample sizes and test-set metrics
    written by train_cv0.py, so no training or raw prediction files are
    required.

    Parameters
    ----------
    experiments_root  Directory containing one "<model>_cv0" subdirectory
                       per model (e.g. PROJECT_ROOT / "experiments")
    output_path       Destination CSV path

    Returns
    -------
    DataFrame with one row per model x fold (9 x 3 = 27 rows).
    """
    rows = []
    for model in MODEL_SORT_ORDER:
        cv_path = experiments_root / f"{model}_cv0" / "cv_results.csv"
        if not cv_path.exists():
            logger.warning(f"  Skipping {model}: {cv_path} not found")
            continue

        cv_df = pd.read_csv(cv_path)
        for _, r in cv_df.iterrows():
            fold_name = str(r["fold_name"])
            rows.append({
                "model":        model,
                "display_name": MODEL_DISPLAY_NAMES.get(model, model),
                "family":       MODEL_FAMILY.get(model, "Unknown"),
                "fold":         int(r["fold"]),
                "fold_name":    fold_name,
                "test_year":    FOLD_TEST_YEAR.get(fold_name, fold_name),
                "n_train":      int(r["train_samples"]),
                "n_val":        int(r["val_samples"]),
                "n_test":       int(r["test_samples"]),
                "r2":           float(r["test_r2"]),
                "mae":          float(r["test_mae"]),
                "rmse":         float(r["test_rmse"]),
            })

    if not rows:
        raise FileNotFoundError(
            f"No cv_results.csv files found under {experiments_root} "
            f"for any of: {', '.join(MODEL_SORT_ORDER)}"
        )

    table = pd.DataFrame(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    logger.info(
        f"  Saved → {output_path}  "
        f"({table['model'].nunique()} models x "
        f"{table['fold_name'].nunique()} folds = {len(table)} rows)"
    )

    # Sanity check: per-model means should match the Figure 3 bar values
    means = (
        table.groupby(["model", "display_name", "family"], sort=False)
        [["r2", "mae", "rmse"]]
        .mean()
        .reindex(MODEL_SORT_ORDER, level="model")
        .reset_index()
    )
    logger.info("  Per-model means (Figure 3 check):")
    for _, r in means.iterrows():
        logger.info(
            f"    [{r['family']:<6}] {r['display_name']:<18} "
            f"R²={r['r2']:.3f}  MAE={r['mae']:.3f}  RMSE={r['rmse']:.3f}"
        )

    return table


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CV0 test predictions and generate publication figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- Single-experiment mode
    parser.add_argument(
        "--experiment-dir", type=str, default=None,
        help=(
            "Path to a completed train_cv0.py output directory "
            "(e.g. experiments/<model>_cv0). "
            "Outputs are written to results/<model>_cv0/evaluation/ by default."
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override the default output directory for single-experiment mode.",
    )

    # ---- Cross-model comparison mode
    parser.add_argument(
        "--compare-dirs", nargs="+", type=str, default=None,
        help=(
            "Two or more experiment directories to compare. "
            "Generates model_comparison.csv and model_comparison.tex."
        ),
    )
    parser.add_argument(
        "--model-labels", nargs="+", type=str, default=None,
        help=(
            "Human-readable model names in the same order as --compare-dirs "
            "(e.g. 'EVA-02-B' 'ConvNeXt V2-B'). Defaults to directory names."
        ),
    )
    parser.add_argument(
        "--force-recompute", action="store_true",
        help=(
            "Re-evaluate all experiments from raw prediction CSVs even when "
            "metrics_aggregate.json already exists."
        ),
    )

    # ---- Aggregate results table mode
    parser.add_argument(
        "--aggregate-table", action="store_true",
        help=(
            "Build a single top-level CSV covering all 9 models x 3 folds x "
            "3 metrics, reproducing Figure 3 and Table 1 from cv_results.csv "
            "(no training or raw predictions required)."
        ),
    )
    parser.add_argument(
        "--experiments-root", type=str, default=None,
        help=(
            "Directory containing one '<model>_cv0' subdirectory per model "
            "(default: <project_root>/experiments)."
        ),
    )
    parser.add_argument(
        "--aggregate-output", type=str, default=None,
        help=(
            "Output CSV path for --aggregate-table "
            "(default: <project_root>/results/cv0_aggregate_results.csv)."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if (args.experiment_dir is None and args.compare_dirs is None
            and not args.aggregate_table):
        logger.error(
            "Provide --experiment-dir, --compare-dirs, and/or --aggregate-table."
        )
        sys.exit(1)

    # ---- Single experiment --------------------------------------------------
    if args.experiment_dir is not None:
        exp_dir = Path(args.experiment_dir)
        if not exp_dir.exists():
            logger.error(f"Experiment directory not found: {exp_dir}")
            sys.exit(1)
        out_dir = Path(args.output_dir) if args.output_dir else None
        evaluate_experiment(exp_dir, output_dir=out_dir)

    # ---- Cross-model comparison ---------------------------------------------
    if args.compare_dirs is not None:
        exp_dirs = [Path(d) for d in args.compare_dirs]
        missing  = [d for d in exp_dirs if not d.exists()]
        if missing:
            logger.error(f"Directories not found: {missing}")
            sys.exit(1)

        out_dir = (Path(args.output_dir) if args.output_dir
                   else RESULTS_ROOT / "model_comparison")
        logger.info(f"Comparing {len(exp_dirs)} models → {out_dir}")

        comp_df = build_comparison_table(
            experiment_dirs=exp_dirs,
            model_labels=args.model_labels,
            output_dir=out_dir,
            force_recompute=args.force_recompute,
        )

        # Print the comparison table to stdout for quick review
        print("\n" + comp_df.to_string(index=False) + "\n")

    # ---- Aggregate results table (Figure 3 / Table 1) -----------------------
    if args.aggregate_table:
        experiments_root = (
            Path(args.experiments_root) if args.experiments_root
            else PROJECT_ROOT / "experiments"
        )
        aggregate_output = (
            Path(args.aggregate_output) if args.aggregate_output
            else RESULTS_ROOT / "cv0_aggregate_results.csv"
        )
        if not experiments_root.exists():
            logger.error(f"Experiments root not found: {experiments_root}")
            sys.exit(1)

        logger.info(f"Building aggregate results table ← {experiments_root}")
        agg_df = build_aggregate_results_table(experiments_root, aggregate_output)
        print("\n" + agg_df.to_string(index=False) + "\n")


if __name__ == "__main__":
    main()
