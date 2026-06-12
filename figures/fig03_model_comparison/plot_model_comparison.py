"""
figures/fig03_model_comparison/plot_model_comparison.py
=======================================================
Produces two publication figures from the CV0 (leave-one-year-out) results:

    Figure 3  — Three-panel aggregate metric bar chart
                (A) mean test R²  ↑
                (B) mean test MAE ↓
                (C) mean test RMSE ↓
                Output: fig03_model_comparison.{pdf,png}

    Figure S3 — Four-panel per-fold test R² bar chart
                One panel per test year (2023, 2024, 2025) plus the mean.
                Output: figures/supplemental/figS3_fold_r2/figS3_fold_r2.{pdf,png}

Data sources
------------
    experiments/<model>/summary.json     — aggregate metrics across all three folds
    experiments/<model>/cv_results.csv   — per-fold metrics

Expected summary.json fields
-----------------------------
    mean_val_r2, std_val_r2
    mean_test_r2, std_test_r2
    mean_test_rmse, std_test_rmse
    mean_test_mae

Expected cv_results.csv columns
---------------------------------
    fold_name, test_r2, test_rmse, test_mae

Usage
-----
    # From repo root:
    python figures/fig03_model_comparison/plot_model_comparison.py

    # Custom results directory:
    python figures/fig03_model_comparison/plot_model_comparison.py \\
        --results_dir /path/to/results
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Shared style — import from figures/style.py via sys.path
# ---------------------------------------------------------------------------
# Add the figures/ directory to the module search path so style.py can be
# imported regardless of the working directory when the script is run.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import apply_style, despine, save_fig, FAMILY_COLORS

apply_style()

# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------
# ROOT resolves to the repository root (two levels above this script).
ROOT = Path(__file__).resolve().parent.parent.parent

RESULTS_DIR_DEFAULT = ROOT / "experiments"
OUTPUT_DIR          = Path(__file__).resolve().parent   # save alongside script
SUPPLEMENTAL_DIR    = ROOT / "figures" / "supplemental" / "figS3_fold_r2"

# ---------------------------------------------------------------------------
# Architecture family membership
# ---------------------------------------------------------------------------
# Maps each results subdirectory name to its architecture family.
# SwinV2-B is classified as Hybrid (shifted-window attention qualifies as a
# hybrid between window-based attention and hierarchical CNNs).
FAMILY: dict[str, str] = {
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

# Display names used on chart axes (model name only; family shown via color)
DISPLAY_NAMES: dict[str, str] = {
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
SORT_ORDER: list[str] = [
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

# CV0 fold identifiers and human-readable year labels
FOLD_CODES  = ["fold_23", "fold_24", "fold_25"]
FOLD_LABELS = {"fold_23": "2023", "fold_24": "2024", "fold_25": "2025"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_fold_r2(results_path: Path) -> dict[str, float]:
    """
    Read per-fold test R² from cv_results.csv inside a model results directory.

    Parameters
    ----------
    results_path : Path to the model's results subdirectory

    Returns
    -------
    dict mapping fold_name (e.g. "fold_23") → test R² (float)
    """
    csv_path = results_path / "cv_results.csv"
    if not csv_path.exists():
        return {}
    result: dict[str, float] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fold_name = row.get("fold_name", "").strip()
            r2_str    = row.get("test_r2",   "").strip()
            if fold_name and r2_str:
                try:
                    result[fold_name] = float(r2_str)
                except ValueError:
                    pass
    return result


def load_summaries(results_dir: Path) -> list[dict]:
    """
    Walk *results_dir* and load summary.json + cv_results.csv for every model.

    Each record dict contains:
      - folder, display, family
      - mean_val_r2, std_val_r2
      - mean_test_r2, std_test_r2, mean_test_rmse, std_test_rmse, mean_test_mae
      - test_r2_fold_23, test_r2_fold_24, test_r2_fold_25

    Records are sorted by SORT_ORDER (family groups, ascending param count).
    Unknown models are appended at the end.
    """
    records: list[dict] = []
    for summary_path in sorted(results_dir.glob("*/summary.json")):
        folder = summary_path.parent.name
        # Result directories are named "<model>_cv0"; strip the CV-run suffix
        # to get the model key used by FAMILY, DISPLAY_NAMES, and SORT_ORDER.
        model_key = folder.removesuffix("_cv0")
        with open(summary_path) as f:
            data = json.load(f)
        fold_r2 = _load_fold_r2(summary_path.parent)
        records.append({
            "folder":           model_key,
            "display":          DISPLAY_NAMES.get(model_key, model_key),
            "family":           FAMILY.get(model_key, "Unknown"),
            # aggregate metrics (mean ± std across the three CV0 folds)
            "mean_val_r2":      data.get("mean_val_r2",   float("nan")),
            "std_val_r2":       data.get("std_val_r2",    float("nan")),
            "mean_test_r2":     data.get("mean_test_r2",  float("nan")),
            "std_test_r2":      data.get("std_test_r2",   float("nan")),
            "mean_test_rmse":   data.get("mean_test_rmse",float("nan")),
            "std_test_rmse":    data.get("std_test_rmse", float("nan")),
            "mean_test_mae":    data.get("mean_test_mae", float("nan")),
            # per-fold test R² (for Figure S3)
            "test_r2_fold_23":  fold_r2.get("fold_23", float("nan")),
            "test_r2_fold_24":  fold_r2.get("fold_24", float("nan")),
            "test_r2_fold_25":  fold_r2.get("fold_25", float("nan")),
        })

    # Sort into the preferred family-grouped order
    records.sort(key=lambda r: (
        SORT_ORDER.index(r["folder"]) if r["folder"] in SORT_ORDER
        else len(SORT_ORDER)
    ))
    return records


# ---------------------------------------------------------------------------
# Architecture-family visual grouping helpers
# ---------------------------------------------------------------------------

def _compute_group_info(records: list[dict]) -> list[tuple[int, int, str]]:
    """
    Return (start_idx, end_idx_inclusive, family_name) for each contiguous
    family block in *records* (which must already be sorted by family).
    """
    if not records:
        return []
    groups: list[tuple[int, int, str]] = []
    start = 0
    for i in range(1, len(records)):
        if records[i]["family"] != records[i - 1]["family"]:
            groups.append((start, i - 1, records[start]["family"]))
            start = i
    groups.append((start, len(records) - 1, records[start]["family"]))
    return groups


def _draw_family_bands(
    ax: plt.Axes,
    group_info: list[tuple[int, int, str]],
) -> None:
    """
    Shade each family group with a faint background and draw a thin
    separator line between consecutive groups.
    """
    for start, end, family in group_info:
        ax.axhspan(
            start - 0.5, end + 0.5,
            facecolor=FAMILY_COLORS.get(family, "#888888"),
            alpha=0.07, zorder=-1,
        )
    for start, end, _ in group_info[:-1]:
        ax.axhline(end + 0.5, color="#888888", lw=0.9,
                   linestyle="-", alpha=0.45, zorder=3)


# ---------------------------------------------------------------------------
# Figure 3 — aggregate metric bar chart
# ---------------------------------------------------------------------------

def _hbar(
    ax: plt.Axes,
    values: np.ndarray,
    colors: list[str],
    labels: list[str],
    xlabel: str,
    title: str,
    higher_is_better: bool,
    xlim: tuple[float, float],
    fmt: str = ".3f",
    group_info: list[tuple[int, int, str]] | None = None,
) -> None:
    """
    Draw a horizontal bar chart with value labels and a star on the best bar.

    Parameters
    ----------
    ax               : target Axes
    values           : bar lengths (NaN → bar omitted but label shown)
    colors           : per-bar fill color (from FAMILY_COLORS)
    labels           : y-axis tick labels (model display names)
    xlabel           : x-axis label string
    title            : panel title (e.g. "A) Mean Test R²  ↑")
    higher_is_better : True → best = highest value; False → best = lowest
    xlim             : (left, right) x-axis limits
    fmt              : format spec for value annotation (e.g. ".3f")
    group_info       : output of _compute_group_info; draws family bands/separators
    """
    n = len(values)
    y = np.arange(n)

    if group_info:
        _draw_family_bands(ax, group_info)

    bars = ax.barh(y, values, height=0.65, color=colors,
                   edgecolor="white", linewidth=0.8)

    # Mark the best bar with a dark outline
    finite = values[~np.isnan(values)]
    if len(finite):
        best_idx = int(np.nanargmax(values) if higher_is_better else np.nanargmin(values))
        bars[best_idx].set_edgecolor("#222222")
        bars[best_idx].set_linewidth(2.2)

    # Value annotations to the right of each bar
    x_offset = (xlim[1] - xlim[0]) * 0.012
    for bar, val in zip(bars, values):
        if np.isnan(val):
            continue
        ax.text(
            bar.get_width() + x_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:{fmt}}",
            va="center", ha="left",
            fontsize=8, fontweight="bold",
        )

    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlim(*xlim)
    ax.invert_yaxis()
    ax.grid(axis="x", which="major", linestyle="--", alpha=0.4)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="x", which="minor", linestyle=":", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.tick_params(axis="both", which="major", labelsize=8, width=1.1)
    despine(ax)


def make_aggregate_figure(records: list[dict]) -> plt.Figure:
    """
    Build the three-panel aggregate metric figure (Figure 3).

    Panels
    ------
    A) Mean test R²   (higher is better)
    B) Mean test MAE  (lower is better)
    C) Mean test RMSE (lower is better)

    Returns
    -------
    matplotlib Figure
    """
    n          = len(records)
    labels     = [r["display"] for r in records]
    colors     = [FAMILY_COLORS.get(r["family"], "#888888") for r in records]
    group_info = _compute_group_info(records)

    r2_vals   = np.array([r["mean_test_r2"]   for r in records])
    mae_vals  = np.array([r["mean_test_mae"]  for r in records])
    rmse_vals = np.array([r["mean_test_rmse"] for r in records])

    fig, axes = plt.subplots(
        1, 3,
        figsize=(13.5, max(4.0, 0.46 * n + 2.2)),
        gridspec_kw={"wspace": 0.06},
    )

    # --- Panel A: R² ---
    r2_floor = max(-0.1, np.nanmin(r2_vals) - 0.08) if np.any(np.isfinite(r2_vals)) else -0.1
    r2_ceil  = min(1.05, np.nanmax(r2_vals) + 0.14) if np.any(np.isfinite(r2_vals)) else 1.05
    _hbar(axes[0], r2_vals, colors, labels,
          xlabel="Mean test R²",
          title="A) Mean test R²  ↑",
          higher_is_better=True,
          xlim=(r2_floor, r2_ceil),
          group_info=group_info)

    # --- Panel B: MAE ---
    _hbar(axes[1], mae_vals, colors, labels,
          xlabel="Mean test MAE (score units)",
          title="B) Mean test MAE  ↓",
          higher_is_better=False,
          xlim=(0, np.nanmax(mae_vals) * 1.22),
          fmt=".3f",
          group_info=group_info)
    axes[1].set_yticklabels([])          # hide y-labels on middle panel
    axes[1].tick_params(axis="y", length=0)
    axes[1].spines["left"].set_visible(False)

    # --- Panel C: RMSE ---
    _hbar(axes[2], rmse_vals, colors, labels,
          xlabel="Mean test RMSE (score units)",
          title="C) Mean test RMSE  ↓",
          higher_is_better=False,
          xlim=(0, np.nanmax(rmse_vals) * 1.22),
          fmt=".3f",
          group_info=group_info)
    axes[2].set_yticklabels([])
    axes[2].tick_params(axis="y", length=0)
    axes[2].spines["left"].set_visible(False)

    # --- Architecture family legend ---
    legend_patches = [
        mpatches.Patch(facecolor=FAMILY_COLORS[fam], edgecolor="#444",
                       linewidth=0.8, label=fam)
        for fam in ["CNN", "ViT", "Hybrid"]
    ]
    fig.legend(
        handles=legend_patches,
        title="Architecture family", title_fontsize=9,
        fontsize=8, loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, -0.06),
        frameon=True, framealpha=0.9, edgecolor="#cccccc",
    )

    fig.tight_layout(w_pad=0)
    return fig


# ---------------------------------------------------------------------------
# Figure S3 — per-fold R² bar chart
# ---------------------------------------------------------------------------

def make_fold_r2_figure(records: list[dict]) -> plt.Figure:
    """
    Build the four-panel per-fold test R² figure (Figure S3).

    Panels
    ------
    Column 1: test year 2023  (fold_23)
    Column 2: test year 2024  (fold_24)
    Column 3: test year 2025  (fold_25)
    Column 4: mean across all three test years

    Returns
    -------
    matplotlib Figure
    """
    n          = len(records)
    labels     = [r["display"] for r in records]
    colors     = [FAMILY_COLORS.get(r["family"], "#888888") for r in records]
    group_info = _compute_group_info(records)
    y          = np.arange(n)
    bar_h      = 0.62

    # Panel specifications: (result dict key, panel title)
    panels = [
        ("test_r2_fold_23", "Test year: 2023"),
        ("test_r2_fold_24", "Test year: 2024"),
        ("test_r2_fold_25", "Test year: 2025"),
        ("mean",            "All years (mean)"),
    ]

    fig, axes = plt.subplots(
        1, len(panels),
        figsize=(4.8 * len(panels), max(4.5, 0.45 * n + 2.5)),
        gridspec_kw={"wspace": 0.06},
    )

    # Shared x-range across all panels so R² values are directly comparable
    all_r2s = []
    for key, _ in panels:
        if key == "mean":
            all_r2s.extend([r["mean_test_r2"] for r in records])
        else:
            all_r2s.extend([r[key] for r in records])
    finite   = [v for v in all_r2s if np.isfinite(v)]
    x_floor  = max(-0.1, min(finite) - 0.08) if finite else -0.1
    x_ceil   = min(1.05, max(finite) + 0.14) if finite else 1.05

    for ax_i, (fold_key, panel_title) in enumerate(panels):
        ax = axes[ax_i]

        r2s = np.array(
            [r["mean_test_r2"] for r in records] if fold_key == "mean"
            else [r[fold_key] for r in records]
        )

        _draw_family_bands(ax, group_info)

        bars = ax.barh(y, r2s, height=bar_h, color=colors,
                       edgecolor="white", linewidth=0.8)

        # Highlight the best model with a bold outline
        valid = r2s[~np.isnan(r2s)]
        if len(valid):
            best_idx = int(np.nanargmax(r2s))
            bars[best_idx].set_edgecolor("#222222")
            bars[best_idx].set_linewidth(2.2)

        # Value labels to the right of each bar
        x_offset = (x_ceil - x_floor) * 0.015 + abs(x_ceil) * 0.005
        for bar, val in zip(bars, r2s):
            if np.isnan(val):
                continue
            ax.text(
                bar.get_width() + x_offset,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center", ha="left",
                fontsize=8, fontweight="bold",
            )

        ax.axvline(0, color="black", lw=0.6)
        ax.set_xlabel("R²", fontsize=10)
        ax.set_title(panel_title, fontsize=10, fontweight="bold", pad=6)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.grid(axis="x", which="major", linestyle="--", alpha=0.4)
        ax.grid(axis="x", which="minor", linestyle=":", alpha=0.2)
        ax.set_axisbelow(True)
        ax.set_xlim(x_floor, x_ceil)
        ax.set_yticks(y)
        ax.spines["left"].set_linewidth(1.4)
        ax.spines["bottom"].set_linewidth(1.4)
        ax.tick_params(axis="both", which="major", labelsize=8, width=1.1)
        ax.invert_yaxis()
        despine(ax)

        if ax_i == 0:
            ax.set_yticklabels(labels, fontsize=9.5)
        else:
            # Remove redundant y-labels on panels 2–4
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)

    # Architecture family legend below all panels
    legend_patches = [
        mpatches.Patch(facecolor=FAMILY_COLORS[fam], edgecolor="#444",
                       linewidth=0.8, label=fam)
        for fam in ["CNN", "ViT", "Hybrid"]
    ]
    fig.legend(
        handles=legend_patches,
        title="Architecture family", title_fontsize=9,
        fontsize=8, loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, -0.07),
        frameon=True, framealpha=0.9, edgecolor="#cccccc",
    )

    fig.tight_layout(w_pad=0.3)
    return fig


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate model comparison figures (Fig 3 and Fig S3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_dir", default=str(RESULTS_DIR_DEFAULT),
        help="Root directory containing one subdirectory per model.",
    )
    parser.add_argument(
        "--out_dir", default=str(OUTPUT_DIR),
        help="Directory where Figure 3 PDF and PNG outputs are written.",
    )
    parser.add_argument(
        "--supplemental_dir", default=str(SUPPLEMENTAL_DIR),
        help="Directory where Figure S3 PDF and PNG outputs are written.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)

    # ── load summaries ───────────────────────────────────────────────────────
    records = load_summaries(results_dir)
    if not records:
        raise FileNotFoundError(
            f"No summary.json files found under {results_dir}. "
            "Run scripts/collect_results.py first."
        )

    print(f"Loaded {len(records)} model(s):")
    for r in records:
        print(f"  [{r['family']:<8}]  {r['folder']:<22}  "
              f"R²={r['mean_test_r2']:.4f}  "
              f"MAE={r['mean_test_mae']:.4f}  "
              f"RMSE={r['mean_test_rmse']:.4f}")

    # ── Figure 3 — aggregate metrics ─────────────────────────────────────────
    print("\nBuilding Figure 3 (aggregate metrics) …")
    fig3 = make_aggregate_figure(records)
    save_fig(fig3, out_dir, "fig03_model_comparison")

    # ── Figure S3 — per-fold R² ───────────────────────────────────────────────
    print("\nBuilding Figure S3 (per-fold R²) …")
    figS3 = make_fold_r2_figure(records)
    save_fig(figS3, args.supplemental_dir, "figS3_fold_r2")

    print("\nDone.")


if __name__ == "__main__":
    main()
