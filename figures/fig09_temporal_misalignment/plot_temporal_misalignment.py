"""
figures/fig09_temporal_misalignment/plot_temporal_misalignment.py
=================================================================
Figure 9 — Residual distribution by signed temporal misalignment.

Temporal misalignment is the number of days between the image flight date and
the human scoring date.

    Signed convention (matches full_dataset.csv):
        signed_days_diff  =  flight_date − scoring_date
        negative  →  image was captured BEFORE the score was assigned
        positive  →  image was captured AFTER the score was assigned
        zero      →  same-day match

    Residual convention:
        residual  =  predicted − actual
        positive  →  model over-predicts
        negative  →  model under-predicts

Raw day differences are bucketed into seven ordered categories so the figure
remains legible even when tail values are sparse:

    ≤ −3 | −2 | −1 | 0 | +1 | +2 | ≥ +3  (days)

The primary output is a violin plot of residuals by category (combined across
all three CV0 folds).  A per-category statistics table is also saved.

Data sources
------------
    Predictions  : results/eva02_base/predictions/cv0_combined_predictions.csv
                   Required columns: image_filename, actual, predicted

    Metadata     : data/labels/full_dataset.csv
                   Required columns: image_filename, signed_days_diff

Output (written to this script's directory)
------------------------------------------
    fig09_temporal_misalignment.{pdf,png}
    fig09_temporal_misalignment_stats.csv

Usage
-----
    python figures/fig09_temporal_misalignment/plot_temporal_misalignment.py

    python figures/fig09_temporal_misalignment/plot_temporal_misalignment.py \\
        --predictions results/eva02_base/predictions/cv0_combined_predictions.csv \\
        --metadata    data/labels/full_dataset.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import apply_style, despine, save_fig

apply_style()

# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent

PREDICTIONS_DEFAULT = (
    ROOT / "experiments" / "eva02_base_cv0" / "predictions" / "cv0_combined_predictions.csv"
)
METADATA_DEFAULT = ROOT / "data" / "labels" / "full_dataset.csv"
OUTPUT_DIR       = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Temporal misalignment bins
# ---------------------------------------------------------------------------
# Raw signed_days_diff values are mapped to seven labelled categories.
# The boundary at ±3 collapses sparse tail values into a single bin each.
BIN_EDGES = [-np.inf, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, np.inf]

# Multi-line descriptive labels — match analyze_residuals_by_temporal_misalignment.py
# so that figures are visually consistent across the analysis suite.
BIN_LABELS = [
    "3 days\nbefore scoring",   # ≤ −3
    "2 days\nbefore scoring",   # −2
    "1 day\nbefore scoring",    # −1
    "Same day",                 #  0
    "1 day\nafter scoring",     # +1
    "2 days\nafter scoring",    # +2
    "3 days\nafter scoring",    # ≥ +3
]

# Integer category codes (used for OLS position mapping)
BIN_CODES = [-3, -2, -1, 0, 1, 2, 3]

# Single violin color — teal consistent with analyze_residuals_by_temporal_misalignment.py
VIOLIN_COLOR = "#1C5A62"

# Zero-line style shared with the analysis suite
ZERO_LINE = dict(color="gray", linestyle="--", linewidth=2, alpha=0.5)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and strip all column names in place."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def load_and_merge(pred_path: Path, meta_path: Path) -> pd.DataFrame:
    """
    Load model predictions and dataset metadata, then join on image_filename.

    The signed_error column is computed if not already present.

    Returns
    -------
    DataFrame with columns: image_filename, actual, predicted, signed_error,
    signed_days_diff, misalignment_bin (ordered Categorical), day_position (int)
    """
    # ── predictions ──────────────────────────────────────────────────────────
    pred = _norm_cols(pd.read_csv(pred_path))

    # Accept either "actual/predicted" or "target/prediction" schemas
    pred = pred.rename(columns={"target": "actual", "prediction": "predicted"})
    for col in ("actual", "predicted"):
        if col not in pred.columns:
            raise ValueError(
                f"Predictions CSV must contain 'actual' and 'predicted'. "
                f"Found columns: {list(pred.columns)}"
            )
    if "signed_error" not in pred.columns:
        pred["signed_error"] = pred["predicted"] - pred["actual"]
    pred = pred.dropna(subset=["actual", "predicted"])
    print(f"[predictions] {len(pred)} images")

    # ── metadata ─────────────────────────────────────────────────────────────
    meta = _norm_cols(pd.read_csv(meta_path))
    if "signed_days_diff" not in meta.columns:
        raise ValueError(
            "Metadata CSV must contain 'signed_days_diff'. "
            f"Found: {list(meta.columns)}"
        )
    meta["signed_days_diff"] = pd.to_numeric(
        meta["signed_days_diff"], errors="coerce"
    )
    meta = meta[["image_filename", "signed_days_diff"]].drop_duplicates(
        subset="image_filename"
    )
    print(
        f"[metadata]    {len(meta)} rows  |  "
        f"signed_days_diff range [{meta['signed_days_diff'].min()}, "
        f"{meta['signed_days_diff'].max()}]"
    )

    # ── join ─────────────────────────────────────────────────────────────────
    merged = pred.merge(meta, on="image_filename", how="inner")
    n_lost = len(pred) - len(merged)
    if n_lost:
        warnings.warn(
            f"{n_lost} predictions had no metadata match.", stacklevel=2
        )
    print(f"[merged]      {len(merged)} images after join")

    # ── binning ──────────────────────────────────────────────────────────────
    # Assign each row a descriptive bin label (ordered categorical) and a
    # sequential integer position (0–6) for OLS fitting.
    merged["misalignment_bin"] = pd.cut(
        merged["signed_days_diff"],
        bins=BIN_EDGES,
        labels=BIN_LABELS,
        ordered=True,
    )

    # Map bin label → 0-indexed axis position (matches analyze script convention)
    label_to_pos = {lbl: i for i, lbl in enumerate(BIN_LABELS)}
    merged["day_position"] = merged["misalignment_bin"].map(label_to_pos)

    return merged


# ---------------------------------------------------------------------------
# Per-category statistics table
# ---------------------------------------------------------------------------

def compute_statistics(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """
    Compute mean, median, std, MAE, RMSE, and count per misalignment bin.

    A one-sample t-test against zero is included to flag bins where the mean
    signed error is significantly non-zero (indicates systematic bias).

    Parameters
    ----------
    df       : merged DataFrame with 'signed_error' and 'misalignment_bin' columns
    out_path : path to write the CSV statistics table

    Returns
    -------
    DataFrame indexed by misalignment_bin
    """
    rows = []
    for bin_lbl in BIN_LABELS:
        sub = df[df["misalignment_bin"] == bin_lbl]["signed_error"].dropna()
        n   = len(sub)
        if n >= 2:
            t_stat, pval = stats.ttest_1samp(sub, popmean=0)
        else:
            t_stat, pval = np.nan, np.nan
        rows.append({
            "misalignment_bin": bin_lbl,
            "n":                n,
            "mean_signed_error":    sub.mean()           if n else np.nan,
            "median_signed_error":  sub.median()         if n else np.nan,
            "std_signed_error":     sub.std()            if n else np.nan,
            "mae":              sub.abs().mean()     if n else np.nan,
            "rmse":             np.sqrt((sub**2).mean()) if n else np.nan,
            "t_stat":           t_stat,
            "p_value":          pval,
        })
    stats_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"[stats] saved → {out_path}")
    return stats_df


# ---------------------------------------------------------------------------
# Figure 9 — violin plot
# ---------------------------------------------------------------------------

def plot_violin(df: pd.DataFrame, stats_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Violin plot of prediction residuals by temporal misalignment category.

    Visual language matches analyze_residuals_by_temporal_misalignment.py:
      - Single teal violin body (VIOLIN_COLOR) for all bins
      - IQR bar (vlines Q25→Q75) in teal
      - Median: white dot with teal edge
      - Mean: solid black diamond
      - Jittered raw observations overlaid at low alpha
      - Dashed black OLS trend line across all seven positions
      - ▲/▼ overestimation / underestimation annotations in panel corners
      - Legend summarising all overlay elements
      - y-axis fixed to (−6, 6); no x-axis label (bin labels are self-describing)

    Parameters
    ----------
    df       : merged DataFrame with 'signed_error', 'misalignment_bin', and
               'day_position' columns
    stats_df : per-bin statistics from compute_statistics()
    out_dir  : output directory
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # ------------------------------------------------------------------
    # Collect residuals and positions for populated bins only
    # (matplotlib raises ValueError on empty arrays in violinplot)
    # ------------------------------------------------------------------
    populated = []
    for i, lbl in enumerate(BIN_LABELS):
        vals = df[df["misalignment_bin"] == lbl]["signed_error"].dropna().values
        if len(vals) >= 2:
            populated.append((i, lbl, vals))

    if not populated:
        print("[violin] No bins contain ≥2 observations — skipping figure.")
        return

    pop_positions = [i for i, lbl, vals in populated]
    pop_labels    = [lbl for i, lbl, vals in populated]
    pop_data      = [vals for i, lbl, vals in populated]

    # ------------------------------------------------------------------
    # Violin bodies
    # ------------------------------------------------------------------
    parts = ax.violinplot(
        pop_data,
        positions=pop_positions,
        widths=0.7,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body in parts["bodies"]:
        body.set_facecolor(VIOLIN_COLOR)
        body.set_edgecolor(VIOLIN_COLOR)
        body.set_alpha(0.45)

    # ------------------------------------------------------------------
    # IQR bar (Q25–Q75) + median dot per bin
    # ------------------------------------------------------------------
    for pos, vals in zip(pop_positions, pop_data):
        q25, med, q75 = np.percentile(vals, [25, 50, 75])
        # Thick teal vertical bar spanning the interquartile range
        ax.vlines(pos, q25, q75, color=VIOLIN_COLOR, lw=3.0, alpha=0.85, zorder=3)
        # White dot with teal outline marks the median
        ax.scatter(
            pos, med,
            color="white", edgecolors=VIOLIN_COLOR,
            s=20, zorder=4, linewidths=1.4,
        )

    # ------------------------------------------------------------------
    # Jittered raw observations overlaid at low opacity
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)   # fixed seed for reproducibility
    for pos, vals in zip(pop_positions, pop_data):
        n_show = min(len(vals), 150)
        sample = rng.choice(vals, n_show, replace=False)
        jitter = rng.uniform(-0.12, 0.12, n_show)
        ax.scatter(
            pos + jitter, sample,
            color=VIOLIN_COLOR, alpha=0.18, s=4,
            linewidths=0, zorder=2,
        )

    # ------------------------------------------------------------------
    # Mean diamond (solid black) per bin
    # ------------------------------------------------------------------
    bin_means = [vals.mean() for vals in pop_data]
    ax.scatter(
        pop_positions, bin_means,
        marker="D", color="black",
        s=16, zorder=5,
    )

    # ------------------------------------------------------------------
    # Zero-error reference lines:
    #   dashed gray line (ZERO_LINE style) — primary visual reference
    #   solid black line at y=0 — axis anchor for readability
    # ------------------------------------------------------------------
    ax.axhline(y=0, **ZERO_LINE, zorder=0)
    ax.axhline(0, color="black", lw=1.3, ls="-", zorder=3)

    # ------------------------------------------------------------------
    # OLS trend line across all populated positions
    # Fit is on the 0-indexed axis positions vs raw residuals so the slope
    # is interpretable in units of (residual units / category step).
    # ------------------------------------------------------------------
    x_pos = df["day_position"].values
    y_err = df["signed_error"].values
    # valid  = ~(np.isnan(x_pos) | np.isnan(y_err))
    slope, intercept, _r, _p, _se = stats.linregress(x_pos, y_err)
    x_range = np.linspace(min(pop_positions), max(pop_positions), 200)
    ax.plot(
        x_range, slope * x_range + intercept,
        color="black", linewidth=2.0, ls="--", alpha=0.7, zorder=4,
    )

    # ------------------------------------------------------------------
    # Axis formatting
    # ------------------------------------------------------------------
    n_cats = len(BIN_LABELS)
    ax.set_xlim(-0.6, n_cats - 0.4)           # 0.6-unit padding on both ends
    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(BIN_LABELS, fontsize=11)
    ax.set_ylim(-6, 6)                         # fixed range shared with analysis suite

    # No x-axis label — bin labels are self-describing
    ax.set_ylabel(
        "Signed Error (Test Score − Visual Score)",
        fontsize=11, fontweight="bold",
    )

    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)
    ax.tick_params(axis="both", which="major", labelsize=11, width=1.3)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)
    despine(ax)

    # ------------------------------------------------------------------
    # Corner annotations — direction of signed error
    # ------------------------------------------------------------------
    ax.text(
        0.01, 0.98, "▼ Underestimation",
        transform=ax.transAxes,
        fontsize=8, va="top", ha="left",
        color="#666666", style="italic",
    )
    ax.text(
        0.01, 0.02, "▲ Overestimation",
        transform=ax.transAxes,
        fontsize=8, va="bottom", ha="left",
        color="#666666", style="italic",
    )

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    legend_handles = [
        Line2D([0], [0], color=VIOLIN_COLOR, linewidth=3,
               alpha=0.85, label="IQR"),
        plt.scatter([], [], color="white", edgecolors=VIOLIN_COLOR,
                    s=18, linewidths=1.4, label="Median"),
        plt.scatter([], [], marker="D", color="black",
                    s=18, label="Mean"),
        Line2D([0], [0], color="black", linestyle="-",
               linewidth=2, label="Zero Error"),
        Line2D([0], [0], color="black", linestyle="--",
               linewidth=2, label=f"OLS Trend (slope={slope:.4f})"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    fig.tight_layout()
    save_fig(fig, out_dir, "fig09_temporal_misalignment")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Figure 9 — prediction residuals by temporal misalignment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predictions", default=str(PREDICTIONS_DEFAULT),
        help="Combined CV0 predictions CSV.",
    )
    parser.add_argument(
        "--metadata", default=str(METADATA_DEFAULT),
        help="Full dataset CSV with signed_days_diff column.",
    )
    parser.add_argument(
        "--out_dir", default=str(OUTPUT_DIR),
        help="Output directory.",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    meta_path = Path(args.metadata)
    out_dir   = Path(args.out_dir)

    print("=" * 60)
    print("FIGURE 9 — RESIDUALS BY TEMPORAL MISALIGNMENT")
    print("=" * 60)
    print(f"Predictions : {pred_path}")
    print(f"Metadata    : {meta_path}")
    print(f"Output      : {out_dir}\n")

    df = load_and_merge(pred_path, meta_path)

    # Per-category statistics
    stats_df = compute_statistics(
        df, out_dir / "fig09_temporal_misalignment_stats.csv"
    )

    # Print a quick summary
    print("\nPer-bin residual summary:")
    print(
        stats_df[
            ["misalignment_bin", "n", "mean_signed_error", "std_signed_error", "mae"]
        ].to_string(index=False)
    )

    # Main figure
    print("\nBuilding Figure 9 (violin plot) …")
    plot_violin(df, stats_df, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
