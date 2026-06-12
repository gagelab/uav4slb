"""
figures/fig04_test_performance/plot_test_performance.py
=======================================================
Figure 4 — EVA-02-B CV0 test performance across all three folds combined.

    Panel A — Violin plot of model test scores grouped by in-field visual score
              (0.5-wide bins, 1.0–9.0).  Each violin shows the distribution of
              model predictions for plots scored at that visual level.  The IQR
              bar, bin median, jittered raw points, bin mean, 1:1 reference
              line, and OLS trend line are overlaid.

    Panel B — Mean signed error (model − visual) per visual-score bin,
              coloured by direction: warm orange = over-prediction,
              dark blue = under-prediction.  The overall mean is annotated.

Residual convention  (consistent across all figure scripts):
    residual  =  predicted − actual  =  test score − visual score
    positive  →  model over-predicts
    negative  →  model under-predicts

Data source
-----------
    results/eva02_base/predictions/cv0_combined_predictions.csv
    Required columns: actual, predicted   (plus optionally fold)

Output (written to this script's directory)
------------------------------------------
    fig04_test_performance.{pdf,png}
    fig04_metrics_summary.csv  — R², RMSE, MAE per fold and combined

Usage
-----
    python figures/fig04_test_performance/plot_test_performance.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import apply_style, despine, save_fig

apply_style()

# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------
ROOT           = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_DEFAULT = (
    ROOT / "experiments" / "eva02_base_cv0" / "cv0_combined_predictions.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCORE_MIN, SCORE_MAX = 1.0, 9.0
# Bins: 0.5-wide steps covering the full 1–9 scale (17 values)
ALL_BINS = np.arange(SCORE_MIN, SCORE_MAX + 0.5, 0.5)

# Violin body color: batlow-86 mid green — a neutral mid-range batlow stop
VIOLIN_COLOR = "#3C6D56"

# Signed-error bar colors — consistent with residual_analysis source notebook
# and with FAMILY_COLORS orientation (warm = over-prediction, cool = under)
COLOR_OVER  = "#D29343"   # batlow-171 — model over-predicts (positive residual)
COLOR_UNDER = "#103F60"   # batlow-29  — model under-predicts (negative residual)

# Minimum images required to draw a violin for a given bin
MIN_BIN_COUNT = 5


# ---------------------------------------------------------------------------
# Data loading and metrics
# ---------------------------------------------------------------------------

def load_predictions(pred_path: Path) -> pd.DataFrame:
    """
    Load the combined CV0 predictions CSV and compute residuals.

    Accepts either 'actual / predicted' or 'target / prediction' column names.
    """
    df = pd.read_csv(pred_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Normalise column name variants
    df = df.rename(columns={"target": "actual", "prediction": "predicted"})
    for col in ("actual", "predicted"):
        if col not in df.columns:
            raise ValueError(
                f"Predictions CSV must contain 'actual' and 'predicted'. "
                f"Found: {list(df.columns)}"
            )
    df["actual"]    = pd.to_numeric(df["actual"],    errors="coerce")
    df["predicted"] = pd.to_numeric(df["predicted"], errors="coerce")
    df = df.dropna(subset=["actual", "predicted"])

    df["signed_error"] = df["predicted"] - df["actual"]
    df["abs_error"]    = df["signed_error"].abs()
    print(f"[predictions]  {len(df):,} images  |  "
          f"R² = {r2_score(df['actual'], df['predicted']):.4f}  |  "
          f"MAE = {mean_absolute_error(df['actual'], df['predicted']):.4f}")
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    """Return a dict of performance metrics for a (sub)set of predictions."""
    y, p = df["actual"].values, df["predicted"].values
    return {
        "n":          len(df),
        "r2":         float(r2_score(y, p)),
        "rmse":       float(np.sqrt(mean_squared_error(y, p))),
        "mae":        float(mean_absolute_error(y, p)),
        "mean_error": float(np.mean(p - y)),
        "pearson_r":  float(stats.pearsonr(y, p)[0]),
    }


def build_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics per fold and for all folds combined."""
    rows = []
    if "fold" in df.columns:
        for fold_name, grp in df.groupby("fold", sort=True):
            m = compute_metrics(grp)
            m["fold"] = fold_name
            rows.append(m)
    m = compute_metrics(df)
    m["fold"] = "all_folds"
    rows.append(m)
    return pd.DataFrame(rows)[["fold", "n", "r2", "rmse", "mae",
                                "mean_error", "pearson_r"]]


# ---------------------------------------------------------------------------
# Per-bin aggregation
# ---------------------------------------------------------------------------

def aggregate_by_bin(df: pd.DataFrame) -> list[dict]:
    """
    For each 0.5-wide visual-score bin, collect predicted values and compute
    the mean signed error.

    Only bins with ≥ MIN_BIN_COUNT images are returned.

    Returns
    -------
    list of dicts, each with keys:
        bin_center, predicted_vals, mean_pred, mean_signed_error
    """
    bins: list[dict] = []
    for b in ALL_BINS:
        # Round actual scores to 0.5 precision before matching
        mask   = ((df["actual"] * 2).round() / 2) == b
        pvals  = df.loc[mask, "predicted"].dropna().values
        avals  = df.loc[mask, "actual"].dropna().values
        if len(pvals) >= MIN_BIN_COUNT:
            bins.append({
                "bin_center":        b,
                "predicted_vals":    pvals,
                "mean_pred":         pvals.mean(),
                "mean_signed_error": float(np.mean(pvals - avals)),
            })
    return bins


# ---------------------------------------------------------------------------
# Figure 4
# ---------------------------------------------------------------------------

def make_figure(df: pd.DataFrame) -> plt.Figure:
    """
    Build the two-panel test performance figure.

    Returns
    -------
    matplotlib Figure
    """
    bins_data = aggregate_by_bin(df)

    if not bins_data:
        raise RuntimeError("No bins had enough data to draw violins.")

    positions  = [b["bin_center"]        for b in bins_data]
    groups     = [b["predicted_vals"]    for b in bins_data]
    means      = [b["mean_pred"]         for b in bins_data]
    mse_vals   = [b["mean_signed_error"] for b in bins_data]

    # OLS slope and intercept for the trend line
    ols_slope, ols_intercept, *_ = stats.linregress(df["actual"], df["predicted"])

    rng = np.random.default_rng(7)   # fixed seed for reproducible jitter

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6))

    # ═════════════════════════════════════════════════════════════════════════
    # Panel A — Violin plot
    # ═════════════════════════════════════════════════════════════════════════
    parts = ax_a.violinplot(
        groups, positions=positions,
        widths=0.38, showmedians=False, showextrema=False,
    )
    for body in parts["bodies"]:
        body.set_facecolor(VIOLIN_COLOR)
        body.set_edgecolor(VIOLIN_COLOR)
        body.set_alpha(0.45)

    # IQR bar + median dot per bin
    for pos, vals in zip(positions, groups):
        q25, med, q75 = np.percentile(vals, [25, 50, 75])
        ax_a.vlines(pos, q25, q75, color=VIOLIN_COLOR, lw=3.0, alpha=0.85, zorder=3)
        ax_a.scatter(pos, med, color="white", edgecolors=VIOLIN_COLOR,
                     s=18, zorder=4, linewidths=1.4)

    # Jittered raw points (subsample large bins to reduce overplotting)
    for pos, vals in zip(positions, groups):
        n_show = min(len(vals), 150)
        sample = rng.choice(vals, n_show, replace=False)
        jitter = rng.uniform(-0.12, 0.12, n_show)
        ax_a.scatter(pos + jitter, sample, color=VIOLIN_COLOR,
                     alpha=0.18, s=4, linewidths=0, zorder=2)

    # Bin mean diamonds
    ax_a.scatter(positions, means, marker="D", color="black",
                 s=16, zorder=5, label="Bin mean")

    # 1:1 reference line
    ax_a.plot([SCORE_MIN, SCORE_MAX], [SCORE_MIN, SCORE_MAX],
              "k-", lw=1.3, label="1:1 (perfect)", zorder=3)

    # OLS trend line
    x_fit = np.array([SCORE_MIN, SCORE_MAX])
    ax_a.plot(x_fit, ols_slope * x_fit + ols_intercept,
              color="black", lw=1.6, ls="--", zorder=4,
              label=f"OLS fit (slope = {ols_slope:.2f})")

    # Axis limits and tick marks
    y_pad = 0.6
    ax_a.set_xlim(SCORE_MIN - 0.5, SCORE_MAX + 0.5)
    ax_a.set_ylim(SCORE_MIN - y_pad, SCORE_MAX + y_pad)
    ax_a.set_xticks(ALL_BINS)
    ax_a.set_xticklabels(
        [f"{b:g}" if b == int(b) else f"{b}" for b in ALL_BINS],
        fontsize=7, rotation=45, ha="right",
    )
    ax_a.set_yticks(np.arange(SCORE_MIN, SCORE_MAX + 1, 1))
    ax_a.set_xlabel("Visual Score", fontsize=12, fontweight="bold")
    ax_a.set_ylabel("Test Score",   fontsize=12, fontweight="bold")

    # Legend with explicit median/IQR proxy handles
    median_handle = mlines.Line2D(
        [0], [0], marker="o", color="white",
        markeredgecolor=VIOLIN_COLOR, markersize=5,
        linewidth=0, markeredgewidth=1.4, label="Bin median",
    )
    iqr_handle = mlines.Line2D(
        [0], [0], color=VIOLIN_COLOR, lw=3.0, alpha=0.85, label="IQR bar",
    )
    handles, labels = ax_a.get_legend_handles_labels()
    ax_a.legend(
        handles=handles + [median_handle, iqr_handle],
        labels=labels  + ["Bin median", "IQR bar"],
        fontsize=8, loc="upper left",
    )

    # Panel letter
    ax_a.text(-0.06, 1.04, "A)", transform=ax_a.transAxes,
              fontsize=16, fontweight="bold", va="bottom", ha="left")
    ax_a.text(0.01, 1.04, "Image-based Test vs. In-field Visual Score",
              transform=ax_a.transAxes,
              fontsize=13, fontweight="bold", va="bottom", ha="left")

    ax_a.spines["left"].set_linewidth(1.5)
    ax_a.spines["bottom"].set_linewidth(1.5)
    ax_a.tick_params(width=1.2)
    ax_a.grid(True, ls="--", alpha=0.35, lw=0.7)
    despine(ax_a)

    # ═════════════════════════════════════════════════════════════════════════
    # Panel B — Mean signed error bars
    # ═════════════════════════════════════════════════════════════════════════
    x_idx      = np.arange(len(positions))
    bar_colors = [COLOR_OVER if v >= 0 else COLOR_UNDER for v in mse_vals]
    bin_labels = [f"{b:g}" if b == int(b) else f"{b}" for b in positions]

    ax_b.bar(x_idx, mse_vals, color=bar_colors, alpha=0.78,
             edgecolor="black", linewidth=0.5, zorder=2)

    # Value annotations above/below each bar
    for j, v in enumerate(mse_vals):
        offset = 0.01 if v >= 0 else -0.01
        va     = "bottom" if v >= 0 else "top"
        ax_b.text(j, v + offset, f"{v:+.2f}",
                  ha="center", va=va, fontsize=7)

    # Zero baseline
    ax_b.axhline(0, color="black", lw=1.2, ls="-", zorder=3)

    # Overall mean signed error dashed reference
    overall_mse = float(np.mean(df["predicted"] - df["actual"]))
    ax_b.axhline(
        overall_mse, color="black", lw=1.6, ls="--",
        label=f"Overall mean = {overall_mse:+.3f}", zorder=4,
    )

    ax_b.set_xticks(x_idx)
    ax_b.set_xticklabels(bin_labels, fontsize=7, rotation=45, ha="right")
    ax_b.set_xlabel("Visual Score",                       fontsize=12, fontweight="bold")
    ax_b.set_ylabel("Mean Signed Error (Test − Visual)", fontsize=12, fontweight="bold")

    # Symmetric y-axis with headroom
    y_abs_max = max(abs(v) for v in mse_vals) * 1.30
    ax_b.set_ylim(-y_abs_max, y_abs_max)

    # Legend
    over_patch  = mpatches.Patch(facecolor=COLOR_OVER,  alpha=0.78,
                                  edgecolor="white", label="Overestimation")
    under_patch = mpatches.Patch(facecolor=COLOR_UNDER, alpha=0.78,
                                  edgecolor="white", label="Underestimation")
    handles_b, labels_b = ax_b.get_legend_handles_labels()
    ax_b.legend(
        handles=[over_patch, under_patch] + handles_b,
        labels=["Overestimation", "Underestimation"] + labels_b,
        fontsize=8,
    )

    ax_b.spines["left"].set_linewidth(1.5)
    ax_b.spines["bottom"].set_linewidth(1.5)
    ax_b.tick_params(width=1.2)
    ax_b.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax_b.grid(True, ls="--", alpha=0.35, lw=0.7)
    despine(ax_b)

    # Panel letter
    ax_b.text(-0.06, 1.04, "B)", transform=ax_b.transAxes,
              fontsize=16, fontweight="bold", va="bottom", ha="left")
    ax_b.text(0.01, 1.04, "Mean Signed Error by Visual Score Bins",
              transform=ax_b.transAxes,
              fontsize=13, fontweight="bold", va="bottom", ha="left")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Figure 4 — EVA-02-B CV0 test performance violin + signed error.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions", default=str(PREDICTIONS_DEFAULT),
                        help="Combined CV0 predictions CSV.")
    parser.add_argument("--out_dir",     default=str(OUTPUT_DIR),
                        help="Output directory.")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    out_dir   = Path(args.out_dir)

    print("=" * 60)
    print("FIGURE 4 — TEST PERFORMANCE")
    print("=" * 60)
    print(f"Predictions : {pred_path}")
    print(f"Output      : {out_dir}\n")

    df = load_predictions(pred_path)

    # Save metrics table
    metrics_tbl = build_metrics_table(df)
    print("\nMetrics summary:")
    print(metrics_tbl.round(4).to_string(index=False))
    csv_path = out_dir / "fig04_metrics_summary.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_tbl.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n[metrics] saved → {csv_path}")

    # Build and save figure
    print("\nBuilding Figure 4 …")
    fig = make_figure(df)
    save_fig(fig, out_dir, "fig04_test_performance")
    print("Done.")


if __name__ == "__main__":
    main()
