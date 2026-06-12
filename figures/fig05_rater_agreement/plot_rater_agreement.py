"""
figures/fig05_rater_agreement/plot_rater_agreement.py
=====================================================
Figure 5 — Rater-to-rater vs. model-to-rater agreement comparison.

Computes all C(5,2) = 10 pairwise Pearson r values among the five raters,
and the 5 pairwise Pearson r values between each rater and the EVA-02-B
model, then plots both distributions as a jitter strip chart with mean lines.

The figure answers: "Is the model's agreement with individual raters
comparable to inter-rater agreement?"

The ICC(2,1) for the five raters is also computed and printed to stdout.

Field context
-------------
Data come from a single field-date: C7B, flight 2025-07-10.  The five raters
are PBK, TZ, CH, AH, SK.  Plots were scored by all five raters on the same
day, providing a clean within-session inter-rater reliability estimate.

Data sources
------------
    Rater scores : data/labels/long_format_ratings.csv
                   Required columns: plot (int), rater (str), score (float)
                   This file contains ratings for C7B 2025-07-10 only.

    Predictions  : results/eva02_base/predictions/cv0_combined_predictions.csv
                   Required columns: image_filename, predicted

    The script filters predictions to the C7B 2025-07-10 flight and
    aggregates the per-image predictions to plot-level means.

Output (written to this script's directory)
------------------------------------------
    fig05_rater_agreement.{pdf,png}
    fig05_rater_agreement_stats.csv  — per-pair r values + ICC summary

Usage
-----
    python figures/fig05_rater_agreement/plot_rater_agreement.py
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr

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

RATINGS_DEFAULT     = ROOT / "data" / "labels" / "long_format_ratings.csv"
PREDICTIONS_DEFAULT = (
    ROOT / "experiments" / "eva02_base_cv0" / "cv0_combined_predictions.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Study parameters
# ---------------------------------------------------------------------------
# The five raters who participated in the C7B 2025-07-10 session.
RATERS: list[str] = ["PBK", "TZ", "CH", "AH", "SK"]

# Image filename prefix used to filter predictions to this field-date.
FLIGHT_DATE = "20250710"
FIELD       = "C7B"

# Colors from the batlow palette
# Rater–rater pairs: dark blue (batlow-29); model–rater pairs: warm orange (batlow-171)
COLOR_RATER_RATER = "#103F60"   # batlow-29
COLOR_MODEL_RATER = "#D29343"   # batlow-171


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def load_ratings(path: Path) -> pd.DataFrame:
    """
    Load the long-format rater scores and pivot to wide format.

    Expected CSV columns: plot, rater, score
    Returns wide DataFrame indexed by plot, with one column per rater.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["plot"]  = df["plot"].astype(int)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    wide = (
        df[df["rater"].isin(RATERS)]
        .pivot(index="plot", columns="rater", values="score")
        .reset_index()
    )
    wide.columns.name = None

    # Consensus statistics
    wide["consensus_mean"] = wide[RATERS].mean(axis=1)
    wide["consensus_std"]  = wide[RATERS].std(axis=1)
    wide["n_raters"]       = wide[RATERS].notna().sum(axis=1)

    print(f"[ratings]  {len(wide)} plots  |  "
          f"raters: {RATERS}")
    return wide


def load_plot_predictions(pred_path: Path) -> pd.DataFrame:
    """
    Load CV0 predictions, filter to C7B 2025-07-10, and aggregate to plot level.

    Returns DataFrame with columns: plot, model_pred, n_images
    """
    preds = pd.read_csv(pred_path)
    preds.columns = [c.strip().lower() for c in preds.columns]

    mask = preds["image_filename"].str.contains(
        f"{FLIGHT_DATE}_{FIELD}", na=False
    )
    preds_sub = preds[mask].copy()
    print(f"[predictions]  {len(preds_sub)} images matched "
          f"({FLIGHT_DATE}_{FIELD})")

    # Extract 5-digit plot number from filename: YYYYMMDD_FIELD_NNNNN.jpg
    preds_sub["plot"] = (
        preds_sub["image_filename"]
        .str.extract(r"_(\d{5})\.jpg")[0]
        .astype(int)
    )

    plot_preds = (
        preds_sub.groupby("plot")
        .agg(model_pred=("predicted", "mean"),
             n_images=("predicted", "count"))
        .reset_index()
    )
    print(f"[predictions]  {len(plot_preds)} unique plots")
    return plot_preds


def merge_data(ratings: pd.DataFrame, plot_preds: pd.DataFrame) -> pd.DataFrame:
    """Merge wide-format ratings with plot-level model predictions."""
    df = ratings.merge(plot_preds, on="plot", how="inner")
    print(f"[merged]  {len(df)} plots with both rater scores and predictions")
    return df


# ---------------------------------------------------------------------------
# Agreement metrics
# ---------------------------------------------------------------------------

def compute_pairwise_rater_r(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson r for all C(5,2) = 10 rater-rater pairs.

    Returns DataFrame with columns: pair, r1, r2, r, p, n
    """
    rows = []
    for r1, r2 in combinations(RATERS, 2):
        sub     = df[[r1, r2]].dropna()
        rv, pv  = pearsonr(sub[r1], sub[r2])
        rows.append({"pair": f"{r1}–{r2}", "r1": r1, "r2": r2,
                     "r": rv, "p": pv, "n": len(sub)})
    return pd.DataFrame(rows)


def compute_model_rater_r(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson r for each of the 5 model-rater pairs.

    Returns DataFrame with columns: pair, rater, r, p, n
    """
    rows = []
    for r in RATERS:
        sub    = df[[r, "model_pred"]].dropna()
        rv, pv = pearsonr(sub[r], sub["model_pred"])
        rows.append({"pair": f"Model–{r}", "rater": r,
                     "r": rv, "p": pv, "n": len(sub)})
    return pd.DataFrame(rows)


def compute_icc_twoway(data_wide: pd.DataFrame, rater_cols: list[str]) -> dict:
    """
    ICC(2,1) — two-way random effects, absolute agreement, single measures.

    Uses the standard Shrout & Fleiss (1979) formula.
    Returns dict with keys: ICC, CI_lower, CI_upper, n, k
    """
    X = data_wide[rater_cols].dropna().values
    n, k       = X.shape
    grand_mean = X.mean()
    subj_means = X.mean(axis=1)
    rater_means= X.mean(axis=0)

    SS_r = k * np.sum((subj_means - grand_mean) ** 2)
    SS_c = n * np.sum((rater_means - grand_mean) ** 2)
    SS_e = np.sum(
        (X - subj_means[:, None] - rater_means[None, :] + grand_mean) ** 2
    )
    df_r, df_c, df_e = n - 1, k - 1, (n - 1) * (k - 1)
    MS_r = SS_r / df_r
    MS_c = SS_c / df_c
    MS_e = SS_e / df_e

    icc = (MS_r - MS_e) / (MS_r + (k - 1) * MS_e + k * (MS_c - MS_e) / n)
    F   = MS_r / MS_e
    FL  = F / stats.f.ppf(0.975, df_r, df_e)
    FU  = F * stats.f.ppf(0.975, df_e, df_r)
    return {
        "ICC":      icc,
        "CI_lower": (FL - 1) / (FL + k - 1),
        "CI_upper": (FU - 1) / (FU + k - 1),
        "n": n, "k": k,
    }


# ---------------------------------------------------------------------------
# Figure 5 — strip chart
# ---------------------------------------------------------------------------

def make_figure(
    rr_df: pd.DataFrame,
    mr_df: pd.DataFrame,
    icc_result: dict,
) -> plt.Figure:
    """
    Build the pairwise Pearson r strip chart.

    Left column  (x = 0): 10 rater–rater pairs (blue dots), mean line.
    Right column (x = 1): 5 model–rater pairs (orange dots), mean line.
    Mean values are annotated to the right of each line.

    Parameters
    ----------
    rr_df      : DataFrame from compute_pairwise_rater_r()
    mr_df      : DataFrame from compute_model_rater_r()
    icc_result : dict from compute_icc_twoway()

    Returns
    -------
    matplotlib Figure
    """
    mean_rr = rr_df["r"].mean()
    mean_mr = mr_df["r"].mean()

    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(8, 5))

    # ── Rater–rater dots (column x ≈ 0) ──────────────────────────────────────
    rr_x_jitter = rng.uniform(-0.08, 0.08, len(rr_df))
    ax.scatter(
        np.zeros(len(rr_df)) + rr_x_jitter,
        rr_df["r"],
        s=60, alpha=0.75, zorder=3,
        color=COLOR_RATER_RATER,
        label="Rater–rater pairs",
    )

    # ── Model–rater dots (column x ≈ 1) ──────────────────────────────────────
    mr_x_jitter = rng.uniform(-0.08, 0.08, len(mr_df))
    ax.scatter(
        np.ones(len(mr_df)) + mr_x_jitter,
        mr_df["r"],
        s=60, alpha=0.85, zorder=3,
        color=COLOR_MODEL_RATER,
        label="Model–rater pairs",
    )

    # ── Mean lines with numeric annotations ───────────────────────────────────
    # Rater–rater mean: spans columns 0 column only
    ax.hlines(mean_rr, -0.35, 0.35,
              colors=COLOR_RATER_RATER, linewidths=2.5, linestyles="-",
              zorder=4, label=f"Rater–rater mean")
    ax.text(0.37, mean_rr, f"{mean_rr:.3f}",
            color=COLOR_RATER_RATER, va="center", ha="left",
            fontsize=9, fontweight="bold")

    # Model–rater mean: spans column 1 only
    ax.hlines(mean_mr, 0.65, 1.35,
              colors=COLOR_MODEL_RATER, linewidths=2.5, linestyles="-",
              zorder=4, label=f"Model–rater mean")
    ax.text(1.37, mean_mr, f"{mean_mr:.3f}",
            color=COLOR_MODEL_RATER, va="center", ha="left",
            fontsize=9, fontweight="bold")

    # ── ICC annotation box ────────────────────────────────────────────────────
    # icc_text = (
    #     f"ICC(2,1) = {icc_result['ICC']:.3f}\n"
    #     f"95% CI [{icc_result['CI_lower']:.3f}, {icc_result['CI_upper']:.3f}]\n"
    #     f"n = {icc_result['n']} plots, k = {icc_result['k']} raters"
    # )
    # ax.text(
    #     0.02, 0.03, icc_text,
    #     transform=ax.transAxes,
    #     ha="left", va="bottom", fontsize=8,
    #     bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", alpha=0.9),
    # )

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [f"Rater–to-Rater\n({len(rr_df)} pairs)",
         f"Model–to-Rater\n({len(mr_df)} pairs)"],
        fontsize=11, fontweight="bold",
    )
    ax.set_ylabel("Pearson r", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0.0, 1.05)

    ax.legend(fontsize=9, loc="lower center",
              framealpha=0.9, bbox_to_anchor=(0.5, -0.30), ncol=2)

    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(width=1.2, labelsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    despine(ax)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Figure 5 — pairwise Pearson r: rater-rater vs. model-rater.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ratings",     default=str(RATINGS_DEFAULT),
                        help="Long-format rater scores CSV.")
    parser.add_argument("--predictions", default=str(PREDICTIONS_DEFAULT),
                        help="Combined CV0 predictions CSV.")
    parser.add_argument("--out_dir",     default=str(OUTPUT_DIR),
                        help="Output directory.")
    args = parser.parse_args()

    ratings_path = Path(args.ratings)
    pred_path    = Path(args.predictions)
    out_dir      = Path(args.out_dir)

    print("=" * 60)
    print("FIGURE 5 — RATER AGREEMENT")
    print("=" * 60)
    print(f"Ratings     : {ratings_path}")
    print(f"Predictions : {pred_path}")
    print(f"Output      : {out_dir}\n")

    # Load and merge
    ratings     = load_ratings(ratings_path)
    plot_preds  = load_plot_predictions(pred_path)
    df          = merge_data(ratings, plot_preds)

    # Pairwise Pearson r
    rr_df = compute_pairwise_rater_r(df)
    mr_df = compute_model_rater_r(df)

    mean_rr = rr_df["r"].mean()
    mean_mr = mr_df["r"].mean()
    print(f"\nRater–rater mean r = {mean_rr:.4f}  (n={len(rr_df)} pairs)")
    print(f"Model–rater  mean r = {mean_mr:.4f}  (n={len(mr_df)} pairs)")
    print(f"Difference (model − rater): {mean_mr - mean_rr:+.4f}")

    # ICC
    icc_result = compute_icc_twoway(df, RATERS)
    icc_interp = (
        "Excellent" if icc_result["ICC"] >= 0.90 else
        "Good"      if icc_result["ICC"] >= 0.75 else
        "Moderate"  if icc_result["ICC"] >= 0.50 else
        "Poor"
    )
    print(f"\nICC(2,1) = {icc_result['ICC']:.4f}  "
          f"[{icc_result['CI_lower']:.4f}, {icc_result['CI_upper']:.4f}]  "
          f"→ {icc_interp}")

    # Save statistics CSV
    stats_rows = []
    for _, row in rr_df.iterrows():
        stats_rows.append({"type": "rater-rater", "pair": row["pair"],
                           "r": row["r"], "p": row["p"], "n": row["n"]})
    for _, row in mr_df.iterrows():
        stats_rows.append({"type": "model-rater", "pair": row["pair"],
                           "r": row["r"], "p": row["p"], "n": row["n"]})
    # Append ICC summary row
    stats_rows.append({
        "type": "ICC(2,1)", "pair": "all 5 raters",
        "r": icc_result["ICC"], "p": None,
        "n": icc_result["n"],
    })
    stats_df = pd.DataFrame(stats_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "fig05_rater_agreement_stats.csv"
    stats_df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n[stats] saved → {csv_path}")

    # Build and save figure
    print("\nBuilding Figure 5 …")
    fig = make_figure(rr_df, mr_df, icc_result)
    save_fig(fig, out_dir, "fig05_rater_agreement")
    print("Done.")


if __name__ == "__main__":
    main()
