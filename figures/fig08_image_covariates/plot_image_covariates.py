"""
figures/fig08_image_covariates/plot_image_covariates.py
=======================================================
Figure 8 — Per-image Spearman correlations between image quality covariates
and absolute prediction error for the EVA-02-B model under CV0.

The primary output is a 2 × 2 scatter plot (absolute error vs each covariate),
with Spearman ρ and p-value annotated on each panel.  A second figure
with signed error on the y-axis is also saved for diagnostic use.

Covariates
----------
    frac_weed       — weed canopy fraction (pixel-classification)
    sf_illuminorm   — shadow fraction (illumination-normalized threshold)
    mean_brightness — per-image mean pixel brightness (0–255)
    contrast_rms    — per-image RMS contrast

Data sources
------------
    Predictions  : results/eva02_base/predictions/cv0_combined_predictions.csv
                   Required columns: image_filename, actual, predicted

    Covariates   : data/covariates/image_covariates.csv
                   Pre-joined image-level features.  Required columns:
                     image_filename, frac_weed, sf_illuminorm,
                     mean_brightness, contrast_rms

    image_covariates.csv is produced by the data-preparation workflow
    documented in data/data_README.md.

Outputs (written to this script's directory)
--------------------------------------------
    fig08_image_covariates_abs.{pdf,png}    — |error| vs covariates (Fig 8)
    fig08_image_covariates_signed.{pdf,png} — signed error vs covariates
    fig08_image_covariates_stats.csv        — Spearman ρ, p, n per covariate

Usage
-----
    python figures/fig08_image_covariates/plot_image_covariates.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import (
    apply_style, despine, save_fig,
    COVARIATE_COLORS, COVARIATE_LABELS,
)

apply_style()

# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent

PREDICTIONS_DEFAULT = (
    ROOT / "results" / "eva02_base" / "predictions" / "cv0_combined_predictions.csv"
)
COVARIATES_DEFAULT  = ROOT / "data" / "covariates" / "image_covariates.csv"
OUTPUT_DIR          = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Covariate order (left-to-right, top-to-bottom in the 2×2 grid)
# ---------------------------------------------------------------------------
# Ordered to match the panel labels in Figure 7 (noise source examples) so
# that readers can visually link the example images to the aggregate statistics.
COVARIATE_ORDER: list[str] = [
    "frac_weed",        # Panel A — Weeds
    "sf_illuminorm",    # Panel B — Shadows
    "mean_brightness",  # Panel C — Brightness
    "contrast_rms",     # Panel D — Contrast
]

PANEL_LABELS = ["A)", "B)", "C)", "D)"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_merge(pred_path: Path, cov_path: Path) -> pd.DataFrame:
    """
    Load predictions and image covariates, then join on image_filename.

    Returns
    -------
    DataFrame with columns: image_filename, actual, predicted, abs_error,
    signed_error, frac_weed, sf_illuminorm, mean_brightness, contrast_rms
    """
    # ── predictions ──────────────────────────────────────────────────────────
    pred = pd.read_csv(pred_path)
    pred.columns = [c.strip().lower() for c in pred.columns]

    if "signed_error" not in pred.columns:
        pred["signed_error"] = pred["predicted"] - pred["actual"]
    pred["abs_error"] = pred["signed_error"].abs()

    print(f"[predictions] {len(pred)} images")

    # ── image covariates ─────────────────────────────────────────────────────
    cov = pd.read_csv(cov_path)
    cov.columns = [c.strip().lower() for c in cov.columns]
    print(f"[covariates]  {len(cov)} image rows")

    # Merge on image_filename; keep only images with both predictions and features
    merged = pred.merge(cov, on="image_filename", how="inner")
    n_lost = len(pred) - len(merged)
    if n_lost > 0:
        print(f"[merged]      {len(merged)} images after inner join "
              f"({n_lost} predictions had no covariate match)")
    else:
        print(f"[merged]      {len(merged)} images — full match")
    return merged


# ---------------------------------------------------------------------------
# Spearman statistics
# ---------------------------------------------------------------------------

def compute_spearman(
    df: pd.DataFrame,
    error_col: str,
) -> pd.DataFrame:
    """
    Compute Spearman ρ and two-sided p-value for each covariate vs *error_col*.

    Parameters
    ----------
    df        : merged DataFrame
    error_col : "abs_error" or "signed_error"

    Returns
    -------
    DataFrame indexed by covariate with columns: rho, pval, n
    """
    rows = []
    for col in COVARIATE_ORDER:
        if col not in df.columns:
            print(f"  [skip] '{col}' not found — check image_covariates.csv")
            rows.append({"covariate": col, "error_type": error_col,
                         "rho": np.nan, "pval": np.nan, "n": 0})
            continue
        sub  = df[[error_col, col]].dropna()
        n    = len(sub)
        rho, pval = stats.spearmanr(sub[error_col], sub[col]) if n >= 3 else (np.nan, np.nan)
        rows.append({"covariate": col, "error_type": error_col,
                     "rho": rho, "pval": pval, "n": n})
        label = COVARIATE_LABELS.get(col, col)
        print(f"  {label:<25s}  ρ={rho:+.4f}  p={pval:.4e}  n={n}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def plot_scatter_2x2(
    df: pd.DataFrame,
    stats_df: pd.DataFrame,
    error_col: str,
    out_dir: Path,
    stem: str,
) -> None:
    """
    Build a 2 × 2 scatter panel (one covariate per panel).

    Each panel shows:
      - Individual data points (subsampled to ≤ 5 000 for legibility)
      - A vertical dashed line at the covariate mean
      - Spearman ρ and p-value annotation in the upper-right corner
      - A panel letter (A, B, C, D) in the upper-left corner

    Parameters
    ----------
    df        : merged DataFrame (image-level)
    stats_df  : Spearman results from compute_spearman()
    error_col : "abs_error" or "signed_error"
    out_dir   : output directory
    stem      : filename stem (e.g. "fig08_image_covariates_abs")
    """
    is_signed = error_col == "signed_error"
    y_label   = ("Signed error  (predicted − actual)"
                 if is_signed else "Absolute error")

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 9.6))
    axes = axes.flatten()

    # Fixed random seed for reproducible subsampling
    rng = np.random.default_rng(42)

    # Build a lookup table for Spearman statistics
    stat_lookup = stats_df.set_index("covariate").to_dict("index")

    for ax, panel_lbl, col in zip(axes, PANEL_LABELS, COVARIATE_ORDER):
        label = COVARIATE_LABELS.get(col, col)
        color = COVARIATE_COLORS.get(col, "#444444")

        if col not in df.columns:
            ax.text(0.5, 0.5, f"'{col}' not available",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="gray")
            continue

        sub = df[[error_col, col]].dropna()

        # Subsample to reduce over-plotting on large datasets
        if len(sub) > 5000:
            idx = rng.choice(len(sub), size=5000, replace=False)
            sub = sub.iloc[idx]

        ax.scatter(
            sub[col], sub[error_col],
            s=5, alpha=0.25, color=color, linewidths=0,
        )

        # OLS fit line (computed on the full non-subsampled data)
        full_sub = df[[error_col, col]].dropna()
        if len(full_sub) >= 2:
            slope, intercept, *_ = stats.linregress(full_sub[col], full_sub[error_col])
            x_range = np.array([full_sub[col].min(), full_sub[col].max()])
            ax.plot(x_range, slope * x_range + intercept,
                    color="black", linewidth=1.5, zorder=4)

        # Zero line for signed error
        if is_signed:
            ax.axhline(0, color="0.4", linewidth=0.8, linestyle="--")

        # Spearman annotation in the upper-right corner
        if col in stat_lookup:
            row = stat_lookup[col]
            rho, pval = row["rho"], row["pval"]
            p_str = f"p = {pval:.2e}" if not np.isnan(pval) else "p = —"
            rho_str = f"ρ = {rho:+.3f}" if not np.isnan(rho) else "ρ = —"
            ann = f"{rho_str}\n{p_str}"
            ax.text(
                0.97, 0.97, ann,
                transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="white", ec="0.8", alpha=0.9),
            )

        # Panel letter in the upper-left corner
        ax.text(
            0.03, 0.97, panel_lbl,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=12, fontweight="bold",
        )

        ax.set_xlabel(label, fontsize=11, fontweight="bold")
        # Only label the y-axis on the left column
        ax.set_ylabel(y_label if col in (COVARIATE_ORDER[0], COVARIATE_ORDER[2])
                      else "", fontsize=11, fontweight="bold")
        ax.spines["left"].set_linewidth(1.6)
        ax.spines["bottom"].set_linewidth(1.6)
        ax.tick_params(axis="both", which="major", labelsize=11, width=1.3)
        ax.grid(alpha=0.25, linestyle="--")
        despine(ax)

    fig.tight_layout()
    save_fig(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Figure 8 — image covariate vs prediction error scatter plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions", default=str(PREDICTIONS_DEFAULT),
                        help="Combined CV0 predictions CSV.")
    parser.add_argument("--covariates",  default=str(COVARIATES_DEFAULT),
                        help="Pre-joined image covariate CSV.")
    parser.add_argument("--out_dir",     default=str(OUTPUT_DIR),
                        help="Output directory.")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    cov_path  = Path(args.covariates)
    out_dir   = Path(args.out_dir)

    print("=" * 60)
    print("FIGURE 8 — IMAGE COVARIATE × PREDICTION ERROR")
    print("=" * 60)
    print(f"Predictions : {pred_path}")
    print(f"Covariates  : {cov_path}")
    print(f"Output      : {out_dir}\n")

    df = load_and_merge(pred_path, cov_path)

    # ── Absolute error ───────────────────────────────────────────────────────
    print("\nSpearman ρ — absolute error vs covariates")
    print("-" * 60)
    stats_abs = compute_spearman(df, "abs_error")
    print("\nBuilding Figure 8 (absolute error) …")
    plot_scatter_2x2(df, stats_abs, "abs_error", out_dir,
                     "fig08_image_covariates_abs")

    # ── Signed error (supplemental) ──────────────────────────────────────────
    print("\nSpearman ρ — signed error vs covariates")
    print("-" * 60)
    stats_signed = compute_spearman(df, "signed_error")
    print("\nBuilding supplemental signed-error figure …")
    plot_scatter_2x2(df, stats_signed, "signed_error", out_dir,
                     "fig08_image_covariates_signed")

    # ── Save statistics table ────────────────────────────────────────────────
    csv_path = out_dir / "fig08_image_covariates_stats.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([stats_abs, stats_signed], ignore_index=True).to_csv(
        csv_path, index=False, float_format="%.6f"
    )
    print(f"\n[stats] saved → {csv_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
