"""
figures/fig02_dataset_composition/plot_dataset_composition.py
=============================================================
Figure 2 — Dataset composition: image counts by visual score and by year.

    Panel A — Bar chart of image counts per SLB score value (1.0–9.0 in 0.5
              increments), with integer scores colored separately from
              half-integer scores.  A KDE-derived null distribution is overlaid
              as a dashed line to visualise rater half-integer underutilisation.

    Panel B — Bar chart of image counts per year (2023, 2024, 2025), colored
              with the shared batlow palette.

Half-integer analysis
---------------------
The 1–9 SLB scale offers 17 legal values (1, 1.5, 2, …, 9): 9 integers and
8 half-integers.  Expert raters systematically underuse the half-integer steps.
The KDE-expected null accounts for the continuous severity distribution:

  1. Fit a boundary-reflected Gaussian KDE to the observed score distribution.
  2. Integrate ±0.25 windows around each of the 17 score values.
  3. Renormalise to obtain expected proportions.

A chi-squared goodness-of-fit test (17 bins) is printed to stdout.

Note: for any smooth density on [1, 9], the ±0.25 windows for integers and
half-integers tile the interval with equal total measure (~4.0 each), so the
KDE null converges to ~0.50, not the naive uniform fraction 8/17 ≈ 0.47.

Data source
-----------
    data/labels/full_dataset.csv
    Required columns: image_filename, score, year

Output (written to this script's directory)
------------------------------------------
    fig02_dataset_composition.{pdf,png}
    fig02_half_integer_stats.csv  — per-score observed vs expected

Usage
-----
    python figures/fig02_dataset_composition/plot_dataset_composition.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import quad
from scipy.stats import chisquare, gaussian_kde

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import apply_style, despine, save_fig, BATLOW_10, YEAR_COLORS

apply_style()

# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent.parent
DATA_CSV   = ROOT / "data" / "labels" / "full_dataset.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Score-type colors  (consistent with residual_analysis publication figure)
# ---------------------------------------------------------------------------
# Integer scores:     dark blue (batlow-29) — majority category, darker anchor
# Half-integer scores: warm orange (batlow-171) — minority, highlights deficit
INT_COLOR  = "#103F60"   # batlow-29  — integer scores
HALF_COLOR = "#D29343"   # batlow-171 — half-integer scores
KDE_COLOR  = "#444444"   # neutral dark grey — KDE-expected null line


# Legal SLB score values
SCORE_MIN, SCORE_MAX = 1.0, 9.0
ALL_SCORES  = np.arange(SCORE_MIN, SCORE_MAX + 0.5, 0.5)   # 17 values
INT_SCORES  = ALL_SCORES[ALL_SCORES % 1 == 0]               # 9 integers
HALF_SCORES = ALL_SCORES[ALL_SCORES % 1 != 0]              # 8 half-integers
HALF_WINDOW = 0.25   # integration half-width around each score step


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_path: Path) -> pd.DataFrame:
    """
    Load and lightly validate the clean dataset CSV.

    The dataset is assumed to be pre-filtered (missing images and white-fill
    tiles already removed by scripts/create_cv_splits.py).
    """
    df = pd.read_csv(data_path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"score", "year"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"full_dataset.csv is missing columns: {sorted(missing)}. "
            f"Found: {list(df.columns)}"
        )
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["year"]  = df["year"].astype(int)
    df = df.dropna(subset=["score"])

    print(f"[data]  {len(df):,} images loaded  |  "
          f"score range [{df['score'].min()}, {df['score'].max()}]  |  "
          f"years {sorted(df['year'].unique())}")
    return df


# ---------------------------------------------------------------------------
# KDE-based half-integer null distribution
# ---------------------------------------------------------------------------

def fit_kde_expected(scores: np.ndarray) -> dict[float, float]:
    """
    Fit a boundary-reflected Gaussian KDE and return expected counts per legal
    score value.

    The boundary-reflection trick mirrors the data at SCORE_MIN and SCORE_MAX
    before fitting, correcting for edge bias.  Scott's bandwidth is computed on
    the ORIGINAL data (not the reflected array) to avoid bandwidth inflation.

    Parameters
    ----------
    scores : 1-D array of observed SLB score values

    Returns
    -------
    dict mapping each legal score value → expected count (float)
    """
    N = len(scores)

    # Reflect at both boundaries
    reflected = np.concatenate([
        scores,
        2 * SCORE_MIN - scores,
        2 * SCORE_MAX - scores,
    ])

    # Scott bandwidth from ORIGINAL data, floored to the 0.5 step size
    scott_bw = scores.std(ddof=1) * N ** (-0.2)
    bw_abs   = max(scott_bw, 0.5)   # floor prevents under-smoothing
    bw_scale = bw_abs / reflected.std(ddof=1)
    kde      = gaussian_kde(reflected, bw_method=bw_scale)

    # Scalar-safe wrapper (gaussian_kde returns ndarray)
    def _kde_raw(x: float) -> float:
        return float(kde(np.atleast_1d(x))[0])

    # Normalise so the KDE integrates to 1.0 over [SCORE_MIN, SCORE_MAX]
    norm_const, _ = quad(_kde_raw, SCORE_MIN, SCORE_MAX)

    def kde_pdf(x: float) -> float:
        return _kde_raw(x) / norm_const

    # Integrate ±HALF_WINDOW around each legal score value
    raw_fracs: dict[float, float] = {}
    for sv in ALL_SCORES:
        lo = max(SCORE_MIN, sv - HALF_WINDOW)
        hi = min(SCORE_MAX, sv + HALF_WINDOW)
        m, _ = quad(kde_pdf, lo, hi)
        raw_fracs[sv] = m

    # Renormalise so fractions sum to exactly 1.0
    total = sum(raw_fracs.values())
    return {sv: raw_fracs[sv] / total * N for sv in ALL_SCORES}


def chi_squared_test(
    observed: np.ndarray,
    expected: np.ndarray,
    score_vals: np.ndarray,
) -> None:
    """Print chi-squared goodness-of-fit results to stdout."""
    chi2_stat, p_val = chisquare(observed, expected)
    df_val = len(score_vals) - 1
    print(f"\nChi-squared goodness-of-fit (observed vs KDE-expected, {df_val} d.f.)")
    print(f"  χ² = {chi2_stat:,.2f}  |  p = {p_val:.2e}  |  "
          f"{'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")


# ---------------------------------------------------------------------------
# Figure 2 — two-panel dataset composition
# ---------------------------------------------------------------------------

def make_figure(df: pd.DataFrame) -> plt.Figure:
    """
    Build the two-panel dataset composition figure.

    Panel A — score distribution bars + KDE-expected null line
    Panel B — image count by year

    Returns
    -------
    matplotlib Figure
    """
    scores = df["score"].dropna().values
    N      = len(scores)

    # ── KDE-expected counts ──────────────────────────────────────────────────
    expected_counts = fit_kde_expected(scores)

    # ── Observed counts per legal score value ────────────────────────────────
    score_vc = (
        pd.Series(scores)
        .value_counts()
        .reindex(ALL_SCORES, fill_value=0)
        .sort_index()
    )
    obs_arr = np.array([score_vc[sv] for sv in ALL_SCORES], dtype=float)
    exp_arr = np.array([expected_counts[sv] for sv in ALL_SCORES], dtype=float)

    # Print summary statistics
    n_half_obs = int(score_vc[HALF_SCORES].sum())
    p_half_obs = n_half_obs / N
    p_half_kde = sum(expected_counts[sv] / N for sv in HALF_SCORES)
    print(f"\nHalf-integer fraction — observed: {p_half_obs:.4f}  |  "
          f"KDE null: {p_half_kde:.4f}  |  "
          f"deficit: {(p_half_kde - p_half_obs) * 100:+.2f} pp")
    chi_squared_test(obs_arr, exp_arr, ALL_SCORES)

    # ── Year counts ──────────────────────────────────────────────────────────
    year_counts = df["year"].value_counts().sort_index()

    # ── Build figure ─────────────────────────────────────────────────────────
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(12, 5),
        gridspec_kw={"width_ratios": [2.2, 1.0]},
    )

    # ─────────── Panel A: score distribution ─────────────────────────────────
    x_pos       = np.arange(len(ALL_SCORES))
    bar_colors  = [INT_COLOR if sv % 1 == 0 else HALF_COLOR for sv in ALL_SCORES]

    ax_a.bar(x_pos, obs_arr, color=bar_colors,
             edgecolor="black", linewidth=0.5, zorder=2)

    # KDE-expected null line
    ax_a.plot(x_pos, exp_arr, color=KDE_COLOR, linewidth=1.6,
              linestyle="--", marker="o", markersize=4, zorder=3,
              label="KDE-expected")

    # x-axis ticks: show every integer, suppress half-integer labels for clarity
    tick_labels = [f"{sv:g}" if sv % 1 == 0 else "" for sv in ALL_SCORES]
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(tick_labels, fontsize=9)
    ax_a.set_xlabel("Visual Score", fontsize=12, fontweight="bold", labelpad=6)
    ax_a.set_ylabel("Image Count", fontsize=12, fontweight="bold", labelpad=6)
    ax_a.set_title("A) Image Counts by Visual Score",
                   fontsize=14, fontweight="bold", pad=10, loc="left")
    ax_a.set_ylim(0, obs_arr.max() * 1.20)
    ax_a.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_a.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.55, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.spines["left"].set_linewidth(1.5)
    ax_a.spines["bottom"].set_linewidth(1.5)
    ax_a.tick_params(width=1.2)
    despine(ax_a)

    # Legend: integer bar, half-integer bar, KDE line
    ax_a.legend(
        handles=[
            mpatches.Patch(facecolor=INT_COLOR,  label="Integer score"),
            mpatches.Patch(facecolor=HALF_COLOR, label="Half-integer score"),
            mlines.Line2D([], [], color=KDE_COLOR, linewidth=1.6,
                          linestyle="--", marker="o", markersize=4,
                          label="KDE-expected null"),
        ],
        fontsize=9, framealpha=0.88, loc="upper right",
    )

    # ─────────── Panel B: year counts ────────────────────────────────────────
    yr_labels = [str(yr) for yr in year_counts.index]
    yr_colors = [YEAR_COLORS.get(int(yr), BATLOW_10[3]) for yr in year_counts.index]

    ax_b.bar(yr_labels, year_counts.values, color=yr_colors,
             edgecolor="black", linewidth=0.5, zorder=2)

    # Value labels above bars
    for i, (lbl, val) in enumerate(zip(yr_labels, year_counts.values)):
        ax_b.text(i, val + year_counts.max() * 0.015,
                  f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax_b.set_xlabel("Year", fontsize=12, fontweight="bold", labelpad=6)
    ax_b.set_ylabel("Image Count", fontsize=12, fontweight="bold", labelpad=6)
    ax_b.set_title("B) Image Counts by Year",
                   fontsize=14, fontweight="bold", pad=10, loc="left")
    ax_b.set_ylim(0, year_counts.max() * 1.22)
    ax_b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_b.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.55, zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.spines["left"].set_linewidth(1.5)
    ax_b.spines["bottom"].set_linewidth(1.5)
    ax_b.tick_params(width=1.2)
    despine(ax_b)

    fig.tight_layout(pad=2.5)
    return fig, score_vc, expected_counts


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Figure 2 — Dataset composition: score distribution and year counts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",    default=str(DATA_CSV),
                        help="Full dataset CSV (data/labels/full_dataset.csv).")
    parser.add_argument("--out_dir", default=str(OUTPUT_DIR),
                        help="Output directory.")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir   = Path(args.out_dir)

    print("=" * 60)
    print("FIGURE 2 — DATASET COMPOSITION")
    print("=" * 60)
    print(f"Data   : {data_path}")
    print(f"Output : {out_dir}\n")

    df = load_data(data_path)
    fig, score_vc, expected_counts = make_figure(df)
    save_fig(fig, out_dir, "fig02_dataset_composition")

    # Save per-score observed-vs-expected table for reference
    stats_rows = []
    for sv in ALL_SCORES:
        obs = int(score_vc.get(sv, 0))
        exp = expected_counts.get(sv, 0)
        stats_rows.append({
            "score":    sv,
            "type":     "integer" if sv % 1 == 0 else "half-integer",
            "observed": obs,
            "expected_kde": round(exp, 2),
            "deviation": round(obs - exp, 2),
        })
    stats_df = pd.DataFrame(stats_rows)
    csv_path = out_dir / "fig02_half_integer_stats.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n[stats] saved → {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
