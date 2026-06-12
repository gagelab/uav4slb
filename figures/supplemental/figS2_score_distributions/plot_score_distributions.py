"""
figures/supplemental/figS2_score_distributions/plot_score_distributions.py
==========================================================================
Figure S2 — SLB severity score distributions by year.

A violin plot with one column per year (2023, 2024, 2025), colored with the
shared batlow palette.  A Kruskal-Wallis H-test annotation indicates whether
score distributions differ significantly across years.

Data source
-----------
    data/labels/full_dataset.csv
    Required columns: score, year

Output (written to this script's directory)
------------------------------------------
    figS2_score_distributions.{pdf,png}
    figS2_score_distributions_stats.csv  — per-year summary + KW test

Usage
-----
    python figures/supplemental/figS2_score_distributions/plot_score_distributions.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from style import apply_style, despine, save_fig, YEAR_COLORS

apply_style()

# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent.parent.parent
DATA_CSV   = ROOT / "data" / "labels" / "full_dataset.csv"
OUTPUT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["year"]  = df["year"].astype(int)
    df = df.dropna(subset=["score"])
    print(f"[data]  {len(df):,} images  |  years {sorted(df['year'].unique())}")
    return df


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-year descriptive statistics."""
    rows = []
    for yr, grp in df.groupby("year"):
        s = grp["score"]
        rows.append({
            "year":   yr,
            "n":      len(s),
            "mean":   s.mean(),
            "median": s.median(),
            "sd":     s.std(),
            "q1":     s.quantile(0.25),
            "q3":     s.quantile(0.75),
            "iqr":    s.quantile(0.75) - s.quantile(0.25),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure S2
# ---------------------------------------------------------------------------

def make_figure(df: pd.DataFrame) -> plt.Figure:
    """
    Violin plot of score distributions by year, with Kruskal-Wallis annotation.

    Returns
    -------
    matplotlib Figure
    """
    years = sorted(df["year"].unique())

    # Kruskal-Wallis H-test
    groups  = [df.loc[df["year"] == yr, "score"].dropna().values for yr in years]
    H_kw, p_kw = kruskal(*groups)
    sig = "***" if p_kw < 0.001 else "**" if p_kw < 0.01 else "*" if p_kw < 0.05 else "ns"
    kw_str = (
        f"Kruskal–Wallis\nH = {H_kw:.1f},  p {'< 0.001' if p_kw < 0.001 else f'= {p_kw:.3e}'}  {sig}"
    )
    print(f"\nKruskal-Wallis: H = {H_kw:.2f},  p = {p_kw:.4e}  {sig}")

    # Build year color palette for seaborn (keyed by year as string)
    palette = {str(yr): YEAR_COLORS.get(yr, "#888888") for yr in years}

    # seaborn expects string x categories
    df_plot       = df.copy()
    df_plot["yr"] = df_plot["year"].astype(str)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    sns.violinplot(
        data=df_plot, x="yr", y="score",
        order=[str(yr) for yr in years],
        palette=palette,
        inner="box", linewidth=0.9,
        ax=ax,
    )

    # Overlay explicit median dots so they match the legend glyph
    for x_pos, yr in enumerate(years):
        med = df.loc[df["year"] == yr, "score"].median()
        ax.scatter(
            x_pos, med,
            s=30, color="white", edgecolors="black", linewidths=0.8,
            zorder=5,
        )

    # Legend — explain the inner-box glyphs drawn by seaborn
    median_handle = mlines.Line2D(
        [], [], marker="o", color="white",
        markerfacecolor="white", markeredgecolor="black",
        markeredgewidth=0.8, markersize=5, linestyle="None",
        label="Median",
    )
    iqr_handle = mlines.Line2D(
        [], [], color="black", linewidth=3.5,
        solid_capstyle="butt", label="IQR (Q1–Q3)",
    )
    whisker_handle = mlines.Line2D(
        [], [], color="black", linewidth=1.0,
        solid_capstyle="butt", label="1.5 × IQR",
    )
    ax.legend(
        handles=[median_handle, iqr_handle, whisker_handle],
        fontsize=9, frameon=True,
        framealpha=0.9, edgecolor="#cccccc",
        loc="upper right",
    )

    ax.set_xlabel("Year",             fontsize=12, fontweight="bold", labelpad=6)
    ax.set_ylabel("SLB Severity Score", fontsize=12, fontweight="bold", labelpad=6)
    ax.set_yticks(range(1, 10))
    ax.tick_params(labelsize=10, width=1.2)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.grid(axis="y", alpha=0.30, linestyle="--")
    despine(ax)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Figure S2 — score distributions by year.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",    default=str(DATA_CSV))
    parser.add_argument("--out_dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir   = Path(args.out_dir)

    print("=" * 60)
    print("FIGURE S2 — SCORE DISTRIBUTIONS BY YEAR")
    print("=" * 60)

    df       = load_data(data_path)
    stats_df = compute_stats(df)
    print("\nPer-year statistics:")
    print(stats_df.round(4).to_string(index=False))

    fig = make_figure(df)
    save_fig(fig, out_dir, "figS2_score_distributions")

    out_dir.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(out_dir / "figS2_score_distributions_stats.csv",
                    index=False, float_format="%.4f")
    print("\nDone.")


if __name__ == "__main__":
    main()
