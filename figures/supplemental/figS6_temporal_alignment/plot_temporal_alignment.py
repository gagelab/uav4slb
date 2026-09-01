"""
figures/supplemental/figS5_temporal_alignment/plot_temporal_alignment.py
========================================================================
Figure S5 — Distribution of score-flight temporal offset, stacked by year.

Shows how many images (y-axis) were captured at each signed day offset
relative to the human scoring date (x-axis).  Bars are stacked by year and
colored with the shared batlow palette.

Signed convention (matches full_dataset.csv):
    signed_days_diff  =  flight_date − rating_date
    negative  →  flight happened BEFORE the score was assigned
    positive  →  flight happened AFTER the score was assigned
    zero      →  same-day collection

A shaded band marks the ±3-day target alignment window.

Data source
-----------
    data/labels/full_dataset.csv
    Required columns: year, signed_days_diff

Output (written to this script's directory)
------------------------------------------
    figS5_temporal_alignment.{pdf,png}
    figS5_temporal_alignment_stats.csv  — per-day-offset × year counts

Usage
-----
    python figures/supplemental/figS5_temporal_alignment/plot_temporal_alignment.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

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

# Images within ±3 days of the rating are considered "well-aligned"
ALIGNMENT_WINDOW = 3


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"year", "signed_days_diff"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"full_dataset.csv missing columns: {sorted(missing)}. "
            f"Found: {list(df.columns)}"
        )
    df["signed_days_diff"] = pd.to_numeric(df["signed_days_diff"], errors="coerce")
    df["year"]             = df["year"].astype(int)
    df = df.dropna(subset=["signed_days_diff"])

    dd = df["signed_days_diff"].astype(int)
    n_aligned = (dd.abs() <= ALIGNMENT_WINDOW).sum()
    print(f"[data]  {len(df):,} images  |  "
          f"offset range [{dd.min()}, {dd.max()}]  |  "
          f"|offset| ≤ {ALIGNMENT_WINDOW} days: "
          f"{n_aligned:,} ({n_aligned / len(df) * 100:.1f}%)")
    return df


# ---------------------------------------------------------------------------
# Figure S5
# ---------------------------------------------------------------------------

def make_figure(df: pd.DataFrame) -> plt.Figure:
    """
    Build the stacked bar chart of score-flight temporal offset.

    Each x-position is a unique signed_days_diff value; bars are stacked by
    year.  A grey shaded band marks the ±3-day alignment window.

    Returns
    -------
    matplotlib Figure
    """
    years    = sorted(df["year"].unique())
    dd       = df["signed_days_diff"].astype(int)
    day_vals = sorted(dd.unique())

    # Per-year counts reshaped into a pivot table
    year_dd = (
        df.groupby(["signed_days_diff", "year"])
        .size()
        .unstack(fill_value=0)
        .reindex(day_vals, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Stacked bars
    bottom = np.zeros(len(day_vals))
    for yr in sorted(years):
        if yr not in year_dd.columns:
            continue
        label, color = str(yr), YEAR_COLORS.get(yr, "#888888")
        heights = year_dd[yr].values
        ax.bar(
            day_vals, heights, bottom=bottom,
            color=color, edgecolor="white", linewidth=0.5,
            width=0.7, label=label, zorder=3,
        )
        bottom += heights

    # ±3-day alignment window (shaded)
    ax.axvspan(
        -(ALIGNMENT_WINDOW + 0.5), ALIGNMENT_WINDOW + 0.5,
        alpha=0.08, color="gray", zorder=1,
        label=f"±{ALIGNMENT_WINDOW}-day window",
    )

    # x-axis: label every integer in the observed range
    ax.set_xticks(day_vals)
    ax.set_xticklabels([str(d) for d in day_vals], fontsize=9, rotation=45, ha="right")

    ax.set_xlabel("Score-Flight Offset (days: flight − scoring)",
                  fontsize=11, fontweight="bold", labelpad=6)
    ax.set_ylabel("Image Count", fontsize=11, fontweight="bold", labelpad=6)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.50, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(title="Year", fontsize=9, title_fontsize=9,
              framealpha=0.88, loc="upper right")
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(width=1.2)
    despine(ax)

    fig.tight_layout()
    return fig, year_dd


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Figure S5 — temporal alignment stacked bar chart.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",    default=str(DATA_CSV))
    parser.add_argument("--out_dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir   = Path(args.out_dir)

    print("=" * 60)
    print("FIGURE S5 — TEMPORAL ALIGNMENT")
    print("=" * 60)

    df = load_data(data_path)

    fig, year_dd = make_figure(df)
    save_fig(fig, out_dir, "figS5_temporal_alignment")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "figS5_temporal_alignment_stats.csv"
    year_dd.to_csv(csv_path, float_format="%.0f")
    print(f"\n[stats] saved → {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
