"""
figures/supplemental/figS1_flight_timeline/plot_flight_timeline.py
==================================================================
Figure S1 — Rating-session and flight-date calendar timeline.

One subplot row per year (2023, 2024, 2025).  Within each row, one bubble per
field on each unique date when either a rating session or a flight occurred:

    ■ Square marker  — visual scoring session (date the field was rated)
    ▲ Triangle marker — UAV flight date (date imagery was captured)

Bubble area is proportional to the number of images collected or rated on that
date-field combination, enabling quick visual assessment of sampling intensity.
All years are plotted on a shared calendar x-axis (month-day normalised to a
reference year) to enable direct comparison of seasonal timing.

Data source
-----------
    data/labels/full_dataset.csv
    Required columns: field, year, date (rating date), flight_date

Output (written to this script's directory)
------------------------------------------
    figS1_flight_timeline.{pdf,png}

Usage
-----
    python figures/supplemental/figS1_flight_timeline/plot_flight_timeline.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from style import apply_style, save_fig, FOLD_COLORS

apply_style()

# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent.parent.parent
DATA_CSV   = ROOT / "data" / "labels" / "full_dataset.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------
# Rating markers: dark blue (anchors the "observation" side)
# Flight markers: warm orange (anchors the "imaging" side)
COLOR_RATING = "#103F60"   # batlow-29
COLOR_FLIGHT = "#D29343"   # batlow-171

# Reference year for normalising all dates to a shared x-axis
REF_YEAR = 2000


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_path: Path) -> pd.DataFrame:
    """
    Load the full dataset CSV and parse date columns.

    Expects columns: field, year, date (rating date), flight_date.
    """
    df = pd.read_csv(data_path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"field", "year", "date", "flight_date"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"full_dataset.csv missing columns: {sorted(missing)}. "
            f"Found: {list(df.columns)}"
        )

    # Parse dates — 'mixed' format handles M/D/YY and YYYY-MM-DD
    for col in ("date", "flight_date"):
        df[col] = pd.to_datetime(df[col], format="mixed", dayfirst=False)

    df["year"] = df["year"].astype(int)
    print(f"[data]  {len(df):,} rows  |  "
          f"years {sorted(df['year'].unique())}  |  "
          f"fields {sorted(df['field'].unique())}")
    return df


def _to_ref_year(series: pd.Series) -> pd.Series:
    """Replace the year component with REF_YEAR, preserving month and day."""
    return pd.to_datetime(
        series.dt.strftime(f"{REF_YEAR}-%m-%d"), errors="coerce"
    )


# ---------------------------------------------------------------------------
# Figure S1
# ---------------------------------------------------------------------------

def make_figure(df: pd.DataFrame) -> plt.Figure:
    """
    Build the timeline bubble chart.

    Returns
    -------
    matplotlib Figure
    """
    years  = sorted(df["year"].unique())
    fields = sorted(df["field"].unique())

    # Global x-axis extent: month-day window spanning all years
    all_dates = pd.concat([
        _to_ref_year(df["date"]),
        _to_ref_year(df["flight_date"]),
    ]).dropna()
    x_min = all_dates.min() - pd.Timedelta(days=4)
    x_max = all_dates.max() + pd.Timedelta(days=4)

    fig, axes = plt.subplots(
        len(years), 1,
        figsize=(14, 2.8 * len(years)),
        sharex=True,
    )
    if len(years) == 1:
        axes = [axes]

    for ax, yr in zip(axes, years):
        sub = df[df["year"] == yr]

        # ── Rating session bubbles ─────────────────────────────────────────
        session_df = (
            sub.groupby(["date", "field"]).size()
            .reset_index(name="n")
        )
        session_df["date_ref"] = _to_ref_year(session_df["date"])

        # ── Flight date bubbles ────────────────────────────────────────────
        flight_df = (
            sub.groupby(["flight_date", "field"]).size()
            .reset_index(name="n")
            .rename(columns={"flight_date": "date"})
        )
        flight_df["date_ref"] = _to_ref_year(flight_df["date"])

        # Scale bubble area relative to maximum count in this year
        max_n = max(session_df["n"].max(), flight_df["n"].max(), 1)

        ax.scatter(
            session_df["date_ref"], session_df["field"],
            s=session_df["n"] / max_n * 300 + 40,
            color=COLOR_RATING, alpha=0.80,
            marker="s", zorder=3,
            label="Visual scoring session",
        )
        ax.scatter(
            flight_df["date_ref"], flight_df["field"],
            s=flight_df["n"] / max_n * 300 + 40,
            color=COLOR_FLIGHT, alpha=0.60,
            marker="^", zorder=2,
            label="Flight date",
        )

        ax.set_xlim(x_min, x_max)
        ax.set_yticks(range(len(fields)))
        ax.set_yticklabels(fields, fontsize=10)
        ax.set_ylabel("Field", fontsize=10)
        ax.set_title(f"Year {yr}  (bubble area ∝ image count)",
                     fontsize=11, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
        ax.tick_params(axis="x", rotation=40, labelsize=9)
        ax.grid(axis="x", alpha=0.30, linestyle="--")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Figure S1 — Rating and flight timeline bubble chart.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",    default=str(DATA_CSV),
                        help="Full dataset CSV.")
    parser.add_argument("--out_dir", default=str(OUTPUT_DIR),
                        help="Output directory.")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir   = Path(args.out_dir)

    print("=" * 60)
    print("FIGURE S1 — RATING / FLIGHT TIMELINE")
    print("=" * 60)
    df  = load_data(data_path)
    fig = make_figure(df)
    save_fig(fig, out_dir, "figS1_flight_timeline")
    print("Done.")


if __name__ == "__main__":
    main()
