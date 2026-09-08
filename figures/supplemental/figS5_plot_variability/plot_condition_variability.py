"""
figures_defense/supplemental/figS5_condition_variability/plot_condition_variability.py
========================================================================================
Figure S5 — Within-flight variability in illumination and weed pressure.

Motivation
----------
This is the qualitative justification for
modeling at the plot/image level rather than the flight level.

Each field panel therefore shows:
    1. A field-level overview cropped to the experimental extent (not the
       full flight footprint), with every plot boundary outlined so the
       actual SLB trial area is unambiguous.
    2. Two highlighted plots from the SAME flight, chosen near the 10th and
       90th percentile of the relevant covariate (not the literal min/max,
       to avoid highlighting a possible artifact/outlier).
    3. Native-resolution insets cropped directly from the source GeoTIFF for
       each highlighted plot, connected to the overview with callout lines.

Panels
------
    Row 1 — C7B, 2025-07-01 (variable illumination within flight)
            covariate: mean_brightness  (source: image_covariates.csv)
    Row 2 — B7A, 2024-07-10 (variable weed pressure within flight)
            covariate: frac_weed        (source: image_covariates.csv)

Data sources
------------
    Orthomosaic GeoTIFFs (full resolution, not committed to the repo):
        <lighting_ortho>   C7B, 2025-07-01
        <weed_ortho>       B7A, 2024-07-10

    Plot boundary shapefiles (one per field):
        <shapefile_dir>/{FIELD}_plot_outline.shp
        Falls back to any .shp in shapefile_dir whose stem contains the
        field name (matches the convention in weed_pressure_pipeline.py).
        NOTE: on-disk shapefiles for B7A may be named with the legacy
        field code 'B7' rather than the canonical 'B7A' used in image
        filenames/covariates -- see FIELD_RENAME below.

    data/covariates/image_covariates.csv
        Required columns: image_filename, flight_id, field, frac_weed,
        mean_brightness (plus sf_illuminorm, contrast_rms, unused here).
        flight_id convention: YYYYMMDD_FIELD (e.g. '20250701_C7B').

Output (written to this script's directory)
--------------------------------------------
    figS5_condition_variability.{pdf,png}
    figS5_condition_variability_selected_plots.csv
        Plot ID, covariate value, and percentile rank for every highlighted
        plot -- keep this alongside the figure so the selection is
        traceable/reviewable, and cite the exact values in the caption.

Usage
-----
    # From repo root, with defaults matching the current data locations:
    python figures/supplemental/figS5_plot_variability/plot_condition_variability.py

    # Overriding paths:
    python figures/supplemental/figS5_plot_variability/plot_condition_variability.py \\
        --lighting-ortho /mnt/research-projects/j/jlgage/RawUAVData01/uavforslb/final_image/final_unsliced/DJI_20250701_C7B-2025_20251103T1727_ortho_dtm.tif \\
        --weed-ortho     /mnt/research-projects/j/jlgage/RawUAVData01/uavforslb/final_image/final_unsliced/DJI_202407101250_005-006_B7A-2024_20241107T1347_ortho_dtm.tif \\
        --shapefile-dir  /mnt/research-projects/j/jlgage/RawUAVData01/uavforslb/final_image/final_outline \\
        --covariates     data/covariates/image_covariates.csv

    # Hiding the shapefile plot-outline overlay on the overview panel:
    python figures/supplemental/figS5_plot_variability/plot_condition_variability.py --no-show-shapefile-outline

    # Hiding the north arrow:
    python figures/supplemental/figS5_plot_variability/plot_condition_variability.py --no-show-north-arrow

Dependencies
------------
    rasterio, geopandas, shapely, matplotlib, pandas, numpy
    conda activate uav_for_slb
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from rasterio.mask import mask as rio_mask
from rasterio.windows import Window, from_bounds
from shapely.geometry import box

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from style import apply_style, despine, save_fig  # noqa: E402

apply_style()

# ---------------------------------------------------------------------------
# Repository / data paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent

RAW_DATA_ROOT = Path("/mnt/research-projects/j/jlgage/RawUAVData01/uavforslb")

LIGHTING_ORTHO_DEFAULT = (
    RAW_DATA_ROOT / "final_image" / "final_unsliced"
    / "DJI_20250701_C7B-2025_20251103T1727_ortho_dtm.tif"
)
WEED_ORTHO_DEFAULT = (
    RAW_DATA_ROOT / "final_image" / "final_unsliced"
    / "DJI_202407101250_005-006_B7A-2024_20241107T1347_ortho_dtm.tif"
)
SHAPEFILE_DIR_DEFAULT = RAW_DATA_ROOT / "final_image" / "final_outline"
COVARIATES_DEFAULT = ROOT / "data" / "covariates" / "image_covariates.csv"

# Known on-disk field-code mismatch: shapefiles may use the legacy code
# while image filenames / covariates use the canonical code.
FIELD_RENAME = {"B7": "B7A"}

# Buffer (raster CRS units -- US survey feet, EPSG:2264) added around the
# trial extent when cropping the field-level overview.
OVERVIEW_BUFFER_FT = 15.0
# Buffer added around a single plot polygon when cropping an inset.
INSET_BUFFER_FT = 3.0
# Longest edge (pixels) of the field-level overview raster read; the
# overview is decimated to this size purely for display efficiency, the
# insets below are always read at full native resolution.
OVERVIEW_MAX_PX = 2200

PERCENTILE_LOW = 10
PERCENTILE_HIGH = 90


# ---------------------------------------------------------------------------
# Panel configuration
# ---------------------------------------------------------------------------
class PanelConfig:
    def __init__(
        self,
        field: str,
        flight_id: str,
        date_label: str,
        ortho_path: Path,
        covariate: str,
        covariate_label: str,
        value_fmt: str,
    ) -> None:
        self.field = field
        self.flight_id = flight_id
        self.date_label = date_label
        self.ortho_path = ortho_path
        self.covariate = covariate
        self.covariate_label = covariate_label
        self.value_fmt = value_fmt


def build_panel_configs(args: argparse.Namespace) -> list[PanelConfig]:
    return [
        PanelConfig(
            field="C7B",
            flight_id="20250701_C7B",
            date_label="C7B — 01 Jul 2025",
            ortho_path=Path(args.lighting_ortho),
            covariate="mean_brightness",
            covariate_label="Mean brightness",
            value_fmt="{:.0f}",
        ),
        PanelConfig(
            field="B7A",
            flight_id="20240710_B7A",
            date_label="B7A — 10 Jul 2024",
            ortho_path=Path(args.weed_ortho),
            covariate="frac_weed",
            covariate_label="Weed fraction",
            value_fmt="{:.2f}",
        ),
    ]


# ---------------------------------------------------------------------------
# Shapefile loading (mirrors the fallback pattern in weed_pressure_pipeline.py)
# ---------------------------------------------------------------------------
def load_plot_shapefile(shapefile_dir: Path, field: str, raster_crs) -> gpd.GeoDataFrame:
    candidates = list(shapefile_dir.glob(f"{field}_plot_outline.shp"))
    if not candidates:
        on_disk_field = next(
            (k for k, v in FIELD_RENAME.items() if v == field), field
        )
        candidates = list(shapefile_dir.glob(f"{on_disk_field}_plot_outline.shp"))
    if not candidates:
        candidates = [
            p for p in shapefile_dir.glob("*.shp")
            if field.lower() in p.stem.lower()
            or field.lower() in FIELD_RENAME.get(p.stem.split("_")[0], "").lower()
        ]
    if not candidates:
        raise FileNotFoundError(
            f"No plot outline shapefile found for field '{field}' in {shapefile_dir}"
        )

    plots = gpd.read_file(candidates[0])
    if plots.crs is not None and raster_crs is not None and plots.crs != raster_crs:
        plots = plots.to_crs(raster_crs)

    plot_id_col = "plot" if "plot" in plots.columns else None
    if plot_id_col is None:
        # Fall back to any column that looks like a plot identifier.
        candidates_cols = [c for c in plots.columns if "plot" in c.lower()]
        if not candidates_cols:
            raise ValueError(
                f"Could not find a plot-ID column in {candidates[0].name}. "
                f"Available columns: {list(plots.columns)}"
            )
        plot_id_col = candidates_cols[0]
    plots = plots.rename(columns={plot_id_col: "plot"})
    plots["plot"] = plots["plot"].astype(int)
    print(f"  Loaded {len(plots)} plot polygons from {candidates[0].name}")
    return plots


# ---------------------------------------------------------------------------
# Covariate selection
# ---------------------------------------------------------------------------
def select_representative_plots(
    covariates: pd.DataFrame, cfg: PanelConfig
) -> pd.DataFrame:
    flight_rows = covariates[covariates["flight_id"] == cfg.flight_id].copy()
    if flight_rows.empty:
        raise ValueError(
            f"No rows in image_covariates.csv with flight_id == '{cfg.flight_id}'. "
            f"Check the flight_id values actually present, e.g.:\n"
            f"{covariates['flight_id'].dropna().unique()[:20]}"
        )
    flight_rows["plot"] = (
        flight_rows["image_filename"]
        .str.extract(r"_(\d+)\.jpg$")[0]
        .astype(int)
    )
    flight_rows = flight_rows.dropna(subset=[cfg.covariate])
    flight_rows = flight_rows.sort_values(cfg.covariate).reset_index(drop=True)
    n = len(flight_rows)
    print(
        f"  {cfg.flight_id}: n={n} plots with valid '{cfg.covariate}', "
        f"range [{flight_rows[cfg.covariate].min():.3f}, "
        f"{flight_rows[cfg.covariate].max():.3f}]"
    )

    idx_low = int(round((PERCENTILE_LOW / 100) * (n - 1)))
    idx_high = int(round((PERCENTILE_HIGH / 100) * (n - 1)))
    low_row = flight_rows.iloc[idx_low].copy()
    high_row = flight_rows.iloc[idx_high].copy()
    low_row["percentile"] = PERCENTILE_LOW
    high_row["percentile"] = PERCENTILE_HIGH
    selected = pd.DataFrame([low_row, high_row]).reset_index(drop=True)
    selected["field"] = cfg.field
    selected["covariate"] = cfg.covariate
    return selected


# ---------------------------------------------------------------------------
# Raster helpers
# ---------------------------------------------------------------------------
def read_overview(src: rasterio.io.DatasetReader, bounds_geom, buffer_ft: float):
    """Windowed, decimated read of the trial extent for the field-level panel."""
    minx, miny, maxx, maxy = bounds_geom.buffer(buffer_ft).bounds
    window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
    window = window.round_offsets().round_lengths()
    win_h, win_w = int(window.height), int(window.width)
    scale = min(1.0, OVERVIEW_MAX_PX / max(win_h, win_w))
    out_h, out_w = max(1, int(win_h * scale)), max(1, int(win_w * scale))

    nbands = min(3, src.count)
    data = src.read(
        indexes=list(range(1, nbands + 1)),
        window=window,
        out_shape=(nbands, out_h, out_w),
    )
    transform = src.window_transform(window) * rasterio.Affine.scale(
        win_w / out_w, win_h / out_h
    )
    return data, transform


def read_inset(src: rasterio.io.DatasetReader, plot_geom, buffer_ft: float):
    """Full-native-resolution crop around a single plot polygon."""
    # join_style="mitre" keeps the plot's corners square; the default
    # "round" join would otherwise round off the buffered corners and
    # crop the inset to a rounded-rectangle shape.
    geom = plot_geom.buffer(buffer_ft, join_style="mitre")
    out_image, out_transform = rio_mask(src, [geom], crop=True, filled=True)
    nbands = min(3, out_image.shape[0])
    return out_image[:nbands], out_transform


def to_display(data: np.ndarray) -> np.ndarray:
    """CHW uint8/uint16 raster -> HWC float [0,1] for imshow, nodata as white."""
    arr = np.moveaxis(data, 0, -1).astype(np.float32)
    if arr.max() > 255:
        arr = arr / arr.max() * 255.0
    nodata_mask = np.all(arr <= 1, axis=-1)  # rio_mask fills outside geom with 0
    arr = arr / 255.0
    arr[nodata_mask] = 1.0  # render masked area as white, not black
    return np.clip(arr, 0, 1)


def add_scale_bar(ax, length_ft: float = 100.0) -> None:
    bar = AnchoredSizeBar(
        ax.transData,
        length_ft,
        f"{int(length_ft)} ft",
        loc="lower right",
        pad=0.4,
        color="white",
        frameon=True,
        size_vertical=length_ft * 0.02,
        fontproperties={"size": 8, "weight": "bold"},
    )
    bar.patch.set_facecolor("black")
    bar.patch.set_alpha(0.5)
    ax.add_artist(bar)


NORTH_ARROW_COLOR = "black"


def add_north_arrow(
    ax, xy=(0.94, 0.94), color=NORTH_ARROW_COLOR, xycoords="axes fraction"
) -> None:
    ax.annotate(
        "N", xy=xy, xytext=(xy[0], xy[1] - 0.08),
        xycoords=xycoords, textcoords=xycoords,
        ha="center", va="bottom", fontsize=12, fontweight="bold", color=color,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=3.0),
        annotation_clip=False,
    )


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------
def build_panel_row(
    fig, gs_row, cfg: PanelConfig, plots: gpd.GeoDataFrame,
    selected: pd.DataFrame, ortho_path: Path, show_shapefile_outline: bool,
) -> None:
    color = "black"

    with rasterio.open(ortho_path) as src:
        trial_bounds = box(*plots.total_bounds)
        overview_data, overview_transform = read_overview(
            src, trial_bounds, OVERVIEW_BUFFER_FT
        )

        ax_overview = fig.add_subplot(gs_row[0])
        h, w = overview_data.shape[1:]
        extent = rasterio.transform.array_bounds(h, w, overview_transform)
        # array_bounds returns (left, bottom, right, top) in a different
        # order than imshow's extent=(left, right, bottom, top)
        left, bottom, right, top = extent
        ax_overview.imshow(
            to_display(overview_data), extent=(left, right, bottom, top)
        )
        if show_shapefile_outline:
            plots.boundary.plot(ax=ax_overview, color="0.9", linewidth=0.5, alpha=0.8)

        inset_axes = []
        for i, (_, sel_row) in enumerate(selected.iterrows()):
            plot_geom = plots.loc[plots["plot"] == sel_row["plot"], "geometry"]
            if plot_geom.empty:
                print(
                    f"  [WARN] plot {sel_row['plot']} not found in shapefile for "
                    f"{cfg.field} -- skipping this inset."
                )
                continue
            plot_geom = plot_geom.iloc[0]

            gpd.GeoSeries([plot_geom], crs=plots.crs).boundary.plot(
                ax=ax_overview, color=color, linewidth=1.8
            )

            inset_data, inset_transform = read_inset(src, plot_geom, INSET_BUFFER_FT)
            ax_inset = fig.add_subplot(gs_row[i + 1])
            ih, iw = inset_data.shape[1:]
            ileft, ibottom, iright, itop = rasterio.transform.array_bounds(
                ih, iw, inset_transform
            )
            ax_inset.imshow(to_display(inset_data), extent=(ileft, iright, ibottom, itop))
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])
            for spine in ax_inset.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.5)

            label = (
                f"Plot {int(sel_row['plot'])}  ({sel_row['percentile']}th pct.)\n"
                f"{cfg.covariate_label} = {cfg.value_fmt.format(sel_row[cfg.covariate])}"
            )
            ax_inset.set_title(label, fontsize=8)
            inset_axes.append(ax_inset)

    ax_overview.set_xticks([])
    ax_overview.set_yticks([])
    for spine in ax_overview.spines.values():
        spine.set_visible(False)
    ax_overview.set_title(cfg.date_label, fontsize=9, fontweight="bold", loc="left")
    add_scale_bar(ax_overview)
    return ax_overview


def make_figure(
    panel_configs: list[PanelConfig],
    plots_by_field: dict[str, gpd.GeoDataFrame],
    selected_by_field: dict[str, pd.DataFrame],
    show_shapefile_outline: bool = True,
    show_north_arrow: bool = True,
):
    fig = plt.figure(figsize=(9.5, 6.4))
    gs = fig.add_gridspec(
        nrows=2, ncols=3, width_ratios=[2.0, 1.0, 1.0],
        hspace=0.35, wspace=0.15,
    )
    overview_axes = []
    for row, cfg in enumerate(panel_configs):
        ax_overview = build_panel_row(
            fig, [gs[row, 0], gs[row, 1], gs[row, 2]],
            cfg, plots_by_field[cfg.field], selected_by_field[cfg.field],
            cfg.ortho_path, show_shapefile_outline,
        )
        overview_axes.append(ax_overview)

    if show_north_arrow:
        # Single north arrow shared by both panel rows, centered in the gap
        # between them rather than repeated on each overview panel. Placed
        # over the overview column (not the inset columns to its right).
        top_row_bottom = overview_axes[0].get_position().y0
        bottom_row_top = overview_axes[1].get_position().y1
        mid_y = (top_row_bottom + bottom_row_top) / 2
        overview_pos = overview_axes[0].get_position()
        mid_x = overview_pos.x0 + -0.1 * (overview_pos.x1 - overview_pos.x0)
        add_north_arrow(
            overview_axes[0], xy=(mid_x, mid_y), xycoords="figure fraction"
        )
    return fig


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Figure S5 -- within-flight condition variability (illumination, weeds).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lighting-ortho", default=str(LIGHTING_ORTHO_DEFAULT))
    parser.add_argument("--weed-ortho", default=str(WEED_ORTHO_DEFAULT))
    parser.add_argument("--shapefile-dir", default=str(SHAPEFILE_DIR_DEFAULT))
    parser.add_argument("--covariates", default=str(COVARIATES_DEFAULT))
    parser.add_argument("--out_dir", default=str(OUTPUT_DIR))
    parser.add_argument(
        "--show-shapefile-outline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw the shapefile plot-boundary outlines on top of the "
             "orthomosaic overview panel.",
    )
    parser.add_argument(
        "--show-north-arrow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw the shared north arrow between the two panel rows.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("FIGURE S5 -- WITHIN-FLIGHT CONDITION VARIABILITY")
    print("=" * 60)

    covariates = pd.read_csv(args.covariates)
    panel_configs = build_panel_configs(args)

    plots_by_field: dict[str, gpd.GeoDataFrame] = {}
    selected_by_field: dict[str, pd.DataFrame] = {}
    for cfg in panel_configs:
        print(f"\n[{cfg.field}] {cfg.ortho_path}")
        with rasterio.open(cfg.ortho_path) as src:
            raster_crs = src.crs
        plots = load_plot_shapefile(Path(args.shapefile_dir), cfg.field, raster_crs)
        selected = select_representative_plots(covariates, cfg)
        print(selected[["plot", cfg.covariate, "percentile"]].to_string(index=False))
        plots_by_field[cfg.field] = plots
        selected_by_field[cfg.field] = selected

    print("\nBuilding figure ...")
    fig = make_figure(
        panel_configs, plots_by_field, selected_by_field,
        show_shapefile_outline=args.show_shapefile_outline,
        show_north_arrow=args.show_north_arrow,
    )

    out_dir = Path(args.out_dir)
    save_fig(fig, out_dir, "figS5_condition_variability")

    all_selected = pd.concat(selected_by_field.values(), ignore_index=True)
    sel_cols = ["field", "plot", "covariate", "percentile", "frac_weed", "mean_brightness"]
    sel_cols = [c for c in sel_cols if c in all_selected.columns]
    out_dir.mkdir(parents=True, exist_ok=True)
    all_selected[sel_cols].to_csv(
        out_dir / "figS5_condition_variability_selected_plots.csv", index=False
    )
    print(f"\n[selected plots] saved -> {out_dir / 'figS5_condition_variability_selected_plots.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
