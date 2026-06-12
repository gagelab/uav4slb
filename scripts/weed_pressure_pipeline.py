#!/usr/bin/env python3
"""
weed_pressure_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Compute per-image weed-zone pixel fraction (``frac_weed``) for every RGB UAV
flight across all six field trials (G3, I3B, B7A, C6B, C10, C7B) in the
2023–2025 growing seasons.

Scientific motivation
─────────────────────
Weed pressure is an image-level noise source for deep-learning SLB severity
prediction.  Weed canopy within a plot boundary displaces maize canopy from
the image tile, reducing the visible lesion area available to the model and
introducing green-tissue signal that is spectrally similar to healthy maize.
Quantifying weed pressure per image allows its correlation with model
residuals to be evaluated in the error analysis (Section 3.2).

Method
──────
A Canopy Height Model (CHM = DSM − reference DTM) is computed for each
flight.  Each pixel is classified into one of three height classes:

    ground : CHM  <  ground_max_ft   (~0.15 m; standing residue / bare soil)
    weed   : ground_max_ft ≤ CHM ≤ weed_max_ft  (~0.15–0.75 m)
    maize  : CHM  >  weed_max_ft     (~0.75 m; expected maize canopy at V8+)

Per-plot, the weed pixel fraction is:

    frac_weed = px_weed / px_valid

where px_valid excludes nodata pixels (outside the plot boundary or masked
by the DTM).

DTM reference strategy
──────────────────────
2024 / 2025
    The earliest flight DTM for a given field-year is used as the bare-ground
    reference for all later flights in the same field-year.  Early-season
    (May/June) flights precede canopy closure, so the DTM accurately
    represents bare ground.

        CHM = DSM_flight_i − DTM_earliest_flight

2023
    The earliest 2023 flights (July 13/20) already had full maize canopy,
    making a cross-flight CHM biologically invalid.  Each flight therefore
    uses its own Metashape-derived DTM.

        CHM = DSM_i − DTM_i   (relative surface; within-date comparison only)

CRS and units
─────────────
All rasters are in EPSG:2264 (NAD83 / North Carolina ftUS).  Metashape
exports elevation in US survey feet for this CRS.  All height thresholds
are therefore specified in feet:
    0.492 ft ≈ 0.15 m  (ground_max_ft default)
    2.461 ft ≈ 0.75 m  (weed_max_ft default)

Image filename convention
─────────────────────────
Plot images follow the naming convention:

    {YYYYMMDD}_{FIELD}_{PLOT:05d}.jpg

This pipeline constructs the image_filename key directly from the flight
date, field name, and integer plot ID — no external lookup table required.

Expected directory structure
────────────────────────────
DSMs and DTMs:
    <base_dir>/
        {YEAR}/
            {FIELD}/
                {FLIGHT_FOLDER}/      # named DJI_YYYYMMDD…
                    *_dsm.tif
                    *_dtm.tif

Plot boundaries:
    <shapefile_dir>/
        {FIELD}_plot_outline.shp      # or any .shp containing the field name

Output
──────
    <output_dir>/
        frac_weed.csv        one row per image; columns: image_filename, frac_weed

    The intermediate per-flight rasters (DSM, DTM) are NOT distributed with
    this repository.  frac_weed.csv is archived in data/covariates/ and is
    the direct input to build_image_covariates.py.

Usage
─────
# Process all fields using the YAML config:
python scripts/weed_pressure_pipeline.py --config configs/weed_pressure_config.yaml

# Override paths at the CLI for a single field:
python scripts/weed_pressure_pipeline.py \\
    --config        configs/weed_pressure_config.yaml \\
    --base-dir      /path/to/processed \\
    --shapefile-dir /path/to/shapefiles \\
    --output-dir    data/covariates \\
    --field         C7B

Dependencies
────────────
    conda install -c conda-forge rasterio geopandas rasterstats numpy pandas pyyaml

Author : Gage Lab — Cole Hammett
"""

import logging
import os
import re
import sys
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterstats import zonal_stats

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    # Required paths — must be set via config.yaml or CLI ─────────────────────
    "base_dir":       None,   # root of DSM/DTM directory tree
    "shapefile_dir":  None,   # directory of plot boundary shapefiles

    # Output ──────────────────────────────────────────────────────────────────
    "output_dir":     "data/covariates",

    # Shapefile attribute that uniquely identifies each plot ──────────────────
    # Run `ogrinfo -al -so your_shapefile.shp` to see available field names.
    "plot_id_field":  "plot",

    # Height classification thresholds (US survey FEET, EPSG:2264) ───────────
    #   Pixels are assigned to the first matching class:
    #     ground : CHM  < ground_max_ft   (≈ 0.15 m)
    #     weed   : ground_max_ft ≤ CHM ≤ weed_max_ft   (≈ 0.15–0.75 m)
    #     maize  : CHM  > weed_max_ft     (≈ 0.75 m)
    "ground_max_ft":  0.492,
    "weed_max_ft":    2.461,

    # Years for which each flight uses its own DTM (see DTM strategy above) ──
    "no_cross_flight_years": [2023],

    # Optional: restrict processing to one field (case-insensitive) ───────────
    "field_filter": None,

    # Target CRS — vectors and rasters are reprojected to this if needed ──────
    "target_epsg": 2264,
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — FILE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

# Regex patterns for parsing DJI flight folder names and processing timestamps.
# Flight folders are named:  DJI_YYYYMMDD…
# Processing timestamps embedded in file names:  …_YYYYMMDDThhmm_…
_FLIGHT_DATE_RE = re.compile(r"DJI_(\d{8})")
_PROC_TS_RE     = re.compile(r"_(\d{8}T\d{4})_")


def _is_cc(path: Path) -> bool:
    """Return True if '_CC_' appears in the path name or its parent folder.

    CC-tagged folders are experimental comparison runs that should not be
    included in the published analysis.
    """
    return "_CC_" in path.name or "_CC_" in str(path.parent.name)


def _parse_flight_date(folder_name: str) -> Optional[date]:
    """Extract the flight date from a DJI folder name (DJI_YYYYMMDD…)."""
    m = _FLIGHT_DATE_RE.search(folder_name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d").date()
        except ValueError:
            return None
    return None


def _parse_proc_timestamp(filename: str) -> Optional[datetime]:
    """Extract the Metashape processing timestamp from a raster filename."""
    m = _PROC_TS_RE.search(filename)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%dT%H%M")
        except ValueError:
            return None
    return None


def _select_latest(candidates: list) -> Optional[Path]:
    """Return the raster with the most recent Metashape processing timestamp.

    Falls back to file modification time when no embedded timestamp is found.
    CC-tagged files are excluded.
    """
    valid = [p for p in candidates if not _is_cc(p)]
    if not valid:
        return None

    def _sort_key(p: Path):
        ts = _parse_proc_timestamp(p.name)
        return ts if ts is not None else datetime.fromtimestamp(p.stat().st_mtime)

    return max(valid, key=_sort_key)


def _find_rasters(flight_dir: Path) -> dict:
    """Return the latest-timestamp DSM and DTM within a flight folder.

    Excludes .aux.xml sidecars, orthomosaic files (_ortho_*.tif), and
    CC-tagged files.
    """
    tifs = [
        f for f in flight_dir.glob("*.tif")
        if not f.name.endswith(".aux.xml")
        and "_ortho_" not in f.name
        and not _is_cc(f)
    ]
    return {
        "dsm": _select_latest([f for f in tifs if f.name.endswith("_dsm.tif")]),
        "dtm": _select_latest([f for f in tifs if f.name.endswith("_dtm.tif")]),
    }


def discover_flights(base_dir: Path, field_filter: Optional[str] = None) -> pd.DataFrame:
    """Walk base_dir/{YEAR}/{FIELD}/{FLIGHT_FOLDER}/ and return one row per
    valid flight.

    A flight is valid if:
      - The folder name starts with 'DJI_' (not a CC folder, not multispectral)
      - At least one DSM or DTM is found inside

    Parameters
    ----------
    base_dir : Path
        Root directory containing year subdirectories (2023/, 2024/, 2025/).
    field_filter : str, optional
        If provided, only folders matching this field name (case-insensitive)
        are processed.

    Returns
    -------
    pd.DataFrame
        Columns: year, field, flight_folder, flight_dir, flight_date,
                 dsm_path, dtm_path, ref_dtm_path (None), dtm_strategy (None).
    """
    records = []

    for year_dir in sorted(base_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        try:
            year = int(year_dir.name)
        except ValueError:
            continue

        for field_dir in sorted(year_dir.iterdir()):
            if not field_dir.is_dir():
                continue
            field = field_dir.name

            # Apply optional field filter (case-insensitive)
            if field_filter and field.upper() != field_filter.upper():
                log.debug("Field filter active — skipping %s", field)
                continue

            for flight_dir in sorted(field_dir.iterdir()):
                if not flight_dir.is_dir():
                    continue
                # Skip non-DJI folders, CC runs, and multispectral flights.
                # Multispectral flights lack RTK positioning and have
                # insufficient spatial accuracy for cross-flight plot alignment.
                if not flight_dir.name.startswith("DJI_"):
                    continue
                if _is_cc(flight_dir):
                    log.debug("Skipping CC folder: %s", flight_dir.name)
                    continue
                if "_multispec" in flight_dir.name.lower():
                    log.debug("Skipping multispectral: %s", flight_dir.name)
                    continue

                flight_date = _parse_flight_date(flight_dir.name)
                if flight_date is None:
                    log.warning("Cannot parse flight date: %s", flight_dir.name)
                    continue

                rasters = _find_rasters(flight_dir)
                if rasters["dsm"] is None and rasters["dtm"] is None:
                    log.debug("No rasters in %s — skipping", flight_dir.name)
                    continue

                records.append({
                    "year":          year,
                    "field":         field,
                    "flight_folder": flight_dir.name,
                    "flight_dir":    flight_dir,
                    "flight_date":   flight_date,
                    "dsm_path":      rasters["dsm"],
                    "dtm_path":      rasters["dtm"],
                    # Filled by assign_dtm_strategy below
                    "ref_dtm_path":  None,
                    "dtm_strategy":  None,
                })

    df = pd.DataFrame(records)
    if df.empty:
        log.error("No flights discovered under %s", base_dir)
    else:
        log.info(
            "Discovered %d flights across %d field-year(s)",
            len(df), df.groupby(["year", "field"]).ngroups,
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DTM REFERENCE STRATEGY
# ─────────────────────────────────────────────────────────────────────────────

def assign_dtm_strategy(inventory: pd.DataFrame,
                        no_cross_flight_years: list) -> pd.DataFrame:
    """Assign dtm_strategy and ref_dtm_path for every flight.

    Strategy labels:
      'own'           — flight uses its own Metashape DTM (2023 flights)
      'own_fallback'  — no valid earliest DTM found; fell back to own DTM
      'earliest_self' — this IS the earliest flight; its DTM is the reference
      'earliest'      — using the earliest flight's DTM as bare-ground reference
    """
    inventory = inventory.copy()

    for (year, field), group in inventory.groupby(["year", "field"]):

        # 2023: canopy was present at earliest flights; use own-DTM strategy
        if int(year) in no_cross_flight_years:
            for i in group.index:
                inventory.at[i, "dtm_strategy"] = "own"
                inventory.at[i, "ref_dtm_path"] = inventory.at[i, "dtm_path"]
            continue

        # 2024/2025: use earliest-flight DTM as bare-ground reference
        has_dtm = group[group["dtm_path"].notna()].sort_values("flight_date")
        if has_dtm.empty:
            log.warning(
                "No DTM found for %s %s — falling back to own DTM", field, year
            )
            for i in group.index:
                inventory.at[i, "dtm_strategy"] = "own_fallback"
                inventory.at[i, "ref_dtm_path"] = inventory.at[i, "dtm_path"]
            continue

        ref_dtm  = has_dtm.iloc[0]["dtm_path"]
        ref_date = has_dtm.iloc[0]["flight_date"]
        log.info(
            "Reference DTM for %s %s: %s (flight %s)",
            field, year, ref_dtm.name if ref_dtm else "None", ref_date,
        )

        for i in group.index:
            strategy = (
                "earliest_self"
                if inventory.at[i, "flight_date"] == ref_date
                else "earliest"
            )
            inventory.at[i, "dtm_strategy"] = strategy
            inventory.at[i, "ref_dtm_path"] = ref_dtm

    return inventory


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — RASTER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _read_single_band(path: Path) -> tuple:
    """Read a single-band raster as a float32 numpy array.

    NoData values are replaced with np.nan.

    Returns
    -------
    tuple : (array, transform, crs, nodata)
    """
    with rasterio.open(path) as src:
        arr    = src.read(1).astype(np.float32)
        trans  = src.transform
        crs    = src.crs
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr, trans, crs, nodata


def _reproject_to_match(src_path: Path, ref_path: Path) -> tuple:
    """Reproject a raster onto the exact pixel grid of a reference raster.

    Uses bilinear resampling (appropriate for continuous elevation data).
    NoData in the source is propagated as np.nan in the output.

    Parameters
    ----------
    src_path : Path  — raster to reproject
    ref_path : Path  — reference raster defining the target grid

    Returns
    -------
    tuple : (array float32, transform)
    """
    with rasterio.open(ref_path) as ref:
        ref_crs       = ref.crs
        ref_transform = ref.transform
        ref_width     = ref.width
        ref_height    = ref.height

    with rasterio.open(src_path) as src:
        dst = np.full((ref_height, ref_width), np.nan, dtype=np.float32)
        reproject(
            source        = rasterio.band(src, 1),
            destination   = dst,
            src_transform = src.transform,
            src_crs       = src.crs,
            dst_transform = ref_transform,
            dst_crs       = ref_crs,
            resampling    = Resampling.bilinear,
            src_nodata    = src.nodata,
            dst_nodata    = np.nan,
        )
    return dst, ref_transform


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CHM COMPUTATION AND HEIGHT CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_chm(dsm: np.ndarray, ref_dtm: np.ndarray) -> np.ndarray:
    """Compute the Canopy Height Model as CHM = DSM − reference_DTM.

    Negative values arising from DSM/DTM co-registration noise are floored to
    zero.  Both arrays must already share the same pixel grid (ensured by
    _reproject_to_match).  Units: US survey feet (EPSG:2264).
    """
    chm = dsm - ref_dtm
    chm[chm < 0] = 0.0
    return chm


def classify_chm(chm: np.ndarray,
                 ground_max_ft: float,
                 weed_max_ft: float) -> dict:
    """Return boolean masks for ground / weed / maize height classes.

    NaN pixels (outside the flight boundary or DTM nodata) are excluded from
    all three classes.

    Parameters
    ----------
    chm          : float32 array, CHM in US survey feet
    ground_max_ft: upper bound of ground class (default 0.492 ft ≈ 0.15 m)
    weed_max_ft  : upper bound of weed class   (default 2.461 ft ≈ 0.75 m)

    Returns
    -------
    dict with keys 'ground', 'weed', 'maize' → boolean numpy arrays
    """
    valid = ~np.isnan(chm)
    return {
        "ground": valid & (chm <  ground_max_ft),
        "weed":   valid & (chm >= ground_max_ft) & (chm <= weed_max_ft),
        "maize":  valid & (chm >  weed_max_ft),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — PER-PLOT ZONAL STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_plot_stats(plots:         gpd.GeoDataFrame,
                       chm:           np.ndarray,
                       transform,
                       plot_id_field: str,
                       cfg:           dict) -> pd.DataFrame:
    """Compute per-plot frac_weed for one flight date.

    Uses rasterstats.zonal_stats to count pixels of each height class within
    each plot polygon.

    Output columns
    ──────────────
    {plot_id_field} — plot identifier from the shapefile attribute
    frac_weed       — px_weed / px_valid (primary output used in analysis)

    Parameters
    ----------
    plots         : GeoDataFrame of plot polygons
    chm           : float32 CHM array aligned to the DTM grid
    transform     : rasterio Affine transform of the CHM
    plot_id_field : attribute column name for the plot identifier
    cfg           : pipeline config dict (ground_max_ft, weed_max_ft)
    """
    classes = classify_chm(chm, cfg["ground_max_ft"], cfg["weed_max_ft"])

    # Count weed-class pixels within each plot polygon.
    # The boolean mask is cast to uint8; nodata=255 ensures pixels outside any
    # polygon boundary are excluded from the sum.
    zs_weed = zonal_stats(
        plots, classes["weed"].astype(np.uint8),
        affine=transform, stats=["sum"], nodata=255,
    )
    px_weed = [r["sum"] if r["sum"] is not None else 0 for r in zs_weed]

    # Count all non-NaN CHM pixels within each plot (denominator for frac_weed)
    valid_mask = (~np.isnan(chm)).astype(np.uint8)
    zs_valid   = zonal_stats(
        plots, valid_mask, affine=transform, stats=["sum"], nodata=255,
    )
    px_valid = [r["sum"] if r["sum"] is not None else 0 for r in zs_valid]

    # Assemble result DataFrame; zero valid pixels → NaN (avoids 0/0)
    result = pd.DataFrame({
        plot_id_field: plots[plot_id_field].values,
        "px_weed":     px_weed,
        "px_valid":    px_valid,
    })
    result["frac_weed"] = (
        result["px_weed"] / result["px_valid"].replace(0, np.nan)
    )

    return result[[plot_id_field, "frac_weed"]]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — IMAGE FILENAME CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_image_filename(flight_date: date, field: str, plot_id) -> str:
    """Construct the canonical image filename from flight metadata.

    Image filenames follow the convention used throughout the repository:

        {YYYYMMDD}_{FIELD}_{PLOT:05d}.jpg

    Parameters
    ----------
    flight_date : datetime.date  — date of the UAV flight
    field       : str            — field identifier (e.g. 'B7A')
    plot_id     : int or str     — plot number (zero-padded to 5 digits)

    Returns
    -------
    str, e.g. '20240703_B7A_05512.jpg'
    """
    date_str = flight_date.strftime("%Y%m%d")
    plot_str = f"{int(plot_id):05d}"
    return f"{date_str}_{field}_{plot_str}.jpg"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — PER-FIELD-YEAR PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_field_year(field:          str,
                       year:           int,
                       flights:        pd.DataFrame,
                       shapefile_dir:  Path,
                       cfg:            dict) -> pd.DataFrame:
    """Run the weed fraction pipeline for one field-year.

    For each flight date in chronological order:
      1. Load DSM and reference DTM; reproject DSM onto DTM grid if necessary.
      2. Compute CHM and classify pixels into ground / weed / maize.
      3. Run zonal statistics against the plot shapefile.
      4. Construct the image_filename key for every plot.

    Returns
    -------
    pd.DataFrame
        Columns: image_filename, frac_weed.
        One row per plot image processed across all flights in this field-year.
    """
    log.info("=" * 60)
    log.info(
        "Processing  field=%-8s  year=%d  (%d flights)", field, year, len(flights)
    )

    # ── Locate plot boundary shapefile ────────────────────────────────────────
    # Primary pattern: {FIELD}_plot_outline.shp
    # Fallback: any .shp whose stem contains the field name
    shp_candidates = list(shapefile_dir.glob(f"{field}_plot_outline.shp"))
    if not shp_candidates:
        shp_candidates = [
            p for p in shapefile_dir.glob("*.shp")
            if field.lower() in p.stem.lower()
        ]
    if not shp_candidates:
        log.error(
            "No shapefile found for %s %d in %s — skipping", field, year, shapefile_dir
        )
        return pd.DataFrame(columns=["image_filename", "frac_weed"])

    plot_id_field = cfg["plot_id_field"]
    plots = gpd.read_file(shp_candidates[0])

    if plot_id_field not in plots.columns:
        log.error(
            "Column '%s' not found in %s.  Available: %s",
            plot_id_field, shp_candidates[0].name, list(plots.columns),
        )
        log.error("Set plot_id_field in config.yaml to the correct column name.")
        return pd.DataFrame(columns=["image_filename", "frac_weed"])

    # Reproject plot polygons to the target CRS if needed
    if cfg.get("target_epsg"):
        plots = plots.to_crs(f"EPSG:{cfg['target_epsg']}")

    log.info("Loaded %d plots from %s", len(plots), shp_candidates[0].name)

    # ── Iterate flights in chronological order ────────────────────────────────
    all_rows = []  # list of DataFrames, one per flight

    for _, row in flights.sort_values("flight_date").iterrows():
        fdate    = row["flight_date"]
        dsm_path = row["dsm_path"]
        ref_dtm  = row["ref_dtm_path"]
        strategy = row["dtm_strategy"]

        log.info("  ── Flight %s  strategy=%s", fdate, strategy)

        if dsm_path is None:
            log.warning("     No DSM found — skipping flight %s", fdate)
            continue
        if ref_dtm is None:
            log.warning("     No reference DTM — skipping flight %s", fdate)
            continue

        # ── Load rasters ──────────────────────────────────────────────────────
        try:
            _, _, _, _  = _read_single_band(ref_dtm)  # validate file is readable
            dtm_arr, _, _, _ = _read_single_band(ref_dtm)

            # Re-open reference DTM to get its authoritative Affine transform
            with rasterio.open(ref_dtm) as src:
                dtm_transform = src.transform

            # If DSM and reference DTM are the same file (own-DTM strategy),
            # avoid re-reading; otherwise reproject DSM onto the DTM grid.
            if str(dsm_path) == str(ref_dtm):
                dsm_arr = dtm_arr.copy()
            else:
                dsm_arr, _ = _reproject_to_match(dsm_path, ref_dtm)

        except Exception as exc:
            log.error("     Raster load error: %s", exc)
            continue

        # ── CHM and zonal statistics ──────────────────────────────────────────
        try:
            chm      = compute_chm(dsm_arr, dtm_arr)
            stats_df = compute_plot_stats(
                plots         = plots,
                chm           = chm,
                transform     = dtm_transform,
                plot_id_field = plot_id_field,
                cfg           = cfg,
            )
            log.info("     Zonal stats OK  (%d plots)", len(stats_df))

        except Exception as exc:
            log.error("     Zonal stats failed: %s", exc)
            continue

        # ── Construct image_filename for each plot in this flight ─────────────
        # The filename encodes flight date, field, and plot number, so it is
        # fully determined without any external lookup table.
        stats_df["image_filename"] = stats_df[plot_id_field].apply(
            lambda pid: build_image_filename(fdate, field, pid)
        )

        all_rows.append(stats_df[["image_filename", "frac_weed"]])

    if not all_rows:
        log.warning("No stats produced for %s %d", field, year)
        return pd.DataFrame(columns=["image_filename", "frac_weed"])

    return pd.concat(all_rows, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(cfg: dict) -> None:
    """Run the full weed pressure pipeline for all discovered field-years.

    Steps
    ─────
    1. Discover flights from the base_dir directory tree.
    2. Assign the DTM reference strategy per field-year.
    3. Process each field-year; accumulate (image_filename, frac_weed) rows.
    4. Write the single consolidated frac_weed.csv to output_dir.
    """
    base_dir      = Path(cfg["base_dir"])
    shapefile_dir = Path(cfg["shapefile_dir"])
    output_dir    = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for p, label in [
        (base_dir,      "base_dir"),
        (shapefile_dir, "shapefile_dir"),
    ]:
        if not p.exists():
            log.error("Path does not exist: %s = %s", label, p)
            sys.exit(1)

    # ── Step 1–2: discover flights and assign DTM strategy ────────────────────
    inventory = discover_flights(base_dir, field_filter=cfg.get("field_filter"))
    if inventory.empty:
        sys.exit(1)

    inventory = assign_dtm_strategy(inventory, cfg["no_cross_flight_years"])

    # ── Step 3: process each field-year ───────────────────────────────────────
    all_results = []

    for (year, field), group in inventory.groupby(["year", "field"]):
        result_df = process_field_year(
            field         = field,
            year          = int(year),
            flights       = group,
            shapefile_dir = shapefile_dir,
            cfg           = cfg,
        )
        if not result_df.empty:
            all_results.append(result_df)

    # ── Step 4: write frac_weed.csv ───────────────────────────────────────────
    if not all_results:
        log.error("No results produced — frac_weed.csv not written.")
        sys.exit(1)

    frac_weed = pd.concat(all_results, ignore_index=True)

    # Verify there are no duplicate image_filename entries.  Duplicates would
    # indicate a flight was matched to the same scored image twice, which should
    # not occur under normal operation.
    n_dupes = frac_weed["image_filename"].duplicated().sum()
    if n_dupes:
        log.warning(
            "%d duplicate image_filename entries found — check inventory for "
            "overlapping flight dates or field filter settings.", n_dupes
        )

    out_path = output_dir / "frac_weed.csv"
    frac_weed[["image_filename", "frac_weed"]].to_csv(out_path, index=False)
    log.info(
        "Wrote %s  (%d images across %d field-years)",
        out_path, len(frac_weed), len(all_results),
    )

    log.info("Pipeline complete.  Output: %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — CLI
# ─────────────────────────────────────────────────────────────────────────────

def _load_config(args) -> dict:
    """Merge DEFAULT_CONFIG → YAML file → CLI overrides (highest precedence)."""
    cfg = DEFAULT_CONFIG.copy()

    if getattr(args, "config", None):
        import yaml
        with open(args.config) as f:
            yaml_cfg = yaml.safe_load(f) or {}
        # Only update keys that are explicitly set in the YAML
        cfg.update({k: v for k, v in yaml_cfg.items() if v is not None})

    # CLI flags override everything
    cli_overrides = {
        "base_dir":       getattr(args, "base_dir",      None),
        "shapefile_dir":  getattr(args, "shapefile_dir", None),
        "output_dir":     getattr(args, "output_dir",    None),
        "plot_id_field":  getattr(args, "plot_id_field", None),
        "field_filter":   getattr(args, "field",         None),
    }
    cfg.update({k: v for k, v in cli_overrides.items() if v is not None})

    missing = [k for k in ("base_dir", "shapefile_dir") if not cfg.get(k)]
    if missing:
        log.error("Required settings not provided: %s", missing)
        log.error("Supply via --config config.yaml or the corresponding CLI flags.")
        sys.exit(1)

    return cfg


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="UAV weed pressure pipeline — outputs frac_weed.csv "
                    "(image_filename, frac_weed) for all field-years.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", metavar="FILE",
        help="YAML config file (configs/weed_pressure_config.yaml). "
             "CLI flags override values in the config.",
    )
    p.add_argument(
        "--base-dir", metavar="DIR",
        help="Root data directory containing {YEAR}/{FIELD}/{FLIGHT}/ tree.",
    )
    p.add_argument(
        "--shapefile-dir", metavar="DIR",
        help="Directory containing {FIELD}_plot_outline.shp files.",
    )
    p.add_argument(
        "--output-dir", metavar="DIR",
        help="Output directory for frac_weed.csv (default: data/covariates).",
    )
    p.add_argument(
        "--plot-id-field", metavar="COLUMN",
        help="Shapefile attribute for plot ID (default: plot). "
             "Run `ogrinfo -al -so <shapefile>` to list available columns.",
    )
    p.add_argument(
        "--field", metavar="NAME",
        help="Process only this field (e.g. C7B). Case-insensitive.",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = p.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = _load_config(args)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
