"""
scripts/build_flight_covariates.py
===================================
Compute per-flight covariates from raw data sources and write the joined
table to ``data/covariates/flight_covariates.csv``.

This script is the single authoritative ETL step that sits between the raw
input files (committed to the repository) and the figure/analysis scripts
that consume ``flight_covariates.csv``. 

Raw inputs (relative to repo root)
------------------------------------
  data/covariates/raw/flight_times.csv
      One row per flight.  Columns: flight_key (YYYYMMDD_FIELD),
      flight_start (UTC-local datetime), flight_end (UTC-local datetime),
      time_from_solar_noon (signed integer minutes, pre-computed).

  data/covariates/raw/irradiance_minute.csv
      One-minute EcoNet downwelling shortwave radiation observations spanning
      all three field seasons.  Columns: datetime, radiation(W/m2),
      par(micromol/m2s).

  data/covariates/raw/gdd.csv
      Daily growing-degree-day increments (base 50 °F) from the EcoNet
      station.  Columns: date (YYYY-MM-DD), gdd (°F·day).  GDD is not
      cumulative in this file; AGDD is computed here by summing from the
      planting date of each field through the flight date.

  data/covariates/raw/planting_dates.csv
      Planting date for each field used as the AGDD reference origin.
      Columns: field (str), planting_date (YYYY-MM-DD).

  data/covariates/raw/inoculation_dates.csv
      Inoculation date for each growing season used as the DPI reference origin.
      Columns: year (int), inoculation_date (YYYY-MM-DD).

External dependency (not committed)
-------------------------------------
  Shapefile at <shapefile_dir>/  — plot boundary outlines used to derive
  per-field row azimuth, which feeds the shadow–row angle calculation.
  Default path: final_image/final_outline  (relative to the HPC data root).
  Pass --shapefile_dir to override.

Outputs
--------
  data/covariates/flight_covariates.csv  (default; override with --out)

      flight_key             : YYYYMMDD_FIELD  (join key for figure scripts)
      flight_date            : YYYY-MM-DD
      field                  : field identifier (e.g., G3, I3B)
      year                   : integer
      flight_start           : datetime string
      flight_end             : datetime string

      -- Solar geometry (pvlib, evaluated at flight midpoint) --
      sza                    : solar zenith angle (°)
      sun_azimuth            : sun azimuth (°, from N clockwise)
      abs_time_from_noon     : |minutes from solar noon| at flight midpoint

      -- Irradiance (EcoNet one-minute data, flight window) --
      irradiance_mean        : mean downwelling shortwave (W m⁻²)
      irradiance_cv          : coefficient of variation (σ/μ)
      irradiance_n           : number of valid one-minute observations

      -- Temporal phenology --
      dpi                    : days post-inoculation (flight date − inoculation date for that year)
      agdd                   : accumulated GDD base-50 °F (planting → flight date)

      -- Shadow geometry (requires shapefile) --
      row_azimuth            : mean plot row azimuth (°, 0–180), NaN if shapefile absent
      shadow_row_angle       : acute angle between shadow direction and plot long axis (°,
                               0–90); 0° = shadow runs along row length, 90° = shadow
                               falls across into the adjacent plot

Usage
------
  # From repo root (all defaults):
  python scripts/build_flight_covariates.py

  # Override paths:
  python scripts/build_flight_covariates.py \\
      --flight_times      data/covariates/raw/flight_times.csv \\
      --irradiance        data/covariates/raw/irradiance_minute.csv \\
      --gdd               data/covariates/raw/gdd.csv \\
      --planting_dates    data/covariates/raw/planting_dates.csv \\
      --inoculation_dates data/covariates/raw/inoculation_dates.csv \\
      --shapefile_dir     /path/to/final_outline \\
      --out               data/covariates/flight_covariates.csv

Dependencies
-------------
  pandas, numpy, pvlib, geopandas (optional — only needed for shadow_row_angle)
  Install: conda activate uav_for_slb
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Site configuration
# ---------------------------------------------------------------------------

# Geographic centroid of the NCSU farm fields used across all three seasons.
# pvlib uses this to compute solar position; sub-field differences in lat/lon
# are <2 km and contribute <0.1° to zenith angle at mid-day.
SITE_LAT  = 35.77   # °N
SITE_LON  = -78.67  # °W
SITE_ALT  = 100     # metres above sea level
SITE_TZ   = "America/New_York"

# Crop row orientation is derived from the plot shapefile at runtime.
# This fallback is used only when the shapefile is unavailable, and the
# resulting shadow_row_angle will be NaN for those flights.
ROW_AZIMUTH_FALLBACK = np.nan

# Minimum number of valid irradiance observations required to compute
# irradiance_mean and irradiance_cv.  Flights with fewer observations are
# assigned NaN for both columns.
MIN_IRRADIANCE_OBS = 5

# ---------------------------------------------------------------------------
# Repository path defaults
# ---------------------------------------------------------------------------

# Resolve paths relative to this script's location (repo_root/scripts/)
_REPO_ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_FLIGHT_TIMES   = _REPO_ROOT / "data" / "covariates" / "raw" / "flight_times.csv"
_DEFAULT_IRRADIANCE     = _REPO_ROOT / "data" / "covariates" / "raw" / "irradiance_minute.csv"
_DEFAULT_GDD            = _REPO_ROOT / "data" / "covariates" / "raw" / "gdd.csv"
_DEFAULT_PLANTING_DATES     = _REPO_ROOT / "data" / "covariates" / "raw" / "planting_dates.csv"
_DEFAULT_INOCULATION_DATES  = _REPO_ROOT / "data" / "covariates" / "raw" / "inoculation_dates.csv"
_DEFAULT_SHAPEFILE_DIR  = Path("/mnt/research-projects/j/jlgage/RawUAVData01"
                               "/uavforslb/final_image/final_outline")
_DEFAULT_OUT            = _REPO_ROOT / "data" / "covariates" / "flight_covariates.csv"


# ===========================================================================
# Section 1 — Load and validate raw inputs
# ===========================================================================

def load_flight_times(path: Path) -> pd.DataFrame:
    """
    Load flight_times.csv and parse datetime columns.

    Returns a DataFrame with columns:
        flight_key, flight_start, flight_end, flight_date (date), field (str), year (int)

    Note: abs_time_from_noon is NOT taken from the pre-computed
    time_from_solar_noon column.  It is calculated from scratch in
    compute_solar_geometry() using pvlib at the exact flight midpoint.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Parse datetime strings — flight times are local Eastern time
    df["flight_start"] = pd.to_datetime(df["flight_start"])
    df["flight_end"]   = pd.to_datetime(df["flight_end"])

    # Localize to Eastern time so pvlib solar position is computed correctly
    df["flight_start"] = df["flight_start"].dt.tz_localize(SITE_TZ, ambiguous="NaT",
                                                            nonexistent="NaT")
    df["flight_end"]   = df["flight_end"].dt.tz_localize(SITE_TZ, ambiguous="NaT",
                                                          nonexistent="NaT")

    # Derived columns from flight_key (format: YYYYMMDD_FIELD)
    df["flight_date"] = pd.to_datetime(df["flight_key"].str[:8], format="%Y%m%d").dt.date
    df["field"]       = df["flight_key"].str.split("_").str[1].str.upper()
    df["year"]        = df["flight_key"].str[:4].astype(int)

    print(f"[flight_times]  {len(df)} flights loaded from {path.name}")
    print(f"                years: {sorted(df['year'].unique())}  "
          f"  fields: {sorted(df['field'].unique())}")
    return df


def load_irradiance(path: Path) -> pd.DataFrame:
    """
    Load irradiance_minute.csv and coerce the radiation column to numeric.

    EcoNet exports occasionally include non-numeric flags (e.g., '-9999',
    'M') for missing observations; these become NaN after coercion.

    Returns a DataFrame with columns: datetime (tz-aware Eastern), radiation
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    df["datetime"]  = pd.to_datetime(df["datetime"], format="mixed")
    df["datetime"]  = df["datetime"].dt.tz_localize(SITE_TZ, ambiguous="NaT",
                                                     nonexistent="NaT")
    df["radiation"] = pd.to_numeric(df["radiation(W/m2)"], errors="coerce")

    n_nan = df["radiation"].isna().sum()
    if n_nan > 0:
        print(f"[irradiance]    {n_nan} NaN observations after coercion "
              f"(non-numeric / flagged values) — excluded from flight summaries")

    print(f"[irradiance]    {len(df)} one-minute rows  "
          f"  range: {df['datetime'].min().date()} → {df['datetime'].max().date()}")
    return df[["datetime", "radiation"]].copy()


def load_gdd(path: Path) -> pd.DataFrame:
    """
    Load gdd.csv containing daily GDD increments (base 50 °F).

    The file stores *daily increments*, not a running cumulative sum.
    AGDD from any reference date is computed downstream by summing the
    increments over the relevant date window.

    Returns a DataFrame with columns: date (datetime64), gdd (float), year (int)
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df = df.sort_values("date").reset_index(drop=True)

    print(f"[gdd]           {len(df)} daily rows  "
          f"  range: {df['date'].min().date()} → {df['date'].max().date()}")
    return df


def load_planting_dates(path: Path) -> pd.DataFrame:
    """
    Load planting_dates.csv.

    Returns a DataFrame with columns: field (str), planting_date (datetime64)
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["planting_date"] = pd.to_datetime(df["planting_date"])
    df["field"]         = df["field"].str.upper()

    print(f"[planting_dates] {len(df)} fields: "
          + "  ".join(f"{r['field']}→{r['planting_date'].date()}"
                      for _, r in df.iterrows()))
    return df


def load_inoculation_dates(path: Path) -> pd.DataFrame:
    """
    Load inoculation_dates.csv.

    Returns a DataFrame with columns: year (int), inoculation_date (datetime64)
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["inoculation_date"] = pd.to_datetime(df["inoculation_date"])
    df["year"]             = df["year"].astype(int)

    print(f"[inoculation_dates] {len(df)} seasons: "
          + "  ".join(f"{r['year']}→{r['inoculation_date'].date()}"
                      for _, r in df.iterrows()))
    return df


# ===========================================================================
# Section 2 — Solar geometry (pvlib)
# ===========================================================================

def compute_solar_geometry(
    flight_df:    pd.DataFrame,
    field_coords: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """
    Compute solar zenith angle, azimuth, and minutes from solar noon for
    each flight, evaluated with pvlib at the exact flight midpoint.

    Using the midpoint rather than a window average is consistent with how
    the pre-computed time_from_solar_noon values in flight_times.csv were
    originally derived, and avoids artificial smoothing of the azimuth
    for asymmetric flight windows.

    Solar noon is located by finding the timestamp of maximum solar elevation
    across a one-minute grid of the full flight day, then computing the signed
    minute offset of the midpoint from that peak.

    Parameters
    ----------
    flight_df    : output of load_flight_times()
    field_coords : optional dict mapping field name (e.g. 'G3') to
                   (latitude, longitude) in decimal degrees.  When provided,
                   per-field coordinates are used instead of the single site
                   centroid, improving accuracy for shadow_row_angle.
                   Populated by _load_and_orient_shapefile() when called first.

    Returns
    -------
    DataFrame with columns: flight_key, sza, sun_azimuth, abs_time_from_noon
    """
    rows = []
    for _, row in flight_df.iterrows():
        field = row["field"]

        # Use per-field centroid coordinates when available; fall back to
        # the site-wide centroid otherwise.  The fields span <2 km so the
        # fallback introduces <0.1° error in SZA.
        if field_coords and field in field_coords:
            lat, lon = field_coords[field]
        else:
            lat, lon = SITE_LAT, SITE_LON

        loc = pvlib.location.Location(
            latitude=lat, longitude=lon, tz=SITE_TZ, altitude=SITE_ALT
        )

        # Flight midpoint — the single representative timestamp
        midpoint = row["flight_start"] + (row["flight_end"] - row["flight_start"]) / 2

        if pd.isna(midpoint):
            rows.append({"flight_key": row["flight_key"],
                         "sza": np.nan, "sun_azimuth": np.nan,
                         "abs_time_from_noon": np.nan})
            continue

        # Solar position at midpoint
        sp_mid = loc.get_solarposition(pd.DatetimeIndex([midpoint]))
        sza      = float(sp_mid["apparent_zenith"].iloc[0])
        sun_az   = float(sp_mid["azimuth"].iloc[0])

        # Solar noon: timestamp of maximum elevation over the full flight day.
        # Evaluated on a one-minute grid so the precision matches the
        # integer-minute values stored in flight_times.csv.
        flight_date = midpoint.normalize()   # midnight of the flight day
        day_times   = pd.date_range(
            start=flight_date,
            end=flight_date + pd.Timedelta(hours=23, minutes=59),
            freq="1min",
            tz=SITE_TZ,
        )
        sp_day   = loc.get_solarposition(day_times)
        noon_ts  = sp_day["elevation"].idxmax()

        # Signed offset in whole minutes; absolute value stored as the covariate
        offset_min      = (midpoint - noon_ts).total_seconds() / 60.0
        abs_from_noon   = abs(offset_min)

        rows.append({
            "flight_key":        row["flight_key"],
            "sza":               sza,
            "sun_azimuth":       sun_az,
            "abs_time_from_noon": abs_from_noon,
        })

    out = pd.DataFrame(rows)
    print(f"[solar_geometry] computed at midpoint for {len(out)} flights  "
          f"  SZA range: {out['sza'].min():.1f}°–{out['sza'].max():.1f}°")
    return out


# ===========================================================================
# Section 3 — Irradiance summary statistics
# ===========================================================================

def compute_irradiance(flight_df: pd.DataFrame,
                       irr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise EcoNet one-minute irradiance observations within each flight window.

    For each flight, observations in [flight_start, flight_end] are extracted
    and summarised as:
        irradiance_mean : arithmetic mean (W m⁻²)
        irradiance_cv   : coefficient of variation (σ/μ); NaN when μ = 0
        irradiance_n    : number of valid (non-NaN) observations

    Flights with fewer than MIN_IRRADIANCE_OBS valid observations are assigned
    NaN for mean and CV, with a warning printed to stdout.

    Parameters
    ----------
    flight_df : output of load_flight_times()
    irr_df    : output of load_irradiance()

    Returns
    -------
    DataFrame with columns: flight_key, irradiance_mean, irradiance_cv, irradiance_n
    """
    rows = []
    for _, row in flight_df.iterrows():
        # Select one-minute observations within the flight window (inclusive)
        mask = (
            (irr_df["datetime"] >= row["flight_start"]) &
            (irr_df["datetime"] <= row["flight_end"])
        )
        sub = irr_df.loc[mask, "radiation"].dropna()
        n   = len(sub)

        if n < MIN_IRRADIANCE_OBS:
            print(f"  [irradiance] WARNING: {row['flight_key']} has only {n} valid "
                  f"observations (threshold={MIN_IRRADIANCE_OBS}) — NaN assigned")
            rows.append({
                "flight_key":      row["flight_key"],
                "irradiance_mean": np.nan,
                "irradiance_cv":   np.nan,
                "irradiance_n":    n,
            })
            continue

        mu = sub.mean()
        cv = sub.std(ddof=1) / mu if mu > 0 else np.nan

        rows.append({
            "flight_key":      row["flight_key"],
            "irradiance_mean": mu,
            "irradiance_cv":   cv,
            "irradiance_n":    n,
        })

    out = pd.DataFrame(rows)
    n_ok = (out["irradiance_n"] >= MIN_IRRADIANCE_OBS).sum()
    print(f"[irradiance]     {n_ok}/{len(out)} flights have sufficient observations")
    return out


# ===========================================================================
# Section 4 — Phenological indices (DPI and AGDD)
# ===========================================================================

def compute_phenology(flight_df: pd.DataFrame,
                      gdd_df: pd.DataFrame,
                      plant_df: pd.DataFrame,
                      inoc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute days post-inoculation (DPI) and accumulated GDD (AGDD) for each
    flight.

    DPI is the integer number of days between the inoculation date for that
    growing season and the flight date (DPI = 0 on the inoculation date itself).
    The inoculation date is looked up by year from inoc_df.

    AGDD is the sum of daily GDD increments (base 50 °F) from the planting
    date through the flight date, inclusive.  The GDD file provides daily
    increments; this function performs the cumulative sum over the relevant
    window.  The planting date (not the inoculation date) is the AGDD origin.

    Parameters
    ----------
    flight_df : output of load_flight_times()
    gdd_df    : output of load_gdd()
    plant_df  : output of load_planting_dates()
    inoc_df   : output of load_inoculation_dates()

    Returns
    -------
    DataFrame with columns: flight_key, dpi, agdd
    """
    # Build lookups
    plant_lookup = plant_df.set_index("field")["planting_date"].to_dict()
    inoc_lookup  = inoc_df.set_index("year")["inoculation_date"].to_dict()

    rows = []
    for _, row in flight_df.iterrows():
        field      = row["field"]
        year       = row["year"]
        fdate      = pd.Timestamp(row["flight_date"])   # date object → Timestamp
        plant_date = plant_lookup.get(field)
        inoc_date  = inoc_lookup.get(year)

        if inoc_date is None:
            print(f"  [phenology] WARNING: no inoculation date for year {year} — DPI NaN")
        if plant_date is None:
            print(f"  [phenology] WARNING: no planting date for field {field} — AGDD NaN")

        if inoc_date is None and plant_date is None:
            rows.append({"flight_key": row["flight_key"], "dpi": np.nan, "agdd": np.nan})
            continue

        # DPI — days from inoculation date to flight date (by year)
        dpi = (fdate - inoc_date).days if inoc_date is not None else np.nan

        # AGDD — sum daily GDD increments from planting date through flight date
        if plant_date is not None:
            mask = (gdd_df["date"] >= plant_date) & (gdd_df["date"] <= fdate)
            agdd = gdd_df.loc[mask, "gdd"].sum()
            if mask.sum() == 0:
                print(f"  [phenology] WARNING: no GDD rows found for "
                      f"{row['flight_key']} between {plant_date.date()} and {fdate.date()}")
                agdd = np.nan
        else:
            agdd = np.nan

        rows.append({"flight_key": row["flight_key"], "dpi": dpi, "agdd": agdd})

    out = pd.DataFrame(rows)
    print(f"[phenology]      DPI range: {out['dpi'].min():.0f}–{out['dpi'].max():.0f} days")
    print(f"                 AGDD range: {out['agdd'].min():.0f}–{out['agdd'].max():.0f} °F·day")
    return out


# ===========================================================================
# Section 5 — Shadow–row angle (geopandas, optional)
# ===========================================================================

# Target projected CRS for all shapefile geometry operations.
# EPSG:2264 = NAD83 / North Carolina State Plane (ft US) — the native CRS of
# the final_outline shapefiles.  Angles are computed in this CRS before
# converting centroids to WGS84.
_PROJ_EPSG = 2264


def _force_2d_crs(gdf, target_epsg: int = _PROJ_EPSG):
    """
    Reproject *gdf* to *target_epsg*, stripping any compound/vertical CRS
    component that prevents a standard to_crs() call.

    The final_outline shapefiles carry a compound vertical datum (NAD83 /
    NC State Plane ftUS + unknown vertical) that raises a pyproj error on
    direct reprojection.  Three fallback strategies are tried in order so
    that execution is never blocked by this metadata artefact.
    """
    # Strategy 1 — normal reprojection
    try:
        return gdf.to_crs(epsg=target_epsg)
    except Exception:
        pass

    # Strategy 2 — override the stored CRS then reproject
    try:
        return gdf.set_crs(epsg=target_epsg, allow_override=True).to_crs(epsg=target_epsg)
    except Exception:
        pass

    # Strategy 3 — rebuild CRS from scratch via pyproj, bypassing any
    # malformed metadata in the .prj file entirely
    try:
        from pyproj import CRS
        gdf = gdf.copy()
        gdf.geometry = gdf.geometry.set_crs(CRS.from_epsg(target_epsg), allow_override=True)
        return gdf.to_crs(epsg=target_epsg)
    except Exception as exc:
        raise RuntimeError(
            f"Could not normalise shapefile CRS to EPSG:{target_epsg}. "
            f"Last error: {exc}"
        )


def _polygon_row_azimuth(geom) -> float:
    """
    Estimate the dominant row orientation of one plot polygon (degrees, 0–180°).

    Method: find the longest edge of the minimum rotated bounding rectangle
    (MRR) and compute its compass bearing.  For maize plots the longest edge
    aligns with the planted row direction.  The result is collapsed to 0–180°
    because rows have two equivalent directions.
    """
    if geom is None or geom.is_empty:
        return np.nan

    mrr    = geom.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)[:-1]   # 4 vertices; drop repeated close
    if len(coords) < 4:
        return np.nan

    sides = [(coords[i], coords[(i + 1) % 4]) for i in range(4)]
    lengths_angles = [
        (
            np.hypot(b[0] - a[0], b[1] - a[1]),
            np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])),
        )
        for a, b in sides
    ]
    _, angle_from_east = max(lengths_angles, key=lambda x: x[0])

    # Convert math angle (CCW from East) → compass bearing (CW from North),
    # then collapse to undirected 0–180° (rows have no preferred direction)
    bearing = (90.0 - angle_from_east) % 360.0
    return bearing % 180.0


def _load_and_orient_shapefile(
    shapefile_dir: Path,
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    """
    Load all per-field plot shapefiles, compute per-field mean row azimuth
    and field centroid coordinates.

    File naming convention
    ----------------------
    Each shapefile covers one field: ``FIELD_plot_outline.shp``.
    The field name is extracted from the filename stem by splitting on
    ``_plot_outline`` (e.g. ``C7B_plot_outline.shp`` → ``C7B``).

    Loading strategy
    ----------------
    Each file is individually CRS-normalised via ``_force_2d_crs`` before
    geometry operations, which strips the compound vertical-datum component
    present in the final_outline shapefiles.

    Row azimuth
    -----------
    Per-polygon azimuths are averaged using the circular mean of angles to
    avoid wrap-around bias near 0°/180°.  This is the correct estimator for
    directional data (rows are undirected: 0–180°).

    Centroids
    ---------
    The union of all plot polygons per field is reprojected to WGS84 and its
    centroid is used as the per-field coordinate for pvlib solar calculations.

    Parameters
    ----------
    shapefile_dir : Path to directory containing the .shp files

    Returns
    -------
    row_azimuths : dict  field → mean row azimuth (°, 0–180)
    field_coords : dict  field → (latitude, longitude) in WGS84 decimal degrees
    """
    import geopandas as gpd

    shp_files = sorted(shapefile_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp files found in {shapefile_dir}")

    row_azimuths: dict[str, float]               = {}
    field_coords: dict[str, tuple[float, float]] = {}

    for shp in shp_files:
        # Extract field name from filename: "G3_plot_outline" → "G3"
        stem  = shp.stem
        field = stem.split("_plot_outline")[0].upper()
        if not field:
            print(f"  [shapefile] WARNING: cannot parse field from {shp.name} — skipped")
            continue

        # Load and reproject to metric CRS for geometry operations
        raw       = gpd.read_file(shp)
        projected = _force_2d_crs(raw, target_epsg=_PROJ_EPSG)

        # Per-polygon row azimuth from minimum rotated rectangle long axis
        projected["_row_az"] = projected.geometry.apply(_polygon_row_azimuth)
        angles = projected["_row_az"].dropna().values
        if len(angles) == 0:
            print(f"  [shapefile] WARNING: no valid polygons for {field} — skipped")
            continue

        # Circular mean: project angles to unit circle, average, project back.
        # Handles wrap-around correctly (e.g. rows near 0°/180°).
        rad     = np.radians(angles)
        mean_az = float(np.degrees(
            np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))
        ) % 180)
        row_azimuths[field] = mean_az

        # Field centroid in WGS84 for per-field pvlib solar position
        union_wgs84  = projected.to_crs(epsg=4326).geometry.union_all()
        centroid     = union_wgs84.centroid
        field_coords[field] = (centroid.y, centroid.x)   # (lat, lon)

        print(f"  [shapefile]  {field:6s}: {len(angles):4d} plots  "
              f"row_azimuth={mean_az:.2f}°  "
              f"centroid=({centroid.y:.5f}°N, {centroid.x:.5f}°E)")

    print(f"[shapefile]      {len(row_azimuths)} fields loaded from {shapefile_dir.name}/")
    return row_azimuths, field_coords


def compute_shadow_row_angle(
    flight_df:     pd.DataFrame,
    solar_df:      pd.DataFrame,
    shapefile_dir: Path | None,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """
    Compute the acute angle between the shadow direction and the plot long axis
    (crop row orientation) for each flight.

    Physical interpretation
    -----------------------
    Shadows are cast in the direction opposite the sun:
        shadow_azimuth = (sun_azimuth + 180°) mod 360°

    The shadow–row angle is the acute angle between the shadow direction and
    the plot long axis (row direction), constrained to [0°, 90°]:

        0°  — shadow runs along the row length, shading within the same plot
        90° — shadow falls perpendicularly across rows, shading adjacent plots

    Both the shadow direction and row azimuth are collapsed to the 0–180°
    half-circle before computing the acute angle, so the result is
    independent of which end of the row is called "north".

    Parameters
    ----------
    flight_df     : output of load_flight_times()
    solar_df      : output of compute_solar_geometry(); must contain sun_azimuth
    shapefile_dir : directory containing per-field plot shapefiles; if None or
                    inaccessible, row_azimuth and shadow_row_angle are NaN

    Returns
    -------
    shadow_df    : DataFrame with columns flight_key, row_azimuth, shadow_row_angle
    field_coords : dict field → (lat, lon) centroids from shapefile (empty dict if
                   shapefile unavailable); used by compute_solar_geometry for
                   per-field pvlib coordinate accuracy
    """
    row_az_lookup: dict[str, float]               = {}
    field_coords:  dict[str, tuple[float, float]] = {}

    if shapefile_dir is not None and shapefile_dir.exists():
        try:
            row_az_lookup, field_coords = _load_and_orient_shapefile(shapefile_dir)
        except Exception as exc:
            print(f"  [shapefile] WARNING: could not read shapefile ({exc}) "
                  f"— shadow_row_angle will be NaN")
    else:
        print(f"  [shapefile] WARNING: shapefile_dir not found "
              f"({shapefile_dir}) — shadow_row_angle will be NaN")

    # Merge solar azimuth into flight table
    merged = flight_df[["flight_key", "field"]].merge(
        solar_df[["flight_key", "sun_azimuth"]], on="flight_key", how="left"
    )

    rows = []
    for _, row in merged.iterrows():
        field   = row["field"]
        sun_az  = row["sun_azimuth"]
        row_az  = row_az_lookup.get(field, ROW_AZIMUTH_FALLBACK)

        if np.isnan(sun_az) or np.isnan(row_az):
            rows.append({
                "flight_key":       row["flight_key"],
                "row_azimuth":      row_az,
                "shadow_row_angle": np.nan,
            })
            continue

        # Shadow direction is opposite the sun; collapse directed bearing
        # (0–360°) to undirected half-circle (0–180°) to match row_azimuth
        shadow_az_180 = (sun_az + 180) % 180

        # Acute angle between shadow direction and plot long axis (0–90°)
        diff  = abs(shadow_az_180 - row_az)
        angle = min(diff, 180 - diff)   # always ≤ 90° by construction

        rows.append({
            "flight_key":       row["flight_key"],
            "row_azimuth":      row_az,
            "shadow_row_angle": angle,
        })

    shadow_df = pd.DataFrame(rows)
    n_valid = shadow_df["shadow_row_angle"].notna().sum()
    print(f"[shadow_row]     {n_valid}/{len(shadow_df)} flights with valid shadow_row_angle")
    return shadow_df, field_coords
# ===========================================================================
# Section 6 — Join all covariates and write output
# ===========================================================================

def build_flight_covariates(
    flight_times_path:      Path,
    irradiance_path:        Path,
    gdd_path:               Path,
    planting_dates_path:    Path,
    inoculation_dates_path: Path,
    shapefile_dir:          Path | None,
    out_path:               Path,
) -> pd.DataFrame:
    """
    Orchestrate all covariate computations and write the merged table.

    The join is performed sequentially on ``flight_key``; all sections produce
    one row per flight.  No covariate is required — missing values are NaN and
    propagate transparently to downstream scripts.

    Parameters correspond to the CLI arguments documented in the module
    docstring.

    Returns
    -------
    The merged DataFrame written to ``out_path``.
    """
    print("=" * 60)
    print("BUILD FLIGHT COVARIATES")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load raw inputs
    # ------------------------------------------------------------------
    print("\n--- Loading raw inputs ---")
    flight_df = load_flight_times(flight_times_path)
    irr_df    = load_irradiance(irradiance_path)
    gdd_df    = load_gdd(gdd_path)
    plant_df  = load_planting_dates(planting_dates_path)
    inoc_df   = load_inoculation_dates(inoculation_dates_path)

    # ------------------------------------------------------------------
    # Compute per-section covariate tables
    # ------------------------------------------------------------------
    # Section 5 (part 1): load shapefile to get per-field centroid coordinates
    # before pvlib solar geometry is computed.  The row azimuth lookup and
    # field centroid dict are extracted here; shadow_row_angle is computed in
    # part 2 after solar azimuth is available.
    print("\n--- Section 5 (part 1): load shapefile → field centroids ---")
    field_coords: dict[str, tuple[float, float]] = {}
    if shapefile_dir is not None and shapefile_dir.exists():
        try:
            _, field_coords = _load_and_orient_shapefile(shapefile_dir)
        except Exception as exc:
            print(f"  [shapefile] WARNING: {exc} — using site centroid for all fields")
    else:
        print(f"  [shapefile] WARNING: shapefile_dir not found ({shapefile_dir})"
              f" — using site centroid for all fields")

    print("\n--- Section 2: Solar geometry ---")
    solar_df = compute_solar_geometry(flight_df, field_coords=field_coords)

    print("\n--- Section 5 (part 2): shadow_row_angle from solar azimuth ---")
    shadow_df, _ = compute_shadow_row_angle(flight_df, solar_df, shapefile_dir)

    print("\n--- Section 3: Irradiance ---")
    irr_summary_df = compute_irradiance(flight_df, irr_df)

    print("\n--- Section 4: Phenology (DPI, AGDD) ---")
    phenology_df = compute_phenology(flight_df, gdd_df, plant_df, inoc_df)

    # ------------------------------------------------------------------
    # Sequential left-join on flight_key
    # ------------------------------------------------------------------
    print("\n--- Section 6: Joining all covariates ---")

    # Start from the flight_times backbone so all flights are present
    result = flight_df[[
        "flight_key", "flight_date", "field", "year",
        "flight_start", "flight_end",
    ]].copy()

    # Convert timestamps back to plain strings for CSV portability
    result["flight_start"] = result["flight_start"].dt.strftime("%Y-%m-%d %H:%M:%S")
    result["flight_end"]   = result["flight_end"].dt.strftime("%Y-%m-%d %H:%M:%S")
    result["flight_date"]  = result["flight_date"].astype(str)

    # Join solar geometry — sza, sun_azimuth, and abs_time_from_noon are all
    # computed by pvlib at the flight midpoint (not taken from flight_times.csv)
    result = result.merge(
        solar_df[["flight_key", "sza", "sun_azimuth", "abs_time_from_noon"]],
        on="flight_key", how="left",
    )

    # Join irradiance summary
    result = result.merge(
        irr_summary_df[["flight_key", "irradiance_mean", "irradiance_cv", "irradiance_n"]],
        on="flight_key", how="left",
    )

    # Join phenology
    result = result.merge(
        phenology_df[["flight_key", "dpi", "agdd"]],
        on="flight_key", how="left",
    )

    # Join shadow–row angle (row_azimuth retained for transparency/audit)
    result = result.merge(
        shadow_df[["flight_key", "row_azimuth", "shadow_row_angle"]],
        on="flight_key", how="left",
    )

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False, float_format="%.4f")

    print(f"\n[output] {len(result)} rows × {len(result.columns)} columns "
          f"written → {out_path}")
    print("\nColumn summary:")
    print(result.describe(include="all").T[["count", "mean", "min", "max"]].to_string())

    return result


# ===========================================================================
# CLI entry point
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-flight covariates from raw inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--flight_times",
        default=str(_DEFAULT_FLIGHT_TIMES),
        help="Path to flight_times.csv",
    )
    parser.add_argument(
        "--irradiance",
        default=str(_DEFAULT_IRRADIANCE),
        help="Path to irradiance_minute.csv",
    )
    parser.add_argument(
        "--gdd",
        default=str(_DEFAULT_GDD),
        help="Path to gdd.csv (daily GDD increments, base 50 °F)",
    )
    parser.add_argument(
        "--planting_dates",
        default=str(_DEFAULT_PLANTING_DATES),
        help="Path to planting_dates.csv",
    )
    parser.add_argument(
        "--inoculation_dates",
        default=str(_DEFAULT_INOCULATION_DATES),
        help="Path to inoculation_dates.csv (DPI reference origin, keyed by year)",
    )
    parser.add_argument(
        "--shapefile_dir",
        default=str(_DEFAULT_SHAPEFILE_DIR),
        help="Directory containing plot boundary shapefile (for row azimuth).",
    )
    parser.add_argument(
        "--out",
        default=str(_DEFAULT_OUT),
        help="Destination path for flight_covariates.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    shapefile_dir = Path(args.shapefile_dir)
    if not shapefile_dir.exists():
        print(f"[shapefile] Note: shapefile_dir does not exist at {shapefile_dir}")
        print("             shadow_row_angle will be NaN — provide --shapefile_dir to fix.")
        shapefile_dir = None   # pass None so compute_shadow_row_angle handles gracefully

    build_flight_covariates(
        flight_times_path      = Path(args.flight_times),
        irradiance_path        = Path(args.irradiance),
        gdd_path               = Path(args.gdd),
        planting_dates_path    = Path(args.planting_dates),
        inoculation_dates_path = Path(args.inoculation_dates),
        shapefile_dir          = shapefile_dir,
        out_path               = Path(args.out),
    )


if __name__ == "__main__":
    main()
