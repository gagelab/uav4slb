"""
solar.py
========
Reusable solar geometry and shapefile utilities for the SLB UAV project.

Public API
----------
_force_2d_crs              — robust CRS normalisation (strips vertical datum)
load_plot_outlines         — load and concatenate plot-outline shapefiles
get_polygon_row_orientation — minimum-rotated-rectangle → compass bearing
compute_field_centroids    — per-population centroid + row orientation table
build_solar_features       — pvlib solar angles + shadow geometry for a flight list

Dependencies: geopandas, shapely, pvlib, pandas, numpy, pyproj
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import pvlib
from shapely.geometry import MultiPolygon, Polygon  # noqa: F401  (re-exported for callers)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants (override via function arguments where provided)
# ---------------------------------------------------------------------------

DEFAULT_LAT = 35.787    # degrees N  (NC State Gage Lab fields, Raleigh NC)
DEFAULT_LON = -78.680   # degrees W  (west is negative)
DEFAULT_ALT = 100.0     # metres above sea level
LOCAL_TZ    = "America/New_York"


# ===========================================================================
# CRS utilities
# ===========================================================================

def _force_2d_crs(gdf: gpd.GeoDataFrame, target_epsg: int = 2264) -> gpd.GeoDataFrame:
    """
    Strip any compound/vertical CRS component and reproject to *target_epsg*.

    Tries three strategies in order so that the compound-datum issue present in
    the final_outline shapefiles (NAD83 / NC ftUS + unknown vertical) cannot
    block execution.
    """
    # Strategy 1 — normal reprojection
    try:
        return gdf.to_crs(epsg=target_epsg)
    except Exception:
        pass

    # Strategy 2 — override CRS then reproject
    try:
        return gdf.set_crs(epsg=target_epsg, allow_override=True).to_crs(epsg=target_epsg)
    except Exception:
        pass

    # Strategy 3 — nuclear: rebuild via pyproj CRS object
    try:
        from pyproj import CRS
        crs_2d = CRS.from_epsg(target_epsg)
        gdf = gdf.copy()
        gdf.geometry = gdf.geometry.set_crs(crs_2d, allow_override=True)
        return gdf.to_crs(epsg=target_epsg)
    except Exception as e:
        raise RuntimeError(
            f"Could not normalise CRS to EPSG:{target_epsg}. Last error: {e}"
        )


# ===========================================================================
# Shapefile loading
# ===========================================================================

def load_plot_outlines(shapefile_dir: str | Path) -> gpd.GeoDataFrame:
    """
    Load all shapefiles in *shapefile_dir* and concatenate into a single GDF.

    Each file is individually normalised to EPSG:2264 (NAD83 / NC State Plane
    ftUS) before concatenation, which strips the compound vertical-datum
    component that prevents geopandas from merging the final_outline shapefiles.

    Prints a diagnostic table of column names and unique values immediately
    after loading so that population-matching problems are visible at runtime.
    """
    shapefile_dir = Path(shapefile_dir)
    shapefiles = list(shapefile_dir.glob("*.shp"))
    if not shapefiles:
        raise FileNotFoundError(
            f"No .shp files found in {shapefile_dir}\n"
            "Verify the path and that you have read access to the volume."
        )

    gdfs = []
    for shp in shapefiles:
        print(f"  Reading {shp.name} ...", end=" ", flush=True)
        raw = gpd.read_file(shp)
        raw["source_file"] = shp.stem
        normalised = _force_2d_crs(raw)
        print(f"{len(normalised):,} polygons  [CRS → EPSG:2264]")
        gdfs.append(normalised)

    combined = gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True), crs="EPSG:2264"
    )
    print(f"\nTotal: {len(combined):,} polygons from {len(shapefiles)} shapefile(s).")

    print("\n  Shapefile attribute columns (use these to verify population matching):")
    for col in combined.columns:
        if col == "geometry":
            continue
        uniq = combined[col].dropna().unique()
        sample = list(uniq[:8])
        suffix = f" … ({len(uniq)} total)" if len(uniq) > 8 else ""
        print(f"    {col:20s}: {sample}{suffix}")

    return combined


# ===========================================================================
# Plot geometry
# ===========================================================================

def get_polygon_row_orientation(geom) -> float:
    """
    Compute the dominant row orientation of a plot polygon in degrees (0–180).

    The row orientation is estimated as the angle of the *longest axis* of the
    minimum rotated bounding rectangle.  For maize plots, this is the direction
    of planted rows.

    Returns
    -------
    float : azimuth of the long axis in degrees from North (0–180°).
    """
    if geom is None or geom.is_empty:
        return np.nan
    rect   = geom.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)[:-1]   # 4 vertices
    sides  = [(coords[i], coords[(i + 1) % 4]) for i in range(4)]
    side_lengths = [np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in sides]
    longest_idx  = int(np.argmax(side_lengths))
    a, b = sides[longest_idx]
    dx, dy = b[0] - a[0], b[1] - a[1]
    angle_from_east = np.degrees(np.arctan2(dy, dx)) % 360
    bearing = (90 - angle_from_east) % 360
    return bearing % 180   # collapse to 0–180 (rows have two equivalent directions)


# ===========================================================================
# Field centroids
# ===========================================================================

def _pick_population_column(gdf: gpd.GeoDataFrame) -> str:
    """Return the column that best identifies field/population groupings."""
    for col in gdf.columns:
        if col.lower() in ("population", "pop", "field", "entry"):
            return col
        if any(kw in col.lower() for kw in ("population", "pop", "field", "entry")):
            return col
    if "source_file" in gdf.columns:
        return "source_file"
    gdf["_pop"] = "FIELD"
    return "_pop"


def compute_field_centroids(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    For each population/field group in the GDF, compute:
      - centroid latitude / longitude (WGS84)
      - mean row orientation angle derived from polygon geometry
      - number of plots

    Row-orientation geometry is computed in the projected CRS (EPSG:2264, feet)
    before reprojecting to WGS84 — the angle is invariant to isotropic scaling.

    Returns
    -------
    DataFrame with columns: population, source_file_stem, field_lat, field_lon,
                            row_orientation, n_plots
    """
    gdf_proj = gdf.copy()
    gdf_proj["_row_angle"] = gdf_proj.geometry.apply(get_polygon_row_orientation)

    gdf_wgs84 = gdf_proj.to_crs(epsg=4326)
    pop_col   = _pick_population_column(gdf_wgs84)
    print(f"\n  Using '{pop_col}' as population grouping column.")

    records = []
    for pop, grp in gdf_wgs84.groupby(pop_col):
        union_geom = grp.geometry.unary_union
        centroid   = union_geom.centroid

        angles_rad = np.radians(grp["_row_angle"].dropna())
        if len(angles_rad) > 0:
            mean_angle = np.degrees(
                np.arctan2(np.mean(np.sin(angles_rad)),
                           np.mean(np.cos(angles_rad)))
            ) % 180
        else:
            mean_angle = np.nan

        records.append({
            "population":       str(pop),
            "source_file_stem": str(pop),
            "field_lat":        centroid.y,
            "field_lon":        centroid.x,
            "row_orientation":  mean_angle,
            "n_plots":          len(grp),
        })

    centroids_df = pd.DataFrame(records)
    print(f"\n  Field centroids ({len(centroids_df)} groups):")
    print(centroids_df.to_string(index=False))
    return centroids_df


# ===========================================================================
# Solar angle computation (pvlib)
# ===========================================================================

def compute_solar_angles(
    date_str: str,
    time_from_noon_min: float,
    lat: float,
    lon: float,
    altitude: float = DEFAULT_ALT,
) -> dict:
    """
    Compute solar zenith angle (SZA) and solar azimuth for a flight.

    Parameters
    ----------
    date_str           : 'YYYYMMDD' string extracted from the flight ID.
    time_from_noon_min : minutes from solar noon (negative = AM, positive = PM).
    lat, lon           : WGS84 decimal degrees of field centroid.
    altitude           : metres above sea level.

    Returns
    -------
    dict with keys: sza, solar_azimuth, solar_elevation, airmass,
                    flight_time_local
    """
    location = pvlib.location.Location(
        latitude=lat, longitude=lon, altitude=altitude, tz=LOCAL_TZ
    )
    date = datetime.strptime(date_str, "%Y%m%d")
    date_range = pd.date_range(
        start=f"{date.year}-{date.month:02d}-{date.day:02d} 00:00",
        end=  f"{date.year}-{date.month:02d}-{date.day:02d} 23:59",
        freq="1min",
        tz=LOCAL_TZ,
    )
    solar_pos = location.get_solarposition(date_range)
    noon_idx  = solar_pos["elevation"].idxmax()
    flight_time = noon_idx + pd.Timedelta(minutes=time_from_noon_min)

    sp        = location.get_solarposition(pd.DatetimeIndex([flight_time], tz=LOCAL_TZ))
    sza       = float(sp["zenith"].iloc[0])
    azimuth   = float(sp["azimuth"].iloc[0])
    elevation = float(sp["elevation"].iloc[0])
    airmass   = pvlib.atmosphere.get_relative_airmass(sza) if sza < 90 else np.nan

    return {
        "sza":               sza,
        "solar_azimuth":     azimuth,
        "solar_elevation":   elevation,
        "airmass":           airmass,
        "flight_time_local": str(flight_time),
    }


def compute_shadow_row_angle(solar_azimuth: float, row_orientation: float) -> dict:
    """
    Compute the angular relationship between shadow direction and row orientation.

    Shadow direction = (solar_azimuth + 180) mod 360.
    shadow_row_angle is the acute angle between the shadow and the row axis
    (0° = shadows along rows; 90° = shadows across rows).

    Returns
    -------
    dict with keys: shadow_azimuth, shadow_row_angle,
                    shadow_component_along_row, shadow_component_across_row,
                    shadow_regime
    """
    shadow_azimuth = (solar_azimuth + 180.0) % 360.0
    diff  = abs(shadow_azimuth % 180 - row_orientation % 180)
    acute = min(diff, 180.0 - diff)

    along  = np.cos(np.radians(acute))
    across = np.sin(np.radians(acute))

    if acute < 20:
        regime = "along_row"
    elif acute > 70:
        regime = "across_row"
    else:
        regime = "diagonal"

    return {
        "shadow_azimuth":              shadow_azimuth,
        "shadow_row_angle":            acute,
        "shadow_component_along_row":  along,
        "shadow_component_across_row": across,
        "shadow_regime":               regime,
    }


def _extract_population_from_filename(name: str) -> str:
    """Extract population code from 'YYYYMMDD_POPULATION' or image filename."""
    parts = name.split("_")
    return parts[1] if len(parts) >= 2 else name


# ===========================================================================
# Solar features table builder
# ===========================================================================

def build_solar_features(
    shadows_df: pd.DataFrame,
    centroids_df: Optional[pd.DataFrame] = None,
    default_lat: float = DEFAULT_LAT,
    default_lon: float = DEFAULT_LON,
) -> pd.DataFrame:
    """
    For every row in *shadows_df*, compute solar angles and shadow geometry.

    Parameters
    ----------
    shadows_df   : DataFrame with columns ['flight', 'time(min)']
                   where 'flight' = 'YYYYMMDD_POPULATION'.
    centroids_df : Optional DataFrame with per-population lat/lon/row_orientation
                   (output of :func:`compute_field_centroids`).  If None, all
                   flights use *default_lat* / *default_lon* and row_orientation
                   is set to NaN.

    Returns
    -------
    DataFrame with one row per flight containing solar and shadow covariates.
    """
    records = []
    for _, row in shadows_df.iterrows():
        flight     = str(row["flight"])
        time_min   = float(row["time(min)"])
        date_str   = flight.split("_")[0]
        population = _extract_population_from_filename(flight)

        lat, lon, row_orient = default_lat, default_lon, np.nan
        matched_pop = None

        if centroids_df is not None and not centroids_df.empty:
            # Tier 1: exact match
            match = centroids_df[centroids_df["population"] == population]

            # Tier 2: population code appears in shapefile stem
            if match.empty:
                mask  = centroids_df["population"].str.contains(
                    population, case=False, regex=False, na=False
                )
                match = centroids_df[mask]

            # Tier 3: stem appears in population code (reverse partial)
            if match.empty:
                mask  = centroids_df["population"].apply(
                    lambda s: str(s).upper() in population.upper()
                )
                match = centroids_df[mask]

            if not match.empty:
                if len(match) > 1:
                    match = match.copy()
                    match["_match_len"] = match["population"].str.len()
                    match = match.sort_values("_match_len", ascending=False).head(1)
                lat         = float(match["field_lat"].iloc[0])
                lon         = float(match["field_lon"].iloc[0])
                row_orient  = float(match["row_orientation"].iloc[0])
                matched_pop = str(match["population"].iloc[0])

        if matched_pop:
            match_note = f"→ matched '{matched_pop}'"
        elif centroids_df is not None and not centroids_df.empty:
            match_note = "→ NO MATCH (using default coords)"
        else:
            match_note = "→ no shapefile (using default coords)"
        print(f"  {flight:25s}  pop={population:8s}  {match_note}")

        try:
            solar = compute_solar_angles(date_str, time_min, lat, lon)
        except Exception as e:
            print(f"  Warning: solar computation failed for {flight}: {e}")
            solar = {
                "sza": np.nan, "solar_azimuth": np.nan,
                "solar_elevation": np.nan, "airmass": np.nan,
                "flight_time_local": "",
            }

        if not np.isnan(solar.get("solar_azimuth", np.nan)) and not np.isnan(row_orient):
            shadow = compute_shadow_row_angle(solar["solar_azimuth"], row_orient)
        else:
            shadow = {
                "shadow_azimuth": np.nan, "shadow_row_angle": np.nan,
                "shadow_component_along_row": np.nan,
                "shadow_component_across_row": np.nan,
                "shadow_regime": "unknown",
            }

        records.append({
            "flight":             flight,
            "population":         population,
            "date":               pd.to_datetime(date_str, format="%Y%m%d"),
            "year":               int(date_str[:4]),
            "time_from_noon":     time_min,
            "abs_time_from_noon": abs(time_min),
            "field_lat":          lat,
            "field_lon":          lon,
            "row_orientation":    row_orient,
            **solar,
            **shadow,
        })

    return pd.DataFrame(records)
