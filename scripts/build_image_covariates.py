"""
build_image_covariates.py
─────────────────────────────────────────────────────────────────────────────
Compute pixel-derived image-level covariates for every .jpg tile in
final_sliced/ and join the CHM-derived frac_weed from frac_weed.csv.

Covariates computed directly from each .jpg tile
─────────────────────────────────────────────────
  sf_illuminorm   Illumination-normalised shadow fraction (HSV, Method 4).
                  Normalises the V channel by the per-image 95th-percentile
                  brightness before thresholding, making the measure robust
                  to flight-level exposure differences.

                      v_norm  = V / (P95_V_valid + ε)
                      shadow  = v_norm < (V_THRESHOLD / 255)
                      sf      = shadow_pixels / valid_pixels

  mean_brightness Per-image mean of the HSV V channel over valid (non-nodata)
                  pixels.  Units: 0–255 (uint8 scale).

  contrast_rms    RMS contrast of the grayscale image: standard deviation of
                  grayscale pixel values over the full tile.
                  contrast_rms = std( gray.astype(float32) )

Covariate joined from weed_frac.csv (optional)
─────────────────────────────────────────────────────
  frac_weed      Pixel fraction of CHM-classified weed-zone vegetation
                  within the plot boundary.  Computed by
                  weed_pressure_pipeline.py; keyed here by image_filename.

Nodata masking
──────────────
White-filled pixels (R, G, B all ≥ WHITE_THRESH = 245) are treated as
nodata and excluded from all pixel-level computations.

Output
──────
image_covariates.csv  (written to --out_dir, default: data/covariates/):

    image_filename, flight_id, field, frac_weed,
    sf_illuminorm, mean_brightness, contrast_rms

Column order matches the existing ground-truth file in the repository.

Usage
─────
# Full run with weed join:
python scripts/build_image_covariates.py \\
    --image_dir  /path/to/final_sliced \\
    --labels_csv data/labels/full_dataset.csv \\
    --weed_csv   data/covariates/raw/frac_weed.csv \\
    --out_dir    data/covariates

# Without weed (omit --weed_csv; frac_weed column will be NaN):
python scripts/build_image_covariates.py \\
    --image_dir  /path/to/final_sliced \\
    --labels_csv data/labels/full_dataset.csv \\
    --out_dir    data/covariates

Author : Gage Lab — Cole Hammett
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ── Shadow fraction parameters ─────────────────────────────────────────────────
# V channel threshold (0–255) below which a pixel is classified as shadow.
# Applied after per-image illumination normalisation (illuminorm method).
V_THRESHOLD = 55

# Pixels with R, G, B all ≥ this value are treated as white-fill nodata.
WHITE_THRESH = 245


# ── Output column order ────────────────────────────────────────────────────────
# Matches the ground-truth image_covariates.csv schema.
OUTPUT_COLS = [
    "image_filename",
    "flight_id",
    "field",
    "frac_weed",
    "sf_illuminorm",
    "mean_brightness",
    "contrast_rms",
]


# =============================================================================
# Pixel-level covariate computation
# =============================================================================

def nodata_mask(rgb: np.ndarray, white_thresh: int = WHITE_THRESH) -> np.ndarray:
    """
    Return a boolean mask that is True where a pixel is white-fill nodata.

    Parameters
    ----------
    rgb : np.ndarray, shape (H, W, 3), dtype uint8
    white_thresh : int
        Pixels with R, G, B all ≥ this value are nodata.

    Returns
    -------
    np.ndarray, shape (H, W), dtype bool
    """
    return np.all(rgb >= white_thresh, axis=-1)


def compute_pixel_covariates(
    img_path: Path,
    v_threshold: int = V_THRESHOLD,
    white_thresh: int = WHITE_THRESH,
) -> dict:
    """
    Compute sf_illuminorm, mean_brightness, and contrast_rms for one .jpg tile.

    Parameters
    ----------
    img_path : Path
        Path to a single plot tile (.jpg).
    v_threshold : int
        HSV V-channel threshold for shadow classification (0–255).
    white_thresh : int
        Pixel brightness threshold for nodata masking (0–255).

    Returns
    -------
    dict with keys: sf_illuminorm, mean_brightness, contrast_rms.
    All values are NaN on read failure or if no valid pixels exist.
    """
    nan_result = {
        "sf_illuminorm":  np.nan,
        "mean_brightness": np.nan,
        "contrast_rms":   np.nan,
    }

    # Load image as BGR, convert to RGB
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return nan_result
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── Nodata mask ────────────────────────────────────────────────────────────
    nd      = nodata_mask(rgb, white_thresh)
    valid   = ~nd
    n_valid = int(valid.sum())

    if n_valid == 0:
        return nan_result

    # ── HSV conversion ─────────────────────────────────────────────────────────
    hsv   = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    v     = hsv[:, :, 2].astype(np.float32)        # brightness channel (0–255)

    # ── mean_brightness ────────────────────────────────────────────────────────
    # Mean HSV V value over valid pixels.
    mean_brightness = float(v[valid].mean())

    # ── sf_illuminorm (illumination-normalised shadow fraction) ────────────────
    # Normalise V by the per-image 95th-percentile brightness so that
    # flight-level exposure differences do not bias the shadow fraction.
    # See: analyze_residuals_shadow_illuminorm.py, Method 4.
    p95_v         = float(np.percentile(v[valid], 95))
    v_norm        = v / (p95_v + 1e-6)             # normalised brightness
    norm_thresh   = v_threshold / 255.0             # threshold in normalised space
    sf_illuminorm = float(
        ((v_norm < norm_thresh) & valid).sum()
    ) / n_valid

    # ── contrast_rms ───────────────────────────────────────────────────────────
    # RMS contrast: standard deviation of grayscale pixel values over the full
    # tile (including nodata border pixels, consistent with ground-truth file).
    # Source: laplacian_variance_analysis.ipynb, compute_image_quality().
    gray         = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    contrast_rms = float(gray.astype(np.float32).std())

    return {
        "sf_illuminorm":   sf_illuminorm,
        "mean_brightness": mean_brightness,
        "contrast_rms":    contrast_rms,
    }


# =============================================================================
# Image directory indexing
# =============================================================================

def index_image_dir(image_dir: Path) -> dict[str, Path]:
    """
    Build a {filename → full_path} lookup for all .jpg tiles in image_dir.

    Only the basename (e.g. '20230713_I3B_09565.jpg') is used as the key so
    the lookup is robust to any subdirectory structure.

    Parameters
    ----------
    image_dir : Path

    Returns
    -------
    dict mapping image_filename (str) → Path
    """
    lookup: dict[str, Path] = {}
    for ext in ("*.jpg", "*.JPG", "*.jpeg"):
        for p in image_dir.rglob(ext):
            # Normalise to lowercase basename for case-insensitive matching
            lookup[p.name.lower()] = p
    return lookup


# =============================================================================
# Metadata helpers
# =============================================================================

def flight_id_from_filename(image_filename: str) -> str:
    """
    Extract flight_id from image_filename.

    Convention: YYYYMMDD_FIELD_PLOT.jpg → YYYYMMDD_FIELD

    Parameters
    ----------
    image_filename : str
        e.g. '20230713_I3B_09565.jpg'

    Returns
    -------
    str  e.g. '20230713_I3B'
    """
    parts = Path(image_filename).stem.split("_")
    # parts[0] = YYYYMMDD, parts[1] = FIELD, parts[2] = PLOT
    return f"{parts[0]}_{parts[1]}"


def field_from_filename(image_filename: str) -> str:
    """
    Extract field from image_filename.

    Convention: YYYYMMDD_FIELD_PLOT.jpg → FIELD
    """
    return Path(image_filename).stem.split("_")[1]


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute image-level covariates and write image_covariates.csv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image_dir",
        required=True,
        type=Path,
        help="Directory of .jpg plot tiles (final_sliced/).",
    )
    parser.add_argument(
        "--labels_csv",
        required=True,
        type=Path,
        help=(
            "Labels CSV containing at minimum an 'image_filename' column "
            "(e.g. data/labels/full_dataset.csv). "
            "Used to define the set of images to process."
        ),
    )
    parser.add_argument(
        "--weed_csv",
        type=Path,
        default=None,
        help=(
            "Optional: frac_weed.csv produced by weed_pressure_pipeline.py. "
            "If omitted, frac_weed column is written as NaN."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/covariates"),
        help="Directory to write image_covariates.csv (default: data/covariates).",
    )
    parser.add_argument(
        "--v_threshold",
        type=int,
        default=V_THRESHOLD,
        help=f"HSV V-channel shadow threshold (default: {V_THRESHOLD}).",
    )
    parser.add_argument(
        "--white_thresh",
        type=int,
        default=WHITE_THRESH,
        help=f"Nodata brightness threshold (default: {WHITE_THRESH}).",
    )
    args = parser.parse_args()

    image_dir: Path           = args.image_dir.resolve()
    labels_csv: Path          = args.labels_csv.resolve()
    weed_csv: Optional[Path]  = args.weed_csv.resolve() if args.weed_csv else None
    out_dir: Path             = args.out_dir.resolve()

    for p, name in [(image_dir, "--image_dir"), (labels_csv, "--labels_csv")]:
        if not p.exists():
            sys.exit(f"ERROR: {name} does not exist: {p}")
    if weed_csv is not None and not weed_csv.exists():
        sys.exit(f"ERROR: --weed_csv does not exist: {weed_csv}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load labels to obtain the canonical image list ─────────────────────────
    print(f"Loading labels: {labels_csv}")
    labels = pd.read_csv(labels_csv)
    if "image_filename" not in labels.columns:
        sys.exit("ERROR: labels_csv must contain an 'image_filename' column.")

    # Unique image filenames across the full dataset
    image_filenames = sorted(labels["image_filename"].dropna().unique())
    print(f"  Unique image filenames: {len(image_filenames):,}")

    # ── Index image directory ──────────────────────────────────────────────────
    print(f"\nIndexing image directory: {image_dir}")
    image_lookup = index_image_dir(image_dir)
    print(f"  Found {len(image_lookup):,} .jpg files")

    # ── Compute pixel covariates ───────────────────────────────────────────────
    print("\nComputing pixel covariates …")
    rows = []
    n_missing = 0

    for fn in tqdm(image_filenames, unit="img", ncols=80):
        fn_lower = fn.lower()
        img_path = image_lookup.get(fn_lower)

        if img_path is None:
            # Image tile not found on disk; record NaNs
            n_missing += 1
            metrics = {
                "sf_illuminorm":   np.nan,
                "mean_brightness": np.nan,
                "contrast_rms":    np.nan,
            }
        else:
            metrics = compute_pixel_covariates(
                img_path,
                v_threshold=args.v_threshold,
                white_thresh=args.white_thresh,
            )

        rows.append({
            "image_filename":  fn,
            "flight_id":       flight_id_from_filename(fn),
            "field":           field_from_filename(fn),
            **metrics,
        })

    result = pd.DataFrame(rows)

    if n_missing > 0:
        print(
            f"\n  WARNING: {n_missing} image filenames not found in {image_dir}. "
            f"Covariates for these rows are NaN."
        )

    # ── Join frac_weed ────────────────────────────────────────────────────────
    if weed_csv is not None:
        print(f"\nJoining frac_weed from: {weed_csv}")
        weed = pd.read_csv(weed_csv, usecols=["image_filename", "frac_weed"])
        n_before = len(result)
        result = result.merge(weed, on="image_filename", how="left")
        assert len(result) == n_before, (
            "Row count changed after weed join — check for duplicate "
            "image_filename values in frac_weed.csv."
        )
        n_null_weed = result["frac_weed"].isna().sum()
        if n_null_weed > 0:
            print(
                f"  WARNING: {n_null_weed} images have no matching frac_weed "
                f"(no weed_pressure_pipeline.py output for that flight/field)."
            )
        else:
            print(f"  Weed join complete — {len(result):,} rows matched.")
    else:
        # No weed CSV supplied; insert NaN column to match output schema
        print("\n--weed_csv not supplied; frac_weed will be NaN.")
        result["frac_weed"] = np.nan

    # ── Enforce output column order ────────────────────────────────────────────
    result = result[OUTPUT_COLS]

    # ── Summary statistics ─────────────────────────────────────────────────────
    print("\nCovariate summary:")
    print(
        result[["sf_illuminorm", "mean_brightness", "contrast_rms", "frac_weed"]]
        .describe()
        .round(4)
        .to_string()
    )

    # ── Write output ───────────────────────────────────────────────────────────
    out_path = out_dir / "image_covariates.csv"
    result.to_csv(out_path, index=False)
    print(f"\nWritten: {out_path}  ({len(result):,} rows)")


if __name__ == "__main__":
    main()
