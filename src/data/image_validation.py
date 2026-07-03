"""
Image Validation for UAV Orthomosaic Tiles.

When an orthomosaic is sliced into plot-level tiles, tiles that fall outside
the mapped area contain solid white fill pixels (255, 255, 255).  These
out-of-bounds tiles introduce systematic noise into SLB severity regression:
  - Fully filled tiles carry no disease signal at all.
  - Partially filled tiles mislead attention maps and inflate prediction error.

This module provides fast, vectorized checks to detect and filter white-fill
tiles before they enter cross-validation splits or the DataLoader.

Detection strategy
------------------
A pixel is considered nodata when ALL three RGB channels meet or exceed a
high threshold (near-white):

    nodata = (R >= white_thresh AND G >= white_thresh AND B >= white_thresh)

Defaults:
    white_thresh = 245  (tolerates mild JPEG compression at tile edges)

A tile is rejected when its nodata fraction exceeds
``max_nodata_fraction`` (default 0.05 — 5% of pixels).

Integration with CV split generation
-------------------------------------
  White-fill validation runs automatically in ``create_cv_splits.py`` when
  ``--image-source-dir`` is provided.  ``filter_incomplete_images()`` is
  called before any splits are made.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core per-image check
# ---------------------------------------------------------------------------

def check_image_completeness(
    image_path: str | Path,
    white_thresh: int = 245,
    max_nodata_fraction: float = 0.05,
) -> Tuple[bool, float, str]:
    """
    Verify that a single image tile contains sufficient valid RGB data.

    A pixel is treated as nodata if it is near-white OR near-black — the two
    common fill values used by orthomosaic slicing tools when a tile extends
    beyond the mapped area.

    Parameters
    ----------
    image_path : str or Path
        Path to the image file (JPG, PNG, TIF, …).
    white_thresh : int
        Per-channel value at or above which a pixel is white nodata.
        Default 245 tolerates mild JPEG compression artefacts at tile edges.
    max_nodata_fraction : float
        Maximum allowable proportion of nodata pixels [0.0–1.0].
        Images exceeding this are flagged as incomplete. Default 0.05 (5%).

    Returns
    -------
    is_valid : bool
        True if the image passes (enough valid pixels present).
    nodata_fraction : float
        Fraction of white nodata pixels actually found.
    reason : str
        'ok', 'fully_oob', 'partially_oob_white', 'load_error', or
        'wrong_channels'.
    """
    path = Path(image_path)

    try:
        img = Image.open(path)

        if img.mode != "RGB":
            img = img.convert("RGB")

        arr = np.asarray(img, dtype=np.uint8)  # (H, W, 3)

        if arr.ndim != 3 or arr.shape[2] != 3:
            return False, 1.0, "wrong_channels"

        n_px = arr.shape[0] * arr.shape[1]

        # Near-white: all three channels ≥ white_thresh
        white_mask = np.all(arr >= white_thresh, axis=2)
        nodata_mask     = white_mask
        nodata_fraction = float(nodata_mask.sum()) / n_px

        if nodata_fraction >= max_nodata_fraction:
            if nodata_fraction >= 0.95:
                return False, nodata_fraction, "fully_oob"
            return False, nodata_fraction, "partially_oob_white"

        return True, nodata_fraction, "ok"

    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load %s: %s", path, exc)
        return False, 1.0, f"load_error: {exc}"


# ---------------------------------------------------------------------------
# Batch scanner
# ---------------------------------------------------------------------------

def scan_images(
    image_paths: List[str | Path],
    white_thresh: int = 245,
    max_nodata_fraction: float = 0.05,
    num_workers: int = 8,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Scan a list of image paths and return a validation report DataFrame.

    Parameters
    ----------
    image_paths : list of str or Path
    white_thresh : int
        Near-white nodata threshold per channel (default 245).
    max_nodata_fraction : float
        Reject images whose nodata fraction exceeds this (default 0.05).
    num_workers : int
        Parallel I/O threads (default 8).
    verbose : bool
        Print progress every 1,000 images.

    Returns
    -------
    pd.DataFrame with columns:
        image_path, filename, is_valid, nodata_fraction,
        white_fraction, reason
    """
    total   = len(image_paths)
    results: List[Dict] = []

    def _check(p: Path) -> Dict:
        path = Path(p)
        try:
            img = Image.open(path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            arr = np.asarray(img, dtype=np.uint8)

            if arr.ndim != 3 or arr.shape[2] != 3:
                return dict(
                    image_path=str(path), filename=path.name,
                    is_valid=False, nodata_fraction=1.0,
                    white_fraction=0.0, 
                    reason="wrong_channels",
                )

            n_px       = arr.shape[0] * arr.shape[1]
            white_mask = np.all(arr >= white_thresh, axis=2)
            nodata_mask = white_mask

            nodata_frac = float(nodata_mask.sum()) / n_px
            white_frac  = float(white_mask.sum())  / n_px
            is_valid    = nodata_frac < max_nodata_fraction

            if not is_valid:
                if nodata_frac >= 0.95:
                    reason = "fully_oob"
                else:
                    reason = "partially_oob_white"
            else:
                reason = "ok"

        except Exception as exc:  # noqa: BLE001
            return dict(
                image_path=str(path), filename=path.name,
                is_valid=False, nodata_fraction=1.0,
                white_fraction=0.0, 
                reason=f"load_error: {exc}",
            )

        return dict(
            image_path=str(path),
            filename=path.name,
            is_valid=is_valid,
            nodata_fraction=round(nodata_frac, 4),
            white_fraction= round(white_frac,  4),
            reason=reason,
        )

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_check, p): p for p in image_paths}
        done = 0
        for future in as_completed(futures):
            results.append(future.result())
            done += 1
            if verbose and done % 1000 == 0:
                print(f"  Validated {done:,}/{total:,} images …", flush=True)

    if verbose:
        print(f"  Validated {total:,}/{total:,} images.", flush=True)

    return pd.DataFrame(results).sort_values("image_path").reset_index(drop=True)


# ---------------------------------------------------------------------------
# DataFrame-level filtering (plugs into create_cv_splits.py)
# ---------------------------------------------------------------------------

def filter_incomplete_images(
    labels_df: pd.DataFrame,
    image_dir: str | Path,
    image_filename_column: str = "image_filename",
    image_extension: str = ".jpg",
    white_thresh: int = 245,
    max_nodata_fraction: float = 0.05,
    num_workers: int = 8,
    auto_confirm: bool = False,
    report_dir: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter a labels DataFrame to remove rows whose images are out-of-bounds.

    This is the primary integration point for ``create_cv_splits.py``.  Call
    this function *after* ``check_and_filter_missing_images()`` and *before*
    creating CV folds so that no bad tile ever appears in any split.

    Parameters
    ----------
    labels_df : pd.DataFrame
        Must contain ``image_filename_column``.
    image_dir : str or Path
        Root directory where sliced images live.
    image_filename_column : str
        Column with image filenames.
    image_extension : str
        Extension to append when image IDs lack one.
    white_thresh : int
        Near-white nodata threshold per channel (default 245).
    max_nodata_fraction : float
        Images with more than this fraction of nodata pixels are removed
        (default 0.05 — 5%).
    num_workers : int
        Parallel I/O threads (default 8).
    auto_confirm : bool
        If False, prompt the user to confirm before dropping rows.
    report_dir : str or Path, optional
        If given, write ``incomplete_images_report.csv`` there.

    Returns
    -------
    clean_df : pd.DataFrame
        Labels DataFrame with incomplete images removed.
    report_df : pd.DataFrame
        Full per-image validation report.
    """
    image_dir = Path(image_dir)

    print("\n" + "=" * 70)
    print("Checking image completeness (white-fill / out-of-bounds detection)")
    print(f"  Directory        : {image_dir}")
    print(f"  Nodata threshold : fraction > {max_nodata_fraction:.1%} → reject")
    print(f"  White pixel ≥    : {white_thresh} (all channels)")
    print("=" * 70)

    def _build_path(image_filename: str) -> Path:
        if Path(image_filename).suffix:
            return image_dir / image_filename
        return image_dir / f"{image_filename}{image_extension}"

    image_paths = [_build_path(row[image_filename_column]) for _, row in labels_df.iterrows()]

    print(f"\nScanning {len(image_paths):,} images with {num_workers} threads …")
    report_df = scan_images(
        image_paths,
        white_thresh=white_thresh,
        max_nodata_fraction=max_nodata_fraction,
        num_workers=num_workers,
        verbose=True,
    )

    labels_df = labels_df.copy()
    labels_df["_abs_path"] = [str(p) for p in image_paths]
    merged = labels_df.merge(
        report_df[["image_path", "is_valid", "nodata_fraction", "reason"]],
        left_on="_abs_path",
        right_on="image_path",
        how="left",
    ).drop(columns=["_abs_path", "image_path"])

    invalid_df = merged[~merged["is_valid"]]
    clean_df   = merged[merged["is_valid"]].drop(
        columns=["is_valid", "nodata_fraction", "reason"]
    )

    n_total   = len(labels_df)
    n_invalid = len(invalid_df)
    n_clean   = len(clean_df)

    reason_counts = invalid_df["reason"].value_counts().to_dict()
    print(f"\n  Total images   : {n_total:>8,}")
    print(f"  Valid          : {n_clean:>8,}  ({n_clean / n_total:.1%})")
    print(f"  Invalid        : {n_invalid:>8,}  ({n_invalid / n_total:.1%})")
    for reason, count in reason_counts.items():
        print(f"    └─ {reason:<28}: {count:,}")

    if n_invalid == 0:
        print("\n✓ All images have complete RGB data — no filtering needed.")
        print("=" * 70 + "\n")
        return (
            labels_df.drop(
                columns=["is_valid", "nodata_fraction", "reason"], errors="ignore"
            ),
            report_df,
        )

    if report_dir is not None:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "incomplete_images_report.csv"
        report_df.to_csv(report_path, index=False)
        print(f"\n  Report saved → {report_path}")

    if not auto_confirm:
        print(f"\n⚠  WARNING: {n_invalid} images will be removed from the dataset.")
        print(f"   Data retained: {n_clean}/{n_total}  ({n_clean / n_total:.1%})")
        while True:
            response = input(
                "\nProceed with removing incomplete images? (yes/no): "
            ).strip().lower()
            if response in ("yes", "y"):
                print("Proceeding …")
                break
            elif response in ("no", "n"):
                import sys
                print("Aborted. No splits created.")
                sys.exit(0)
            else:
                print("Please type 'yes' or 'no'.")
    else:
        print(f"\nAuto-confirm: removing {n_invalid} incomplete images.")

    print(f"✓ Clean dataset: {n_clean:,} samples")
    print("=" * 70 + "\n")

    return clean_df.reset_index(drop=True), report_df


# ---------------------------------------------------------------------------
# Quick summary helper (useful in notebooks)
# ---------------------------------------------------------------------------

def summarise_validation_report(report_df: pd.DataFrame) -> None:
    """Print a concise summary of a scan_images() report."""
    total     = len(report_df)
    n_valid   = int(report_df["is_valid"].sum())
    n_invalid = total - n_valid

    print(f"{'─' * 55}")
    print(f"  Total images  : {total:>8,}")
    print(f"  Valid         : {n_valid:>8,}  ({n_valid / total:.1%})")
    print(f"  Invalid       : {n_invalid:>8,}  ({n_invalid / total:.1%})")
    print(f"{'─' * 55}")

    if n_invalid > 0:
        print("  Breakdown by failure reason:")
        for reason, cnt in (
            report_df[~report_df["is_valid"]]["reason"].value_counts().items()
        ):
            print(f"    {reason:<30}: {cnt:>6,}  ({cnt / total:.1%})")
        print(f"{'─' * 55}")

        worst = report_df.nlargest(5, "nodata_fraction")[
            ["filename", "nodata_fraction", "white_fraction", "reason"]
        ]
        print("\n  Top 5 worst tiles (highest nodata fraction):")
        print(worst.to_string(index=False))
    print()
