#!/usr/bin/env python3
"""
create_cv_splits.py
===================
Build CV0 (leave-one-year-out) train/val/test split files for the UAV-SLB
disease severity prediction pipeline.

CV0 is the primary cross-validation strategy in this study: each fold holds
out a single season (year) as the test set and trains on all other years,
directly assessing temporal generalizability of the learned representations.

Three folds are created, one per year in the dataset (2023, 2024, 2025):
  fold_YYYY/metadata/train_labels.csv
  fold_YYYY/metadata/val_labels.csv
  fold_YYYY/metadata/test_labels.csv

The validation set is a stratified 15% subsample of the non-test years,
used only for early stopping and checkpoint selection during training.

Image validation (optional but recommended)
-------------------------------------------
When --image-source-dir is provided the script performs two checks before
creating splits:

  1. Existence check: rows whose image file cannot be found on disk are
     dropped so that DataLoader workers never encounter missing tiles.

  2. White-fill / no-data check: tiles with more than --max-nodata-fraction
     white-fill pixels (indicating orthomosaic boundary artefacts) are also
     dropped.  This delegates to src.data.image_validation.filter_incomplete_images,
     which scans tiles in parallel using --validation-workers threads.

Both checks report dropped rows and save a detailed text report alongside the
output splits for reproducibility documentation.

Outputs
-------
  <output_dir>/
    cv0/
      fold_2023/metadata/{train,val,test}_labels.csv
      fold_2024/metadata/{train,val,test}_labels.csv
      fold_2025/metadata/{train,val,test}_labels.csv
    cv_splits_summary.txt
    missing_images_report.txt     (only if any images are missing)
    low_quality_images_report.txt (only if any tiles fail the no-data check)

Usage
-----
  python scripts/create_cv_splits.py \\
      --labels-csv data/full_dataset.csv \\
      --output-dir data/cv_splits \\
      --image-source-dir /path/to/final_sliced \\
      --image-id-column image_filename \\
      --target-column score \\
      --year-column year \\
      --cv-strategies cv0 \\
      --auto-confirm-missing
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Project path bootstrap — allows running from any working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.image_validation import filter_incomplete_images


# ===========================================================================
# Image existence validation
# ===========================================================================

def check_and_filter_missing_images(
    labels_df: pd.DataFrame,
    source_dir: str,
    image_filename_column: str,
    image_extension: str = ".tif",
    auto_confirm: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Verify that every image referenced in the labels CSV exists on disk.

    Rows for which the image file cannot be found are optionally dropped before
    splits are created.  This prevents DataLoader workers from failing at
    runtime due to missing tiles.

    The function tries the exact path first, then probes common alternative
    extensions (.tif, .tiff, .jpg, .jpeg, .png) so that column values recorded
    without an extension still resolve correctly.

    Parameters
    ----------
    labels_df             DataFrame loaded from the labels CSV
    source_dir            Root directory that contains the image files
    image_filename_column Name of the column holding image filenames / IDs
    image_extension       Primary extension to append when the column value
                          has no suffix (e.g. '.tif')
    auto_confirm          When True, drop missing rows without prompting.
                          Set to True for non-interactive HPC runs.

    Returns
    -------
    filtered_df     DataFrame with missing-image rows removed
    missing_images  List of image IDs that could not be resolved
    """
    source_dir = Path(source_dir)
    missing_images: List[str] = []
    existing_indices: List[int] = []

    print("\n" + "=" * 70)
    print("Checking for missing images ...")
    print("=" * 70)

    for idx, row in labels_df.iterrows():
        image_id = row[image_filename_column]

        # If the column value already carries an extension, use it directly
        if Path(image_id).suffix:
            candidate = source_dir / image_id
            if candidate.exists():
                existing_indices.append(idx)
            else:
                missing_images.append(image_id)
            continue

        # No extension in column value — try the specified extension first,
        # then fall back to other common formats
        found = False
        for ext in [image_extension, ".tif", ".tiff", ".jpg", ".jpeg", ".png"]:
            candidate = source_dir / f"{image_id}{ext}"
            if candidate.exists():
                existing_indices.append(idx)
                found = True
                break
        if not found:
            missing_images.append(image_id)

    # Report findings
    total         = len(labels_df)
    n_missing     = len(missing_images)
    n_existing    = len(existing_indices)

    print(f"\n  Total labels in CSV : {total}")
    print(f"  Images found        : {n_existing}")
    print(f"  Images missing      : {n_missing}")

    if n_missing == 0:
        print("  ✓ All images found — no filtering required.")
        print("=" * 70 + "\n")
        return labels_df, []

    # Display up to 20 missing IDs, then summarise the rest
    print(f"\n  ⚠️  Missing images ({n_missing} total):")
    print("-" * 70)
    display_limit = 20
    for i, img in enumerate(missing_images[:display_limit], 1):
        print(f"    {i}. {img}")
    if n_missing > display_limit:
        print(f"    ... and {n_missing - display_limit} more")
    print("-" * 70)

    # Confirm with the user (or proceed automatically)
    if not auto_confirm:
        print(f"\n  ⚠️  WARNING: proceeding will drop {n_missing} rows "
              f"({n_missing / total * 100:.1f}% of original data).")
        print(f"  Remaining samples after filtering: {n_existing}")
        while True:
            resp = input("\n  Drop missing rows and continue? [yes/no]: ").strip().lower()
            if resp in ("yes", "y"):
                break
            if resp in ("no", "n"):
                print("  Aborting — no splits created.")
                sys.exit(0)
            print("  Please type 'yes' or 'no'.")
    else:
        print(f"\n  Auto-confirm enabled. Dropping {n_missing} rows ...")

    filtered_df = labels_df.iloc[existing_indices].copy()
    print(f"  ✓ Filtered dataset: {len(filtered_df)} samples remaining.")
    print("=" * 70 + "\n")

    return filtered_df, missing_images


def save_missing_images_report(
    missing_images: List[str],
    output_dir: Path,
    filename: str = "missing_images_report.txt",
) -> None:
    """Write a text file listing all missing image IDs for reproducibility records."""
    if not missing_images:
        return
    report_path = output_dir / filename
    with open(report_path, "w") as f:
        f.write("Missing Images Report\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total missing: {len(missing_images)}\n")
        f.write("=" * 70 + "\n\n")
        for img in missing_images:
            f.write(f"{img}\n")
    print(f"  ✓ Missing images report saved → {report_path}")


# ===========================================================================
# Dataset loading and validation
# ===========================================================================

def load_and_validate_data(
    labels_csv: str,
    image_filename_column: str,
    target_column: str,
    year_column: str,
    image_source_dir: Optional[str] = None,
    image_extension: str = ".tif",
    auto_confirm_missing: bool = False,
    max_nodata_fraction: float = 0.05,
    validation_num_workers: int = 8,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load the master labels CSV, validate required columns, and optionally
    filter rows for which images are missing or of insufficient quality.

    Quality filtering steps (only when --image-source-dir is provided)
    -------------------------------------------------------------------
    1. Existence check via check_and_filter_missing_images
    2. White-fill / no-data fraction check via
       src.data.image_validation.filter_incomplete_images (parallelised)

    Parameters
    ----------
    labels_csv              Path to the master labels CSV
    image_filename_column   Column name for image filenames / IDs
    target_column           Column name for the SLB severity score
    year_column             Column name for the flight year
    image_source_dir        Directory containing image tiles; if None, skip
                            all image-level validation
    image_extension         Extension to resolve bare image IDs
    auto_confirm_missing    Skip interactive prompts (set True on HPC)
    max_nodata_fraction     Tiles with more white-fill than this fraction
                            are removed (default 5%)
    validation_num_workers  Threads for parallel white-fill scanning

    Returns
    -------
    df              Validated (and filtered) DataFrame
    missing_images  List of image IDs dropped by the existence check
    """
    print("\n" + "=" * 70)
    print("Loading and validating dataset")
    print("=" * 70)

    df = pd.read_csv(labels_csv)
    print(f"  Loaded {len(df)} rows from {labels_csv}")

    # Verify that all columns needed for CV0 are present
    required = [image_filename_column, target_column, year_column]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    print(f"  Required columns found: {required}")

    # Warn about any NaN values in key columns
    for col in required:
        n_na = df[col].isna().sum()
        if n_na > 0:
            print(f"  ⚠️  {n_na} missing values in column '{col}'")

    # Image validation (optional — skipped when no source directory given)
    missing_images: List[str] = []
    if image_source_dir is not None:
        print(f"\n  Image source directory: {image_source_dir}")

        # Step 1: existence check
        df, missing_images = check_and_filter_missing_images(
            labels_df=df,
            source_dir=image_source_dir,
            image_filename_column=image_filename_column,
            image_extension=image_extension,
            auto_confirm=auto_confirm_missing,
        )

        # Step 2: white-fill / no-data fraction check (parallelised)
        # Tiles at the orthomosaic boundary may be mostly white-filled;
        # these are identified and removed here before splits are created.
        df, _ = filter_incomplete_images(
            labels_df=df,
            image_dir=image_source_dir,
            image_filename_column=image_filename_column,
            image_extension=image_extension,
            max_nodata_fraction=max_nodata_fraction,
            num_workers=validation_num_workers,
            auto_confirm=auto_confirm_missing,
            report_dir=Path(labels_csv).parent,
        )

    # Print data distribution by year (informational only)
    print(f"\n  Data distribution by year:")
    for year in sorted(df[year_column].unique()):
        count = (df[year_column] == year).sum()
        print(f"    {year}: {count} samples")

    print("=" * 70 + "\n")
    return df, missing_images


# ===========================================================================
# Train / validation splitting
# ===========================================================================

def create_train_val_split(
    train_df: pd.DataFrame,
    target_column: str,
    val_ratio: float = 0.15,
    random_seed: int = 42,
    stratify: bool = True,
    n_bins: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a training DataFrame into train and validation subsets.

    Stratification is applied by binning the continuous SLB severity score
    into `n_bins` quantile bins, so that the severity distribution is
    approximately preserved in both splits.  Falls back to a simple random
    split if quantile binning fails (e.g., too few unique values in a small
    fold).

    Parameters
    ----------
    train_df       DataFrame to split
    target_column  Name of the SLB severity column
    val_ratio      Fraction of rows to place in the validation set (0–1)
    random_seed    Seed for reproducibility
    stratify       Whether to stratify by binned severity scores
    n_bins         Number of quantile bins used for stratification

    Returns
    -------
    train_split, val_split : DataFrames
    """
    if len(train_df) == 0:
        return train_df, pd.DataFrame()

    stratify_labels = None
    if stratify:
        try:
            # Bin the continuous severity scores into n_bins quantile intervals.
            # duplicates='drop' handles cases where multiple quantile edges
            # coincide (common at extreme score values).
            stratify_labels = pd.qcut(
                train_df[target_column], q=n_bins,
                labels=False, duplicates="drop",
            )
        except ValueError:
            print("  Warning: stratification failed (too few unique values); "
                  "using simple random split.")
            stratify_labels = None

    try:
        train_split, val_split = train_test_split(
            train_df,
            test_size=val_ratio,
            random_state=random_seed,
            stratify=stratify_labels,
        )
    except ValueError as exc:
        print(f"  Warning: stratified split failed ({exc}); retrying without stratification.")
        train_split, val_split = train_test_split(
            train_df,
            test_size=val_ratio,
            random_state=random_seed,
            stratify=None,
        )

    return train_split, val_split


# ===========================================================================
# CV0: leave-one-year-out
# ===========================================================================

def create_cv0_splits(
    df: pd.DataFrame,
    year_column: str,
    target_column: str,
    output_dir: Path,
    val_ratio: float = 0.15,
    random_seed: int = 42,
    stratify: bool = True,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Create CV0 (leave-one-year-out) splits for temporal generalizability testing.

    For each unique year in the dataset, one fold is created:
      - Test set  : all samples from that year (temporal hold-out)
      - Train+val : all samples from the remaining years, then split into
                    train (85%) and val (15%) by create_train_val_split

    Splits are written to disk in the standard layout expected by train_cv0.py:
      <output_dir>/cv0/fold_YYYY/metadata/{train,val,test}_labels.csv

    Parameters
    ----------
    df            Validated full-dataset DataFrame
    year_column   Column name for the flight year
    target_column Column name for the SLB severity score
    output_dir    Root output directory (cv0/ subdirectory created within it)
    val_ratio     Fraction of non-test data to use as validation
    random_seed   Seed for reproducibility
    stratify      Whether to stratify the train/val split

    Returns
    -------
    splits : dict mapping fold_name → {'train': df, 'val': df, 'test': df}
    """
    print("\n" + "=" * 70)
    print("Creating CV0 (Leave-One-Year-Out) Splits")
    print("=" * 70)

    years = sorted(df[year_column].unique())
    print(f"  Years found: {years}")
    print(f"  Folds to create: {len(years)}")

    cv0_dir = output_dir / "cv0"
    cv0_dir.mkdir(parents=True, exist_ok=True)

    splits: Dict[str, Dict[str, pd.DataFrame]] = {}

    for test_year in years:
        fold_name = f"fold_{test_year}"
        print(f"\n  [{fold_name}]  test year = {test_year}")

        # Test set: the entire held-out year
        test_df = df[df[year_column] == test_year].copy()

        # Training pool: all other years
        train_pool = df[df[year_column] != test_year].copy()

        print(f"    Training pool (all other years) : {len(train_pool)} samples")
        print(f"    Test set ({test_year})             : {len(test_df)} samples")

        # Split the training pool into train and validation subsets
        train_df, val_df = create_train_val_split(
            train_pool,
            target_column=target_column,
            val_ratio=val_ratio,
            random_seed=random_seed,
            stratify=stratify,
        )
        print(f"    After split — train : {len(train_df)}  val : {len(val_df)}")

        # Write CSVs to the expected directory layout
        fold_meta_dir = cv0_dir / fold_name / "metadata"
        fold_meta_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(fold_meta_dir / "train_labels.csv", index=False)
        val_df.to_csv(fold_meta_dir / "val_labels.csv",   index=False)
        test_df.to_csv(fold_meta_dir / "test_labels.csv",  index=False)

        print(f"    ✓ Saved to {fold_meta_dir}")

        splits[fold_name] = {"train": train_df, "val": val_df, "test": test_df}

    print("\n" + "=" * 70 + "\n")
    return splits


# ===========================================================================
# Summary report
# ===========================================================================

def generate_summary_report(
    output_dir: Path,
    cv0_splits: Dict[str, Dict[str, pd.DataFrame]],
    df: pd.DataFrame,
    year_column: str,
    target_column: str,
    missing_images: List[str],
) -> None:
    """
    Write a human-readable summary of the CV0 splits to a text file.

    The report records dataset statistics, per-fold sample counts, and
    any images that were dropped during validation.  It is saved alongside
    the splits for reproducibility documentation.
    """
    report_path = output_dir / "cv_splits_summary.txt"

    print("\n" + "=" * 70)
    print("Generating CV splits summary report ...")
    print("=" * 70)

    with open(report_path, "w") as f:
        # ── Dataset overview ─────────────────────────────────────────────────
        f.write("=" * 70 + "\n")
        f.write("CV0 SPLITS SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write("DATASET\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Total samples      : {len(df)}\n")
        f.write(f"  Years              : {sorted(df[year_column].unique())}\n")
        f.write(f"  Target column      : {target_column}\n")
        f.write(f"  Target range       : [{df[target_column].min():.2f}, "
                f"{df[target_column].max():.2f}]\n")
        f.write(f"  Target mean ± std  : {df[target_column].mean():.3f} ± "
                f"{df[target_column].std():.3f}\n")

        if missing_images:
            f.write(f"\n  ⚠️  {len(missing_images)} images were dropped "
                    f"(see missing_images_report.txt)\n")
        else:
            f.write("\n  ✓ All images present — no rows dropped.\n")

        f.write("\n")

        # ── CV0 fold breakdown ───────────────────────────────────────────────
        f.write("=" * 70 + "\n")
        f.write("CV0 FOLDS (Leave-One-Year-Out)\n")
        f.write("=" * 70 + "\n")
        f.write(f"  Total folds: {len(cv0_splits)}\n\n")

        for fold_name, fold_splits in cv0_splits.items():
            f.write(f"  {fold_name}:\n")
            f.write(f"    Train : {len(fold_splits['train']):>5} samples\n")
            f.write(f"    Val   : {len(fold_splits['val']):>5} samples\n")
            f.write(f"    Test  : {len(fold_splits['test']):>5} samples\n")
            if len(fold_splits["test"]) > 0:
                test_mean = fold_splits["test"][target_column].mean()
                test_std  = fold_splits["test"][target_column].std()
                f.write(f"    Test {target_column} : "
                        f"{test_mean:.3f} ± {test_std:.3f}\n")
            f.write("\n")

    print(f"  ✓ Summary report saved → {report_path}")
    print("=" * 70 + "\n")


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    """Command-line entry point for creating CV0 splits."""
    parser = argparse.ArgumentParser(
        description="Create CV0 (leave-one-year-out) splits for the UAV-SLB dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Required ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--labels-csv", type=str, required=True,
        help="Path to the master labels CSV (must contain year and score columns).",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Root directory where cv0/ subdirectory and summary files are written.",
    )

    # ── Image validation ─────────────────────────────────────────────────────
    parser.add_argument(
        "--image-source-dir", type=str, default=None,
        help=(
            "Directory containing image tiles. When provided, existence and "
            "white-fill quality checks are performed before creating splits."
        ),
    )
    parser.add_argument(
        "--image-extension", type=str, default=".tif",
        help="File extension to use when resolving bare image IDs (default: .tif).",
    )
    parser.add_argument(
        "--max-nodata-fraction", type=float, default=0.05,
        help=(
            "Tiles with a white-fill fraction above this threshold are dropped. "
            "Default: 0.05 (5%%). Only used when --image-source-dir is provided."
        ),
    )
    parser.add_argument(
        "--auto-confirm-missing", action="store_true",
        help="Drop missing / low-quality images without prompting (for HPC use).",
    )
    parser.add_argument(
        "--no-missing-report", action="store_true",
        help="Do not write the missing_images_report.txt file.",
    )
    parser.add_argument(
        "--validation-workers", type=int, default=8,
        help="Number of threads for parallel white-fill scanning (default: 8).",
    )

    # ── Column names ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--image-id-column", type=str, default="image_filename",
        help="Column name for image filenames / IDs (default: image_filename).",
    )
    parser.add_argument(
        "--target-column", type=str, default="score",
        help="Column name for the SLB severity score (default: score).",
    )
    parser.add_argument(
        "--year-column", type=str, default="year",
        help="Column name for the flight year (default: year).",
    )

    # ── Split parameters ──────────────────────────────────────────────────────
    parser.add_argument(
        "--val-ratio", type=float, default=0.15,
        help=(
            "Fraction of non-test samples to reserve for validation "
            "(default: 0.15 = 15%%)."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the train/val split (default: 42).",
    )
    parser.add_argument(
        "--no-stratify", action="store_true",
        help="Disable stratified splitting of the validation set.",
    )

    args = parser.parse_args()

    # ── Print configuration ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CV0 CROSS-VALIDATION SPLITS CREATION")
    print("=" * 70)
    print(f"  Input CSV          : {args.labels_csv}")
    print(f"  Output directory   : {args.output_dir}")
    print(f"  Validation ratio   : {args.val_ratio:.0%}")
    print(f"  Random seed        : {args.seed}")
    print(f"  Stratify val split : {not args.no_stratify}")
    if args.image_source_dir:
        print(f"  Image source dir   : {args.image_source_dir}")
        print(f"  Image extension    : {args.image_extension}")
        print(f"  Max white-fill     : {args.max_nodata_fraction:.0%}")
        print(f"  Auto-confirm       : {args.auto_confirm_missing}")
    else:
        print("  ⚠️  Image validation skipped (--image-source-dir not provided)")
    print("=" * 70)

    # ── Load and validate data ────────────────────────────────────────────────
    df, missing_images = load_and_validate_data(
        labels_csv=args.labels_csv,
        image_filename_column=args.image_id_column,
        target_column=args.target_column,
        year_column=args.year_column,
        image_source_dir=args.image_source_dir,
        image_extension=args.image_extension,
        auto_confirm_missing=args.auto_confirm_missing,
        max_nodata_fraction=args.max_nodata_fraction,
        validation_num_workers=args.validation_workers,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save missing-images report if any were detected
    if missing_images and not args.no_missing_report:
        save_missing_images_report(missing_images, output_dir)

    # ── Create CV0 splits ─────────────────────────────────────────────────────
    cv0_splits = create_cv0_splits(
        df=df,
        year_column=args.year_column,
        target_column=args.target_column,
        output_dir=output_dir,
        val_ratio=args.val_ratio,
        random_seed=args.seed,
        stratify=not args.no_stratify,
    )

    # ── Generate summary report ───────────────────────────────────────────────
    generate_summary_report(
        output_dir=output_dir,
        cv0_splits=cv0_splits,
        df=df,
        year_column=args.year_column,
        target_column=args.target_column,
        missing_images=missing_images,
    )

    # ── Final status ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("✓ CV0 SPLITS CREATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\n  {output_dir}/")
    print(f"    cv0/")
    for fold_name in sorted(cv0_splits.keys()):
        n_tr = len(cv0_splits[fold_name]["train"])
        n_va = len(cv0_splits[fold_name]["val"])
        n_te = len(cv0_splits[fold_name]["test"])
        print(f"      {fold_name}/metadata/  "
              f"[train={n_tr}  val={n_va}  test={n_te}]")
    print(f"    cv_splits_summary.txt")
    if missing_images and not args.no_missing_report:
        print(f"    missing_images_report.txt")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
