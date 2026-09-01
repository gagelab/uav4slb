#!/usr/bin/env python3
"""
create_cv_splits_downsample.py
===============================
Build CV0 (leave-one-year-out) train/val/test split files from a
per-year-downsampled version of the UAV-SLB dataset.

Requested by reviewers: the raw dataset has a very unbalanced number of
images per year (2023, 2024, 2025), so year is confounded with sample size.
This script randomly downsamples each year to the same number of plot
images before building the CV0 splits, so that any performance differences
across folds can't be attributed to year sample-size imbalance.

By default every year is downsampled to the size of the smallest year
(--samples-per-year lets you override this, e.g. to test a smaller common
size). Downsampling is a uniform random selection of rows per year, using
--seed for reproducibility, and is applied to the full labels table (after
image existence / quality filtering) before CV0 folds are constructed.

All other behavior (image validation, train/val split, output layout,
summary report) is identical to create_cv_splits.py; this script re-uses
that module's functions directly.

Outputs
-------
  <output_dir>/
    cv0/
      fold_2023/metadata/{train,val,test}_labels.csv
      fold_2024/metadata/{train,val,test}_labels.csv
      fold_2025/metadata/{train,val,test}_labels.csv
    cv_splits_summary.txt
    downsample_report.txt
    missing_images_report.txt     (only if any images are missing)
    low_quality_images_report.txt (only if any tiles fail the no-data check)

Usage
-----
  python scripts/create_cv_splits_downsample.py \
      --labels-csv data/labels/full_dataset.csv \
      --output-dir data/cv_splits/cv0_downsampled \
      --image-source-dir /mnt/research-projects/j/jlgage/RawUAVData01/uavforslb/final_image/final_sliced \
      --image-id-column image_filename \
      --target-column score \
      --year-column year \
      --auto-confirm-missing
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Project path bootstrap — allows running from any working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from create_cv_splits import (
    load_and_validate_data,
    save_missing_images_report,
    create_cv0_splits,
    generate_summary_report,
)


# ===========================================================================
# Per-year downsampling
# ===========================================================================

def downsample_by_year(
    df: pd.DataFrame,
    year_column: str,
    samples_per_year: Optional[int] = None,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, int]:
    """
    Randomly downsample each year to the same number of rows.

    Parameters
    ----------
    df                Validated full-dataset DataFrame
    year_column       Column name for the flight year
    samples_per_year  Number of rows to keep per year. Defaults to the size
                      of the smallest year in the dataset.
    random_seed       Seed for reproducible random sampling

    Returns
    -------
    downsampled_df, samples_per_year
    """
    print("\n" + "=" * 70)
    print("Downsampling to equal images per year")
    print("=" * 70)

    year_counts = df[year_column].value_counts().sort_index()
    print("  Available samples per year:")
    for year, count in year_counts.items():
        print(f"    {year}: {count}")

    if samples_per_year is None:
        samples_per_year = int(year_counts.min())
        print(f"\n  --samples-per-year not set; using smallest year: {samples_per_year}")
    else:
        too_small = year_counts[year_counts < samples_per_year]
        if not too_small.empty:
            raise ValueError(
                f"--samples-per-year={samples_per_year} exceeds the available "
                f"sample count for year(s): "
                f"{ {int(y): int(c) for y, c in too_small.items()} }"
            )

    sampled = pd.concat(
        [
            group.sample(n=samples_per_year, random_state=random_seed)
            for _, group in df.groupby(year_column)
        ],
        ignore_index=True,
    )

    print(f"\n  Downsampled to {samples_per_year} samples per year "
          f"({len(sampled)} total, seed={random_seed}).")
    print("=" * 70 + "\n")

    return sampled, samples_per_year


def save_downsample_report(
    output_dir: Path,
    original_df: pd.DataFrame,
    downsampled_df: pd.DataFrame,
    year_column: str,
    samples_per_year: int,
    random_seed: int,
) -> None:
    """Write a text report documenting the downsampling for reproducibility."""
    report_path = output_dir / "downsample_report.txt"
    original_counts = original_df[year_column].value_counts().sort_index()
    downsampled_counts = downsampled_df[year_column].value_counts().sort_index()

    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("DOWNSAMPLE REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"  Random seed          : {random_seed}\n")
        f.write(f"  Samples per year     : {samples_per_year}\n\n")
        f.write("  Year   Original   Downsampled\n")
        f.write("  " + "-" * 30 + "\n")
        for year in original_counts.index:
            f.write(f"  {year:<6} {original_counts[year]:>8}   "
                    f"{downsampled_counts.get(year, 0):>11}\n")
        f.write(f"\n  Total  {original_counts.sum():>8}   {downsampled_counts.sum():>11}\n")

    print(f"  ✓ Downsample report saved → {report_path}")


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    """Command-line entry point for creating downsampled CV0 splits."""
    parser = argparse.ArgumentParser(
        description=(
            "Create CV0 (leave-one-year-out) splits from a per-year-downsampled "
            "version of the UAV-SLB dataset"
        ),
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

    # ── Downsampling ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--samples-per-year", type=int, default=None,
        help=(
            "Number of images to keep per year. Defaults to the size of the "
            "smallest year in the dataset (after image validation)."
        ),
    )

    # ── Image validation ─────────────────────────────────────────────────────
    parser.add_argument(
        "--image-source-dir", type=str, default=None,
        help=(
            "Directory containing image tiles. When provided, existence and "
            "white-fill quality checks are performed before downsampling and "
            "creating splits."
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
        help="Random seed for downsampling and the train/val split (default: 42).",
    )
    parser.add_argument(
        "--no-stratify", action="store_true",
        help="Disable stratified splitting of the validation set.",
    )

    args = parser.parse_args()

    # ── Print configuration ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DOWNSAMPLED CV0 CROSS-VALIDATION SPLITS CREATION")
    print("=" * 70)
    print(f"  Input CSV          : {args.labels_csv}")
    print(f"  Output directory   : {args.output_dir}")
    print(f"  Samples per year   : {args.samples_per_year or 'auto (smallest year)'}")
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

    # ── Downsample to equal images per year ───────────────────────────────────
    downsampled_df, samples_per_year = downsample_by_year(
        df=df,
        year_column=args.year_column,
        samples_per_year=args.samples_per_year,
        random_seed=args.seed,
    )

    save_downsample_report(
        output_dir=output_dir,
        original_df=df,
        downsampled_df=downsampled_df,
        year_column=args.year_column,
        samples_per_year=samples_per_year,
        random_seed=args.seed,
    )

    # ── Create CV0 splits from the downsampled data ───────────────────────────
    cv0_splits = create_cv0_splits(
        df=downsampled_df,
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
        df=downsampled_df,
        year_column=args.year_column,
        target_column=args.target_column,
        missing_images=missing_images,
    )

    # ── Final status ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("✓ DOWNSAMPLED CV0 SPLITS CREATED SUCCESSFULLY")
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
    print(f"    downsample_report.txt")
    if missing_images and not args.no_missing_report:
        print(f"    missing_images_report.txt")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
