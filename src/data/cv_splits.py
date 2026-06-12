"""
Create cross-validation splits for UAV SLB disease prediction research.

This script generates fold splits for four cross-validation strategies:
- CV0: Leave-one-year-out (temporal generalization)
- CV1: Leave-one-population-out (genotype generalization)
- CV00: Leave-one-year-and-population-out (combined challenges)
- CV2: Incomplete trials k-fold (within-distribution interpolation)

Includes validation to check for missing images and filter them out before
creating splits.

Usage:

python scripts/create_cv_splits.py \
  --labels-csv /mnt/research-projects/j/jlgage/RawUAVData01/uavforslb/uav_for_slb/data/full_dataset.csv \
  --output-dir data/cv_splits \
  --image-source-dir /mnt/research-projects/j/jlgage/RawUAVData01/uavforslb/final_image/final_sliced \
  --image-id-column image_filename \
  --target-column score \
  --year-column year \
  --population-column pop \
  --image-extension .jpg \
  --cv-strategies cv0 \
  --auto-confirm-missing

Author: Cole Hamment
Date: 2025-01-16
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from typing import Dict, List, Tuple, Optional
import argparse
import sys
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.image_validation import filter_incomplete_images


def check_and_filter_missing_images(
    labels_df: pd.DataFrame,
    source_dir: str,
    image_filename_column: str,
    image_extension: str = '.tif',
    auto_confirm: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Check for missing images and optionally filter them out.
    
    This function checks which images in the labels CSV actually exist in the
    source directory. It displays missing images to the user and asks for
    confirmation before dropping those labels.
    f
    Args:
        labels_df: DataFrame with labels
        source_dir: Directory containing images
        image_filename_column: Column name for image IDs
        image_extension: Image file extension (e.g., '.tif', '.jpg')
        auto_confirm: If True, automatically proceed without user confirmation
        
    Returns:
        Tuple of (filtered_df, missing_images_list)
    """
    source_dir = Path(source_dir)
    missing_images = []
    existing_indices = []
    
    print("\n" + "="*70)
    print("Checking for missing images...")
    print("="*70)
    
    # Check each image
    for idx, row in labels_df.iterrows():
        image_filename = row[image_filename_column]
        
        # Handle case where image_filename already has extension
        if Path(image_filename).suffix:
            image_path = source_dir / image_filename
        else:
            # Try with specified extension
            image_path = source_dir / f"{image_filename}{image_extension}"
            
            # If not found, try other common extensions
            if not image_path.exists():
                found = False
                for ext in ['.tif', '.tiff', '.jpg', '.jpeg', '.png']:
                    alt_path = source_dir / f"{Path(image_filename).stem}{ext}"
                    if alt_path.exists():
                        image_path = alt_path
                        found = True
                        break
                
                if not found:
                    missing_images.append(image_filename)
                    continue
        
        if image_path.exists():
            existing_indices.append(idx)
        else:
            missing_images.append(image_filename)
    
    # Report findings
    total_images = len(labels_df)
    missing_count = len(missing_images)
    existing_count = len(existing_indices)
    
    print(f"\nTotal labels in CSV: {total_images}")
    print(f"Images found: {existing_count}")
    print(f"Images missing: {missing_count}")
    
    if missing_count == 0:
        print("✓ All images found! No filtering needed.")
        print("="*70 + "\n")
        return labels_df, []
    
    # Display missing images
    print(f"\n⚠️  Missing images ({missing_count} total):")
    print("-"*70)
    
    # Show first 20, then summarize if more
    display_limit = 20
    for i, img in enumerate(missing_images[:display_limit], 1):
        print(f"  {i}. {img}")
    
    if missing_count > display_limit:
        print(f"  ... and {missing_count - display_limit} more")
        print(f"\n  [First {display_limit} of {missing_count} missing images shown above]")
    
    print("-"*70)
    
    # Get user confirmation
    if not auto_confirm:
        print(f"\n⚠️  WARNING: This will drop {missing_count} labels from your dataset!")
        print(f"Remaining samples after filtering: {existing_count}")
        print(f"Loss: {missing_count/total_images*100:.1f}% of original data")
        
        while True:
            response = input("\nDo you want to proceed with dropping these labels? (yes/no): ").strip().lower()
            if response in ['yes', 'y']:
                print("Proceeding with filtered dataset...")
                break
            elif response in ['no', 'n']:
                print("Aborting. No splits created.")
                print("Please check your image directory and CSV file.")
                sys.exit(0)
            else:
                print("Please enter 'yes' or 'no'")
    else:
        print(f"\nAuto-confirm enabled. Proceeding with filtered dataset...")
    
    # Filter dataframe
    filtered_df = labels_df.iloc[existing_indices].copy()
    
    print(f"✓ Filtered dataset created: {len(filtered_df)} samples")
    print("="*70 + "\n")
    
    return filtered_df, missing_images


def save_missing_images_report(
    missing_images: List[str],
    output_dir: Path,
    filename: str = 'missing_images_report.txt'
) -> None:
    """
    Save a report of missing images to a text file.
    
    Args:
        missing_images: List of missing image filenames
        output_dir: Directory to save the report
        filename: Name of the report file
    """
    if not missing_images:
        return
    
    report_path = output_dir / filename
    
    with open(report_path, 'w') as f:
        f.write(f"Missing Images Report\n")
        f.write(f"{'='*70}\n")
        f.write(f"Total missing: {len(missing_images)}\n")
        f.write(f"{'='*70}\n\n")
        
        for img in missing_images:
            f.write(f"{img}\n")
    
    print(f"✓ Missing images report saved to: {report_path}")


def load_and_validate_data(
    labels_csv: str,
    image_filename_column: str,
    target_column: str,
    year_column: str,
    population_column: str,
    required_columns: Optional[List[str]] = None,
    image_source_dir: Optional[str] = None,
    image_extension: str = '.tif',
    auto_confirm_missing: bool = False,
    save_missing_report: bool = True,
    max_nodata_fraction: float = 0.05,
    validation_num_workers: int = 8
) -> Tuple[pd.DataFrame, List[str]]:

    """
    Load and validate the dataset, optionally checking for missing images.
    
    Args:
        labels_csv: Path to CSV with labels
        image_filename_column: Column name for image IDs
        target_column: Column name for target variable
        year_column: Column name for year information
        population_column: Column name for population/genotype information
        required_columns: Additional required columns
        image_source_dir: Directory containing images (if None, skip image checking)
        image_extension: Image file extension for checking
        auto_confirm_missing: If True, automatically drop missing/invalid images
        save_missing_report: If True, save report of missing images
        max_nodata_fraction: Images with more white-fill than this fraction are removed
        validation_num_workers: Parallel threads for image scanning

    Returns:
        Tuple of (validated_df, missing_images_list)
    """
    print("\n" + "="*70)
    print("Loading and validating dataset")
    print("="*70)
    
    # Load data
    df = pd.read_csv(labels_csv)
    print(f"Loaded {len(df)} samples from {labels_csv}")
    
    # Check required columns
    required = [image_filename_column, target_column, year_column, population_column]
    if required_columns:
        required.extend(required_columns)
    
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Found all required columns: {required}")
    
    # Check for missing values in key columns
    for col in required:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            print(f"⚠️  Warning: {n_missing} missing values in '{col}'")
    
    # Check for missing images if source directory provided
    missing_images = []
    if image_source_dir is not None:
        print(f"\nImage source directory: {image_source_dir}")
        df, missing_images = check_and_filter_missing_images(
            labels_df=df,
            source_dir=image_source_dir,
            image_filename_column=image_filename_column,
            image_extension=image_extension,
            auto_confirm=auto_confirm_missing
        )
    
    if image_source_dir is not None:
        report_dir = Path(labels_csv).parent
        df, _ = filter_incomplete_images(
            labels_df=df,
            image_dir=image_source_dir,
            image_filename_column=image_filename_column,
            image_extension=image_extension,
            max_nodata_fraction=max_nodata_fraction,
            num_workers=validation_num_workers,
            auto_confirm=auto_confirm_missing,
            report_dir=report_dir,
        )
    
    # Display data distribution
    print(f"\nData distribution:")
    print(f"  Years: {sorted(df[year_column].unique())}")
    print(f"  Populations: {sorted(df[population_column].unique())}")
    print(f"  Samples per year:")
    for year in sorted(df[year_column].unique()):
        count = (df[year_column] == year).sum()
        print(f"    {year}: {count}")
    print(f"  Samples per population:")
    for pop in sorted(df[population_column].unique()):
        count = (df[population_column] == pop).sum()
        print(f"    {pop}: {count}")
    
    print("="*70 + "\n")
    return df, missing_images


def create_train_val_split(
    train_df: pd.DataFrame,
    target_column: str,
    val_ratio: float = 0.15,
    random_seed: int = 42,
    stratify: bool = True,
    n_bins: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split training data into train and validation sets.
    
    Args:
        train_df: Training DataFrame
        target_column: Name of target column
        val_ratio: Proportion for validation set
        random_seed: Random seed for reproducibility
        stratify: Whether to stratify by target values
        n_bins: Number of bins for stratifying continuous targets
        
    Returns:
        Tuple of (train_df, val_df)
    """
    if len(train_df) == 0:
        return train_df, pd.DataFrame()
    
    # Determine stratification
    stratify_labels = None
    if stratify:
        if train_df[target_column].dtype in [np.float32, np.float64]:
            # Regression: bin values for stratification
            try:
                stratify_labels = pd.qcut(
                    train_df[target_column],
                    q=n_bins,
                    labels=False,
                    duplicates='drop'
                )
            except ValueError:
                # If qcut fails (e.g., too few unique values), don't stratify
                print(f"  Warning: Could not stratify - insufficient unique values")
                stratify_labels = None
        else:
            # Classification: use labels directly
            stratify_labels = train_df[target_column]
    
    # Split
    try:
        train_split, val_split = train_test_split(
            train_df,
            test_size=val_ratio,
            random_state=random_seed,
            stratify=stratify_labels
        )
        return train_split, val_split
    except ValueError as e:
        print(f"  Warning: Stratified split failed ({e}), using random split")
        train_split, val_split = train_test_split(
            train_df,
            test_size=val_ratio,
            random_state=random_seed,
            stratify=None
        )
        return train_split, val_split


def create_cv0_splits(
    df: pd.DataFrame,
    year_column: str,
    target_column: str,
    output_dir: Path,
    val_ratio: float = 0.15,
    random_seed: int = 42,
    stratify: bool = True
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Create CV0 (leave-one-year-out) splits for temporal generalization.
    
    Args:
        df: Full dataset
        year_column: Column name for year
        target_column: Column name for target variable
        output_dir: Output directory for splits
        val_ratio: Validation set ratio within training data
        random_seed: Random seed
        stratify: Whether to stratify validation split
        
    Returns:
        Dictionary of fold splits
    """
    print("\n" + "="*70)
    print("Creating CV0 (Leave-One-Year-Out) Splits")
    print("="*70)
    
    years = sorted(df[year_column].unique())
    print(f"Years found: {years}")
    
    cv0_dir = output_dir / 'cv0'
    cv0_dir.mkdir(parents=True, exist_ok=True)
    
    splits = {}
    
    for test_year in years:
        fold_name = f"fold_{test_year}"
        print(f"\nCreating fold: {fold_name}")
        print(f"  Test year: {test_year}")
        
        # Create test set (held-out year)
        test_df = df[df[year_column] == test_year].copy()
        
        # Create training set (all other years)
        train_full_df = df[df[year_column] != test_year].copy()
        
        print(f"  Train samples (all other years): {len(train_full_df)}")
        print(f"  Test samples ({test_year}): {len(test_df)}")
        
        # Split training into train/val
        train_df, val_df = create_train_val_split(
            train_full_df,
            target_column=target_column,
            val_ratio=val_ratio,
            random_seed=random_seed,
            stratify=stratify
        )
        
        print(f"  Final train: {len(train_df)}")
        print(f"  Final val: {len(val_df)}")
        
        # Save splits
        fold_dir = cv0_dir / fold_name / 'metadata'
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        train_df.to_csv(fold_dir / 'train_labels.csv', index=False)
        val_df.to_csv(fold_dir / 'val_labels.csv', index=False)
        test_df.to_csv(fold_dir / 'test_labels.csv', index=False)
        
        print(f"  ✓ Saved to {fold_dir}")
        
        # Store in memory
        splits[fold_name] = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
    
    print("="*70 + "\n")
    return splits


def create_cv1_splits(
    df: pd.DataFrame,
    population_column: str,
    target_column: str,
    output_dir: Path,
    val_ratio: float = 0.15,
    random_seed: int = 42,
    stratify: bool = True,
    min_samples_per_pop: int = 10
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Create CV1 (leave-one-population-out) splits for genotype generalization.
    
    Args:
        df: Full dataset
        population_column: Column name for population
        target_column: Column name for target variable
        output_dir: Output directory for splits
        val_ratio: Validation set ratio within training data
        random_seed: Random seed
        stratify: Whether to stratify validation split
        min_samples_per_pop: Minimum samples per population to create fold
        
    Returns:
        Dictionary of fold splits
    """
    print("\n" + "="*70)
    print("Creating CV1 (Leave-One-Population-Out) Splits")
    print("="*70)
    
    populations = sorted(df[population_column].unique())
    print(f"Populations found: {populations}")
    
    # Check sample counts per population
    pop_counts = df[population_column].value_counts()
    print(f"\nSamples per population:")
    for pop in populations:
        count = pop_counts.get(pop, 0)
        print(f"  {pop}: {count} samples")
    
    cv1_dir = output_dir / 'cv1'
    cv1_dir.mkdir(parents=True, exist_ok=True)
    
    splits = {}
    skipped_pops = []
    
    for test_pop in populations:
        test_pop_count = pop_counts.get(test_pop, 0)
        
        # Skip populations with too few samples
        if test_pop_count < min_samples_per_pop:
            print(f"\n⚠️  Skipping {test_pop}: only {test_pop_count} samples (min: {min_samples_per_pop})")
            skipped_pops.append(test_pop)
            continue
        
        fold_name = f"fold_{test_pop}"
        print(f"\nCreating fold: {fold_name}")
        print(f"  Test population: {test_pop}")
        
        # Create test set (held-out population)
        test_df = df[df[population_column] == test_pop].copy()
        
        # Create training set (all other populations)
        train_full_df = df[df[population_column] != test_pop].copy()
        
        print(f"  Train samples (all other populations): {len(train_full_df)}")
        print(f"  Test samples ({test_pop}): {len(test_df)}")
        
        # Split training into train/val
        train_df, val_df = create_train_val_split(
            train_full_df,
            target_column=target_column,
            val_ratio=val_ratio,
            random_seed=random_seed,
            stratify=stratify
        )
        
        print(f"  Final train: {len(train_df)}")
        print(f"  Final val: {len(val_df)}")
        
        # Save splits
        fold_dir = cv1_dir / fold_name / 'metadata'
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        train_df.to_csv(fold_dir / 'train_labels.csv', index=False)
        val_df.to_csv(fold_dir / 'val_labels.csv', index=False)
        test_df.to_csv(fold_dir / 'test_labels.csv', index=False)
        
        print(f"  ✓ Saved to {fold_dir}")
        
        # Store in memory
        splits[fold_name] = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
    
    if skipped_pops:
        print(f"\n⚠️  Skipped {len(skipped_pops)} populations with insufficient samples:")
        for pop in skipped_pops:
            print(f"    {pop}")
    
    print("="*70 + "\n")
    return splits


def create_cv00_splits(
    df: pd.DataFrame,
    year_column: str,
    population_column: str,
    target_column: str,
    output_dir: Path,
    val_ratio: float = 0.15,
    random_seed: int = 42,
    stratify: bool = True,
    min_samples_per_fold: int = 50
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Create CV00 (leave-one-year-and-population-out) splits for combined challenges.
    
    Args:
        df: Full dataset
        year_column: Column name for year
        population_column: Column name for population
        target_column: Column name for target variable
        output_dir: Output directory for splits
        val_ratio: Validation set ratio within training data
        random_seed: Random seed
        stratify: Whether to stratify validation split
        min_samples_per_fold: Minimum training samples to create fold
        
    Returns:
        Dictionary of fold splits
    """
    print("\n" + "="*70)
    print("Creating CV00 (Leave-One-Year-and-Population-Out) Splits")
    print("="*70)
    
    years = sorted(df[year_column].unique())
    populations = sorted(df[population_column].unique())
    
    print(f"Years: {years}")
    print(f"Populations: {populations}")
    
    cv00_dir = output_dir / 'cv00'
    cv00_dir.mkdir(parents=True, exist_ok=True)
    
    splits = {}
    skipped_combinations = []
    
    for test_year in years:
        for test_pop in populations:
            fold_name = f"fold_{test_year}_{test_pop}"
            
            # Create test set (specific year-population combination)
            test_df = df[
                (df[year_column] == test_year) & 
                (df[population_column] == test_pop)
            ].copy()
            
            # Create training set (exclude test year AND test population)
            train_full_df = df[
                (df[year_column] != test_year) & 
                (df[population_column] != test_pop)
            ].copy()
            
            # Skip if test set is empty or training set is too small
            if len(test_df) == 0:
                skipped_combinations.append((test_year, test_pop, "no_test_samples"))
                continue
            
            if len(train_full_df) < min_samples_per_fold:
                skipped_combinations.append((test_year, test_pop, f"insufficient_train_{len(train_full_df)}"))
                continue
            
            print(f"\nCreating fold: {fold_name}")
            print(f"  Test year: {test_year}, Test population: {test_pop}")
            print(f"  Train samples: {len(train_full_df)}")
            print(f"  Test samples: {len(test_df)}")
            
            # Split training into train/val
            train_df, val_df = create_train_val_split(
                train_full_df,
                target_column=target_column,
                val_ratio=val_ratio,
                random_seed=random_seed,
                stratify=stratify
            )
            
            print(f"  Final train: {len(train_df)}")
            print(f"  Final val: {len(val_df)}")
            
            # Save splits
            fold_dir = cv00_dir / fold_name / 'metadata'
            fold_dir.mkdir(parents=True, exist_ok=True)
            
            train_df.to_csv(fold_dir / 'train_labels.csv', index=False)
            val_df.to_csv(fold_dir / 'val_labels.csv', index=False)
            test_df.to_csv(fold_dir / 'test_labels.csv', index=False)
            
            print(f"  ✓ Saved to {fold_dir}")
            
            # Store in memory
            splits[fold_name] = {
                'train': train_df,
                'val': val_df,
                'test': test_df
            }
    
    if skipped_combinations:
        print(f"\n⚠️  Skipped {len(skipped_combinations)} year-population combinations:")
        for year, pop, reason in skipped_combinations[:10]:  # Show first 10
            print(f"    {year}-{pop}: {reason}")
        if len(skipped_combinations) > 10:
            print(f"    ... and {len(skipped_combinations) - 10} more")
    
    print("="*70 + "\n")
    return splits


def create_cv2_splits(
    df: pd.DataFrame,
    target_column: str,
    output_dir: Path,
    n_folds: int = 5,
    val_ratio: float = 0.15,
    random_seed: int = 42,
    stratify: bool = True,
    n_bins: int = 5
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Create CV2 (incomplete trials / k-fold) splits for within-distribution interpolation.

    Randomly withholds image-score pairs across the full dataset using k-fold
    cross-validation. Unlike CV0/CV1, no entire year or population is excluded,
    so this tests whether the model can interpolate within its training distribution.

    Args:
        df: Full dataset
        target_column: Column name for target variable
        output_dir: Output directory for splits
        n_folds: Number of folds (default: 5)
        val_ratio: Validation set ratio within the per-fold training data
        random_seed: Random seed for reproducibility
        stratify: Whether to stratify folds by binned target values
        n_bins: Number of bins used when stratifying a continuous target

    Returns:
        Dictionary of fold splits
    """
    print("\n" + "="*70)
    print(f"Creating CV2 (Incomplete Trials / {n_folds}-Fold) Splits")
    print("="*70)
    print(f"Total samples: {len(df)}")

    cv2_dir = output_dir / 'cv2'
    cv2_dir.mkdir(parents=True, exist_ok=True)

    # Build stratification labels for the k-fold splitter
    fold_stratify_labels = None
    if stratify:
        if df[target_column].dtype in [np.float32, np.float64]:
            try:
                fold_stratify_labels = pd.qcut(
                    df[target_column],
                    q=n_bins,
                    labels=False,
                    duplicates='drop'
                ).values
            except ValueError:
                print("  Warning: Could not stratify k-fold — insufficient unique values")
        else:
            fold_stratify_labels = df[target_column].values

    if fold_stratify_labels is not None:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        split_iter = splitter.split(df, fold_stratify_labels)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        split_iter = splitter.split(df)

    splits = {}

    for fold_idx, (train_val_idx, test_idx) in enumerate(split_iter):
        fold_name = f"fold_{fold_idx}"
        print(f"\nCreating fold: {fold_name}")

        test_df = df.iloc[test_idx].copy()
        train_full_df = df.iloc[train_val_idx].copy()

        print(f"  Train+val samples: {len(train_full_df)}")
        print(f"  Test samples:      {len(test_df)}")

        train_df, val_df = create_train_val_split(
            train_full_df,
            target_column=target_column,
            val_ratio=val_ratio,
            random_seed=random_seed,
            stratify=stratify,
            n_bins=n_bins
        )

        print(f"  Final train: {len(train_df)}")
        print(f"  Final val:   {len(val_df)}")

        fold_dir = cv2_dir / fold_name / 'metadata'
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(fold_dir / 'train_labels.csv', index=False)
        val_df.to_csv(fold_dir / 'val_labels.csv', index=False)
        test_df.to_csv(fold_dir / 'test_labels.csv', index=False)

        print(f"  ✓ Saved to {fold_dir}")

        splits[fold_name] = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }

    print("="*70 + "\n")
    return splits


def generate_summary_report(
    output_dir: Path,
    cv0_splits: Dict,
    cv1_splits: Dict,
    cv00_splits: Dict,
    cv2_splits: Dict,
    df: pd.DataFrame,
    year_column: str,
    population_column: str,
    target_column: str,
    missing_images: List[str]
) -> None:
    """Generate a summary report of all splits."""
    report_path = output_dir / 'cv_splits_summary.txt'
    
    print("\n" + "="*70)
    print("Generating summary report")
    print("="*70)
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CROSS-VALIDATION SPLITS SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        # Dataset info
        f.write("DATASET INFORMATION\n")
        f.write("-"*70 + "\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Years: {sorted(df[year_column].unique())}\n")
        f.write(f"Populations: {sorted(df[population_column].unique())}\n")
        f.write(f"Target column: {target_column}\n")
        f.write(f"Target range: [{df[target_column].min():.2f}, {df[target_column].max():.2f}]\n")
        f.write(f"Target mean: {df[target_column].mean():.2f}\n")
        f.write(f"Target std: {df[target_column].std():.2f}\n")
        
        # Missing images info
        if missing_images:
            f.write(f"\n⚠️  Missing images: {len(missing_images)}\n")
            f.write(f"These images were filtered out before creating splits.\n")
            f.write(f"See missing_images_report.txt for full list.\n")
        else:
            f.write(f"\n✓ All images present\n")
        
        f.write("\n")
        
        # CV0 summary
        if cv0_splits:
            f.write("="*70 + "\n")
            f.write("CV0: LEAVE-ONE-YEAR-OUT SPLITS\n")
            f.write("="*70 + "\n")
            f.write(f"Total folds: {len(cv0_splits)}\n\n")
            
            for fold_name, splits in cv0_splits.items():
                f.write(f"{fold_name}:\n")
                f.write(f"  Train: {len(splits['train']):>5} samples\n")
                f.write(f"  Val:   {len(splits['val']):>5} samples\n")
                f.write(f"  Test:  {len(splits['test']):>5} samples\n")
                if len(splits['test']) > 0:
                    f.write(f"  Test {target_column} mean: {splits['test'][target_column].mean():.3f}\n")
                f.write("\n")
        
        # CV1 summary
        if cv1_splits:
            f.write("="*70 + "\n")
            f.write("CV1: LEAVE-ONE-POPULATION-OUT SPLITS\n")
            f.write("="*70 + "\n")
            f.write(f"Total folds: {len(cv1_splits)}\n\n")
            
            for fold_name, splits in cv1_splits.items():
                f.write(f"{fold_name}:\n")
                f.write(f"  Train: {len(splits['train']):>5} samples\n")
                f.write(f"  Val:   {len(splits['val']):>5} samples\n")
                f.write(f"  Test:  {len(splits['test']):>5} samples\n")
                if len(splits['test']) > 0:
                    f.write(f"  Test {target_column} mean: {splits['test'][target_column].mean():.3f}\n")
                f.write("\n")
        
        # CV00 summary
        if cv00_splits:
            f.write("="*70 + "\n")
            f.write("CV00: LEAVE-ONE-YEAR-AND-POPULATION-OUT SPLITS\n")
            f.write("="*70 + "\n")
            f.write(f"Total folds: {len(cv00_splits)}\n\n")
            
            for fold_name, splits in cv00_splits.items():
                f.write(f"{fold_name}:\n")
                f.write(f"  Train: {len(splits['train']):>5} samples\n")
                f.write(f"  Val:   {len(splits['val']):>5} samples\n")
                f.write(f"  Test:  {len(splits['test']):>5} samples\n")
                if len(splits['test']) > 0:
                    f.write(f"  Test {target_column} mean: {splits['test'][target_column].mean():.3f}\n")
                f.write("\n")
        
        # CV2 summary
        if cv2_splits:
            f.write("="*70 + "\n")
            f.write("CV2: INCOMPLETE TRIALS (K-FOLD) SPLITS\n")
            f.write("="*70 + "\n")
            f.write(f"Total folds: {len(cv2_splits)}\n\n")

            for fold_name, splits in cv2_splits.items():
                f.write(f"{fold_name}:\n")
                f.write(f"  Train: {len(splits['train']):>5} samples\n")
                f.write(f"  Val:   {len(splits['val']):>5} samples\n")
                f.write(f"  Test:  {len(splits['test']):>5} samples\n")
                if len(splits['test']) > 0:
                    f.write(f"  Test {target_column} mean: {splits['test'][target_column].mean():.3f}\n")
                f.write("\n")

        # Summary statistics
        f.write("="*70 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("="*70 + "\n")

        total_folds = len(cv0_splits) + len(cv1_splits) + len(cv00_splits) + len(cv2_splits)
        f.write(f"Total folds created: {total_folds}\n")
        f.write(f"  CV0:  {len(cv0_splits)} folds\n")
        f.write(f"  CV1:  {len(cv1_splits)} folds\n")
        f.write(f"  CV00: {len(cv00_splits)} folds\n")
        f.write(f"  CV2:  {len(cv2_splits)} folds\n")
        
        f.write("\n")
        f.write("="*70 + "\n")
    
    print(f"✓ Summary report saved to: {report_path}")
    print("="*70 + "\n")


def main():
    """Command-line interface for creating CV splits."""
    parser = argparse.ArgumentParser(
        description='Create cross-validation splits for UAV SLB dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python create_cv_splits.py \\
    --labels-csv /path/to/full_dataset.csv \\
    --output-dir /path/to/cv_splits \\
    --image-source-dir /path/to/images \\
    --image-id-column image_filename \\
    --target-column score \\
    --year-column year \\
    --population-column population \\
    --cv-strategies cv0 cv1 cv00

  # Create CV0 splits only with image validation
  python src/data/create_cv_splits.py \\
    --labels-csv /mnt/research-projects/j/jlgage/RawUAVData01/uavforslb/final_image/clean_scores_with_images_dedup.csv \\
    --output-dir data/cv_splits \\
    --image-source-dir /mnt/research-projects/j/jlgage/RawUAVData01/uavforslb/final_image \\
    --image-id-column image_filename \\
    --target-column score \\
    --year-column year \\
    --population-column pop \\
    --image-extension .tif \\
    --cv-strategies cv0 \\
    --auto-confirm-missing
    
This will create three sets of cross-validation splits:
  - CV0: Leave-one-year-out (temporal generalization)
  - CV1: Leave-one-population-out (genotype generalization)
  - CV00: Leave-one-year-and-population-out (combined challenges)
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--labels-csv',
        type=str,
        required=True,
        help='Path to CSV file with full dataset labels'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for CV splits'
    )
    
    # Image validation arguments
    parser.add_argument(
        '--image-source-dir',
        type=str,
        default=None,
        help='Directory containing images (if provided, will check for missing images)'
    )
    
    parser.add_argument(
        '--image-extension',
        type=str,
        default='.tif',
        help='Image file extension (default: .tif)'
    )
    
    parser.add_argument(
        '--auto-confirm-missing',
        action='store_true',
        help='Automatically drop missing images without prompting'
    )
    
    parser.add_argument(
        '--no-missing-report',
        action='store_true',
        help='Do not save a report of missing images'
    )
    
    # Column name arguments
    parser.add_argument(
        '--image-id-column',
        type=str,
        default='image_filename',
        help='Column name for image IDs (default: image_filename)'
    )
    
    parser.add_argument(
        '--target-column',
        type=str,
        default='score',
        help='Column name for target variable (default: score)'
    )
    
    parser.add_argument(
        '--year-column',
        type=str,
        default='year',
        help='Column name for year information (default: year)'
    )
    
    parser.add_argument(
        '--population-column',
        type=str,
        default='pop',
        help='Column name for population/genotype (default: population)'
    )
    
    # CV strategy selection
    parser.add_argument(
        '--cv-strategies',
        nargs='+',
        choices=['cv0', 'cv1', 'cv00', 'cv2', 'all'],
        default=['all'],
        help='Which CV strategies to create (default: all)'
    )
    
    # Split parameters
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio within training data (default: 0.15)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--no-stratify',
        action='store_true',
        help='Disable stratified splitting for validation sets'
    )
    
    # CV1 parameters
    parser.add_argument(
        '--min-samples-per-population',
        type=int,
        default=10,
        help='Minimum samples per population to create CV1 fold (default: 10)'
    )
    
    # CV00 parameters
    parser.add_argument(
        '--min-samples-per-cv00-fold',
        type=int,
        default=50,
        help='Minimum training samples for CV00 fold (default: 50)'
    )

    parser.add_argument(
        '--validation-workers',
        type=int,
        default=8,
        help='Threads for parallel image white-fill validation (default: 8).'
    )

    # CV2 parameters
    parser.add_argument(
        '--cv2-n-folds',
        type=int,
        default=5,
        help='Number of folds for CV2 incomplete-trials strategy (default: 5)'
    )

    parser.add_argument(
        "--max-nodata-fraction",
        type=float,
        default=0.05,
        help=(
            "Images with more than this fraction of white-fill pixels are "
            "removed. Default: 0.05 (5%%). Only used when --image-source-dir "
            "is provided."
        ),
    )
    
    args = parser.parse_args()
    
    # Determine which strategies to run
    if 'all' in args.cv_strategies:
        strategies = ['cv0', 'cv1', 'cv00', 'cv2']
    else:
        strategies = args.cv_strategies
    
    print("\n" + "="*70)
    print("CROSS-VALIDATION SPLITS CREATION")
    print("="*70)
    print(f"Input CSV: {args.labels_csv}")
    print(f"Output directory: {args.output_dir}")
    print(f"CV strategies: {', '.join(strategies)}")
    print(f"Validation ratio: {args.val_ratio}")
    print(f"Random seed: {args.seed}")
    if args.image_source_dir:
        print(f"Image source directory: {args.image_source_dir}")
        print(f"Image extension: {args.image_extension}")
        print(f"Max white-fill fraction: {args.max_nodata_fraction:.1%}")
        print(f"Auto-confirm missing/invalid: {args.auto_confirm_missing}")
    else:
        print("⚠️  No image validation (--image-source-dir not provided)")
    print("="*70)
    
    # Load and validate data
    df, missing_images = load_and_validate_data(
        labels_csv=args.labels_csv,
        image_filename_column=args.image_filename_column,
        target_column=args.target_column,
        year_column=args.year_column,
        population_column=args.population_column,
        image_source_dir=args.image_source_dir,
        image_extension=args.image_extension,
        auto_confirm_missing=args.auto_confirm_missing,
        save_missing_report=not args.no_missing_report,
        max_nodata_fraction=args.max_nodata_fraction,
        validation_num_workers=args.validation_workers,
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save missing images report if there are any
    if missing_images and not args.no_missing_report:
        save_missing_images_report(missing_images, output_dir)
    
    # Create splits for each strategy
    cv0_splits = {}
    cv1_splits = {}
    cv00_splits = {}
    cv2_splits = {}
    
    if 'cv0' in strategies:
        cv0_splits = create_cv0_splits(
            df=df,
            year_column=args.year_column,
            target_column=args.target_column,
            output_dir=output_dir,
            val_ratio=args.val_ratio,
            random_seed=args.seed,
            stratify=not args.no_stratify
        )
    
    if 'cv1' in strategies:
        cv1_splits = create_cv1_splits(
            df=df,
            population_column=args.population_column,
            target_column=args.target_column,
            output_dir=output_dir,
            val_ratio=args.val_ratio,
            random_seed=args.seed,
            stratify=not args.no_stratify,
            min_samples_per_pop=args.min_samples_per_population
        )
    
    if 'cv00' in strategies:
        cv00_splits = create_cv00_splits(
            df=df,
            year_column=args.year_column,
            population_column=args.population_column,
            target_column=args.target_column,
            output_dir=output_dir,
            val_ratio=args.val_ratio,
            random_seed=args.seed,
            stratify=not args.no_stratify,
            min_samples_per_fold=args.min_samples_per_cv00_fold
        )

    if 'cv2' in strategies:
        cv2_splits = create_cv2_splits(
            df=df,
            target_column=args.target_column,
            output_dir=output_dir,
            n_folds=args.cv2_n_folds,
            val_ratio=args.val_ratio,
            random_seed=args.seed,
            stratify=not args.no_stratify
        )

    # Generate summary report
    generate_summary_report(
        output_dir=output_dir,
        cv0_splits=cv0_splits,
        cv1_splits=cv1_splits,
        cv00_splits=cv00_splits,
        cv2_splits=cv2_splits,
        df=df,
        year_column=args.year_column,
        population_column=args.population_column,
        target_column=args.target_column,
        missing_images=missing_images
    )
    
    print("\n" + "="*70)
    print("✓ ALL CROSS-VALIDATION SPLITS CREATED SUCCESSFULLY")
    print("="*70)
    print(f"\nOutput directory structure:")
    print(f"  {output_dir}/")
    if cv0_splits:
        print(f"    cv0/")
        for fold_name in sorted(cv0_splits.keys()):
            print(f"      {fold_name}/metadata/")
    if cv1_splits:
        print(f"    cv1/")
        for fold_name in sorted(list(cv1_splits.keys())[:3]):  # Show first 3
            print(f"      {fold_name}/metadata/")
        if len(cv1_splits) > 3:
            print(f"      ... and {len(cv1_splits) - 3} more folds")
    if cv00_splits:
        print(f"    cv00/")
        for fold_name in sorted(list(cv00_splits.keys())[:3]):  # Show first 3
            print(f"      {fold_name}/metadata/")
        if len(cv00_splits) > 3:
            print(f"      ... and {len(cv00_splits) - 3} more folds")
    if cv2_splits:
        print(f"    cv2/")
        for fold_name in sorted(cv2_splits.keys()):
            print(f"      {fold_name}/metadata/")
    print(f"    cv_splits_summary.txt")
    if missing_images and not args.no_missing_report:
        print(f"    missing_images_report.txt")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
