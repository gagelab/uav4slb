"""
CORRECT Cross-Validation for Agricultural Forecasting

This implements the standard agricultural ML workflow:

For each CV fold:
1. Hold out ONE YEAR completely as test set (never touched)
2. Use OTHER YEARS for model development:
   - Split into train (80%) and validation (20%)
   - Validation used for early stopping, hyperparameter tuning
3. After training, evaluate on held-out test year

This answers: "If I train on 2 years, can I predict the 3rd?"

KEY POINT: The held-out test year is NEVER seen during training,
not even for validation or early stopping.

Author: Cole H. Hammett  
Date: January 15, 2026
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class CV0_TrueLeaveOneYearOut:
    """
    CV0: True leave-one-year-out for temporal generalization.
    
    For each year as test:
    - Test set: ALL samples from that year (completely held out)
    - Development set: ALL samples from other years
        - Split development into train (80%) + val (20%)
        - Use train for gradient updates
        - Use val for early stopping and model selection
    - After training: Evaluate on test set for true generalization
    
    This is the STANDARD approach in agricultural forecasting.
    """
    
    def __init__(
        self,
        year_column: str = 'year',
        val_ratio: float = 0.2,
        random_seed: int = 42
    ):
        """
        Initialize CV0 strategy.
        
        Args:
            year_column: Column name containing year
            val_ratio: Proportion of development data for validation
            random_seed: Random seed for reproducibility
        """
        self.year_column = year_column
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        self.name = 'CV0_TrueLeaveOneYearOut'
    
    def split(self, df: pd.DataFrame) -> List[Dict]:
        """
        Generate train/val/test splits for each year.
        
        Args:
            df: Complete dataset with year column
            
        Returns:
            List of fold dictionaries with keys:
                - 'fold': Fold number
                - 'train': Training DataFrame (for gradient updates)
                - 'val': Validation DataFrame (for early stopping)
                - 'test': Test DataFrame (NEVER SEEN until final evaluation)
                - 'train_years': Years used for development
                - 'test_year': Year held out for testing
                - 'description': Human-readable description
        """
        # Get unique years
        years = sorted(df[self.year_column].unique())
        n_years = len(years)
        
        if n_years < 3:
            raise ValueError(
                f"CV0 requires at least 3 years for proper train/val/test split. "
                f"Found {n_years} years: {years}"
            )
        
        logger.info(f"CV0: Found {n_years} years: {years}")
        logger.info(f"Will create {n_years} folds (one per test year)")
        
        folds = []
        
        for test_year in years:
            fold_num = len(folds) + 1
            
            # Test set: ALL samples from this year (NEVER TOUCHED DURING TRAINING)
            test_df = df[df[self.year_column] == test_year].copy()
            
            # Development set: ALL samples from OTHER years
            train_years = [y for y in years if y != test_year]
            dev_df = df[df[self.year_column] != test_year].copy()
            
            # Split development set into train and validation
            train_df, val_df = train_test_split(
                dev_df,
                test_size=self.val_ratio,
                random_state=self.random_seed,
                shuffle=True  # Shuffle to mix years in train and val
            )
            
            # Create fold dictionary
            fold = {
                'fold': fold_num,
                'train': train_df.reset_index(drop=True),
                'val': val_df.reset_index(drop=True),
                'test': test_df.reset_index(drop=True),
                'train_years': train_years,
                'test_year': test_year,
                'n_train': len(train_df),
                'n_val': len(val_df),
                'n_test': len(test_df),
                'description': f"Train on {train_years} (train={len(train_df)}, val={len(val_df)}), Test on {test_year} (n={len(test_df)})"
            }
            
            folds.append(fold)
            
            # Log fold information
            logger.info(f"\nFold {fold_num}:")
            logger.info(f"  Test year (held out): {test_year}")
            logger.info(f"  Development years: {train_years}")
            logger.info(f"    - Train: {len(train_df)} samples ({len(train_df)/len(dev_df)*100:.1f}%)")
            logger.info(f"    - Val:   {len(val_df)} samples ({len(val_df)/len(dev_df)*100:.1f}%)")
            logger.info(f"  Test: {len(test_df)} samples (NEVER SEEN until final evaluation)")
            
            # Verify no test year in train or val
            assert test_year not in train_df[self.year_column].values, \
                f"ERROR: Test year {test_year} found in training set!"
            assert test_year not in val_df[self.year_column].values, \
                f"ERROR: Test year {test_year} found in validation set!"
            
            logger.info(f"  ✓ Verified: Year {test_year} NOT in train or val sets")
        
        logger.info(f"\n✓ Created {len(folds)} folds")
        logger.info("Ready for training with proper temporal holdout")
        
        return folds
    
    def get_n_splits(self, df: pd.DataFrame) -> int:
        """Get number of CV splits (one per year)."""
        return len(df[self.year_column].unique())


class CV1_TrueLeaveOnePopOut:
    """
    CV1: True leave-one-population-out for genotype generalization.
    
    Similar to CV0 but for genotypes within populations:
    - For each population, hold out some genotypes as test
    - Split remaining genotypes into train and val
    - Test genotypes are NEVER seen during training
    """
    
    def __init__(
        self,
        population_column: str = 'pop',
        genotype_column: str = 'genotype',
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        min_pop_size: int = 50,
        random_seed: int = 42
    ):
        """Initialize CV1 strategy."""
        self.population_column = population_column
        self.genotype_column = genotype_column
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.min_pop_size = min_pop_size
        self.random_seed = random_seed
        self.name = 'CV1_TrueLeaveOnePopOut'
        
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")
    
    def split(self, df: pd.DataFrame) -> List[Dict]:
        """
        Generate train/val/test splits for each population.
        
        For each population:
        1. Hold out test_ratio of genotypes (test set, never seen)
        2. Split remaining genotypes into train and val
        3. All images from each genotype go to respective set
        """
        # Filter populations by size
        pop_sizes = df.groupby(self.population_column).size()
        valid_pops = pop_sizes[pop_sizes >= self.min_pop_size].index.tolist()
        
        logger.info(f"CV1: Found {len(valid_pops)} populations with >= {self.min_pop_size} samples")
        
        folds = []
        
        for pop in valid_pops:
            fold_num = len(folds) + 1
            
            # Get all data for this population
            pop_df = df[df[self.population_column] == pop].copy()
            
            # Get unique genotypes
            genotypes = pop_df[self.genotype_column].unique()
            n_genotypes = len(genotypes)
            
            # Calculate genotype splits
            n_train = int(n_genotypes * self.train_ratio)
            n_val = int(n_genotypes * self.val_ratio)
            # n_test gets remainder
            
            # Shuffle genotypes
            np.random.seed(self.random_seed)
            shuffled_genotypes = np.random.permutation(genotypes)
            
            # Split genotypes
            train_genotypes = shuffled_genotypes[:n_train]
            val_genotypes = shuffled_genotypes[n_train:n_train + n_val]
            test_genotypes = shuffled_genotypes[n_train + n_val:]
            
            # Create dataframes (all images from each genotype)
            train_df = pop_df[pop_df[self.genotype_column].isin(train_genotypes)].copy()
            val_df = pop_df[pop_df[self.genotype_column].isin(val_genotypes)].copy()
            test_df = pop_df[pop_df[self.genotype_column].isin(test_genotypes)].copy()
            
            fold = {
                'fold': fold_num,
                'train': train_df.reset_index(drop=True),
                'val': val_df.reset_index(drop=True),
                'test': test_df.reset_index(drop=True),
                'population': pop,
                'n_train_genotypes': len(train_genotypes),
                'n_val_genotypes': len(val_genotypes),
                'n_test_genotypes': len(test_genotypes),
                'description': f"Population {pop}: {n_genotypes} genotypes split into train/val/test"
            }
            
            folds.append(fold)
            
            logger.info(f"\nFold {fold_num} (Population: {pop}):")
            logger.info(f"  Total genotypes: {n_genotypes}")
            logger.info(f"  Train: {len(train_df)} samples from {len(train_genotypes)} genotypes")
            logger.info(f"  Val:   {len(val_df)} samples from {len(val_genotypes)} genotypes")
            logger.info(f"  Test:  {len(test_df)} samples from {len(test_genotypes)} genotypes (NEVER SEEN)")
            
            # Verify no overlap
            assert len(set(train_genotypes) & set(test_genotypes)) == 0
            assert len(set(val_genotypes) & set(test_genotypes)) == 0
            logger.info(f"  ✓ Verified: No genotype overlap between train/val/test")
        
        logger.info(f"\n✓ Created {len(folds)} folds (one per population)")
        
        return folds
    
    def get_n_splits(self, df: pd.DataFrame) -> int:
        """Get number of splits."""
        pop_sizes = df.groupby(self.population_column).size()
        return len(pop_sizes[pop_sizes >= self.min_pop_size])


def create_cv_strategy(cv_type: str, **kwargs):
    """
    Factory function to create CV strategy.
    
    Args:
        cv_type: 'cv0' or 'cv1'
        **kwargs: Arguments for the strategy
        
    Returns:
        CV strategy instance
    """
    strategies = {
        'cv0': CV0_TrueLeaveOneYearOut,
        'cv1': CV1_TrueLeaveOnePopOut
    }
    
    if cv_type not in strategies:
        raise ValueError(
            f"Unknown CV type: {cv_type}. "
            f"Available: {list(strategies.keys())}"
        )
    
    return strategies[cv_type](**kwargs)


# Testing and validation
if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data with realistic distribution
    np.random.seed(42)
    data = []
    
    # Simulate your actual data distribution
    year_sizes = {
        2023: 225,   # 10% (like your ~2,250 samples)
        2024: 706,   # 31% (like your ~7,060 samples)
        2025: 1316   # 59% (like your ~13,160 samples)
    }
    
    for year, n_samples in year_sizes.items():
        for i in range(n_samples):
            data.append({
                'year': year,
                'image_filename': f'img_{year}_{i:04d}.tif',
                'score': np.random.uniform(1, 9),
                'pop': np.random.choice(['pop_a', 'pop_b']),
                'genotype': f'geno_{np.random.randint(1, 30)}'
            })
    
    df = pd.DataFrame(data)
    
    print("\n" + "="*80)
    print("TEST: CV0_TrueLeaveOneYearOut")
    print("="*80)
    print(f"\nDataset: {len(df)} samples across {df['year'].nunique()} years")
    print(f"Year distribution:")
    for year in sorted(df['year'].unique()):
        n = len(df[df['year'] == year])
        pct = n / len(df) * 100
        print(f"  {year}: {n:5d} samples ({pct:5.1f}%)")
    
    # Test CV0
    cv0 = CV0_TrueLeaveOneYearOut(year_column='year', val_ratio=0.2, random_seed=42)
    folds = cv0.split(df)
    
    print("\n" + "-"*80)
    print("FOLD SUMMARY:")
    print("-"*80)
    
    for fold in folds:
        print(f"\nFold {fold['fold']}: Test year = {fold['test_year']}")
        print(f"  Train: {fold['n_train']:5d} samples from years {fold['train_years']}")
        print(f"  Val:   {fold['n_val']:5d} samples from years {fold['train_years']}")
        print(f"  Test:  {fold['n_test']:5d} samples from year {fold['test_year']} ← HELD OUT")
        
        # Verify correctness
        train_years_in_data = fold['train']['year'].unique()
        val_years_in_data = fold['val']['year'].unique()
        test_year_in_data = fold['test']['year'].unique()
        
        # Check train and val have same years (both from development set)
        assert set(train_years_in_data) == set(fold['train_years'])
        assert set(val_years_in_data) == set(fold['train_years'])
        
        # Check test has only test year
        assert len(test_year_in_data) == 1
        assert test_year_in_data[0] == fold['test_year']
        
        # Check no overlap
        assert fold['test_year'] not in train_years_in_data
        assert fold['test_year'] not in val_years_in_data
        
        print(f"  ✓ All checks passed")
    
    print("\n" + "="*80)
    print("✓ CV0 implementation is CORRECT")
    print("="*80)
    print("\nThis is the standard agricultural forecasting workflow:")
    print("1. Hold out one year as test set (never touched)")
    print("2. Split other years into train (80%) and val (20%)")
    print("3. Train model using train set, early stop using val set")
    print("4. Evaluate on held-out test year for true generalization")
    print("="*80)
