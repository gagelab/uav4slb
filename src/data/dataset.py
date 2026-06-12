"""
PyTorch Dataset class for UAV imagery.

This module provides a PyTorch Dataset implementation for loading and
preprocessing UAV imagery for disease severity prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig
import logging

from .preprocessing import load_image, preprocess_image
from .augmentation import create_augmentation_pipeline
from ..utils.reproducibility import seed_worker, create_generator


class UAVDataset(Dataset):
    """
    PyTorch Dataset for UAV imagery with disease severity labels.
    
    Supports both classification and regression tasks.
    
    Args:
        data_dir: Directory containing images
        labels_csv: Path to CSV file with labels
        image_filename_column: Column name for image identifiers
        target_column: Column name for target variable (labels/scores)
        task_type: Task type - 'classification' or 'regression'
        image_size: Target image size as (height, width)
        transform: Albumentations transform pipeline
        bands: List of bands to use for multispectral imagery
        cache_images: Whether to cache loaded images in memory
        image_extension: Image file extension (e.g., '.tif', '.jpg')
    """
    
    def __init__(
        self,
        data_dir: str,
        labels_csv: str,
        image_filename_column: str = 'image_filename',
        target_column: str = 'score',
        task_type: str = 'regression',
        image_size: Tuple[int, int] = (224, 224),
        transform: Optional[Any] = None,
        bands: Optional[List[int]] = None,
        cache_images: bool = False,
        image_extension: str = '.tif',
        filter_incomplete: bool = False,
        max_nodata_fraction: float = 0.05
    ):
        """Initialize UAVDataset."""
        self.data_dir = Path(data_dir)
        self.labels_csv = Path(labels_csv)
        self.image_filename_column = image_filename_column
        self.target_column = target_column
        self.task_type = task_type
        self.image_size = image_size
        self.transform = transform
        self.bands = bands
        self.cache_images = cache_images
        self.image_extension = image_extension
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load labels
        self.df = pd.read_csv(self.labels_csv)
        self.logger.info(f"Loaded {len(self.df)} samples from {self.labels_csv}")
        
        # Verify required columns exist
        if self.image_filename_column not in self.df.columns:
            raise ValueError(
                f"Image ID column '{self.image_filename_column}' not found in CSV. "
                f"Available columns: {list(self.df.columns)}"
            )
        if self.target_column not in self.df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in CSV. "
                f"Available columns: {list(self.df.columns)}"
            )
        
        if filter_incomplete:
            from .image_validation import filter_incomplete_images
            self.df, _ = filter_incomplete_images(
                labels_df=self.df,
                image_dir=self.data_dir,
                image_filename_column=self.image_filename_column,
                image_extension=self.image_extension,
                max_nodata_fraction=max_nodata_fraction,
                auto_confirm=True,   # non-interactive inside DataLoader
                report_dir=None,
            )
            self.logger.info(
                f"After completeness filter: {len(self.df)} samples remain"
            )
        
        # Handle classification: encode labels
        if self.task_type == 'classification':
            self.label_encoder = {
                label: idx for idx, label in enumerate(sorted(self.df[self.target_column].unique()))
            }
            self.inverse_label_encoder = {v: k for k, v in self.label_encoder.items()}
            self.num_classes = len(self.label_encoder)
            self.logger.info(f"Classification task with {self.num_classes} classes")
            self.logger.info(f"Label encoding: {self.label_encoder}")
        else:
            self.label_encoder = None
            self.num_classes = 1
        
        # Image cache
        self._cache = {} if cache_images else None
        
        # Verify data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, label_tensor)
        """
        # Get sample info
        sample = self.df.iloc[idx]
        image_filename = sample[self.image_filename_column]
        label = sample[self.target_column]
        
        # Load image (from cache or disk)
        if self._cache is not None and idx in self._cache:
            image = self._cache[idx]
        else:
            # Construct image path
            image_path = self._get_image_path(image_filename)
            
            # Load image
            try:
                image = load_image(str(image_path), bands=self.bands)
            except Exception as e:
                self.logger.error(f"Error loading image {image_path}: {e}")
                raise
            
            # Cache if enabled
            if self._cache is not None:
                self._cache[idx] = image
        
        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Convert to tensor if no transform
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        # Process label
        if self.task_type == 'classification':
            label = self.label_encoder[label]
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = torch.tensor(label, dtype=torch.float32)
        
        return image, label
    
    def _get_image_path(self, image_filename: str) -> Path:
        """
        Get full path to image file.
        
        Args:
            image_filename: Image identifier (with or without extension)
            
        Returns:
            Path to image file
        """
        # Check if image_filename already has extension
        if Path(image_filename).suffix:
            image_path = self.data_dir / image_filename
        else:
            image_path = self.data_dir / f"{image_filename}{self.image_extension}"
        
        if not image_path.exists():
            # Try other common extensions
            for ext in ['.tif', '.tiff', '.jpg', '.jpeg', '.png']:
                alt_path = self.data_dir / f"{Path(image_filename).stem}{ext}"
                if alt_path.exists():
                    return alt_path
            
            raise FileNotFoundError(
                f"Image not found: {image_path}. "
                f"Checked extensions: {['.tif', '.tiff', '.jpg', '.jpeg', '.png']}"
            )
        
        return image_path
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        Get detailed information about a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with sample information
        """
        sample = self.df.iloc[idx]
        
        info = {
            'index': idx,
            'image_filename': sample[self.image_filename_column],
            'label': sample[self.target_column],
        }
        
        if self.task_type == 'classification':
            info['label_encoded'] = self.label_encoder[sample[self.target_column]]
        
        # Add all other columns
        for col in self.df.columns:
            if col not in [self.image_filename_column, self.target_column]:
                info[col] = sample[col]
        
        return info
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'num_samples': len(self),
            'task_type': self.task_type,
        }
        
        if self.task_type == 'classification':
            # Class distribution
            value_counts = self.df[self.target_column].value_counts()
            stats['num_classes'] = self.num_classes
            stats['class_distribution'] = value_counts.to_dict()
            stats['class_balance'] = (value_counts / len(self.df)).to_dict()
        else:
            # Regression statistics
            labels = self.df[self.target_column]
            stats['mean'] = float(labels.mean())
            stats['std'] = float(labels.std())
            stats['min'] = float(labels.min())
            stats['max'] = float(labels.max())
            stats['median'] = float(labels.median())
        
        return stats
    
    def get_label_encoder(self) -> Optional[Dict]:
        """Get label encoder for classification tasks."""
        return self.label_encoder
    
    def get_inverse_label_encoder(self) -> Optional[Dict]:
        """Get inverse label encoder for classification tasks."""
        return getattr(self, 'inverse_label_encoder', None)


def create_dataloaders(
    config: DictConfig,
    split: str = 'all'
) -> Tuple[DataLoader, ...]:
    """
    Create DataLoaders from configuration.
    
    Args:
        config: Data configuration (from data_config.yaml)
        split: Which split(s) to create - 'train', 'val', 'test', or 'all'
        
    Returns:
        Tuple of DataLoader(s) for requested split(s)
        If split='all', returns (train_loader, val_loader, test_loader)
    """
    loaders = []
    
    splits_to_create = {
        'train': split in ['train', 'all'],
        'val': split in ['val', 'all'],
        'test': split in ['test', 'all']
    }
    
    for split_name, should_create in splits_to_create.items():
        if not should_create:
            continue
        
        # Get paths for this split
        if split_name == 'train':
            data_dir = config.data.train_dir
            labels_csv = config.data.train_labels
        elif split_name == 'val':
            data_dir = config.data.val_dir
            labels_csv = config.data.val_labels
        else:  # test
            data_dir = config.data.test_dir
            labels_csv = config.data.test_labels
        
        # Create augmentation pipeline
        transform = create_augmentation_pipeline(config, split=split_name)
        
        # Create dataset
        dataset = UAVDataset(
            data_dir=data_dir,
            labels_csv=labels_csv,
            image_filename_column=config.task.image_filename_column,
            target_column=config.task.target_column,
            task_type=config.task.type,
            image_size=tuple(config.preprocessing.image_size),
            transform=transform,
            bands=config.preprocessing.bands.get('use_bands', None),
            cache_images=config.caching.get('load_to_memory', False),
            image_extension='.tif'  # Could be made configurable
        )
        
        # DataLoader settings
        if split_name == 'train':
            batch_size = config.dataloader.batch_size
            shuffle = config.dataloader.shuffle_train
            drop_last = config.dataloader.drop_last_train
        else:
            batch_size = config.dataloader.get(f'batch_size_{split_name}', config.dataloader.batch_size)
            shuffle = False
            drop_last = False
        
        # Create DataLoader with reproducibility
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=config.dataloader.num_workers,
            pin_memory=config.dataloader.pin_memory,
            drop_last=drop_last,
            worker_init_fn=seed_worker,
            generator=create_generator(config.splits.get('random_seed', 42))
        )
        
        loaders.append(loader)
    
    # Return based on what was requested
    if len(loaders) == 1:
        return loaders[0]
    else:
        return tuple(loaders)


def visualize_batch(
    dataloader: DataLoader,
    num_samples: int = 8,
    denormalize: bool = True,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize a batch of images from a DataLoader.
    
    Args:
        dataloader: DataLoader to visualize
        num_samples: Number of samples to show
        denormalize: Whether to denormalize images
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    from ..data.preprocessing import denormalize_image
    
    # Get a batch
    images, labels = next(iter(dataloader))
    
    # Limit to num_samples
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Calculate grid size
    ncols = min(4, num_samples)
    nrows = (num_samples + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // ncols
        col = idx % ncols
        
        # Convert to numpy and denormalize
        if denormalize:
            # Get mean/std from dataset if available
            img_np = denormalize_image(img)
        else:
            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
        
        # Plot
        axes[row, col].imshow(img_np)
        axes[row, col].set_title(f"Label: {label.item():.2f}")
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for idx in range(num_samples, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


# Example usage
if __name__ == "__main__":
    print("Dataset module loaded.")
    print("\nExample usage:")
    print("""
    from src.utils.config import load_config
    from src.data import create_dataloaders
    
    # Load configuration
    config = load_config('configs/data_config.yaml')
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Use in training
    for images, labels in train_loader:
        # images: (batch_size, 3, 224, 224)
        # labels: (batch_size,)
        pass
    """)
