"""
Data pipeline for UAV SLB project.
"""

from .preprocessing import (
    load_image,
    load_geotiff,
    load_image_simple,
    resize_image,
    normalize_image,
    denormalize_image,
    apply_clahe,
    preprocess_image,
)

from .augmentation import (
    create_train_augmentation,
    create_val_augmentation,
    create_augmentation_pipeline,
    create_custom_augmentation,
    visualize_augmentations,
)

from .dataset import (
    UAVDataset,
    create_dataloaders,
)

from .image_validation import (
    check_image_completeness,
    scan_images,
    filter_incomplete_images,
    summarise_validation_report,
)

__all__ = [
    # Preprocessing
    'load_image',
    'load_geotiff',
    'load_image_simple',
    'resize_image',
    'normalize_image',
    'denormalize_image',
    'apply_clahe',
    'preprocess_image',
    # Augmentation
    'create_train_augmentation',
    'create_val_augmentation',
    'create_augmentation_pipeline',
    'create_custom_augmentation',
    'visualize_augmentations',
    # Dataset
    'UAVDataset',
    'create_dataloaders',
    # Image validation
    'check_image_completeness',
    'scan_images',
    'filter_incomplete_images',
    'summarise_validation_report'
]
