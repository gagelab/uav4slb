"""
Data augmentation pipelines using Albumentations.

This module creates augmentation pipelines for training, validation, and testing.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from typing import Optional, List


def create_train_augmentation(
    config: DictConfig,
    image_size: tuple = (224, 224),
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> A.Compose:
    """
    Create training augmentation pipeline.
    
    Args:
        config: Augmentation configuration (from data_config.yaml)
        image_size: Target image size (height, width)
        mean: Mean values for normalization
        std: Std values for normalization
        
    Returns:
        Albumentations Compose object with augmentation pipeline
    """
    transforms = []
    
    # Resize to target size
    transforms.append(
        A.Resize(height=image_size[0], width=image_size[1])
    )
    
    # Geometric transformations
    if config.augmentation.train.get('horizontal_flip', {}).get('enabled', False):
        transforms.append(
            A.HorizontalFlip(p=config.augmentation.train.horizontal_flip.p)
        )
    
    if config.augmentation.train.get('vertical_flip', {}).get('enabled', False):
        transforms.append(
            A.VerticalFlip(p=config.augmentation.train.vertical_flip.p)
        )
    
    if config.augmentation.train.get('rotation', {}).get('enabled', False):
        transforms.append(
            A.Rotate(
                limit=config.augmentation.train.rotation.limit,
                p=config.augmentation.train.rotation.p,
                border_mode=0
            )
        )
    
    # if config.augmentation.train.get('shift_scale_rotate', {}).get('enabled', False):
    #     ssr = config.augmentation.train.shift_scale_rotate
    #     transforms.append(
    #         A.ShiftScaleRotate(
    #             shift_limit=ssr.shift_limit,
    #             scale_limit=ssr.scale_limit,
    #             rotate_limit=ssr.rotate_limit,
    #             p=ssr.p,
    #             border_mode=0
    #         )
    #     )
    
    # Color/intensity augmentations
    if config.augmentation.train.get('brightness_contrast', {}).get('enabled', False):
        bc = config.augmentation.train.brightness_contrast
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=bc.brightness_limit,
                contrast_limit=bc.contrast_limit,
                p=bc.p
            )
        )
    
    if config.augmentation.train.get('hue_saturation', {}).get('enabled', False):
        hs = config.augmentation.train.hue_saturation
        transforms.append(
            A.HueSaturationValue(
                hue_shift_limit=hs.hue_shift_limit,
                sat_shift_limit=hs.sat_shift_limit,
                val_shift_limit=hs.val_shift_limit,
                p=hs.p
            )
        )
    
    if config.augmentation.train.get('random_gamma', {}).get('enabled', False):
        rg = config.augmentation.train.random_gamma
        transforms.append(
            A.RandomGamma(
                gamma_limit=rg.gamma_limit,
                p=rg.p
            )
        )
    
    # Advanced augmentations
    if config.augmentation.train.get('gaussian_blur', {}).get('enabled', False):
        gb = config.augmentation.train.gaussian_blur
        transforms.append(
            A.GaussianBlur(
                blur_limit=gb.blur_limit,
                p=gb.p
            )
        )
    
    if config.augmentation.train.get('gaussian_noise', {}).get('enabled', False):
        gn = config.augmentation.train.gaussian_noise
        transforms.append(
            A.GaussNoise(
                var_limit=gn.var_limit,
                p=gn.p
            )
        )
    
    if config.augmentation.train.get('coarse_dropout', {}).get('enabled', False):
        cd = config.augmentation.train.coarse_dropout
        transforms.append(
            A.CoarseDropout(
                max_holes=cd.max_holes,
                max_height=cd.max_height,
                max_width=cd.max_width,
                fill_value=0,
                p=cd.p
            )
        )
    
    # Normalization and tensor conversion
    transforms.append(A.Normalize(mean=mean, std=std))
    transforms.append(ToTensorV2())
    
    return A.Compose(transforms)


def create_val_augmentation(
    image_size: tuple = (224, 224),
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> A.Compose:
    """
    Create validation/test augmentation pipeline.
    
    Only resizing and normalization, no data augmentation.
    
    Args:
        image_size: Target image size (height, width)
        mean: Mean values for normalization
        std: Std values for normalization
        
    Returns:
        Albumentations Compose object with preprocessing pipeline
    """
    transforms = [
        A.Resize(height=image_size[0], width=image_size[1], always_apply=True),
        A.Normalize(mean=mean, std=std, always_apply=True),
        ToTensorV2()
    ]
    
    return A.Compose(transforms)


def create_augmentation_pipeline(
    config: DictConfig,
    split: str = 'train'
) -> A.Compose:
    """
    Create augmentation pipeline based on config and data split.
    
    Args:
        config: Data configuration with augmentation settings
        split: Data split - 'train', 'val', or 'test'
        
    Returns:
        Albumentations Compose object
    """
    image_size = tuple(config.preprocessing.image_size)
    mean = config.preprocessing.normalize.mean
    std = config.preprocessing.normalize.std
    
    if split == 'train' and config.augmentation.train.enabled:
        return create_train_augmentation(config, image_size, mean, std)
    else:
        return create_val_augmentation(image_size, mean, std)


def create_custom_augmentation(
    image_size: tuple = (224, 224),
    horizontal_flip: bool = True,
    vertical_flip: bool = True,
    rotation_limit: int = 45,
    brightness_contrast: bool = True,
    blur: bool = True,
    noise: bool = True,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    p: float = 0.5
) -> A.Compose:
    """
    Create custom augmentation pipeline with simple parameters.
    
    Useful for quick experimentation without modifying config files.
    
    Args:
        image_size: Target image size
        horizontal_flip: Enable horizontal flip
        vertical_flip: Enable vertical flip
        rotation_limit: Maximum rotation angle in degrees
        brightness_contrast: Enable brightness/contrast adjustment
        blur: Enable Gaussian blur
        noise: Enable Gaussian noise
        mean: Normalization mean
        std: Normalization std
        p: Probability for each augmentation
        
    Returns:
        Albumentations Compose object
    """
    transforms = [
        A.Resize(height=image_size[0], width=image_size[1], always_apply=True)
    ]
    
    if horizontal_flip:
        transforms.append(A.HorizontalFlip(p=p))
    
    if vertical_flip:
        transforms.append(A.VerticalFlip(p=p))
    
    if rotation_limit > 0:
        transforms.append(A.Rotate(limit=rotation_limit, p=p, border_mode=0))
    
    if brightness_contrast:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=p
            )
        )
    
    if blur:
        transforms.append(A.GaussianBlur(blur_limit=(3, 7), p=p))
    
    if noise:
        transforms.append(A.GaussNoise(var_limit=(10.0, 50.0), p=p))
    
    transforms.extend([
        A.Normalize(mean=mean, std=std, always_apply=True),
        ToTensorV2()
    ])
    
    return A.Compose(transforms)


def visualize_augmentations(
    image_path: str,
    augmentation: A.Compose,
    n_samples: int = 4,
    figsize: tuple = (15, 10)
) -> None:
    """
    Visualize augmentations on a sample image.
    
    Args:
        image_path: Path to image
        augmentation: Albumentations pipeline
        n_samples: Number of augmented samples to show
        figsize: Figure size for matplotlib
    """
    import matplotlib.pyplot as plt
    from src.data.preprocessing import load_image, denormalize_image
    import numpy as np
    
    # Load original image
    original = load_image(image_path)
    
    # Create figure
    fig, axes = plt.subplots(1, n_samples + 1, figsize=figsize)
    
    # Show original
    axes[0].imshow(original)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Show augmented versions
    for i in range(n_samples):
        # Apply augmentation
        augmented = augmentation(image=original)
        
        # Convert tensor back to image for visualization
        if isinstance(augmented['image'], tuple):
            # Handle case where ToTensorV2 returns (tensor,)
            aug_image = augmented['image'][0]
        else:
            aug_image = augmented['image']
        
        # Denormalize
        # Extract mean and std from augmentation pipeline
        normalize_transform = None
        for t in augmentation.transforms:
            if isinstance(t, A.Normalize):
                normalize_transform = t
                break
        
        if normalize_transform:
            mean = normalize_transform.mean
            std = normalize_transform.std
            aug_image_np = denormalize_image(aug_image, mean, std)
        else:
            # No normalization, just convert
            aug_image_np = (aug_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        axes[i + 1].imshow(aug_image_np)
        axes[i + 1].set_title(f'Augmented {i + 1}')
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    print("Augmentation utilities loaded.")
    print("\nExample usage:")
    print("""
    from src.data.augmentation import create_custom_augmentation
    from src.utils.config import load_config
    
    # Simple custom augmentation
    transform = create_custom_augmentation(
        image_size=(224, 224),
        horizontal_flip=True,
        rotation_limit=30
    )
    
    # Or from config file
    config = load_config('configs/data_config.yaml')
    transform = create_augmentation_pipeline(config, split='train')
    
    # Apply to image
    import numpy as np
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    augmented = transform(image=image)
    print(f"Augmented image shape: {augmented['image'].shape}")
    """)
