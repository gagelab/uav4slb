"""
Image preprocessing utilities for UAV imagery.

This module handles loading images from various formats (GeoTIFF, PNG, JPG)
and preprocessing them for deep learning models.
"""

import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
import cv2
from pathlib import Path
from typing import Tuple, Optional, List, Union
import warnings


def load_geotiff(
    image_path: str,
    bands: Optional[List[int]] = None,
    window: Optional[Window] = None,
    normalize_16bit: bool = True
) -> np.ndarray:
    """
    Load a GeoTIFF image using rasterio.
    
    Args:
        image_path: Path to GeoTIFF file
        bands: List of band indices to load (1-indexed). If None, loads all bands.
        window: Optional window to load only a portion of the image
        normalize_16bit: If True, normalize 16-bit images to 0-255 range
        
    Returns:
        Numpy array of shape (H, W, C) with values in [0, 255]
    """
    with rasterio.open(image_path) as src:
        # Determine which bands to read
        if bands is None:
            bands = list(range(1, src.count + 1))  # All bands
        
        # Read bands
        if window is not None:
            img = src.read(bands, window=window)
        else:
            img = src.read(bands)
        
        # Transpose from (C, H, W) to (H, W, C)
        img = np.transpose(img, (1, 2, 0))
        
        # Handle nodata values if present
        if src.nodata is not None:
            mask = np.any(img == src.nodata, axis=-1, keepdims=True)
            img = np.where(mask, 0, img)
        
        # Normalize 16-bit to 8-bit if needed
        if normalize_16bit and img.dtype == np.uint16:
            # Normalize to 0-255 range
            img = (img / 256).astype(np.uint8)
        elif img.dtype == np.float32 or img.dtype == np.float64:
            # Assume float images are in 0-1 range
            img = (img * 255).astype(np.uint8)
        
        return img.astype(np.uint8)


def load_image_simple(image_path: str) -> np.ndarray:
    """
    Load a standard image file (PNG, JPG, etc.) using PIL.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Numpy array of shape (H, W, C) with RGB values in [0, 255]
    """
    img = Image.open(image_path)
    
    # Convert to RGB if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    return np.array(img)


def load_image(
    image_path: str,
    bands: Optional[List[int]] = None,
    force_rgb: bool = True
) -> np.ndarray:
    """
    Universal image loader that handles GeoTIFF and standard formats.
    
    Args:
        image_path: Path to image file
        bands: List of band indices for GeoTIFF (1-indexed). Ignored for standard images.
        force_rgb: If True, ensure output is 3-channel RGB
        
    Returns:
        Numpy array of shape (H, W, C) with values in [0, 255]
    """
    path = Path(image_path)
    
    # Check file extension
    if path.suffix.lower() in ['.tif', '.tiff']:
        img = load_geotiff(image_path, bands=bands)
    else:
        img = load_image_simple(image_path)
    
    # Ensure RGB (3 channels)
    if force_rgb:
        if len(img.shape) == 2:
            # Grayscale to RGB
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] == 1:
            # Single channel to RGB
            img = np.concatenate([img, img, img], axis=-1)
        elif img.shape[-1] == 4:
            # RGBA to RGB
            img = img[:, :, :3]
        elif img.shape[-1] > 4:
            # Multispectral: take first 3 bands
            warnings.warn(
                f"Image has {img.shape[-1]} channels. Using first 3 for RGB."
            )
            img = img[:, :, :3]
    
    return img


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    method: str = 'resize',
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        image: Input image (H, W, C)
        target_size: Target size as (height, width)
        method: Resizing method - 'resize', 'center_crop', or 'random_crop'
        interpolation: OpenCV interpolation method
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    if method == 'resize':
        # Direct resize
        return cv2.resize(image, (target_w, target_h), interpolation=interpolation)
    
    elif method == 'center_crop':
        # Resize to make smallest dimension match target, then center crop
        scale = max(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        
        # Center crop
        start_h = (new_h - target_h) // 2
        start_w = (new_w - target_w) // 2
        return image[start_h:start_h + target_h, start_w:start_w + target_w]
    
    elif method == 'random_crop':
        # Similar to center_crop but random offset (for training)
        scale = max(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        
        # Random crop
        max_h = new_h - target_h
        max_w = new_w - target_w
        start_h = np.random.randint(0, max_h + 1) if max_h > 0 else 0
        start_w = np.random.randint(0, max_w + 1) if max_w > 0 else 0
        return image[start_h:start_h + target_h, start_w:start_w + target_w]
    
    else:
        raise ValueError(f"Unknown resize method: {method}")


def normalize_image(
    image: np.ndarray,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    to_tensor: bool = True
) -> Union[np.ndarray, 'torch.Tensor']:
    """
    Normalize image with mean and std.
    
    Args:
        image: Input image (H, W, C) with values in [0, 255]
        mean: Mean values for each channel
        std: Std values for each channel
        to_tensor: If True, convert to PyTorch tensor (C, H, W)
        
    Returns:
        Normalized image as numpy array or torch tensor
    """
    # Convert to float and scale to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Normalize
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    image = (image - mean) / std
    
    if to_tensor:
        import torch
        # Convert to tensor and transpose to (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)
    
    return image


def denormalize_image(
    image: Union[np.ndarray, 'torch.Tensor'],
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> np.ndarray:
    """
    Denormalize image back to [0, 255] range.
    
    Args:
        image: Normalized image (C, H, W) as tensor or (H, W, C) as numpy
        mean: Mean values used for normalization
        std: Std values used for normalization
        
    Returns:
        Denormalized image as numpy array (H, W, C) with values in [0, 255]
    """
    # Handle torch tensor
    try:
        import torch
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
            # Transpose from (C, H, W) to (H, W, C)
            if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
                image = np.transpose(image, (1, 2, 0))
    except ImportError:
        pass
    
    # Denormalize
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    image = (image * std) + mean
    
    # Scale to [0, 255] and clip
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    return image


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Useful for enhancing contrast in UAV imagery.
    
    Args:
        image: Input RGB image (H, W, C)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        Enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return enhanced


def preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
    bands: Optional[List[int]] = None,
    resize_method: str = 'resize',
    normalize: bool = True,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    enhance_contrast: bool = False,
    to_tensor: bool = True
) -> Union[np.ndarray, 'torch.Tensor']:
    """
    Complete preprocessing pipeline for a single image.
    
    Args:
        image_path: Path to image file
        target_size: Target size as (height, width)
        bands: List of band indices for GeoTIFF
        resize_method: Method for resizing
        normalize: Whether to normalize with mean/std
        mean: Mean values for normalization
        std: Std values for normalization
        enhance_contrast: Whether to apply CLAHE
        to_tensor: Whether to convert to PyTorch tensor
        
    Returns:
        Preprocessed image
    """
    # Load image
    image = load_image(image_path, bands=bands)
    
    # Enhance contrast if requested
    if enhance_contrast:
        image = apply_clahe(image)
    
    # Resize
    image = resize_image(image, target_size, method=resize_method)
    
    # Normalize
    if normalize:
        image = normalize_image(image, mean=mean, std=std, to_tensor=to_tensor)
    elif to_tensor:
        import torch
        # Just convert to tensor without normalization
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
    
    return image


# Example usage
if __name__ == "__main__":
    print("Preprocessing utilities loaded.")
    print("Example usage:")
    print("""
    from src.data.preprocessing import preprocess_image
    
    # Load and preprocess an image
    img_tensor = preprocess_image(
        'path/to/image.tif',
        target_size=(224, 224),
        bands=[1, 2, 3],
        normalize=True
    )
    print(f"Preprocessed image shape: {img_tensor.shape}")
    """)
