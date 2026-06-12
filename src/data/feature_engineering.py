"""
Feature Engineering Module for Maize Image Analysis

This module provides feature extraction capabilities for agricultural image data,
specifically designed for maize/corn plant analysis. It generates additional feature
channels beyond RGB to enhance model learning.

Author: TM 4/9/25
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging


class FeatureEngineer:
    """
    Feature engineering class for extracting handcrafted features from agricultural images.
    
    This class provides methods to compute gradient-based, color-based, and texture-based
    features that can be stacked with the original RGB channels to create enhanced
    multi-channel representations.
    """
    
    def __init__(self, 
                 brown_thresh: int = 30,
                 green_thresh: int = 30,
                 dilate_size: int = 15,
                 kernel_size: int = 5,
                 log_level: Optional[int] = None):
        """
        Initialize the FeatureEngineer with configurable parameters.
        
        Args:
            brown_thresh: Minimum R-G difference to consider a pixel "brownish"
            green_thresh: Minimum G-R difference to consider a pixel "greenish"
            dilate_size: Kernel size for dilation around green areas
            kernel_size: Kernel size for color difference computation
            log_level: Logging level (e.g., logging.INFO, logging.DEBUG)
        """
        self.brown_thresh = brown_thresh
        self.green_thresh = green_thresh
        self.dilate_size = dilate_size
        self.kernel_size = kernel_size
        
        self.logger = logging.getLogger(__name__)
        if log_level:
            self.logger.setLevel(log_level)
    
    def compute_gradient_magnitude(self, img: np.ndarray) -> np.ndarray:
        """
        Compute gradient magnitude edge map using Sobel operator.
        
        Returns an edge map by calculating the edge strength at each pixel
        in terms of the gradient in x and y directions.
        
        Args:
            img: Input RGB image (H, W, 3)
            
        Returns:
            Edge magnitude map (H, W) normalized to [0, 255]
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    def compute_color_gradient_magnitude(self, img: np.ndarray) -> np.ndarray:
        """
        Compute gradient magnitude on color channels.
        
        Returns an edge map by calculating the edge strength at each pixel
        on the color image directly (without converting to grayscale first).
        
        Args:
            img: Input RGB image (H, W, 3)
            
        Returns:
            Color edge magnitude map (H, W, 3) normalized to [0, 255]
        """
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    def compute_color_difference(self, img: np.ndarray, kernel_size: Optional[int] = None) -> np.ndarray:
        """
        Compute pixel-wise color difference from local mean.
        
        Highlights regions where pixel colors differ from their local neighborhood,
        useful for detecting anomalies or variations.
        
        Args:
            img: Input RGB image (H, W, 3)
            kernel_size: Size of blur kernel (uses instance default if None)
            
        Returns:
            Color difference map (H, W, 3) normalized to [0, 255]
        """
        if kernel_size is None:
            kernel_size = self.kernel_size
            
        mean_color = cv2.blur(img, (kernel_size, kernel_size))
        diff = cv2.absdiff(img, mean_color)
        return cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    def compute_texture_variation(self, img: np.ndarray) -> np.ndarray:
        """
        Compute texture variation using Laplacian operator.
        
        Captures fine-grained texture patterns and high-frequency variations
        in the image.
        
        Args:
            img: Input RGB image (H, W, 3)
            
        Returns:
            Texture variation map (H, W) normalized to [0, 255]
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return cv2.normalize(np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    def highlight_brown_near_green(self, 
                                   image_rgb: np.ndarray,
                                   brown_thresh: Optional[int] = None,
                                   green_thresh: Optional[int] = None,
                                   dilate_size: Optional[int] = None) -> np.ndarray:
        """
        Highlight brown-enriched pixels near green areas.
        
        This is particularly useful for detecting disease, stress, or senescence
        in maize plants, where brownish discoloration appears near healthy green tissue.
        
        Args:
            image_rgb: Input RGB image (H, W, 3)
            brown_thresh: Minimum R-G difference for "brownish" pixels (uses instance default if None)
            green_thresh: Minimum G-R difference for "greenish" pixels (uses instance default if None)
            dilate_size: Dilation kernel size around green areas (uses instance default if None)
            
        Returns:
            Binary mask (H, W) where 1 = brown near green, 0 = otherwise
        """
        if brown_thresh is None:
            brown_thresh = self.brown_thresh
        if green_thresh is None:
            green_thresh = self.green_thresh
        if dilate_size is None:
            dilate_size = self.dilate_size
        
        R = image_rgb[:, :, 0].astype(np.int16)
        G = image_rgb[:, :, 1].astype(np.int16)
        
        # Step 1: Detect brownish pixels (R much higher than G)
        brown_mask = (R - G) > brown_thresh
        
        # Step 2: Detect green areas (G much higher than R)
        green_mask = (G - R) > green_thresh
        
        # Step 3: Expand green area using dilation
        green_dilated = cv2.dilate(
            green_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        )
        
        # Step 4: Final mask = brown pixels that are close to green
        brown_near_green = np.logical_and(brown_mask, green_dilated).astype(np.uint8)
        
        return brown_near_green
    
    def build_enhanced_image(self, img: np.ndarray) -> np.ndarray:
        """
        Build enhanced multi-channel image representation.
        
        Combines original RGB channels with three additional feature channels:
        - Gradient magnitude (edge strength)
        - Color difference from local mean
        - Texture variation
        
        Args:
            img: Input RGB image (H, W, 3)
            
        Returns:
            Enhanced 6-channel image (H, W, 6): [R, G, B, gradient, color_diff, texture]
        """
        grad_mag = self.compute_gradient_magnitude(img)
        color_diff = self.compute_color_difference(img)
        texture = self.compute_texture_variation(img)
        
        # Collapse multi-channel color_diff to grayscale average
        color_diff_gray = cv2.cvtColor(color_diff, cv2.COLOR_BGR2GRAY)
        
        # Stack: original RGB + 3 additional channels
        enhanced_img = np.dstack((img, grad_mag, color_diff_gray, texture))
        
        self.logger.debug(f"Enhanced image shape: {enhanced_img.shape}")
        return enhanced_img
    
    def build_feature_stack(self, 
                           img: np.ndarray,
                           include_brown_mask: bool = False) -> np.ndarray:
        """
        Build complete feature stack with optional brown-near-green mask.
        
        Args:
            img: Input RGB image (H, W, 3)
            include_brown_mask: Whether to include brown-near-green detection as 7th channel
            
        Returns:
            Feature stack (H, W, 6) or (H, W, 7) if brown mask included
        """
        enhanced = self.build_enhanced_image(img)
        
        if include_brown_mask:
            brown_mask = self.highlight_brown_near_green(img)
            # Expand dims to match other channels
            brown_mask = brown_mask[..., np.newaxis] * 255  # Scale to [0, 255]
            enhanced = np.concatenate([enhanced, brown_mask], axis=-1)
        
        return enhanced
    
    def get_feature_names(self, include_brown_mask: bool = False) -> list:
        """
        Get human-readable names for each feature channel.
        
        Args:
            include_brown_mask: Whether brown-near-green mask is included
            
        Returns:
            List of feature channel names
        """
        names = ['R', 'G', 'B', 'gradient_magnitude', 'color_difference', 'texture_variation']
        if include_brown_mask:
            names.append('brown_near_green_mask')
        return names


# Standalone utility functions for backward compatibility
def compute_gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """Standalone function - see FeatureEngineer.compute_gradient_magnitude"""
    engineer = FeatureEngineer()
    return engineer.compute_gradient_magnitude(img)


def compute_color_gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """Standalone function - see FeatureEngineer.compute_color_gradient_magnitude"""
    engineer = FeatureEngineer()
    return engineer.compute_color_gradient_magnitude(img)


def compute_color_difference(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Standalone function - see FeatureEngineer.compute_color_difference"""
    engineer = FeatureEngineer(kernel_size=kernel_size)
    return engineer.compute_color_difference(img)


def compute_texture_variation(img: np.ndarray) -> np.ndarray:
    """Standalone function - see FeatureEngineer.compute_texture_variation"""
    engineer = FeatureEngineer()
    return engineer.compute_texture_variation(img)


def build_enhanced_img(img: np.ndarray) -> np.ndarray:
    """Standalone function - see FeatureEngineer.build_enhanced_image"""
    engineer = FeatureEngineer()
    return engineer.build_enhanced_image(img)


def highlight_brown_near_green(image_rgb: np.ndarray,
                               brown_thresh: int = 30,
                               green_thresh: int = 30,
                               dilate_size: int = 15) -> np.ndarray:
    """Standalone function - see FeatureEngineer.highlight_brown_near_green"""
    engineer = FeatureEngineer(brown_thresh=brown_thresh,
                              green_thresh=green_thresh,
                              dilate_size=dilate_size)
    return engineer.highlight_brown_near_green(image_rgb)
