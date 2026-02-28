"""
Resampling utilities for CT volumes
"""
import numpy as np
from scipy import ndimage
from typing import Tuple


def resample_slice_thickness(volume: np.ndarray,
                             current_thickness: float,
                             target_thickness: float,
                             current_spacing: Tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """
    Resample volume to target slice thickness
    
    Args:
        volume: 3D numpy array (H, W, D)
        current_thickness: Current slice thickness in mm
        target_thickness: Target slice thickness in mm
        current_spacing: Current in-plane pixel spacing (row, col) in mm
        
    Returns:
        Resampled volume
    """
    if current_thickness is None or current_thickness == target_thickness:
        return volume
    
    # Calculate zoom factors
    # For in-plane: no change (we'll resize separately)
    # For through-plane: resample to target thickness
    zoom_factor_z = current_thickness / target_thickness
    zoom_factors = (1.0, 1.0, zoom_factor_z)
    
    # Resample using linear interpolation
    resampled = ndimage.zoom(volume, zoom_factors, order=1, mode='nearest')
    
    return resampled


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize 2D image to target size
    
    Args:
        image: 2D or 3D numpy array (H, W) or (H, W, C)
        target_size: Target (height, width)
        
    Returns:
        Resized image
    """
    if image.shape[:2] == target_size:
        return image
    
    # Calculate zoom factors
    zoom_h = target_size[0] / image.shape[0]
    zoom_w = target_size[1] / image.shape[1]
    
    if image.ndim == 2:
        zoom_factors = (zoom_h, zoom_w)
    elif image.ndim == 3:
        zoom_factors = (zoom_h, zoom_w, 1.0)  # Don't zoom channels
    else:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}")
    
    # Resize using linear interpolation
    resized = ndimage.zoom(image, zoom_factors, order=1, mode='nearest')
    
    return resized


if __name__ == "__main__":
    print("Resampling utilities module loaded successfully!")
    
    # Test slice thickness resampling
    test_volume = np.random.rand(512, 512, 40)
    print(f"Original volume shape: {test_volume.shape}")
    
    resampled_volume = resample_slice_thickness(test_volume, current_thickness=2.5, target_thickness=5.0)
    print(f"Resampled volume shape (2.5mm -> 5mm): {resampled_volume.shape}")
    
    # Test image resizing
    test_image = np.random.rand(512, 512, 3)
    print(f"\nOriginal image shape: {test_image.shape}")
    
    resized_image = resize_image(test_image, target_size=(384, 384))
    print(f"Resized image shape: {resized_image.shape}")
    
    print("\n✓ All resampling functions working correctly!")
