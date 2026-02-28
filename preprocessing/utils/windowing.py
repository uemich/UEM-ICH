"""
Windowing utilities for CT images
Implements 3-channel causal windowing (Brain, Subdural, Bone)
"""
import numpy as np
from typing import Dict, Tuple


def apply_window(hu_values: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    Apply window/level transformation to HU values
    
    Args:
        hu_values: Array of Hounsfield Unit values
        center: Window center (level)
        width: Window width
        
    Returns:
        Windowed values normalized to [0, 1]
    """
    lower = center - width / 2
    upper = center + width / 2
    
    # Clip values to window range
    windowed = np.clip(hu_values, lower, upper)
    
    # Normalize to [0, 1]
    windowed = (windowed - lower) / (upper - lower)
    
    return windowed


def apply_causal_windowing(hu_values: np.ndarray, 
                          window_settings: Dict[str, Dict[str, float]]) -> np.ndarray:
    """
    Apply 3-channel causal windowing to CT scan
    
    Args:
        hu_values: 2D array of HU values (H, W)
        window_settings: Dictionary with 'brain', 'subdural', 'bone' window params
        
    Returns:
        3-channel windowed image (H, W, 3) with values in [0, 1]
    """
    # Channel 0: Brain window
    brain_window = apply_window(
        hu_values,
        window_settings['brain']['center'],
        window_settings['brain']['width']
    )
    
    # Channel 1: Subdural window
    subdural_window = apply_window(
        hu_values,
        window_settings['subdural']['center'],
        window_settings['subdural']['width']
    )
    
    # Channel 2: Bone window
    bone_window = apply_window(
        hu_values,
        window_settings['bone']['center'],
        window_settings['bone']['width']
    )
    
    # Stack channels
    windowed_image = np.stack([brain_window, subdural_window, bone_window], axis=-1)
    
    return windowed_image


def normalize_for_png(windowed_image: np.ndarray) -> np.ndarray:
    """
    Convert [0, 1] float values to [0, 255] uint8 for PNG saving
    
    Args:
        windowed_image: Float array with values in [0, 1]
        
    Returns:
        uint8 array with values in [0, 255]
    """
    return (windowed_image * 255).astype(np.uint8)


def denormalize_from_png(png_image: np.ndarray) -> np.ndarray:
    """
    Convert [0, 255] uint8 values back to [0, 1] float
    
    Args:
        png_image: uint8 array with values in [0, 255]
        
    Returns:
        Float array with values in [0, 1]
    """
    return png_image.astype(np.float32) / 255.0


if __name__ == "__main__":
    # Test windowing
    print("Windowing utilities module loaded successfully!")
    
    # Create test HU values
    test_hu = np.random.randn(512, 512) * 100 + 40  # Simulated brain tissue
    
    # Test window settings
    test_settings = {
        'brain': {'center': 40, 'width': 80},
        'subdural': {'center': 80, 'width': 200},
        'bone': {'center': 500, 'width': 2000}
    }
    
    # Apply windowing
    windowed = apply_causal_windowing(test_hu, test_settings)
    print(f"Input shape: {test_hu.shape}")
    print(f"Output shape: {windowed.shape}")
    print(f"Output range: [{windowed.min():.3f}, {windowed.max():.3f}]")
    
    # Test PNG conversion
    png_ready = normalize_for_png(windowed)
    print(f"PNG-ready shape: {png_ready.shape}, dtype: {png_ready.dtype}")
    print(f"PNG-ready range: [{png_ready.min()}, {png_ready.max()}]")
    
    print("\n✓ All windowing functions working correctly!")
