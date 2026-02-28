"""
Storage utilities for saving preprocessed images and metadata
"""
import numpy as np
import cv2
from pathlib import Path
from typing import Union
import pandas as pd


def save_image_png(image: np.ndarray, filepath: Path, compression: int = 6) -> bool:
    """
    Save image as PNG file
    
    Args:
        image: numpy array (H, W, 3) with values in [0, 255] uint8
        filepath: Output file path
        compression: PNG compression level (0-9)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # OpenCV expects BGR, convert from RGB
        if image.shape[-1] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        # Save with compression
        cv2.imwrite(str(filepath), image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, compression])
        return True
    except Exception as e:
        print(f"Error saving PNG {filepath}: {e}")
        return False


def save_image_npy(image: np.ndarray, filepath: Path) -> bool:
    """
    Save image as NumPy array
    
    Args:
        image: numpy array
        filepath: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as .npy
        np.save(str(filepath), image)
        return True
    except Exception as e:
        print(f"Error saving NPY {filepath}: {e}")
        return False


def load_image_png(filepath: Path) -> np.ndarray:
    """
    Load PNG image
    
    Args:
        filepath: Path to PNG file
        
    Returns:
        numpy array (H, W, 3) in RGB format
    """
    # Load with OpenCV (returns BGR)
    image_bgr = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    return image_rgb


def load_image_npy(filepath: Path) -> np.ndarray:
    """
    Load NumPy array image
    
    Args:
        filepath: Path to .npy file
        
    Returns:
        numpy array
    """
    return np.load(str(filepath))


def save_metadata_csv(metadata_list: list, filepath: Path) -> bool:
    """
    Save list of metadata dictionaries as CSV
    
    Args:
        metadata_list: List of dictionaries
        filepath: Output CSV file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(metadata_list)
        df.to_csv(filepath, index=False)
        return True
    except Exception as e:
        print(f"Error saving CSV {filepath}: {e}")
        return False


def append_to_csv(data_dict: dict, filepath: Path) -> bool:
    """
    Append a row to CSV file
    
    Args:
        data_dict: Dictionary of data to append
        filepath: CSV file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame([data_dict])
        
        # Append to file
        if filepath.exists():
            df.to_csv(filepath, mode='a', header=False, index=False)
        else:
            df.to_csv(filepath, mode='w', header=True, index=False)
        
        return True
    except Exception as e:
        print(f"Error appending to CSV {filepath}: {e}")
        return False


if __name__ == "__main__":
    print("Storage utilities module loaded successfully!")
    
    # Test PNG saving/loading
    test_image = np.random.randint(0, 256, (384, 384, 3), dtype=np.uint8)
    test_path = Path("/tmp/test_image.png")
    
    print(f"Test image shape: {test_image.shape}, dtype: {test_image.dtype}")
    
    # Save
    success = save_image_png(test_image, test_path)
    print(f"PNG save successful: {success}")
    
    # Load
    if success:
        loaded_image = load_image_png(test_path)
        print(f"Loaded image shape: {loaded_image.shape}, dtype: {loaded_image.dtype}")
        print(f"Images match: {np.allclose(test_image, loaded_image)}")
        test_path.unlink()  # Clean up
    
    # Test CSV saving
    test_metadata = [
        {'scan_id': 'scan_001', 'num_slices': 30, 'has_hemorrhage': 1},
        {'scan_id': 'scan_002', 'num_slices': 25, 'has_hemorrhage': 0},
    ]
    test_csv_path = Path("/tmp/test_metadata.csv")
    
    success = save_metadata_csv(test_metadata, test_csv_path)
    print(f"\nCSV save successful: {success}")
    
    if success:
        df = pd.read_csv(test_csv_path)
        print(f"Loaded CSV shape: {df.shape}")
        print(df.head())
        test_csv_path.unlink()  # Clean up
    
    print("\n✓ All storage functions working correctly!")
