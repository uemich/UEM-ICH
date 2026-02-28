"""
MBH Segmentation Scan Preprocessing

Preprocesses NIfTI volumes to:
- 384x384 PNG images with 3-channel causal windowing
- 5mm slice thickness resampling

Input:  raw_data/MBH/Train_case
Output: preprocessed_data/MBH_Seg_scan/
        ├── images/          # 3-channel CT PNGs
        └── metadata/        # Slice metadata
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm
import pandas as pd
from scipy import ndimage
import cv2
import config


# =============================================================================
# Configuration
# =============================================================================
RAW_DATA_DIR = config.MBH_TRAIN_CASE_DIR
OUTPUT_DIR = config.MBH_SCAN_OUTPUT_DIR

TARGET_SIZE = config.TARGET_SIZE
TARGET_THICKNESS = config.SLICE_THICKNESS_MM
WINDOW_SETTINGS = config.WINDOW_SETTINGS


# =============================================================================
# Utility Functions
# =============================================================================
def create_output_dirs():
    """Create output directories."""
    dirs = [
        OUTPUT_DIR,
        OUTPUT_DIR / "images",
        OUTPUT_DIR / "metadata",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")


def apply_window(ct_array, center, width):
    """Apply CT windowing."""
    min_val = center - width / 2
    max_val = center + width / 2
    windowed = np.clip(ct_array, min_val, max_val)
    normalized = (windowed - min_val) / (max_val - min_val)
    return (normalized * 255).astype(np.uint8)


def apply_causal_windowing(ct_slice):
    """Apply 3-channel causal windowing (brain, subdural, bone)."""
    brain = apply_window(ct_slice,
                         WINDOW_SETTINGS['brain']['center'],
                         WINDOW_SETTINGS['brain']['width'])
    subdural = apply_window(ct_slice,
                            WINDOW_SETTINGS['subdural']['center'],
                            WINDOW_SETTINGS['subdural']['width'])
    bone = apply_window(ct_slice,
                        WINDOW_SETTINGS['bone']['center'],
                        WINDOW_SETTINGS['bone']['width'])
    return np.stack([brain, subdural, bone], axis=-1)


def resize_image(img, target_size):
    """Resize image using high-quality interpolation."""
    return cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)


def resample_volume(volume, current_spacing, target_spacing):
    """Resample volume to target spacing along z-axis."""
    if abs(current_spacing - target_spacing) < 0.1:
        return volume
    zoom_factor = current_spacing / target_spacing
    return ndimage.zoom(volume, (1, 1, zoom_factor), order=1)


def get_slice_thickness(nifti_img):
    """Extract slice thickness from NIfTI header."""
    header = nifti_img.header
    pixdim = header.get_zooms()
    if len(pixdim) >= 3:
        return pixdim[2]
    return 5.0


def process_scan(nii_path, output_dir):
    """Process a single NIfTI scan."""
    scan_id = nii_path.stem.replace('.nii', '')
    metadata_list = []

    try:
        ct_nii = nib.load(nii_path)
        ct_volume = ct_nii.get_fdata()
        slice_thickness = get_slice_thickness(ct_nii)
    except Exception as e:
        print(f"Error loading {nii_path}: {e}")
        return []

    # Resample
    if abs(slice_thickness - TARGET_THICKNESS) >= 0.5:
        ct_volume = resample_volume(ct_volume, slice_thickness, TARGET_THICKNESS)

    num_slices = ct_volume.shape[2]

    for slice_idx in range(num_slices):
        ct_slice = ct_volume[:, :, slice_idx]
        ct_windowed = apply_causal_windowing(ct_slice)
        ct_windowed = np.rot90(ct_windowed)
        ct_resized = resize_image(ct_windowed, TARGET_SIZE)

        image_filename = f"{scan_id}_slice_{slice_idx:03d}.png"
        ct_img = Image.fromarray(ct_resized)
        ct_img.save(output_dir / "images" / image_filename)

        metadata_list.append({
            'scan_id': scan_id,
            'slice_idx': slice_idx,
            'image_filename': image_filename,
            'original_thickness': slice_thickness,
            'resampled_thickness': TARGET_THICKNESS,
        })

    return metadata_list


def main():
    print("=" * 60)
    print("MBH Seg Scan Preprocessing")
    print("=" * 60)

    create_output_dirs()

    nii_files = sorted(list(RAW_DATA_DIR.glob("*.nii.gz")))
    print(f"\nFound {len(nii_files)} NII files")

    all_metadata = []
    for nii_path in tqdm(nii_files, desc="Processing scans"):
        metadata = process_scan(nii_path, OUTPUT_DIR)
        all_metadata.extend(metadata)

    metadata_df = pd.DataFrame(all_metadata)
    metadata_path = OUTPUT_DIR / "metadata" / "slice_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)

    print("\nPreprocessing Complete")
    print(f"Total slices: {len(all_metadata)}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
