"""
MBH Segmentation Dataset Preprocessing

Preprocesses NIfTI volumes to:
- 384x384 PNG images with 3-channel causal windowing
- 384x384 PNG masks (resized)
- 5mm slice thickness resampling

Output structure:
preprocessed_data/mbh_seg/
├── images/          # 3-channel CT PNGs
├── masks_annot_2/   # Mask annotation 2
├── masks_annot_3/   # Mask annotation 3
├── masks_combined/  # Combined masks (union of all annotations)
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
RAW_DATA_DIR = config.MBH_TRAIN_VOXEL_DIR
OUTPUT_DIR = config.MBH_SEG_OUTPUT_DIR

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
        OUTPUT_DIR / "masks_annot_2",
        OUTPUT_DIR / "masks_annot_3",
        OUTPUT_DIR / "masks_combined",
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


def resize_mask(mask, target_size):
    """Resize mask using nearest neighbor (preserve label values)."""
    return cv2.resize(mask.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)


def resample_volume(volume, current_spacing, target_spacing):
    """Resample volume to target spacing along z-axis."""
    if abs(current_spacing - target_spacing) < 0.1:
        return volume
    zoom_factor = current_spacing / target_spacing
    return ndimage.zoom(volume, (1, 1, zoom_factor), order=1)


def resample_mask(mask, current_spacing, target_spacing):
    """Resample mask volume using nearest neighbor."""
    if abs(current_spacing - target_spacing) < 0.1:
        return mask
    zoom_factor = current_spacing / target_spacing
    return ndimage.zoom(mask, (1, 1, zoom_factor), order=0)


def get_slice_thickness(nifti_img):
    """Extract slice thickness from NIfTI header."""
    header = nifti_img.header
    pixdim = header.get_zooms()
    if len(pixdim) >= 3:
        return pixdim[2]
    return 5.0


def process_scan(scan_dir, output_dir):
    """
    Process a single scan and its masks.

    Returns:
        List of metadata dictionaries
    """
    scan_id = scan_dir.name
    metadata_list = []

    ct_path = scan_dir / "image.nii.gz"
    if not ct_path.exists():
        print(f"  Warning: No CT found for {scan_id}")
        return []

    ct_nii = nib.load(ct_path)
    ct_volume = ct_nii.get_fdata()
    slice_thickness = get_slice_thickness(ct_nii)

    # Load masks
    mask_files = sorted(scan_dir.glob("label_*.nii.gz"))
    masks = {}
    for mf in mask_files:
        mask_name = mf.stem.replace('.nii', '').replace('label_', '')
        mask_nii = nib.load(mf)
        masks[mask_name] = mask_nii.get_fdata()

    # Resample to target thickness if needed
    if abs(slice_thickness - TARGET_THICKNESS) >= 0.5:
        ct_volume = resample_volume(ct_volume, slice_thickness, TARGET_THICKNESS)
        for mask_name in masks:
            masks[mask_name] = resample_mask(masks[mask_name], slice_thickness, TARGET_THICKNESS)

    num_slices = ct_volume.shape[2]

    for slice_idx in range(num_slices):
        ct_slice = ct_volume[:, :, slice_idx]
        ct_windowed = apply_causal_windowing(ct_slice)
        ct_windowed = np.rot90(ct_windowed)
        ct_resized = resize_image(ct_windowed, TARGET_SIZE)

        image_filename = f"{scan_id}_slice_{slice_idx:03d}.png"

        ct_img = Image.fromarray(ct_resized)
        ct_img.save(output_dir / "images" / image_filename)

        combined_mask = np.zeros(TARGET_SIZE, dtype=np.uint8)

        for mask_name, mask_volume in masks.items():
            mask_slice = mask_volume[:, :, slice_idx]
            mask_slice = np.rot90(mask_slice)
            mask_resized = resize_mask(mask_slice, TARGET_SIZE)
            mask_multiclass = mask_resized.astype(np.uint8)

            mask_dir = output_dir / f"masks_{mask_name}"
            mask_dir.mkdir(parents=True, exist_ok=True)

            mask_filename = f"{scan_id}_slice_{slice_idx:03d}.png"
            mask_img = Image.fromarray(mask_multiclass)
            mask_img.save(mask_dir / mask_filename)

            combined_mask = np.maximum(combined_mask, mask_multiclass)

        combined_img = Image.fromarray(combined_mask)
        combined_img.save(output_dir / "masks_combined" / image_filename)

        has_hemorrhage = combined_mask.max() > 0

        metadata_list.append({
            'scan_id': scan_id,
            'slice_idx': slice_idx,
            'image_filename': image_filename,
            'original_thickness': slice_thickness,
            'resampled_thickness': TARGET_THICKNESS,
            'has_hemorrhage': has_hemorrhage,
            'num_mask_annotations': len(masks),
        })

    return metadata_list


def main():
    """Main preprocessing function."""
    print("=" * 60)
    print("MBH Segmentation Dataset Preprocessing")
    print("=" * 60)
    print(f"Target size: {TARGET_SIZE}")
    print(f"Target thickness: {TARGET_THICKNESS}mm")
    print(f"Windowing: Brain={WINDOW_SETTINGS['brain']}, "
          f"Subdural={WINDOW_SETTINGS['subdural']}, "
          f"Bone={WINDOW_SETTINGS['bone']}")
    print("=" * 60)

    create_output_dirs()

    scan_dirs = sorted([d for d in RAW_DATA_DIR.iterdir() if d.is_dir()])
    print(f"\nFound {len(scan_dirs)} scans to process")

    all_metadata = []
    for scan_dir in tqdm(scan_dirs, desc="Processing scans"):
        metadata = process_scan(scan_dir, OUTPUT_DIR)
        all_metadata.extend(metadata)

    metadata_df = pd.DataFrame(all_metadata)
    metadata_path = OUTPUT_DIR / "metadata" / "slice_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)

    print("\n" + "=" * 60)
    print("Preprocessing Complete")
    print("=" * 60)
    print(f"Total scans: {len(scan_dirs)}")
    print(f"Total slices: {len(all_metadata)}")
    print(f"Slices with hemorrhage: {metadata_df['has_hemorrhage'].sum()}")
    print(f"Images saved to: {OUTPUT_DIR / 'images'}")
    print(f"Masks saved to: {OUTPUT_DIR / 'masks_*'}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
