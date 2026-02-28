"""
PhysioNet CT-ICH Dataset Preprocessing

Preprocesses NIfTI volumes to:
- 384x384 PNG images with 3-channel causal windowing
- 384x384 PNG binary masks
- Slice-level labels CSV with ICH subtypes + fracture

Output structure:
preprocessed_data/physionet/
├── images/              # 3-channel CT PNGs
├── masks/               # Binary mask PNGs
└── labels/
    └── slice_labels.csv # Slice-level labels
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
import cv2
import config


# =============================================================================
# Configuration
# =============================================================================
RAW_DATA_DIR = config.PHYSIONET_RAW_DIR
OUTPUT_DIR = config.PHYSIONET_OUTPUT_DIR

TARGET_SIZE = config.TARGET_SIZE
WINDOW_SETTINGS = config.WINDOW_SETTINGS

# Label columns
ICH_COLS = ['Intraventricular', 'Intraparenchymal', 'Subarachnoid', 'Epidural', 'Subdural']
ALL_LABEL_COLS = ICH_COLS + ['any', 'Fracture_Yes_No']


# =============================================================================
# Utility Functions
# =============================================================================
def create_output_dirs():
    """Create output directories."""
    dirs = [
        OUTPUT_DIR,
        OUTPUT_DIR / "images",
        OUTPUT_DIR / "masks",
        OUTPUT_DIR / "labels",
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
    """Resize mask using nearest neighbor (preserve binary values)."""
    return cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)


def load_labels_csv():
    """Load and process the hemorrhage diagnosis CSV."""
    csv_path = RAW_DATA_DIR / "hemorrhage_diagnosis_raw_ct.csv"
    df = pd.read_csv(csv_path)
    df['any'] = df[ICH_COLS].max(axis=1)
    return df


def process_patient(patient_id, labels_df, output_dir):
    """
    Process a single patient's CT scan and mask.

    Returns list of slice metadata.
    """
    ct_path = RAW_DATA_DIR / "ct_scans" / f"{patient_id:03d}.nii"
    mask_path = RAW_DATA_DIR / "masks" / f"{patient_id:03d}.nii"

    if not ct_path.exists():
        print(f"  Skipping patient {patient_id}: CT not found")
        return []

    # Load NIfTI volumes
    ct_nii = nib.load(ct_path)
    ct_data = ct_nii.get_fdata()

    mask_nii = nib.load(mask_path)
    mask_data = mask_nii.get_fdata()
    mask_data = (mask_data > 127).astype(np.uint8)

    # Get patient's labels
    patient_labels = labels_df[labels_df['PatientNumber'] == patient_id]

    slice_metadata = []
    num_slices = ct_data.shape[2]

    for slice_idx in range(num_slices):
        slice_num = slice_idx + 1

        ct_slice = ct_data[:, :, slice_idx]
        ct_rgb = apply_causal_windowing(ct_slice)
        ct_resized = resize_image(ct_rgb, TARGET_SIZE)

        mask_slice = mask_data[:, :, slice_idx]
        mask_resized = resize_mask(mask_slice, TARGET_SIZE)

        filename = f"{patient_id:03d}_{slice_num:03d}.png"

        img_pil = Image.fromarray(ct_resized)
        img_pil.save(output_dir / "images" / filename)

        mask_pil = Image.fromarray(mask_resized * 255)
        mask_pil.save(output_dir / "masks" / filename)

        slice_labels = patient_labels[patient_labels['SliceNumber'] == slice_num]

        if len(slice_labels) == 1:
            row = slice_labels.iloc[0]
            metadata = {
                'filename': filename,
                'patient_id': patient_id,
                'slice_num': slice_num,
                'Intraventricular': int(row['Intraventricular']),
                'Intraparenchymal': int(row['Intraparenchymal']),
                'Subarachnoid': int(row['Subarachnoid']),
                'Epidural': int(row['Epidural']),
                'Subdural': int(row['Subdural']),
                'any': int(row['any']),
                'Fracture_Yes_No': int(row['Fracture_Yes_No']),
            }
        else:
            metadata = {
                'filename': filename,
                'patient_id': patient_id,
                'slice_num': slice_num,
                'Intraventricular': 0, 'Intraparenchymal': 0,
                'Subarachnoid': 0, 'Epidural': 0, 'Subdural': 0,
                'any': 0, 'Fracture_Yes_No': 0,
            }

        slice_metadata.append(metadata)

    return slice_metadata


def main():
    print("=" * 60)
    print("PhysioNet CT-ICH Dataset Preprocessing")
    print("=" * 60)

    create_output_dirs()

    print("\nLoading labels CSV...")
    labels_df = load_labels_csv()
    print(f"  Total slices in CSV: {len(labels_df)}")

    ct_dir = RAW_DATA_DIR / "ct_scans"
    patient_ids = sorted([int(f.stem) for f in ct_dir.glob("*.nii")])
    print(f"  Patients found: {len(patient_ids)}")

    all_metadata = []
    for patient_id in tqdm(patient_ids, desc="Processing patients"):
        patient_metadata = process_patient(patient_id, labels_df, OUTPUT_DIR)
        all_metadata.extend(patient_metadata)

    labels_out_path = OUTPUT_DIR / "labels" / "slice_labels.csv"
    df_out = pd.DataFrame(all_metadata)
    df_out.to_csv(labels_out_path, index=False)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Total slices processed: {len(all_metadata)}")
    print(f"Labels saved to: {labels_out_path}")

    print("\nLabel distribution:")
    for col in ALL_LABEL_COLS:
        count = df_out[col].sum()
        pct = 100 * count / len(df_out)
        print(f"  {col}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
