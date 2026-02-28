"""
CQ500 Bounding Box Preprocessing (Fast Version)

Re-preprocesses the 405 non-contrast CQ500 scans WITHOUT z-resampling
to maintain 1:1 DICOM slice mapping for bounding box label alignment.

Key optimization: processes slices independently (no 3D volume assembly)
with multiprocessing for ~10x speedup.

Pipeline per slice:
  1. Load single DICOM → HU values
  2. Apply 3-channel causal windowing (brain/subdural/bone)
  3. Resize to 384×384
  4. Save as 8-bit PNG
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))  # Add preprocessing/ to path

import config

import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from utils import (
    read_dicom_file,
    apply_causal_windowing,
    normalize_for_png,
    resize_image,
    save_image_png,
)
from utils.dicom_utils import get_hu_values, get_slice_metadata

# =============================================================================
# Configuration
# =============================================================================
CQ500_UNZIPPED_DIR = config.CQ500_UNZIPPED_DIR
NC_SERIES_CSV = config.CQ500_OUTPUT_DIR / "metadata" / "non_contrast_series.csv"

OUTPUT_DIR = config.PREPROCESSED_DIR / "cq500_bbox"
IMAGES_DIR = OUTPUT_DIR / "images"
METADATA_DIR = OUTPUT_DIR / "metadata"

TARGET_SIZE = config.TARGET_SIZE
PNG_COMPRESSION = config.PNG_COMPRESSION
WINDOW_SETTINGS = config.WINDOW_SETTINGS
NUM_WORKERS = 8


# =============================================================================
# Per-slice processing (runs in worker process)
# =============================================================================
def process_single_slice(args):
    """Process a single DICOM file to a windowed PNG. Runs in a worker process."""
    dcm_path, clean_pid, slice_idx, series_uid, images_dir = args
    
    try:
        ds = read_dicom_file(Path(dcm_path))
        if ds is None:
            return None
        
        hu_values = get_hu_values(ds)
        metadata = get_slice_metadata(ds)
        
        # 3-channel causal windowing
        windowed = apply_causal_windowing(hu_values, WINDOW_SETTINGS)
        
        # Resize to 384x384
        resized = resize_image(windowed, TARGET_SIZE)
        
        # Convert to uint8
        png_image = normalize_for_png(resized)
        
        # Save
        image_filename = f"{clean_pid}_slice_{slice_idx:03d}.png"
        image_path = Path(images_dir) / image_filename
        save_image_png(png_image, image_path, compression=PNG_COMPRESSION)
        
        return {
            'patient_id': clean_pid,
            'series_uid': series_uid,
            'slice_idx': slice_idx,
            'image_filename': image_filename,
            'sop_instance_uid': metadata.get('SOPInstanceUID'),
            'slice_position': metadata.get('SlicePosition'),
            'instance_number': metadata.get('InstanceNumber'),
            'slice_thickness': metadata.get('SliceThickness'),
            'original_size': f"{hu_values.shape[0]}x{hu_values.shape[1]}",
            'final_size': f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}",
        }
    except Exception as e:
        return {'error': f"{clean_pid} slice {slice_idx}: {e}"}


# =============================================================================
# Find series directory efficiently
# =============================================================================
def find_series_dir(patient_dir, series_uid):
    """Find the subdirectory containing the target series by reading one DICOM per subdir."""
    for sub_dir in patient_dir.rglob("*"):
        if not sub_dir.is_dir():
            continue
        dcm_files = list(sub_dir.glob("*.dcm"))
        if not dcm_files:
            continue
        ds = read_dicom_file(dcm_files[0])
        if ds and hasattr(ds, 'SeriesInstanceUID'):
            if str(ds.SeriesInstanceUID) == series_uid:
                return dcm_files
    return []


# =============================================================================
# Sort DICOM files by slice position
# =============================================================================
def sort_dicom_files(dcm_files):
    """Sort DICOM files by slice position (read only headers, not pixel data)."""
    import pydicom
    file_positions = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            pos = None
            if hasattr(ds, 'ImagePositionPatient') and ds.ImagePositionPatient:
                pos = float(ds.ImagePositionPatient[2])
            elif hasattr(ds, 'InstanceNumber'):
                pos = float(ds.InstanceNumber)
            file_positions.append((f, pos if pos is not None else 0))
        except:
            file_positions.append((f, 0))
    
    file_positions.sort(key=lambda x: x[1])
    return [f for f, _ in file_positions]


# =============================================================================
# Main
# =============================================================================
def preprocess_bbox_volumes():
    print("=" * 60)
    print("CQ500 BBox Preprocessing (Fast, No Z-Resampling)")
    print("=" * 60)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    if not NC_SERIES_CSV.exists():
        print(f"ERROR: Non-contrast series list not found: {NC_SERIES_CSV}")
        return

    filtered_df = pd.read_csv(NC_SERIES_CSV)
    print(f"\nLoaded {len(filtered_df)} non-contrast series to process")

    patient_groups = filtered_df.groupby('patient_id')
    print(f"Processing {len(patient_groups)} patients with {NUM_WORKERS} workers\n")

    # Phase 1: Collect all slice tasks (sequential, fast — only reads headers)
    all_tasks = []
    errors = []

    for patient_id, patient_series in tqdm(patient_groups, desc="Scanning DICOM dirs"):
        for _, row in patient_series.iterrows():
            series_uid = row['series_uid']
            patient_dir = CQ500_UNZIPPED_DIR / patient_id

            if not patient_dir.exists():
                errors.append(f"Directory not found: {patient_dir}")
                continue

            dcm_files = find_series_dir(patient_dir, series_uid)
            if not dcm_files:
                errors.append(f"No DICOM files for {patient_id}, series {series_uid}")
                continue

            # Sort by slice position (reads headers only, no pixel data)
            sorted_files = sort_dicom_files(dcm_files)
            clean_pid = patient_id.split()[0]

            for slice_idx, dcm_path in enumerate(sorted_files):
                all_tasks.append((
                    str(dcm_path), clean_pid, slice_idx, series_uid, str(IMAGES_DIR)
                ))

    print(f"\nTotal slices to process: {len(all_tasks)}")

    # Phase 2: Process all slices in parallel
    all_metadata = []
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_single_slice, task) for task in all_tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing slices"):
            result = future.result()
            if result is None:
                continue
            if 'error' in result:
                errors.append(result['error'])
            else:
                all_metadata.append(result)

    # Sort metadata by patient_id and slice_idx for deterministic output
    all_metadata.sort(key=lambda x: (x['patient_id'], x['slice_idx']))

    # Save metadata
    metadata_path = METADATA_DIR / "slice_metadata.csv"
    pd.DataFrame(all_metadata).to_csv(metadata_path, index=False)

    # Summary
    print("\n" + "=" * 60)
    print("Preprocessing Statistics")
    print("=" * 60)
    unique_patients = len(set(m['patient_id'] for m in all_metadata))
    print(f"Total patients processed: {unique_patients}")
    print(f"Total slices processed:   {len(all_metadata)}")
    print(f"Images saved to:          {IMAGES_DIR}")
    print(f"Metadata saved to:        {metadata_path}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  - {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    # BHX verification (optional)
    try:
        bhx_path = config.RAW_DATA_DIR / "bhx" / "1_Initial_Manual_Labeling.csv"
        if bhx_path.exists():
            bhx = pd.read_csv(bhx_path)
            our_sops = set(m['sop_instance_uid'] for m in all_metadata if m['sop_instance_uid'])
            bhx_sops = set(bhx['SOPInstanceUID'].unique())
            overlap = our_sops & bhx_sops
            print(f"\nBHX File 1 Verification:")
            print(f"  Our SOPInstanceUIDs:  {len(our_sops)}")
            print(f"  BHX SOPInstanceUIDs:  {len(bhx_sops)}")
            print(f"  Overlap:              {len(overlap)}")
            print(f"  BHX coverage:         {100 * len(overlap) / len(bhx_sops):.1f}%")
    except Exception as e:
        print(f"\nSkipping BHX verification: {e}")


if __name__ == "__main__":
    preprocess_bbox_volumes()
    print("\n" + "=" * 60)
    print("✓ CQ500 BBox Preprocessing Complete!")
    print("=" * 60)
