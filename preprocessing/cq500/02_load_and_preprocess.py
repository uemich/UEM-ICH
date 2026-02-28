"""
CQ500 Step 2: Load DICOM volumes and preprocess to PNG images

This script:
1. Loads non-contrast DICOM series
2. Applies 3-channel causal windowing
3. Resamples to 5mm slice thickness
4. Resizes to 384x384
5. Saves as PNG images
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm
import config
from utils import (
    load_dicom_series,
    apply_causal_windowing,
    normalize_for_png,
    resample_slice_thickness,
    resize_image,
    save_image_png,
    save_metadata_csv
)


def preprocess_cq500_volumes():
    """
    Load and preprocess all CQ500 non-contrast volumes
    """
    print("=" * 60)
    print("CQ500 Step 2: Load and Preprocess Volumes")
    print("=" * 60)
    
    # Load filtered series list
    filtered_series_path = config.CQ500_OUTPUT_DIR / "metadata" / "non_contrast_series.csv"
    
    if not filtered_series_path.exists():
        print("ERROR: Filtered series list not found!")
        print("Please run 01_filter_series.py first")
        return
    
    filtered_df = pd.read_csv(filtered_series_path)
    print(f"\nLoaded {len(filtered_df)} non-contrast series to process")
    
    # Group by patient to process one patient at a time
    patient_groups = filtered_df.groupby('patient_id')
    print(f"Processing {len(patient_groups)} patients")
    
    # Storage for slice metadata
    all_slice_metadata = []
    total_slices_processed = 0
    
    # Process each patient
    for patient_id, patient_series in tqdm(patient_groups, desc="Processing patients"):
        
        # For each series in this patient (usually just one non-contrast series)
        for idx, row in patient_series.iterrows():
            series_uid = row['series_uid']
            
            # Get all DICOM files for this series
            patient_dir = config.CQ500_UNZIPPED_DIR / patient_id
            
            # Find all DICOM files in patient directory
            all_dicom_files = list(patient_dir.rglob("*.dcm"))
            
            # Filter to this specific series
            series_files = []
            for dcm_file in all_dicom_files:
                from utils import read_dicom_file
                ds = read_dicom_file(dcm_file)
                if ds and hasattr(ds, 'SeriesInstanceUID'):
                    if str(ds.SeriesInstanceUID) == series_uid:
                        series_files.append(dcm_file)
            
            if len(series_files) == 0:
                print(f"Warning: No DICOM files found for {patient_id}, series {series_uid}")
                continue
            
            # Load volume
            try:
                volume, metadata_list = load_dicom_series(series_files)
            except Exception as e:
                print(f"Error loading {patient_id}: {e}")
                continue
            
            # Get slice thickness from first slice
            slice_thickness = metadata_list[0].get('SliceThickness', None)
            
            # Resample to 5mm slice thickness if needed
            if slice_thickness and slice_thickness != config.SLICE_THICKNESS_MM:
                volume = resample_slice_thickness(
                    volume,
                    current_thickness=slice_thickness,
                    target_thickness=config.SLICE_THICKNESS_MM
                )
            
            # Process each slice
            num_slices = volume.shape[2]
            
            for slice_idx in range(num_slices):
                # Extract slice
                slice_hu = volume[:, :, slice_idx]
                
                # Apply 3-channel windowing
                windowed = apply_causal_windowing(slice_hu, config.WINDOW_SETTINGS)
                
                # Resize to 384x384
                resized = resize_image(windowed, config.TARGET_SIZE)
                
                # Convert to PNG format (uint8)
                png_image = normalize_for_png(resized)
                
                # Create filename
                image_filename = f"{patient_id}_slice_{slice_idx:03d}.png"
                image_path = config.CQ500_OUTPUT_DIR / "images" / image_filename
                
                # Save image
                save_image_png(png_image, image_path, compression=config.PNG_COMPRESSION)
                
                # Store metadata
                slice_metadata = {
                    'patient_id': patient_id,
                    'series_uid': series_uid,
                    'slice_idx': slice_idx,
                    'image_filename': image_filename,
                    'original_slice_thickness': slice_thickness,
                    'resampled_slice_thickness': config.SLICE_THICKNESS_MM,
                    'original_size': f"{slice_hu.shape[0]}x{slice_hu.shape[1]}",
                    'final_size': f"{config.TARGET_SIZE[0]}x{config.TARGET_SIZE[1]}",
                }
                
                # Add original metadata if available
                if slice_idx < len(metadata_list):
                    orig_meta = metadata_list[slice_idx]
                    slice_metadata.update({
                        'sop_instance_uid': orig_meta.get('SOPInstanceUID'),
                        'slice_position': orig_meta.get('SlicePosition'),
                        'instance_number': orig_meta.get('InstanceNumber'),
                    })
                
                all_slice_metadata.append(slice_metadata)
                total_slices_processed += 1
    
    # Save slice metadata
    metadata_path = config.CQ500_OUTPUT_DIR / "metadata" / "slice_metadata.csv"
    save_metadata_csv(all_slice_metadata, metadata_path)
    
    print("\n" + "=" * 60)
    print("Preprocessing Statistics")
    print("=" * 60)
    print(f"Total patients processed: {len(patient_groups)}")
    print(f"Total slices processed: {total_slices_processed}")
    print(f"Images saved to: {config.CQ500_OUTPUT_DIR / 'images'}")
    print(f"Metadata saved to: {metadata_path}")
    
    # Calculate storage
    avg_size_per_image = 250  # KB (estimated)
    total_storage_gb = (total_slices_processed * avg_size_per_image) / (1024 * 1024)
    print(f"Estimated storage: ~{total_storage_gb:.2f} GB")
    
    return all_slice_metadata


if __name__ == "__main__":
    slice_metadata = preprocess_cq500_volumes()
    print("\n" + "=" * 60)
    print("✓ CQ500 Step 2 Complete!")
    print("=" * 60)
