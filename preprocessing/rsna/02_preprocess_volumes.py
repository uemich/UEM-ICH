"""
RSNA Step 2: Preprocess volumes to PNG images

This script:
1. Loads volumes using the volume index
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
    read_dicom_file,
    get_hu_values,
    apply_causal_windowing,
    normalize_for_png,
    resample_slice_thickness,
    resize_image,
    save_image_png,
    save_metadata_csv
)


def preprocess_rsna_volumes():
    """
    Load and preprocess all RSNA volumes
    """
    print("=" * 60)
    print("RSNA Step 2: Preprocess Volumes")
    print("=" * 60)
    
    # Load volume mapping
    volume_mapping_path = config.RSNA_OUTPUT_DIR / "metadata" / "volume_mapping.csv"
    
    if not volume_mapping_path.exists():
        print("ERROR: Volume mapping not found!")
        print("Please run 01_build_volume_index.py first")
        return
    
    volume_mapping_df = pd.read_csv(volume_mapping_path)
    print(f"\nLoaded {len(volume_mapping_df)} volumes to process")
    
    # Load slice index for metadata
    slice_index_path = config.RSNA_OUTPUT_DIR / "metadata" / "slice_index.csv"
    slice_index_df = pd.read_csv(slice_index_path)
    
    # Storage for slice metadata
    all_slice_metadata = []
    total_slices_processed = 0
    skipped_volumes = 0
    corrupted_files = []
    
    # Process each volume
    for idx, vol_row in tqdm(volume_mapping_df.iterrows(), total=len(volume_mapping_df), desc="Processing volumes"):
        volume_id = vol_row['volume_id']
        series_uid = vol_row['series_uid']
        
        # Check if this volume is already processed (for resuming)
        first_slice_path = config.RSNA_OUTPUT_DIR / "images" / f"{volume_id}_slice_000.png"
        if first_slice_path.exists():
            # Volume already processed, skip
            skipped_volumes += 1
            continue
        
        # Get slice filenames for this volume
        slice_filenames = vol_row['slice_filenames'].split(',')
        
        # Get file paths
        slice_paths = []
        for filename in slice_filenames:
            filepath = config.RSNA_TRAIN_DIR / f"{filename}.dcm"
            if filepath.exists():
                slice_paths.append(filepath)
        
        if len(slice_paths) == 0:
            continue
        
        # Load slices and build volume
        slices_data = []
        for slice_path in slice_paths:
            try:
                ds = read_dicom_file(slice_path)
                if ds is None:
                    continue
                
                hu_values = get_hu_values(ds)
                slices_data.append({
                    'hu': hu_values,
                    'filename': slice_path.stem,
                    'thickness': float(ds.SliceThickness) if hasattr(ds, 'SliceThickness') else None
                })
            except Exception as e:
                # Handle corrupted DICOM files
                corrupted_files.append({
                    'volume_id': volume_id,
                    'filename': slice_path.stem,
                    'error': str(e)
                })
                continue
        
        if len(slices_data) == 0:
            print(f"\nWarning: No valid slices for volume {volume_id}")
            continue
        
        # Stack into volume
        try:
            volume = np.stack([s['hu'] for s in slices_data], axis=-1)  # (H, W, D)
        except Exception as e:
            print(f"\nError stacking volume {volume_id}: {e}")
            continue
            
        slice_thickness = slices_data[0]['thickness']
        
        # Resample to 5mm if needed
        if slice_thickness and slice_thickness != config.SLICE_THICKNESS_MM:
            try:
                volume = resample_slice_thickness(
                    volume,
                    current_thickness=slice_thickness,
                    target_thickness=config.SLICE_THICKNESS_MM
                )
            except Exception as e:
                print(f"\nError resampling volume {volume_id}: {e}")
                continue
        
        # Process each slice
        num_slices = volume.shape[2]
        
        for slice_idx in range(num_slices):
            try:
                # Extract slice
                slice_hu = volume[:, :, slice_idx]
                
                # Apply 3-channel windowing
                windowed = apply_causal_windowing(slice_hu, config.WINDOW_SETTINGS)
                
                # Resize to 384x384
                resized = resize_image(windowed, config.TARGET_SIZE)
                
                # Convert to PNG format
                png_image = normalize_for_png(resized)
                
                # Create filename
                image_filename = f"{volume_id}_slice_{slice_idx:03d}.png"
                image_path = config.RSNA_OUTPUT_DIR / "images" / image_filename
                
                # Save image
                save_image_png(png_image, image_path, compression=config.PNG_COMPRESSION)
                
                # Store metadata
                original_filename = slices_data[slice_idx]['filename'] if slice_idx < len(slices_data) else None
                
                slice_metadata = {
                    'volume_id': volume_id,
                    'series_uid': series_uid,
                    'slice_idx': slice_idx,
                    'image_filename': image_filename,
                    'original_filename': original_filename,
                    'original_slice_thickness': slice_thickness,
                    'resampled_slice_thickness': config.SLICE_THICKNESS_MM,
                    'final_size': f"{config.TARGET_SIZE[0]}x{config.TARGET_SIZE[1]}",
                }
                
                all_slice_metadata.append(slice_metadata)
                total_slices_processed += 1
            except Exception as e:
                print(f"\nError processing slice {slice_idx} of volume {volume_id}: {e}")
                continue
        
        # Save metadata periodically (every 1000 volumes)
        if (idx + 1) % 1000 == 0:
            temp_metadata_path = config.RSNA_OUTPUT_DIR / "metadata" / f"slice_metadata_temp_{idx+1}.csv"
            save_metadata_csv(all_slice_metadata[-num_slices:], temp_metadata_path)
    
    # Save final slice metadata
    metadata_path = config.RSNA_OUTPUT_DIR / "metadata" / "slice_metadata.csv"
    save_metadata_csv(all_slice_metadata, metadata_path)
    
    # Save corrupted files log if any
    if len(corrupted_files) > 0:
        corrupted_log_path = config.RSNA_OUTPUT_DIR / "metadata" / "corrupted_files.csv"
        save_metadata_csv(corrupted_files, corrupted_log_path)
        print(f"\n⚠ Saved corrupted files log to: {corrupted_log_path}")
    
    print("\n" + "=" * 60)
    print("Preprocessing Statistics")
    print("=" * 60)
    print(f"Total volumes to process: {len(volume_mapping_df)}")
    print(f"Volumes already processed (skipped): {skipped_volumes}")
    print(f"Volumes newly processed: {len(volume_mapping_df) - skipped_volumes}")
    print(f"Total slices processed: {total_slices_processed}")
    if len(corrupted_files) > 0:
        print(f"⚠ Corrupted DICOM files encountered: {len(corrupted_files)}")
    print(f"Images saved to: {config.RSNA_OUTPUT_DIR / 'images'}")
    print(f"Metadata saved to: {metadata_path}")
    
    # Calculate storage
    avg_size_per_image = 250  # KB (estimated)
    total_storage_gb = (total_slices_processed * avg_size_per_image) / (1024 * 1024)
    print(f"Estimated storage: ~{total_storage_gb:.2f} GB")
    
    return all_slice_metadata


if __name__ == "__main__":
    slice_metadata = preprocess_rsna_volumes()
    print("\n" + "=" * 60)
    print("✓ RSNA Step 2 Complete!")
    print("=" * 60)
