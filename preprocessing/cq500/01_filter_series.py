"""
CQ500 Step 1: Filter and identify non-contrast CT series

This script scans all CQ500 patient folders, identifies non-contrast series,
and creates a log of which series to use for preprocessing.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from tqdm import tqdm
import config
from utils import get_series_from_directory, read_dicom_file, is_non_contrast_series


def filter_cq500_series():
    """
    Scan all CQ500 patient folders and filter for non-contrast series
    """
    print("=" * 60)
    print("CQ500 Step 1: Filtering Non-Contrast Series")
    print("=" * 60)
    
    # Create output directory
    config.create_output_directories()
    
    # Get all patient folders
    patient_folders = sorted([d for d in config.CQ500_UNZIPPED_DIR.iterdir() if d.is_dir()])
    print(f"\nFound {len(patient_folders)} patient folders")
    
    # Results storage
    series_info = []
    
    # Process each patient
    for patient_dir in tqdm(patient_folders, desc="Scanning patients"):
        patient_id = patient_dir.name
        
        # Get all series in this patient folder
        series_dict = get_series_from_directory(patient_dir)
        
        for series_uid, dicom_files in series_dict.items():
            if len(dicom_files) == 0:
                continue
            
            # Read first DICOM to get series description
            ds = read_dicom_file(dicom_files[0])
            if ds is None:
                continue
            
            series_desc = str(ds.SeriesDescription) if hasattr(ds, 'SeriesDescription') else ""
            
            # Check if non-contrast
            is_non_contrast = is_non_contrast_series(
                series_desc,
                config.CQ500_SERIES_KEYWORDS,
                config.CQ500_EXCLUDE_KEYWORDS
            )
            
            # Store info
            series_info.append({
                'patient_id': patient_id,
                'series_uid': series_uid,
                'series_description': series_desc,
                'num_slices': len(dicom_files),
                'is_non_contrast': is_non_contrast,
                'first_dicom_path': str(dicom_files[0])
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(series_info)
    
    # Save full log
    log_path = config.CQ500_OUTPUT_DIR / "metadata" / "series_filter_log.csv"
    df.to_csv(log_path, index=False)
    print(f"\n✓ Saved series filter log to: {log_path}")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Series Filtering Statistics")
    print("=" * 60)
    print(f"Total series found: {len(df)}")
    print(f"Non-contrast series: {df['is_non_contrast'].sum()}")
    print(f"Contrast series (excluded): {(~df['is_non_contrast']).sum()}")
    
    # Show series descriptions
    print("\nNon-contrast series descriptions:")
    non_contrast_descs = df[df['is_non_contrast']]['series_description'].value_counts()
    for desc, count in non_contrast_descs.items():
        print(f"  '{desc}': {count} series")
    
    if (~df['is_non_contrast']).sum() > 0:
        print("\nExcluded series descriptions:")
        contrast_descs = df[~df['is_non_contrast']]['series_description'].value_counts()
        for desc, count in contrast_descs.items():
            print(f"  '{desc}': {count} series")
    
    # Save filtered list (non-contrast only)
    filtered_df = df[df['is_non_contrast']].copy()
    filtered_path = config.CQ500_OUTPUT_DIR / "metadata" / "non_contrast_series.csv"
    filtered_df.to_csv(filtered_path, index=False)
    print(f"\n✓ Saved filtered series list to: {filtered_path}")
    
    print(f"\n✓ Total patients with non-contrast scans: {filtered_df['patient_id'].nunique()}")
    print(f"✓ Total non-contrast series: {len(filtered_df)}")
    print(f"✓ Total slices to process: {filtered_df['num_slices'].sum()}")
    
    return filtered_df


if __name__ == "__main__":
    filtered_series = filter_cq500_series()
    print("\n" + "=" * 60)
    print("✓ CQ500 Step 1 Complete!")
    print("=" * 60)
