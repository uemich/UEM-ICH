"""
RSNA Step 1: Build volume index

This script scans all DICOM files in stage_2_train and creates a mapping
of slices to volumes (scans) using SeriesInstanceUID.

This is necessary because RSNA files are not organized by patient/scan.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from tqdm import tqdm
import config
from utils import read_dicom_file, get_slice_metadata


def build_rsna_volume_index():
    """
    Build index mapping DICOM slices to volumes
    """
    print("=" * 60)
    print("RSNA Step 1: Building Volume Index")
    print("=" * 60)
    
    # Create output directory
    config.create_output_directories()
    
    # Get all DICOM files
    print(f"\nScanning DICOM files in: {config.RSNA_TRAIN_DIR}")
    dicom_files = sorted(config.RSNA_TRAIN_DIR.glob("*.dcm"))
    print(f"Found {len(dicom_files)} DICOM files")
    
    if len(dicom_files) == 0:
        print("ERROR: No DICOM files found!")
        return
    
    # Read metadata from each file
    slice_info = []
    
    print("\nReading DICOM metadata...")
    for dcm_path in tqdm(dicom_files, desc="Scanning files"):
        ds = read_dicom_file(dcm_path)
        if ds is None:
            continue
        
        metadata = get_slice_metadata(ds)
        
        # Extract filename (ID)
        filename = dcm_path.stem  # e.g., "ID_000012eaf"
        
        slice_info.append({
            'filename': filename,
            'filepath': str(dcm_path),
            'series_uid': metadata.get('SeriesInstanceUID'),
            'study_uid': metadata.get('StudyInstanceUID'),
            'patient_id': metadata.get('PatientID'),
            'instance_number': metadata.get('InstanceNumber'),
            'slice_position': metadata.get('SlicePosition'),
            'slice_thickness': metadata.get('SliceThickness'),
            'rows': metadata.get('Rows'),
            'columns': metadata.get('Columns'),
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(slice_info)
    
    print(f"\nProcessed {len(df)} DICOM files")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    print(f"Unique series (scans): {df['series_uid'].nunique()}")
    
    # Create volume mapping (group by SeriesInstanceUID)
    print("\nGrouping slices by series (volume)...")
    volume_groups = df.groupby('series_uid')
    
    volume_mapping = []
    for series_uid, group in tqdm(volume_groups, desc="Creating volume mapping"):
        # Sort by slice position or instance number
        group_sorted = group.sort_values(
            by='slice_position' if group['slice_position'].notna().all() 
            else 'instance_number'
        )
        
        volume_id = f"volume_{len(volume_mapping):05d}"
        patient_id = group_sorted['patient_id'].iloc[0]
        
        volume_mapping.append({
            'volume_id': volume_id,
            'series_uid': series_uid,
            'patient_id': patient_id,
            'num_slices': len(group_sorted),
            'slice_filenames': ','.join(group_sorted['filename'].tolist()),
            'median_slice_thickness': group_sorted['slice_thickness'].median(),
        })
    
    # Save volume mapping
    volume_mapping_df = pd.DataFrame(volume_mapping)
    volume_mapping_path = config.RSNA_OUTPUT_DIR / "metadata" / "volume_mapping.csv"
    volume_mapping_df.to_csv(volume_mapping_path, index=False)
    
    print(f"\n✓ Saved volume mapping to: {volume_mapping_path}")
    
    # Save full slice index
    slice_index_path = config.RSNA_OUTPUT_DIR / "metadata" / "slice_index.csv"
    df.to_csv(slice_index_path, index=False)
    print(f"✓ Saved slice index to: {slice_index_path}")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Volume Index Statistics")
    print("=" * 60)
    print(f"Total DICOM files: {len(df)}")
    print(f"Total volumes (scans): {len(volume_mapping_df)}")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    print(f"\nSlices per volume:")
    print(f"  Mean: {volume_mapping_df['num_slices'].mean():.1f}")
    print(f"  Median: {volume_mapping_df['num_slices'].median():.0f}")
    print(f"  Min: {volume_mapping_df['num_slices'].min()}")
    print(f"  Max: {volume_mapping_df['num_slices'].max()}")
    
    print(f"\nSlice thickness:")
    median_thickness = volume_mapping_df['median_slice_thickness'].median()
    print(f"  Median: {median_thickness:.2f}mm")
    
    # Estimate resampled slice count
    avg_slices = volume_mapping_df['num_slices'].mean()
    if median_thickness and median_thickness > 0:
        resampled_slices = avg_slices * (median_thickness / config.SLICE_THICKNESS_MM)
        total_resampled = len(volume_mapping_df) * resampled_slices
        print(f"\nEstimated slices after 5mm resampling: ~{total_resampled:,.0f}")
    
    return volume_mapping_df


if __name__ == "__main__":
    volume_mapping = build_rsna_volume_index()
    print("\n" + "=" * 60)
    print("✓ RSNA Step 1 Complete!")
    print("=" * 60)
