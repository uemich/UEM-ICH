"""
RSNA Step 3: Aggregate labels

This script:
1. Parses stage_2_train.csv
2. Creates scan-level labels (max pooling across slices)
3. Creates slice-level labels
4. Maps to volume IDs
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from tqdm import tqdm
import config


def aggregate_rsna_labels():
    """
    Process and aggregate RSNA labels
    """
    print("=" * 60)
    print("RSNA Step 3: Aggregate Labels")
    print("=" * 60)
    
    # Load labels
    print(f"\nLoading labels from: {config.RSNA_LABELS_CSV}")
    labels_df = pd.read_csv(config.RSNA_LABELS_CSV)
    print(f"Loaded {len(labels_df)} label rows")
    
    # Parse ID column (format: "ID_sliceid_labeltype")
    print("\nParsing label format...")
    labels_df['slice_id'] = labels_df['ID'].apply(lambda x: '_'.join(x.split('_')[:2]))
    labels_df['label_type'] = labels_df['ID'].apply(lambda x: x.split('_')[-1])
    
    # Pivot to wide format - use pivot_table to handle any duplicates (take max)
    labels_pivot = labels_df.pivot_table(
        index='slice_id', 
        columns='label_type', 
        values='Label',
        aggfunc='max'  # If duplicates exist, take max
    )
    labels_pivot = labels_pivot.reset_index()
    
    print(f"Pivoted to {len(labels_pivot)} slices with {len(config.RSNA_LABEL_COLUMNS)} label columns")
    
    # Load volume mapping and slice metadata
    volume_mapping_path = config.RSNA_OUTPUT_DIR / "metadata" / "volume_mapping.csv"
    slice_metadata_path = config.RSNA_OUTPUT_DIR / "metadata" / "slice_metadata.csv"
    
    if not volume_mapping_path.exists():
        print("Warning: Volume mapping not found. Creating labels without volume mapping.")
        volume_mapping_df = None
    else:
        volume_mapping_df = pd.read_csv(volume_mapping_path)
        print(f"Loaded volume mapping for {len(volume_mapping_df)} volumes")
    
    # Create mapping from original filename to volume_id
    if volume_mapping_df is not None:
        filename_to_volume = {}
        for _, row in volume_mapping_df.iterrows():
            filenames = row['slice_filenames'].split(',')
            for filename in filenames:
                filename_to_volume[filename] = row['volume_id']
        
        # Add volume_id to labels
        labels_pivot['volume_id'] = labels_pivot['slice_id'].map(filename_to_volume)
    
    # Save slice-level labels
    slice_labels_path = config.RSNA_OUTPUT_DIR / "labels" / "slice_labels.csv"
    labels_pivot.to_csv(slice_labels_path, index=False)
    print(f"\n✓ Saved slice-level labels to: {slice_labels_path}")
    
    # Create scan-level labels (max pooling)
    if volume_mapping_df is not None and 'volume_id' in labels_pivot.columns:
        print("\nCreating scan-level labels (max pooling)...")
        
        scan_labels_list = []
        for volume_id in tqdm(labels_pivot['volume_id'].dropna().unique(), desc="Aggregating scans"):
            volume_slices = labels_pivot[labels_pivot['volume_id'] == volume_id]
            
            scan_label = {'volume_id': volume_id}
            
            # Max pooling: if any slice has the label, the scan has it
            for label in config.RSNA_LABEL_COLUMNS:
                if label in volume_slices.columns:
                    scan_label[label] = int(volume_slices[label].max())
                else:
                    scan_label[label] = 0
            
            scan_labels_list.append(scan_label)
        
        scan_labels_df = pd.DataFrame(scan_labels_list)
        
        # Save scan-level labels
        scan_labels_path = config.RSNA_OUTPUT_DIR / "labels" / "scan_labels.csv"
        scan_labels_df.to_csv(scan_labels_path, index=False)
        print(f"✓ Saved scan-level labels to: {scan_labels_path}")
        
        # Print statistics
        print("\n" + "=" * 60)
        print("Label Statistics (Scan-Level)")
        print("=" * 60)
        for label in config.RSNA_LABEL_COLUMNS:
            if label in scan_labels_df.columns:
                count = scan_labels_df[label].sum()
                pct = count / len(scan_labels_df) * 100
                print(f"  {label}: {count}/{len(scan_labels_df)} ({pct:.1f}%)")
    
    # Print slice-level statistics
    print("\n" + "=" * 60)
    print("Label Statistics (Slice-Level)")
    print("=" * 60)
    for label in config.RSNA_LABEL_COLUMNS:
        if label in labels_pivot.columns:
            count = labels_pivot[label].sum()
            pct = count / len(labels_pivot) * 100
            print(f"  {label}: {count}/{len(labels_pivot)} ({pct:.2f}%)")
    
    return labels_pivot


if __name__ == "__main__":
    labels = aggregate_rsna_labels()
    print("\n" + "=" * 60)
    print("✓ RSNA Step 3 Complete!")
    print("=" * 60)
