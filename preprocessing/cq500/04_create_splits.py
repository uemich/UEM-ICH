"""
CQ500 Step 4: Create train/val/test splits

This script:
1. Loads scan labels
2. Creates stratified 80:10:10 split by ICH presence
3. Saves split assignments
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import config
from utils import create_stratified_splits, save_splits_to_csv


def create_cq500_splits():
    """
    Create train/val/test splits for CQ500 dataset
    """
    print("=" * 60)
    print("CQ500 Step 4: Create Dataset Splits")
    print("=" * 60)
    
    # Load scan labels
    scan_labels_path = config.CQ500_OUTPUT_DIR / "labels" / "scan_labels.csv"
    
    if not scan_labels_path.exists():
        print("ERROR: Scan labels not found!")
        print("Please run 03_process_labels.py first")
        return
    
    scan_labels_df = pd.read_csv(scan_labels_path)
    print(f"\nLoaded labels for {len(scan_labels_df)} scans")
    
    # Get scan IDs and ICH labels for stratification
    scan_ids = scan_labels_df['scan_id'].tolist()
    ich_labels = scan_labels_df['ICH'].tolist()
    
    print(f"ICH positive scans: {sum(ich_labels)}/{len(ich_labels)} ({sum(ich_labels)/len(ich_labels)*100:.1f}%)")
    
    # Create stratified splits
    print(f"\nCreating splits with seed={config.RANDOM_SEED}...")
    print(f"  Train: {config.TRAIN_RATIO*100:.0f}%")
    print(f"  Val: {config.VAL_RATIO*100:.0f}%")
    print(f"  Test: {config.TEST_RATIO*100:.0f}%")
    
    train_ids, val_ids, test_ids = create_stratified_splits(
        scan_ids,
        ich_labels,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        random_seed=config.RANDOM_SEED
    )
    
    # Save splits
    splits_dir = config.CQ500_OUTPUT_DIR / "splits"
    save_splits_to_csv(train_ids, val_ids, test_ids, splits_dir)
    
    # Verify stratification
    print("\n" + "=" * 60)
    print("Split Verification")
    print("=" * 60)
    
    # Get ICH rates for each split
    train_labels = [ich_labels[scan_ids.index(sid)] for sid in train_ids]
    val_labels = [ich_labels[scan_ids.index(sid)] for sid in val_ids]
    test_labels = [ich_labels[scan_ids.index(sid)] for sid in test_ids]
    
    print(f"\nTrain set:")
    print(f"  Size: {len(train_ids)} scans ({len(train_ids)/len(scan_ids)*100:.1f}%)")
    print(f"  ICH positive: {sum(train_labels)}/{len(train_labels)} ({sum(train_labels)/len(train_labels)*100:.1f}%)")
    
    print(f"\nVal set:")
    print(f"  Size: {len(val_ids)} scans ({len(val_ids)/len(scan_ids)*100:.1f}%)")
    print(f"  ICH positive: {sum(val_labels)}/{len(val_labels)} ({sum(val_labels)/len(val_labels)*100:.1f}%)")
    
    print(f"\nTest set:")
    print(f"  Size: {len(test_ids)} scans ({len(test_ids)/len(scan_ids)*100:.1f}%)")
    print(f"  ICH positive: {sum(test_labels)}/{len(test_labels)} ({sum(test_labels)/len(test_labels)*100:.1f}%)")
    
    # Additional statistics for key labels
    print("\n" + "=" * 60)
    print("Key Label Distribution Across Splits")
    print("=" * 60)
    
    for label_name in ['MidlineShift', 'MassEffect', 'IPH', 'SDH']:
        if label_name in scan_labels_df.columns:
            label_values = scan_labels_df.set_index('scan_id')[label_name]
            
            train_count = sum([label_values[sid] for sid in train_ids if sid in label_values.index])
            val_count = sum([label_values[sid] for sid in val_ids if sid in label_values.index])
            test_count = sum([label_values[sid] for sid in test_ids if sid in label_values.index])
            
            print(f"\n{label_name}:")
            print(f"  Train: {train_count}/{len(train_ids)} ({train_count/len(train_ids)*100:.1f}%)")
            print(f"  Val: {val_count}/{len(val_ids)} ({val_count/len(val_ids)*100:.1f}%)")
            print(f"  Test: {test_count}/{len(test_ids)} ({test_count/len(test_ids)*100:.1f}%)")
    
    return train_ids, val_ids, test_ids


if __name__ == "__main__":
    train, val, test = create_cq500_splits()
    print("\n" + "=" * 60)
    print("✓ CQ500 Step 4 Complete!")
    print("=" * 60)
    print("\n✓ CQ500 preprocessing pipeline complete!")
    print(f"✓ Output directory: {config.CQ500_OUTPUT_DIR}")
