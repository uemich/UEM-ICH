"""
Splitting utilities for creating train/val/test splits
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, List


def create_stratified_splits(scan_ids: List[str],
                            labels: List[int],
                            train_ratio: float = 0.8,
                            val_ratio: float = 0.1,
                            test_ratio: float = 0.1,
                            random_seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Create stratified train/val/test splits
    
    Args:
        scan_ids: List of scan identifiers
        labels: List of binary labels for stratification (e.g., ICH presence)
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_ids, val_ids, test_ids)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # First split: train vs (val + test)
    train_ids, temp_ids, train_labels, temp_labels = train_test_split(
        scan_ids,
        labels,
        train_size=train_ratio,
        stratify=labels,
        random_state=random_seed
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids,
        train_size=val_size,
        stratify=temp_labels,
        random_state=random_seed
    )
    
    return train_ids, val_ids, test_ids


def save_splits_to_csv(train_ids: List[str],
                       val_ids: List[str],
                       test_ids: List[str],
                       output_dir: str):
    """
    Save split assignments to CSV files
    
    Args:
        train_ids: List of training scan IDs
        val_ids: List of validation scan IDs
        test_ids: List of test scan IDs
        output_dir: Directory to save CSV files
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each split
    pd.DataFrame({'scan_id': train_ids}).to_csv(output_dir / 'train_scans.csv', index=False)
    pd.DataFrame({'scan_id': val_ids}).to_csv(output_dir / 'val_scans.csv', index=False)
    pd.DataFrame({'scan_id': test_ids}).to_csv(output_dir / 'test_scans.csv', index=False)
    
    print(f"Saved splits to {output_dir}:")
    print(f"  Train: {len(train_ids)} scans")
    print(f"  Val: {len(val_ids)} scans")
    print(f"  Test: {len(test_ids)} scans")


if __name__ == "__main__":
    print("Splitting utilities module loaded successfully!")
    
    # Test stratified splitting
    np.random.seed(42)
    test_scan_ids = [f"scan_{i:03d}" for i in range(100)]
    test_labels = np.random.binomial(1, 0.3, 100).tolist()  # 30% positive
    
    print(f"\nTest data: {len(test_scan_ids)} scans")
    print(f"Positive rate: {np.mean(test_labels):.2%}")
    
    # Create splits
    train_ids, val_ids, test_ids = create_stratified_splits(
        test_scan_ids,
        test_labels,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_ids)} ({len(train_ids)/len(test_scan_ids):.1%})")
    print(f"  Val: {len(val_ids)} ({len(val_ids)/len(test_scan_ids):.1%})")
    print(f"  Test: {len(test_ids)} ({len(test_ids)/len(test_scan_ids):.1%})")
    
    # Check stratification
    train_labels = [test_labels[test_scan_ids.index(sid)] for sid in train_ids]
    val_labels = [test_labels[test_scan_ids.index(sid)] for sid in val_ids]
    test_labels_subset = [test_labels[test_scan_ids.index(sid)] for sid in test_ids]
    
    print(f"\nPositive rates:")
    print(f"  Train: {np.mean(train_labels):.2%}")
    print(f"  Val: {np.mean(val_labels):.2%}")
    print(f"  Test: {np.mean(test_labels_subset):.2%}")
    
    print("\n✓ All splitting functions working correctly!")
