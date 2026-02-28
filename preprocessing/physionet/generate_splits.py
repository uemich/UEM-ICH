"""
Generate K-Fold Splits for PhysioNet CT-ICH Dataset

Creates stratified k-fold splits at the PATIENT level to ensure:
1. No patient overlap between folds
2. Balanced class distribution across folds

Output: splits/ directory with train_{fold}.csv and val_{fold}.csv
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import config


# =============================================================================
# Configuration
# =============================================================================
DATA_DIR = config.PHYSIONET_OUTPUT_DIR
OUTPUT_DIR = DATA_DIR / "splits"
N_FOLDS = 3
RANDOM_SEED = 966

ICH_COLS = ['Intraventricular', 'Intraparenchymal', 'Subarachnoid', 'Epidural', 'Subdural']


# =============================================================================
# Functions
# =============================================================================
def create_patient_level_labels(df):
    """
    Create patient-level labels for stratification.

    Strategy: For each patient, take the OR of all slices' labels.
    This ensures patients with ANY positive slice are labeled as positive.
    """
    patient_labels = df.groupby('patient_id')[ICH_COLS + ['any', 'Fracture_Yes_No']].max().reset_index()
    return patient_labels


def create_stratification_key(row):
    """
    Create a composite key for multi-label stratification.

    Use 'any' (hemorrhage yes/no) as primary stratification to ensure
    hemorrhage/non-hemorrhage balance. With only 75 patients and rare classes,
    we can't stratify on individual ICH types without causing empty folds.
    """
    key = f"{row['any']}_{row['Fracture_Yes_No']}"
    return key


def main():
    print("=" * 60)
    print("Generating K-Fold Splits for PhysioNet CT-ICH")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clean up old fold files
    for old_file in OUTPUT_DIR.glob("*_fold*.csv"):
        old_file.unlink()
        print(f"  Removed old file: {old_file.name}")

    # Load slice labels
    labels_path = DATA_DIR / "labels" / "slice_labels.csv"
    if not labels_path.exists():
        print(f"ERROR: Labels file not found: {labels_path}")
        print("Please run preprocess.py first!")
        return

    df = pd.read_csv(labels_path)
    print(f"\nTotal slices: {len(df)}")
    print(f"Total patients: {df['patient_id'].nunique()}")

    # Create patient-level labels
    patient_labels = create_patient_level_labels(df)
    print(f"\nPatient-level label distribution:")
    for col in ICH_COLS + ['any', 'Fracture_Yes_No']:
        count = patient_labels[col].sum()
        pct = 100 * count / len(patient_labels)
        print(f"  {col}: {count}/{len(patient_labels)} ({pct:.1f}%)")

    # Create stratification key
    patient_labels['strat_key'] = patient_labels.apply(create_stratification_key, axis=1)

    print(f"\nStratification key distribution:")
    key_counts = Counter(patient_labels['strat_key'])
    for key, count in sorted(key_counts.items()):
        print(f"  {key}: {count} patients")

    # Handle rare stratification keys
    min_samples_per_key = N_FOLDS
    rare_keys = [k for k, v in key_counts.items() if v < min_samples_per_key]
    if rare_keys:
        print(f"\n  Merging rare keys (< {min_samples_per_key} samples) with similar groups...")
        for rare_key in rare_keys:
            any_val = rare_key.split('_')[0]
            patient_labels.loc[patient_labels['strat_key'] == rare_key, 'strat_key'] = f"{any_val}_OTHER"

    # Create k-fold splits
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        print("\nUsing MultilabelStratifiedKFold (iterative-stratification installed)")
        strat_cols = ICH_COLS + ['any', 'Fracture_Yes_No']
        skf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        strat_labels = patient_labels[strat_cols].values
        patient_ids = patient_labels['patient_id'].values
    except ImportError:
        print("\nWARNING: iterative-stratification not installed. Falling back to simple StratifiedKFold.")
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        patient_ids = patient_labels['patient_id'].values
        strat_labels = patient_labels['strat_key'].values

    print(f"\n\nGenerating {N_FOLDS}-fold splits...")

    if 'MultilabelStratifiedKFold' in str(type(skf)):
        split_gen = skf.split(patient_ids, strat_labels)
    else:
        split_gen = skf.split(patient_ids, strat_labels)

    for fold_idx, (train_indices, val_indices) in enumerate(split_gen):
        train_patients = patient_ids[train_indices]
        val_patients = patient_ids[val_indices]

        train_df = df[df['patient_id'].isin(train_patients)].copy()
        val_df = df[df['patient_id'].isin(val_patients)].copy()

        train_path = OUTPUT_DIR / f"train_fold{fold_idx}.csv"
        val_path = OUTPUT_DIR / f"val_fold{fold_idx}.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        print(f"\nFold {fold_idx}:")
        print(f"  Train: {len(train_df)} slices from {len(train_patients)} patients")
        print(f"  Val:   {len(val_df)} slices from {len(val_patients)} patients")

        all_cols = ['any'] + ICH_COLS + ['Fracture_Yes_No']
        print(f"  {'Class':<22s} | {'Train Count':>11s} {'Train %':>8s} | {'Val Count':>9s} {'Val %':>7s}")
        print(f"  {'-'*22}-+-{'-'*11}-{'-'*8}-+-{'-'*9}-{'-'*7}")
        for col in all_cols:
            train_count = int(train_df[col].sum())
            val_count = int(val_df[col].sum())
            train_pct = 100 * train_df[col].mean()
            val_pct = 100 * val_df[col].mean()
            print(f"  {col:<22s} | {train_count:>11d} {train_pct:>7.1f}% | {val_count:>9d} {val_pct:>6.1f}%")

    print("\n" + "=" * 60)
    print("K-FOLD SPLITS GENERATED")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Files created:")
    for fold_idx in range(N_FOLDS):
        print(f"  - train_fold{fold_idx}.csv, val_fold{fold_idx}.csv")


if __name__ == "__main__":
    main()
