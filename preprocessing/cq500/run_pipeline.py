"""
CQ500 Master Pipeline Script

Run all CQ500 preprocessing steps in sequence
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
from datetime import timedelta


def run_cq500_pipeline():
    """
    Run complete CQ500 preprocessing pipeline
    """
    print("\n" + "=" * 70)
    print(" " * 20 + "CQ500 PREPROCESSING PIPELINE")
    print("=" * 70)
    
    start_time = time.time()
    
    # Step 1: Filter series
    print("\n" + "▶" * 35)
    print("STEP 1/4: Filtering Non-Contrast Series")
    print("▶" * 35)
    step1_start = time.time()
    
    import importlib
    filter_series = importlib.import_module('cq500.01_filter_series')
    filter_series.filter_cq500_series()
    
    step1_time = time.time() - step1_start
    print(f"\n✓ Step 1 completed in {timedelta(seconds=int(step1_time))}")
    
    # Step 2: Load and preprocess
    print("\n" + "▶" * 35)
    print("STEP 2/4: Loading and Preprocessing Volumes")
    print("▶" * 35)
    step2_start = time.time()
    
    load_and_preprocess = importlib.import_module('cq500.02_load_and_preprocess')
    load_and_preprocess.preprocess_cq500_volumes()
    
    step2_time = time.time() - step2_start
    print(f"\n✓ Step 2 completed in {timedelta(seconds=int(step2_time))}")
    
    # Step 3: Process labels
    print("\n" + "▶" * 35)
    print("STEP 3/4: Processing Labels")
    print("▶" * 35)
    step3_start = time.time()
    
    process_labels = importlib.import_module('cq500.03_process_labels')
    process_labels.process_cq500_labels()
    
    step3_time = time.time() - step3_start
    print(f"\n✓ Step 3 completed in {timedelta(seconds=int(step3_time))}")
    
    # Step 4: Create splits
    print("\n" + "▶" * 35)
    print("STEP 4/4: Creating Dataset Splits")
    print("▶" * 35)
    step4_start = time.time()
    
    create_splits = importlib.import_module('cq500.04_create_splits')
    create_splits.create_cq500_splits()
    
    step4_time = time.time() - step4_start
    print(f"\n✓ Step 4 completed in {timedelta(seconds=int(step4_time))}")
    
    # Summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print(" " * 25 + "PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nTotal time: {timedelta(seconds=int(total_time))}")
    print(f"  Step 1 (Filter): {timedelta(seconds=int(step1_time))}")
    print(f"  Step 2 (Preprocess): {timedelta(seconds=int(step2_time))}")
    print(f"  Step 3 (Labels): {timedelta(seconds=int(step3_time))}")
    print(f"  Step 4 (Splits): {timedelta(seconds=int(step4_time))}")
    
    import config
    print(f"\n✓ All preprocessed data saved to: {config.CQ500_OUTPUT_DIR}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_cq500_pipeline()
