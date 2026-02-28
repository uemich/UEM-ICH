"""
RSNA Master Pipeline Script

Run all RSNA preprocessing steps in sequence
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
from datetime import timedelta


def run_rsna_pipeline():
    """
    Run complete RSNA preprocessing pipeline
    """
    print("\n" + "=" * 70)
    print(" " * 20 + "RSNA PREPROCESSING PIPELINE")
    print("=" * 70)
    
    start_time = time.time()
    
    # Step 1: Build volume index
    print("\n" + "▶" * 35)
    print("STEP 1/3: Building Volume Index")
    print("▶" * 35)
    step1_start = time.time()
    
    import importlib
    build_index = importlib.import_module('rsna.01_build_volume_index')
    build_index.build_rsna_volume_index()
    
    step1_time = time.time() - step1_start
    print(f"\n✓ Step 1 completed in {timedelta(seconds=int(step1_time))}")
    
    # Step 2: Preprocess volumes
    print("\n" + "▶" * 35)
    print("STEP 2/3: Preprocessing Volumes")
    print("▶" * 35)
    step2_start = time.time()
    
    preprocess = importlib.import_module('rsna.02_preprocess_volumes')
    preprocess.preprocess_rsna_volumes()
    
    step2_time = time.time() - step2_start
    print(f"\n✓ Step 2 completed in {timedelta(seconds=int(step2_time))}")
    
    # Step 3: Aggregate labels
    print("\n" + "▶" * 35)
    print("STEP 3/3: Aggregating Labels")
    print("▶" * 35)
    step3_start = time.time()
    
    aggregate = importlib.import_module('rsna.03_aggregate_labels')
    aggregate.aggregate_rsna_labels()
    
    step3_time = time.time() - step3_start
    print(f"\n✓ Step 3 completed in {timedelta(seconds=int(step3_time))}")
    
    # Summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print(" " * 25 + "PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nTotal time: {timedelta(seconds=int(total_time))}")
    print(f"  Step 1 (Index): {timedelta(seconds=int(step1_time))}")
    print(f"  Step 2 (Preprocess): {timedelta(seconds=int(step2_time))}")
    print(f"  Step 3 (Labels): {timedelta(seconds=int(step3_time))}")
    
    import config
    print(f"\n✓ All preprocessed data saved to: {config.RSNA_OUTPUT_DIR}")
    print("\nNote: RSNA uses existing train/val/test splits from:")
    print(f"  - {config.RSNA_TRAIN_SPLIT}")
    print(f"  - {config.RSNA_VAL_SPLIT}")
    print(f"  - {config.RSNA_TEST_SPLIT}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_rsna_pipeline()
