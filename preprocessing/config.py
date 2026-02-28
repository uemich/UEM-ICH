"""
Preprocessing Configuration

All paths are relative to the repository root.
"""
from pathlib import Path

# =============================================================================
# BASE PATHS
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = REPO_ROOT / "raw_data"
PREPROCESSED_DIR = REPO_ROOT / "preprocessed_data"

# =============================================================================
# RSNA PATHS
# =============================================================================
RSNA_RAW_DIR = RAW_DATA_DIR / "rsna"
RSNA_TRAIN_DIR = RSNA_RAW_DIR / "stage_2_train"
RSNA_LABELS_CSV = RSNA_RAW_DIR / "stage_2_train.csv"
RSNA_TRAIN_SPLIT = RSNA_RAW_DIR / "train_split.csv"
RSNA_VAL_SPLIT = RSNA_RAW_DIR / "val_split.csv"
RSNA_TEST_SPLIT = RSNA_RAW_DIR / "test_split.csv"

RSNA_OUTPUT_DIR = PREPROCESSED_DIR / "rsna"

# =============================================================================
# CQ500 PATHS
# =============================================================================
CQ500_RAW_DIR = RAW_DATA_DIR / "cq500"
CQ500_UNZIPPED_DIR = CQ500_RAW_DIR / "unzipped"
CQ500_LABELS_CSV = CQ500_RAW_DIR / "reads.csv"

CQ500_OUTPUT_DIR = PREPROCESSED_DIR / "cq500"

# =============================================================================
# PHYSIONET PATHS
# =============================================================================
PHYSIONET_RAW_DIR = RAW_DATA_DIR / "physionet" / "computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.3.1"
PHYSIONET_OUTPUT_DIR = PREPROCESSED_DIR / "physionet"

# =============================================================================
# MBH PATHS
# =============================================================================
MBH_RAW_DIR = RAW_DATA_DIR / "MBH"
MBH_TRAIN_VOXEL_DIR = MBH_RAW_DIR / "Train_voxel"
MBH_TRAIN_CASE_DIR = MBH_RAW_DIR / "Train_case"
MBH_VAL_DIR = MBH_RAW_DIR / "mbh_seg_val"

MBH_SEG_OUTPUT_DIR = PREPROCESSED_DIR / "mbh_seg"
MBH_SCAN_OUTPUT_DIR = PREPROCESSED_DIR / "MBH_Seg_scan"

# =============================================================================
# PREPROCESSING PARAMETERS
# =============================================================================
TARGET_SIZE = (384, 384)
SLICE_THICKNESS_MM = 5.0
NUM_CHANNELS = 3

WINDOW_SETTINGS = {
    'brain': {'center': 40, 'width': 80},
    'subdural': {'center': 80, 'width': 200},
    'bone': {'center': 500, 'width': 2000}
}

IMAGE_FORMAT = 'png'
PNG_COMPRESSION = 6

# =============================================================================
# CQ500 SERIES FILTERING
# =============================================================================
CQ500_SERIES_KEYWORDS = ['CT 5mm', 'non-contrast', 'plain', 'CT BRAIN']
CQ500_EXCLUDE_KEYWORDS = ['contrast', 'angio', 'CTA', 'perfusion']

# =============================================================================
# DATASET SPLITS
# =============================================================================
RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# =============================================================================
# LABEL MAPPINGS
# =============================================================================
RSNA_LABEL_COLUMNS = [
    'epidural', 'intraparenchymal', 'intraventricular',
    'subarachnoid', 'subdural', 'any'
]

CQ500_HEMORRHAGE_LABELS = ['ICH', 'IPH', 'IVH', 'SDH', 'EDH', 'SAH']
CQ500_LOCATION_LABELS = ['BleedLocation-Left', 'BleedLocation-Right', 'ChronicBleed']
CQ500_FRACTURE_LABELS = ['Fracture', 'CalvarialFracture', 'OtherFracture']
CQ500_GEOMETRY_LABELS = ['MassEffect', 'MidlineShift']
CQ500_ALL_LABELS = (
    CQ500_HEMORRHAGE_LABELS + CQ500_LOCATION_LABELS +
    CQ500_FRACTURE_LABELS + CQ500_GEOMETRY_LABELS
)

# =============================================================================
# PROCESSING
# =============================================================================
BATCH_SIZE = 100
NUM_WORKERS = 4
VERBOSE = True

# =============================================================================
# HELPERS
# =============================================================================
def create_output_directories():
    """Create all necessary output directories."""
    dirs_to_create = [
        PREPROCESSED_DIR,
        RSNA_OUTPUT_DIR, RSNA_OUTPUT_DIR / "images",
        RSNA_OUTPUT_DIR / "metadata", RSNA_OUTPUT_DIR / "labels",
        CQ500_OUTPUT_DIR, CQ500_OUTPUT_DIR / "images",
        CQ500_OUTPUT_DIR / "metadata", CQ500_OUTPUT_DIR / "labels",
        CQ500_OUTPUT_DIR / "splits",
        PHYSIONET_OUTPUT_DIR, PHYSIONET_OUTPUT_DIR / "images",
        PHYSIONET_OUTPUT_DIR / "masks", PHYSIONET_OUTPUT_DIR / "labels",
        MBH_SEG_OUTPUT_DIR, MBH_SEG_OUTPUT_DIR / "images",
        MBH_SEG_OUTPUT_DIR / "metadata",
        MBH_SCAN_OUTPUT_DIR, MBH_SCAN_OUTPUT_DIR / "images",
        MBH_SCAN_OUTPUT_DIR / "metadata",
    ]
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        if VERBOSE:
            print(f"  Created: {dir_path}")


if __name__ == "__main__":
    print("Preprocessing Configuration")
    print("=" * 60)
    print(f"Repo root:       {REPO_ROOT}")
    print(f"Raw data dir:    {RAW_DATA_DIR}")
    print(f"Preprocessed:    {PREPROCESSED_DIR}")
    print(f"Target size:     {TARGET_SIZE}")
    print(f"Slice thickness: {SLICE_THICKNESS_MM}mm")
    print("=" * 60)
    create_output_directories()
    print("\nAll output directories created!")
