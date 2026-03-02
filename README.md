# From Specialists to Generalists : A Unified, Efficient, and Multi-Task Framework for Intracranial Hemorrhage Diagnosis


A unified multi-task framework for Intracranial Hemorrhage (ICH) analysis, supporting detection, subtype classification, and segmentation from non-contrast CT scans.

## Abstract

Intracranial hemorrhage (ICH) is a critical neurological emergency that requires rapid and accurate interpretation of non-contrast head CT scans. Despite advances in deep learning, clinical deployment remains constrained by fragmented labeling standards and task-specific specialist models with high computational overhead. We propose a unified, efficient, and multi-task framework (UEM-ICH) for automated head CT analysis that jointly operates at slice and scan levels using a single lightweight backbone. The architecture integrates a head CT–specialized ConvNeXt-v2 Tiny encoder with coordinate attention for slice-level modeling and a transformer-based volumetric aggregator to capture anatomical symmetry and inter-slice dependencies for scan-level reasoning, enabling classification, segmentation, localization, and clinical parameter prediction within a single framework. The model follows a two-stage curriculum: texture-aware self-supervised pretraining (MSE + GLCM reconstruction via SparK), followed by supervised multi-task alignment. Extensive evaluation across heterogeneous datasets demonstrates competitive performance in classification, segmentation, and localization tasks. These results indicate that a unified and efficient model can serve as a scalable alternative to multiple specialist pipelines for practical clinical deployment.

## Repository Structure

```
UEM-ICH/
├── models/               # Model architecture definitions (ConvNeXtV2, SparK, GLCM)
├── training/             # Training scripts (SSL pretraining, supervised training)
│   ├── ssl/              # Self-supervised learning (GLCM-MAE)
│   └── supervised/       # Supervised fine-tuning (classification, segmentation, aggregator)
├── evaluation/           # Evaluation scripts (K-fold cross-validation)
│   ├── classification/   # RSNA and PhysioNet classification
│   ├── segmentation/     # PhysioNet and MBH segmentation/inference
│   └── aggregator/       # MBH and CQ500 scan-level aggregation
├── preprocessing/        # Data preprocessing pipelines (DICOM/NIfTI to PNG)
├── weights/              # Model weights (Download required)
└── preprocessed_data/    # Preprocessed datasets (Download required)
```

## Setup

### Environment

We recommend using a Conda environment:

```bash
conda create -n uem-ich python=3.10
conda activate uem-ich
pip install torch torchvision timm albumentations pandas numpy pillow scikit-learn tqdm nibabel
```

### Data & Weights

1.  **Weights**: Download the `.pth` files from the [shared drive link](https://drive.google.com/drive/folders/1XXjdAhx_tHO04cD1leVL0W1cJjpJuYgI?usp=sharing) and place them in the `weights/` directory. See `weights/README.md` for details.
2.  **Preprocessed Data**: Download the datasets from the [shared drive link](https://drive.google.com/drive/folders/1XXjdAhx_tHO04cD1leVL0W1cJjpJuYgI?usp=sharing) and extract them into the `preprocessed_data/` directory. See `preprocessed_data/README.md` for details.

## Running Evaluation

All evaluation scripts are designed to be run from the repository root.

### Slice-Level Classification (RSNA)
```bash
python evaluation/classification/kfold_rsna.py
```

### Slice-Level Classification (PhysioNet)
```bash
python evaluation/classification/kfold_physionet.py
```

### Segmentation (PhysioNet)
```bash
python evaluation/segmentation/kfold_physionet_seg.py
```

### Scan-Level Aggregation (MBH / CQ500)
```bash
python evaluation/aggregator/kfold_mbh.py
python evaluation/aggregator/kfold_cq500.py
```

## Training

For detailed instructions on training the models from scratch, see the `training/` directory.

- **SSL Pretraining**: `python training/ssl/ssl_train.py`
- **Supervised Training**: 
  - Classification: `python training/supervised/train_classification.py`
  - Segmentation: `python training/supervised/train_segmentation.py`
  - Joint Multitask: `python training/supervised/train_multitask.py`
  - Aggregator: `python training/supervised/train_aggregator.py`
