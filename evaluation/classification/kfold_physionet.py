"""
PhysioNet K-Fold Cross-Validation (Classification)

Evaluates the joint-trained encoder + fresh classifier on PhysioNet dataset
using stratified 3-fold cross-validation at patient level.

Features:
- 3-fold stratified sampling (using pre-generated splits in preprocessed_data/physionet/splits/)
- Configurable freeze/unfreeze encoder
- 20 epochs per fold
- Fresh classification head (7 classes: 5 ICH + any + fracture)
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
from datetime import datetime
from PIL import Image
from torchvision import transforms

# Add repo root for model imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# Configuration
# =============================================================================
CONFIG = {
    # Paths (relative to repo root)
    'images_dir': str(REPO_ROOT / 'preprocessed_data' / 'physionet' / 'images'),
    'splits_dir': str(REPO_ROOT / 'preprocessed_data' / 'physionet' / 'splits'),

    # Checkpoint (joint-trained encoder)
    'multitask_checkpoint': str(REPO_ROOT / 'weights' / 'best_multitask.pth'),

    # Output
    'output_dir': str(REPO_ROOT / 'evaluation' / 'classification' / 'results_physionet'),

    # K-Fold settings
    'n_folds': 3,  # Reduced from 5 to ensure rare classes have positives in each fold
    'epochs_per_fold': 10,
    'seed': 42,

    # Training
    'batch_size': 4,
    'encoder_lr': 5e-5,   # Lower LR for pre-trained encoder
    'head_lr': 1e-4,      # Higher for fresh classifier head
    'min_lr': 1e-7,
    'weight_decay': 1e-4,

    # *** TOGGLE: Freeze encoder or not ***
    'freeze_encoder': False,

    # Focal loss
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,

    # Device
    'device': 'cuda',
    'num_workers': 4,
}

# PhysioNet label columns (different from RSNA)
LABEL_COLS = ['Intraventricular', 'Intraparenchymal', 'Subarachnoid',
              'Epidural', 'Subdural', 'any', 'Fracture_Yes_No']


# =============================================================================
# Dataset
# =============================================================================
class PhysioNetSliceDataset(Dataset):
    """PhysioNet dataset for slice-level classification."""

    def __init__(self, images_dir, df, image_size=384):
        self.images_dir = images_dir
        self.filenames = df['filename'].tolist()
        self.labels = df[LABEL_COLS].values.astype(np.float32)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.filenames[idx])
        image = Image.open(image_path).convert('RGB')
        image = image.rotate(90, expand=True)  # 90° CCW to align with RSNA orientation
        image = self.transform(image)
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, labels


# =============================================================================
# Model Loading
# =============================================================================
def load_encoder_and_fresh_head(checkpoint_path, device, freeze_encoder=False):
    """Load encoder from multitask checkpoint and create fresh classifier head."""
    from training.ssl.ssl_config import SSLConfig
    from models import ConvNeXtV2Encoder

    # Initialize encoder
    config = SSLConfig()
    config.encoder_pretrained = False
    encoder = ConvNeXtV2Encoder(config)

    # FRESH classification head for PhysioNet (7 classes)
    num_classes = len(LABEL_COLS)
    cls_head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.1),
        nn.Linear(768, 256),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(256, num_classes)
    )

    # Load encoder weights from checkpoint
    print(f"Loading encoder from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    # Note: We do NOT load classifier weights - using fresh head

    encoder = encoder.to(device)
    cls_head = cls_head.to(device)

    # Freeze encoder if requested
    if freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
        print("Encoder FROZEN")
    else:
        print("Encoder UNFROZEN")

    print(f"Encoder: {sum(p.numel() for p in encoder.parameters())/1e6:.2f}M params")
    print(f"Classifier: {sum(p.numel() for p in cls_head.parameters())/1e6:.2f}M params")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")

    return encoder, cls_head


# =============================================================================
# Loss Function
# =============================================================================
class FocalLoss(nn.Module):
    """Multi-label focal loss."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_weight = (1 - pt) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * focal_weight * bce).mean()


# =============================================================================
# Training Functions
# =============================================================================
def train_one_epoch(encoder, cls_head, loader, criterion, optimizer, device, freeze_encoder):
    if not freeze_encoder:
        encoder.train()
    else:
        encoder.eval()
    cls_head.train()

    total_loss = 0
    num_batches = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        if freeze_encoder:
            with torch.no_grad():
                features = encoder(images)
        else:
            features = encoder(images)

        logits = cls_head(features[-1])
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(cls_head.parameters()), 1.0
        )
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def validate(encoder, cls_head, loader, criterion, device):
    encoder.eval()
    cls_head.eval()

    total_loss = 0
    all_logits = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Validating", leave=False):
        images, labels = images.to(device), labels.to(device)
        features = encoder(images)
        logits = cls_head(features[-1])
        loss = criterion(logits, labels)

        total_loss += loss.item()
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    probs = torch.sigmoid(all_logits).numpy()
    labels_np = all_labels.numpy()

    metrics = compute_all_metrics(probs, labels_np)
    metrics['loss'] = total_loss / len(loader)

    return metrics, probs, labels_np


# =============================================================================
# Metrics Computation
# =============================================================================
def compute_optimal_threshold_metrics(probs, labels, label_col_idx):
    """Find optimal threshold that maximizes F1 score and compute all metrics."""
    p = probs[:, label_col_idx]
    l = labels[:, label_col_idx]

    best_f1 = 0
    best_thresh = 0.5

    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (p >= thresh).astype(int)
        f1 = f1_score(l, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    preds = (p >= best_thresh).astype(int)

    return {
        'auc': roc_auc_score(l, p) if len(np.unique(l)) > 1 else 0.5,
        'f1': f1_score(l, preds, zero_division=0),
        'accuracy': accuracy_score(l, preds),
        'precision': precision_score(l, preds, zero_division=0),
        'recall': recall_score(l, preds, zero_division=0),
        'threshold': best_thresh
    }


def compute_all_metrics(probs, labels):
    """Compute metrics for all label columns including micro and macro metrics."""
    metrics = {}

    for i, col in enumerate(LABEL_COLS):
        col_metrics = compute_optimal_threshold_metrics(probs, labels, i)
        for metric_name, value in col_metrics.items():
            metrics[f'{col}_{metric_name}'] = value

    # Macro metrics
    for metric in ['auc', 'f1', 'accuracy', 'precision', 'recall']:
        values = [metrics[f'{col}_{metric}'] for col in LABEL_COLS]
        metrics[f'macro_{metric}'] = np.mean(values)

    # Micro metrics
    probs_flat = probs.ravel()
    labels_flat = labels.ravel()

    try:
        metrics['micro_auc'] = roc_auc_score(labels_flat, probs_flat)
    except ValueError:
        metrics['micro_auc'] = 0.5

    best_micro_f1 = 0
    best_micro_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds_flat = (probs_flat >= thresh).astype(int)
        f1 = f1_score(labels_flat, preds_flat, zero_division=0)
        if f1 > best_micro_f1:
            best_micro_f1 = f1
            best_micro_thresh = thresh

    preds_flat = (probs_flat >= best_micro_thresh).astype(int)
    metrics['micro_f1'] = f1_score(labels_flat, preds_flat, zero_division=0)
    metrics['micro_accuracy'] = accuracy_score(labels_flat, preds_flat)
    metrics['micro_precision'] = precision_score(labels_flat, preds_flat, zero_division=0)
    metrics['micro_recall'] = recall_score(labels_flat, preds_flat, zero_division=0)
    metrics['micro_threshold'] = best_micro_thresh

    return metrics


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 70)
    print("PhysioNet CT-ICH K-Fold Cross-Validation")
    print("=" * 70)
    print(f"Freeze encoder: {CONFIG['freeze_encoder']}")
    print(f"Epochs per fold: {CONFIG['epochs_per_fold']}")
    print(f"Head LR: {CONFIG['head_lr']}, Encoder LR: {CONFIG['encoder_lr']}")
    print()

    device = torch.device(CONFIG['device'])

    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    if CONFIG['freeze_encoder']:
        output_dir = output_dir.parent / 'results_physionet_frozen'
    else:
        output_dir = output_dir.parent / 'results_physionet_unfrozen'
    output_dir.mkdir(parents=True, exist_ok=True)

    all_fold_results = []

    for fold_idx in range(CONFIG['n_folds']):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx}")
        print('='*70)

        # Load fold data — PhysioNet splits use train_fold{N}.csv / val_fold{N}.csv
        train_path = Path(CONFIG['splits_dir']) / f'train_fold{fold_idx}.csv'
        val_path = Path(CONFIG['splits_dir']) / f'val_fold{fold_idx}.csv'

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        print(f"Train: {len(train_df)} slices")
        print(f"Val: {len(val_df)} slices")

        # Create datasets
        train_dataset = PhysioNetSliceDataset(CONFIG['images_dir'], train_df)
        val_dataset = PhysioNetSliceDataset(CONFIG['images_dir'], val_df)

        train_loader = DataLoader(train_dataset, CONFIG['batch_size'], shuffle=True,
                                  num_workers=CONFIG['num_workers'], pin_memory=True)
        val_loader = DataLoader(val_dataset, CONFIG['batch_size'], shuffle=False,
                               num_workers=CONFIG['num_workers'], pin_memory=True)

        # Load encoder and create fresh head
        encoder, cls_head = load_encoder_and_fresh_head(
            CONFIG['multitask_checkpoint'], device, CONFIG['freeze_encoder']
        )

        # Optimizer
        if CONFIG['freeze_encoder']:
            optimizer = AdamW(cls_head.parameters(), lr=CONFIG['head_lr'],
                            weight_decay=CONFIG['weight_decay'])
        else:
            optimizer = AdamW([
                {'params': encoder.parameters(), 'lr': CONFIG['encoder_lr']},
                {'params': cls_head.parameters(), 'lr': CONFIG['head_lr']}
            ], weight_decay=CONFIG['weight_decay'])

        scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs_per_fold'],
                                      eta_min=CONFIG['min_lr'])
        criterion = FocalLoss(CONFIG['focal_alpha'], CONFIG['focal_gamma'])

        best_auc = 0
        best_metrics = None

        for epoch in range(1, CONFIG['epochs_per_fold'] + 1):
            train_loss = train_one_epoch(encoder, cls_head, train_loader, criterion,
                                        optimizer, device, CONFIG['freeze_encoder'])
            val_metrics, probs, labels_np = validate(encoder, cls_head, val_loader,
                                                     criterion, device)
            scheduler.step()

            print(f"  Epoch {epoch}: Train Loss={train_loss:.4f}, "
                  f"Val Macro AUC={val_metrics['macro_auc']:.4f}, "
                  f"Val F1={val_metrics['macro_f1']:.4f}")

            if val_metrics['macro_auc'] > best_auc:
                best_auc = val_metrics['macro_auc']
                best_metrics = val_metrics.copy()

        # Store results
        fold_result = {
            'fold': fold_idx,
            **{f'best_{k}': v for k, v in best_metrics.items()}
        }
        all_fold_results.append(fold_result)

        # Print summary for this fold
        print(f"\nFold {fold_idx} Results (BEST epoch, at optimal thresholds):")
        print(f"  {'Class':<25} {'AUC':>8} {'F1':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8}")
        print("  " + "-" * 70)
        for col in LABEL_COLS:
            print(f"  {col:<25} "
                  f"{best_metrics[f'{col}_auc']:>8.4f} "
                  f"{best_metrics[f'{col}_f1']:>8.4f} "
                  f"{best_metrics[f'{col}_accuracy']:>8.4f} "
                  f"{best_metrics[f'{col}_precision']:>8.4f} "
                  f"{best_metrics[f'{col}_recall']:>8.4f}")
        print("  " + "-" * 70)
        print(f"  {'MACRO':<25} "
              f"{best_metrics['macro_auc']:>8.4f} "
              f"{best_metrics['macro_f1']:>8.4f} "
              f"{best_metrics['macro_accuracy']:>8.4f} "
              f"{best_metrics['macro_precision']:>8.4f} "
              f"{best_metrics['macro_recall']:>8.4f}")
        print(f"  {'MICRO':<25} "
              f"{best_metrics['micro_auc']:>8.4f} "
              f"{best_metrics['micro_f1']:>8.4f} "
              f"{best_metrics['micro_accuracy']:>8.4f} "
              f"{best_metrics['micro_precision']:>8.4f} "
              f"{best_metrics['micro_recall']:>8.4f}")

    # Overall summary
    print("\n" + "=" * 70)
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    results_df = pd.DataFrame(all_fold_results)

    print(f"\nMetrics across {CONFIG['n_folds']} folds (mean ± std):")
    print(f"  {'Metric':<20} {'Mean':>12} {'Std':>12}")
    print("  " + "-" * 44)

    for metric in ['auc', 'f1', 'accuracy', 'precision', 'recall']:
        key = f'best_macro_{metric}'
        mean = results_df[key].mean()
        std = results_df[key].std()
        print(f"  MACRO {metric.upper():<13} {mean:>12.4f} {std:>12.4f}")

    print("  " + "-" * 44)

    for metric in ['auc', 'f1', 'accuracy', 'precision', 'recall']:
        key = f'best_micro_{metric}'
        mean = results_df[key].mean()
        std = results_df[key].std()
        print(f"  MICRO {metric.upper():<13} {mean:>12.4f} {std:>12.4f}")

    # Per-class summary
    print(f"\n\nPer-class metrics (BEST epoch, mean ± std across folds):")
    print(f"  {'Class':<25} {'AUC':>12} {'F1':>12} {'Precision':>12} {'Recall':>12}")
    print("  " + "-" * 65)
    for col in LABEL_COLS:
        auc_mean = results_df[f'best_{col}_auc'].mean()
        auc_std = results_df[f'best_{col}_auc'].std()
        f1_mean = results_df[f'best_{col}_f1'].mean()
        f1_std = results_df[f'best_{col}_f1'].std()
        prec_mean = results_df[f'best_{col}_precision'].mean()
        prec_std = results_df[f'best_{col}_precision'].std()
        rec_mean = results_df[f'best_{col}_recall'].mean()
        rec_std = results_df[f'best_{col}_recall'].std()

        print(f"  {col:<25} "
              f"{auc_mean:.4f}±{auc_std:.4f} "
              f"{f1_mean:.4f}±{f1_std:.4f} "
              f"{prec_mean:.4f}±{prec_std:.4f} "
              f"{rec_mean:.4f}±{rec_std:.4f}")

    # Save results
    csv_suffix = 'frozen' if CONFIG['freeze_encoder'] else 'unfrozen'
    results_df.to_csv(output_dir / f'kfold_results_{csv_suffix}.csv', index=False)

    summary = {
        'freeze_encoder': CONFIG['freeze_encoder'],
        'n_folds': CONFIG['n_folds'],
        'epochs_per_fold': CONFIG['epochs_per_fold'],
        'macro_auc_mean': results_df['best_macro_auc'].mean(),
        'macro_auc_std': results_df['best_macro_auc'].std(),
        'macro_f1_mean': results_df['best_macro_f1'].mean(),
        'macro_f1_std': results_df['best_macro_f1'].std(),
        'micro_auc_mean': results_df['best_micro_auc'].mean(),
        'micro_auc_std': results_df['best_micro_auc'].std(),
        'micro_f1_mean': results_df['best_micro_f1'].mean(),
        'micro_f1_std': results_df['best_micro_f1'].std(),
    }
    pd.DataFrame([summary]).to_csv(output_dir / f'summary_{csv_suffix}.csv', index=False)

    print(f"\n\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
