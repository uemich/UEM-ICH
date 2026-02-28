"""
RSNA K-Fold Cross-Validation (Unfrozen Encoder)

Evaluates the joint-trained encoder + classifier on the RSNA test set
using stratified 5-fold cross-validation at volume level.

Features:
- 5-fold stratified sampling at volume level (pre-generated splits)
- Unfrozen encoder with differential learning rates
- Loads encoder + pre-trained classifier from multitask checkpoint
- Focal Loss, optimal F1 threshold per class
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
    'images_dir': str(REPO_ROOT / 'preprocessed_data' / 'rsna' / 'images'),
    'labels_path': str(REPO_ROOT / 'preprocessed_data' / 'rsna' / 'labels' / 'slice_labels.csv'),
    'metadata_path': str(REPO_ROOT / 'preprocessed_data' / 'rsna' / 'metadata' / 'slice_metadata.csv'),
    'splits_dir': str(REPO_ROOT / 'evaluation' / 'classification' / 'splits'),

    # Checkpoint (joint-trained encoder + classifier)
    'multitask_checkpoint': str(REPO_ROOT / 'weights' / 'best_multitask.pth'),

    # Output
    'output_dir': str(REPO_ROOT / 'evaluation' / 'classification' / 'results_rsna'),

    # K-Fold settings
    'n_folds': 5,
    'epochs_per_fold': 5,
    'seed': 42,

    # Training
    'batch_size': 4,
    'encoder_lr': 1e-5,
    'head_lr': 5e-5,
    'min_lr': 1e-7,
    'weight_decay': 1e-4,

    # Freeze encoder toggle
    'freeze_encoder': False,

    # Focal loss
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,

    # Device
    'device': 'cuda',
    'num_workers': 4,
}

LABEL_COLS = ['epidural', 'intraparenchymal', 'intraventricular',
              'subarachnoid', 'subdural', 'any']


# =============================================================================
# Dataset
# =============================================================================
class RSNASliceDataset(Dataset):
    """RSNA dataset for slice-level classification."""

    def __init__(self, images_dir, image_filenames, labels_array, image_size=384):
        self.images_dir = images_dir
        self.image_filenames = image_filenames
        self.labels_array = labels_array

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_filenames[idx])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        labels = torch.tensor(self.labels_array[idx], dtype=torch.float32)
        return image, labels


# =============================================================================
# Model Loading
# =============================================================================
def load_models(checkpoint_path, device, freeze_encoder=False):
    """Load encoder and classifier from multitask checkpoint."""
    from training.ssl.ssl_config import SSLConfig
    from models import ConvNeXtV2Encoder

    config = SSLConfig()
    config.encoder_pretrained = False
    encoder = ConvNeXtV2Encoder(config)

    cls_head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.1),
        nn.Linear(768, 256),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(256, 6)
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    cls_head.load_state_dict(checkpoint['classifier_state_dict'])

    encoder = encoder.to(device)
    cls_head = cls_head.to(device)

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
# Metrics Computation
# =============================================================================
def compute_optimal_threshold_metrics(probs, labels, label_col_idx):
    p = probs[:, label_col_idx]
    l = labels[:, label_col_idx]
    best_f1, best_thresh = 0, 0.5
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
    metrics = {}
    for i, col in enumerate(LABEL_COLS):
        col_metrics = compute_optimal_threshold_metrics(probs, labels, i)
        for k, v in col_metrics.items():
            metrics[f'{col}_{k}'] = v
    for metric in ['auc', 'f1', 'accuracy', 'precision', 'recall']:
        values = [metrics[f'{col}_{metric}'] for col in LABEL_COLS]
        metrics[f'macro_{metric}'] = np.mean(values)
    probs_flat = probs.ravel()
    labels_flat = labels.ravel()
    try: metrics['micro_auc'] = roc_auc_score(labels_flat, probs_flat)
    except ValueError: metrics['micro_auc'] = 0.5
    best_micro_f1, best_micro_thresh = 0, 0.5
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
# Training Functions
# =============================================================================
def train_one_epoch(encoder, cls_head, loader, criterion, optimizer, device, freeze_encoder):
    if not freeze_encoder:
        encoder.train()
    else:
        encoder.eval()
    cls_head.train()
    total_loss, num_batches = 0, 0
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
def evaluate(encoder, cls_head, loader, criterion, device):
    encoder.eval()
    cls_head.eval()
    total_loss, all_logits, all_labels = 0, [], []
    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
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
# Main
# =============================================================================
def main():
    device = CONFIG['device']
    Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RSNA K-Fold Cross-Validation (Unfrozen Encoder)")
    print("=" * 70)
    print(f"Freeze encoder: {CONFIG['freeze_encoder']}")
    print(f"Folds: {CONFIG['n_folds']}, Epochs/fold: {CONFIG['epochs_per_fold']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print("=" * 70)

    # Load data
    labels_df = pd.read_csv(CONFIG['labels_path'])
    metadata_df = pd.read_csv(CONFIG['metadata_path'])

    # Create lookups
    # 1. ID (e.g. ID_xxx) -> Disk filename (e.g. volume_xxx_slice_yyy.png)
    id_to_file = dict(zip(metadata_df['original_filename'], metadata_df['image_filename']))
    
    # 2. ID (e.g. ID_xxx) -> Label vector
    labels_df['slice_id_clean'] = labels_df['slice_id'] # already clean
    id_to_labels = {}
    for _, row in labels_df.iterrows():
        id_to_labels[row['slice_id']] = [row[col] for col in LABEL_COLS]

    print(f"Metadata loaded: {len(id_to_file)} mappings")
    print(f"Labels loaded: {len(id_to_labels)} mappings")

    all_fold_results = []

    for fold_idx in range(CONFIG['n_folds']):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1}/{CONFIG['n_folds']}")
        print(f"{'='*70}")

        # Load fold splits
        train_df = pd.read_csv(Path(CONFIG['splits_dir']) / f'fold_{fold_idx}_train.csv')
        val_df = pd.read_csv(Path(CONFIG['splits_dir']) / f'fold_{fold_idx}_val.csv')

        def get_files_and_labels(df):
            files, labels = [], []
            for _, row in df.iterrows():
                # Split ID might have .png extension
                clean_id = row['image_filename'].replace('.png', '')
                if clean_id in id_to_file and clean_id in id_to_labels:
                    files.append(id_to_file[clean_id])
                    labels.append(id_to_labels[clean_id])
            return files, np.array(labels, dtype=np.float32)

        train_files, train_labels = get_files_and_labels(train_df)
        val_files, val_labels = get_files_and_labels(val_df)

        print(f"Train: {len(train_files)} slices, Val: {len(val_files)} slices")

        train_dataset = RSNASliceDataset(CONFIG['images_dir'], train_files, train_labels)
        val_dataset = RSNASliceDataset(CONFIG['images_dir'], val_files, val_labels)

        train_loader = DataLoader(train_dataset, CONFIG['batch_size'], shuffle=True,
                                  num_workers=CONFIG['num_workers'], pin_memory=True)
        val_loader = DataLoader(val_dataset, CONFIG['batch_size'], shuffle=False,
                                num_workers=CONFIG['num_workers'], pin_memory=True)

        # Fresh models per fold
        encoder, cls_head = load_models(
            CONFIG['multitask_checkpoint'], device, CONFIG['freeze_encoder']
        )

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

        best_f1, best_metrics = 0, None

        for epoch in range(1, CONFIG['epochs_per_fold'] + 1):
            train_loss = train_one_epoch(encoder, cls_head, train_loader, criterion,
                                        optimizer, device, CONFIG['freeze_encoder'])
            val_metrics, _, _ = evaluate(encoder, cls_head, val_loader, criterion, device)
            scheduler.step()
            print(f"  Epoch {epoch}: Loss={train_loss:.4f}, "
                  f"Macro AUC={val_metrics['macro_auc']:.4f}, "
                  f"F1={val_metrics['macro_f1']:.4f}")
            if val_metrics['macro_f1'] > best_f1:
                best_f1 = val_metrics['macro_f1']
                best_metrics = val_metrics.copy()

        fold_result = {'fold': fold_idx, **{f'best_{k}': v for k, v in best_metrics.items()}}
        all_fold_results.append(fold_result)

        # Per-fold summary
        print(f"\nFold {fold_idx} Best:")
        print(f"  {'Class':<20} {'AUC':>8} {'F1':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8}")
        print("  " + "-" * 60)
        for col in LABEL_COLS:
            print(f"  {col:<20} "
                  f"{best_metrics[f'{col}_auc']:>8.4f} "
                  f"{best_metrics[f'{col}_f1']:>8.4f} "
                  f"{best_metrics[f'{col}_accuracy']:>8.4f} "
                  f"{best_metrics[f'{col}_precision']:>8.4f} "
                  f"{best_metrics[f'{col}_recall']:>8.4f}")

    # Overall summary
    results_df = pd.DataFrame(all_fold_results)
    results_df.to_csv(Path(CONFIG['output_dir']) / 'kfold_results.csv', index=False)

    print("\n" + "=" * 70)
    print("K-FOLD SUMMARY")
    print("=" * 70)
    for metric in ['auc', 'f1', 'accuracy', 'precision', 'recall']:
        k = f'best_macro_{metric}'
        print(f"MACRO {metric.upper():<10}: {results_df[k].mean():.4f} ± {results_df[k].std():.4f}")
    print("-" * 50)
    for metric in ['auc', 'f1', 'accuracy', 'precision', 'recall']:
        k = f'best_micro_{metric}'
        print(f"MICRO {metric.upper():<10}: {results_df[k].mean():.4f} ± {results_df[k].std():.4f}")

    print(f"\nResults saved to: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
