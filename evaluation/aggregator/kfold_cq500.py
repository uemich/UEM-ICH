"""
CQ500 Aggregator K-Fold Validation (Unfrozen Encoder + Pre-trained Aggregator)

Same experiment as MBH aggregator K-Fold, adapted for CQ500:
- 5-Fold Cross Validation
- Unfrozen Encoder (from best_multitask.pth)
- Pre-trained Aggregator (from best_aggregator.pth)
- Fresh 14-class Head (all CQ500 labels)
- Differential Learning Rates (encoder 5e-5, aggregator 5e-5, head 1e-4)

Splits:
  aggregator/splits/cq500/kfold_train_fold{N}.csv
  aggregator/splits/cq500/kfold_val_fold{N}.csv
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
from pathlib import Path

# Add repo root for model imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.ssl.ssl_config import SSLConfig
from models import ConvNeXtV2Encoder
from models.aggregator import SpatialTransformerAggregator


# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    # Paths (relative to repo root)
    'image_dir': str(REPO_ROOT / 'preprocessed_data' / 'cq500' / 'images'),
    'metadata_csv': str(REPO_ROOT / 'preprocessed_data' / 'cq500' / 'metadata' / 'slice_metadata.csv'),
    'splits_dir': str(REPO_ROOT / 'evaluation' / 'aggregator' / 'splits' / 'cq500'),
    'encoder_checkpoint': str(REPO_ROOT / 'weights' / 'best_multitask.pth'),
    'aggregator_checkpoint': str(REPO_ROOT / 'weights' / 'best_aggregator.pth'),
    'output_dir': str(REPO_ROOT / 'evaluation' / 'aggregator' / 'results_cq500'),

    # Model
    'feature_dim': 768,
    'spatial_size': 12,
    'num_classes': 14,  # All CQ500 label columns
    'freeze_encoder': False,

    # Training
    'folds': 5,
    'epochs': 5,
    'batch_size': 1,
    'grad_accum': 8,

    # Learning Rates
    'lr_encoder': 5e-5,
    'lr_aggregator': 5e-5,
    'lr_head': 1e-4,
    'weight_decay': 1e-4,

    'num_workers': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # Loss
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'image_size': 384,
}

# All 14 CQ500 label columns
LABELS = [
    'ICH', 'IPH', 'IVH', 'SDH', 'EDH', 'SAH',
    'BleedLocation-Left', 'BleedLocation-Right', 'ChronicBleed',
    'Fracture', 'CalvarialFracture', 'OtherFracture',
    'MassEffect', 'MidlineShift'
]


# ==============================================================================
# UTILS & MODELS
# ==============================================================================
def setup_logging(log_dir, fold_idx):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/fold_{fold_idx}.log"),
            logging.StreamHandler()
        ],
        force=True
    )
    return logging.getLogger(f"Fold_{fold_idx}")


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return loss.mean()


class CQ500AggregatorDataset(Dataset):
    """CQ500 scan-level dataset. Groups slices by patient_id."""
    def __init__(self, split_csv, image_dir, metadata_csv):
        self.split_df = pd.read_csv(split_csv)
        self.meta_df = pd.read_csv(metadata_csv)
        self.image_dir = Path(image_dir)
        self.patient_ids = self.split_df['patient_id'].values
        self.labels = self.split_df[LABELS].values.astype(np.float32)
        self.patient_to_slices = self.meta_df.groupby('patient_id')['image_filename'].apply(list).to_dict()

    def __len__(self):
        return len(self.split_df)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        image_filenames = self.patient_to_slices.get(patient_id, [])
        image_filenames.sort()

        slices = []
        if not image_filenames:
            slices.append(torch.zeros((3, 384, 384)))
        else:
            for fname in image_filenames:
                img_path = self.image_dir / fname
                try:
                    img = cv2.imread(str(img_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype(np.float32) / 255.0
                    img = (img - 0.5) / 0.5
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1)
                    slices.append(img_tensor)
                except Exception:
                    pass

        if not slices:
            slices.append(torch.zeros((3, 384, 384)))
        scan_tensor = torch.stack(slices)

        if scan_tensor.shape[0] > 64:
            indices = np.linspace(0, scan_tensor.shape[0] - 1, 64).astype(int)
            scan_tensor = scan_tensor[indices]

        return {'scan': scan_tensor, 'label': torch.tensor(self.labels[idx]), 'id': patient_id}


def collate_fn_scan(batch):
    max_slices = max(item['scan'].shape[0] for item in batch)
    padded_scans, masks, labels, ids = [], [], [], []
    for item in batch:
        scan = item['scan']
        d, c, h, w = scan.shape
        padded = torch.zeros((max_slices, c, h, w))
        padded[:d] = scan
        padded_scans.append(padded)
        mask = torch.zeros(max_slices)
        mask[:d] = 1
        masks.append(mask)
        labels.append(item['label'])
        ids.append(item['id'])
    return {
        'scan': torch.stack(padded_scans), 'mask': torch.stack(masks),
        'label': torch.stack(labels), 'id': ids
    }


class CQ500AggregatorModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        ssl_config = SSLConfig()
        ssl_config.encoder_pretrained = False
        self.encoder = ConvNeXtV2Encoder(ssl_config)
        self._load_encoder(config['encoder_checkpoint'])

        if config['freeze_encoder']:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.aggregator = SpatialTransformerAggregator(
            feature_dim=config['feature_dim'], spatial_size=config['spatial_size'],
            max_slices=64, num_heads=8, num_layers=4, dropout=0.1
        )

        self.head = nn.Sequential(
            nn.Linear(config['feature_dim'], config['feature_dim'] // 2),
            nn.GELU(), nn.Dropout(0.4),
            nn.Linear(config['feature_dim'] // 2, config['num_classes'])
        )

        if config.get('aggregator_checkpoint'):
            self._load_aggregator(config['aggregator_checkpoint'])

    def _load_encoder(self, path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('encoder_state_dict', ckpt)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.encoder.load_state_dict(state_dict, strict=False)

    def _load_aggregator(self, path):
        print(f"Loading aggregator weights from: {path}")
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            state_dict = ckpt.get('model_state_dict', ckpt)
            agg_state_dict = {k: v for k, v in state_dict.items() if k.startswith('aggregator.')}
            if not agg_state_dict:
                print("WARNING: No 'aggregator.' keys found in checkpoint!")
                return
            msg = self.load_state_dict(agg_state_dict, strict=False)
            missing_agg = [k for k in msg.missing_keys if k.startswith('aggregator.')]
            if missing_agg:
                print(f"WARNING: Missing aggregator keys: {missing_agg}")
            else:
                print(f"Aggregator weights loaded ({len(agg_state_dict)} keys).")
        except Exception as e:
            print(f"Error loading aggregator weights: {e}")

    def forward(self, scans, mask):
        B, D, C, H, W = scans.shape
        scans_flat = scans.view(B * D, C, H, W)

        if not any(p.requires_grad for p in self.encoder.parameters()):
            with torch.no_grad():
                features_flat = self.encoder.get_final_features(scans_flat)
        else:
            features_flat = self.encoder.get_final_features(scans_flat)

        features = features_flat.view(B, D, 768, 12, 12)
        pooled = self.aggregator(features, mask, return_spatial=False)
        return self.head(pooled)


# ==============================================================================
# METRICS
# ==============================================================================
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
    try: auc = roc_auc_score(l, p)
    except: auc = 0.5
    return {
        'auc': auc, 'f1': f1_score(l, preds, zero_division=0),
        'accuracy': accuracy_score(l, preds),
        'precision': precision_score(l, preds, zero_division=0),
        'recall': recall_score(l, preds, zero_division=0),
        'threshold': best_thresh
    }

def compute_all_metrics(probs, labels):
    metrics = {}
    for i, col in enumerate(LABELS):
        col_metrics = compute_optimal_threshold_metrics(probs, labels, i)
        for k, v in col_metrics.items():
            metrics[f'{col}_{k}'] = v
    for metric in ['auc', 'f1', 'accuracy', 'precision', 'recall']:
        values = [metrics[f'{col}_{metric}'] for col in LABELS]
        metrics[f'macro_{metric}'] = np.mean(values)
    probs_flat = probs.ravel()
    labels_flat = labels.ravel()
    try: metrics['micro_auc'] = roc_auc_score(labels_flat, probs_flat)
    except: metrics['micro_auc'] = 0.5
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


# ==============================================================================
# TRAINING & EVALUATION
# ==============================================================================
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_probs, all_labels = 0, [], []
    for batch in loader:
        scans = batch['scan'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['label'].to(device)
        with autocast():
            logits = model(scans, mask)
            loss = criterion(logits, labels)
        total_loss += loss.item()
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    probs = np.vstack(all_probs)
    targs = np.vstack(all_labels)
    metrics = compute_all_metrics(probs, targs)
    metrics['loss'] = total_loss / len(loader)
    return metrics

def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    if CONFIG['freeze_encoder']: model.encoder.eval()
    total_loss, steps = 0, 0
    optimizer.zero_grad()
    for i, batch in enumerate(tqdm(loader, desc="Training")):
        scans = batch['scan'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['label'].to(device)
        with autocast():
            logits = model(scans, mask)
            loss = criterion(logits, labels) / CONFIG['grad_accum']
        scaler.scale(loss).backward()
        if (i + 1) % CONFIG['grad_accum'] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item() * CONFIG['grad_accum']
        steps += 1
    return total_loss / steps


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
    device = torch.device(CONFIG['device'])
    fold_metrics = []

    for fold in range(CONFIG['folds']):
        logger = setup_logging(CONFIG['output_dir'], fold)
        logger.info(f"========== Starting Fold {fold} ==========")

        # kfold_train_fold{N}.csv / kfold_val_fold{N}.csv
        train_csv = f"{CONFIG['splits_dir']}/kfold_train_fold{fold}.csv"
        val_csv = f"{CONFIG['splits_dir']}/kfold_val_fold{fold}.csv"

        train_ds = CQ500AggregatorDataset(train_csv, CONFIG['image_dir'], CONFIG['metadata_csv'])
        val_ds = CQ500AggregatorDataset(val_csv, CONFIG['image_dir'], CONFIG['metadata_csv'])

        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
                                  num_workers=CONFIG['num_workers'], collate_fn=collate_fn_scan)
        val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                                num_workers=CONFIG['num_workers'], collate_fn=collate_fn_scan)

        model = CQ500AggregatorModel(CONFIG).to(device)

        params = [
            {'params': model.aggregator.parameters(), 'lr': CONFIG['lr_aggregator']},
            {'params': model.head.parameters(), 'lr': CONFIG['lr_head']}
        ]
        if not CONFIG['freeze_encoder']:
            params.append({'params': model.encoder.parameters(), 'lr': CONFIG['lr_encoder']})

        optimizer = torch.optim.AdamW(params, weight_decay=CONFIG['weight_decay'])
        scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
        criterion = FocalLoss(alpha=CONFIG['focal_alpha'], gamma=CONFIG['focal_gamma'])
        scaler = GradScaler()

        best_auc, best_metrics = 0.0, {}

        for epoch in range(1, CONFIG['epochs'] + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
            metrics = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            logger.info(f"Epoch {epoch}: Loss={train_loss:.4f} | "
                        f"Macro AUC={metrics['macro_auc']:.4f} | Macro F1={metrics['macro_f1']:.4f}")
            if metrics['macro_auc'] > best_auc:
                best_auc = metrics['macro_auc']
                best_metrics = metrics
                torch.save(model.state_dict(), f"{CONFIG['output_dir']}/fold{fold}_best.pth")
                logger.info("  >> Saved New Best Model")

        best_metrics['fold'] = fold
        fold_metrics.append(best_metrics)
        del model, optimizer, scaler
        torch.cuda.empty_cache()

    results_df = pd.DataFrame(fold_metrics)
    results_df.to_csv(f"{CONFIG['output_dir']}/kfold_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("CQ500 AGGREGATOR K-FOLD SUMMARY")
    print("=" * 70)
    for metric in ['auc', 'f1', 'accuracy', 'precision', 'recall']:
        k = f'macro_{metric}'
        print(f"MACRO {metric.upper():<14} {results_df[k].mean():>10.4f} {results_df[k].std():>10.4f}")
    print("-" * 42)
    for metric in ['auc', 'f1', 'accuracy', 'precision', 'recall']:
        k = f'micro_{metric}'
        print(f"MICRO {metric.upper():<14} {results_df[k].mean():>10.4f} {results_df[k].std():>10.4f}")
    print(f"\n{'Class':<25} {'AUC':>10} {'F1':>10} {'Prec':>10} {'Rec':>10}")
    print("-" * 68)
    for label in LABELS:
        auc_m = results_df[f'{label}_auc'].mean()
        f1_m = results_df[f'{label}_f1'].mean()
        prec_m = results_df[f'{label}_precision'].mean()
        rec_m = results_df[f'{label}_recall'].mean()
        print(f"{label:<25} {auc_m:>10.4f} {f1_m:>10.4f} {prec_m:>10.4f} {rec_m:>10.4f}")


if __name__ == "__main__":
    main()
