"""
Supervised Training: Slice-Level Segmentation (Cached Features)

Trains SegFormer decoder using pre-cached encoder features.

Features:
- SegFormer decoder with Coordinate Attention
- Dice + Focal loss with Deep Supervision
- Soft dice, hard dice, and IoU metrics
- Positive-only metric tracking

Usage:
    python train_segmentation.py
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import csv
from datetime import datetime
import h5py

# Add repo root to path for model imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)

from models import SegFormerHead


# =============================================================================
# Configuration
# =============================================================================
CONFIG = {
    # Paths (relative to repo root)
    'cache_file': os.path.join(REPO_ROOT, 'training', 'supervised', 'cached_features', 'features.h5'),
    'checkpoint_dir': os.path.join(REPO_ROOT, 'weights', 'supervised'),
    'log_dir': os.path.join(REPO_ROOT, 'training', 'supervised', 'logs'),
    
    # Training
    'epochs': 50,
    'batch_size': 16,
    'grad_accum': 4,       # Effective batch = 64
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'min_lr': 1e-6,
    
    # Loss weights
    'dice_weight': 0.8,
    'focal_weight': 0.2,
    'focal_alpha': 0.75,   # Higher weight for positive (hemorrhage)
    'focal_gamma': 2.0,
    
    # Model
    'embed_dim': 256,
    'dropout': 0.1,
    'in_channels': [96, 192, 384, 768],  # ConvNeXt-Tiny stages
    
    # Device
    'device': 'cuda',
    'num_workers': 4,
}


# =============================================================================
# Cached Dataset
# =============================================================================
class CachedFeatureDataset(Dataset):
    """Dataset that loads pre-cached encoder features."""
    
    def __init__(self, cache_file: str, split: str = 'train'):
        self.cache_file = cache_file
        self.split = split
        
        with h5py.File(cache_file, 'r') as f:
            self.sample_keys = list(f[split].keys())
        
        print(f"[Cached] {split}: {len(self.sample_keys)} samples")
    
    def __len__(self):
        return len(self.sample_keys)
    
    def __getitem__(self, idx):
        key = self.sample_keys[idx]
        
        with h5py.File(self.cache_file, 'r') as f:
            grp = f[self.split][key]
            
            features = [
                torch.from_numpy(grp[f'feat_{i}'][:]) 
                for i in range(4)
            ]
            mask = torch.from_numpy(grp['mask'][:])
        
        return features, mask


def collate_fn(batch):
    """Custom collate for list of features."""
    features_list = [item[0] for item in batch]
    masks = torch.stack([item[1] for item in batch])
    
    batched_features = []
    for i in range(4):
        batched_features.append(
            torch.stack([f[i] for f in features_list])
        )
    
    return batched_features, masks


# =============================================================================
# Losses
# =============================================================================
class DiceLoss(nn.Module):
    """Dice Loss for segmentation."""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1).float()
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice


class SegFocalLoss(nn.Module):
    """Focal Loss for segmentation class imbalance."""
    
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        target = target.float()
        
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        focal_loss = alpha_weight * focal_weight * bce
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Weighted Dice + Focal loss with Deep Supervision."""
    
    def __init__(self, dice_weight=0.5, focal_weight=0.5, alpha=0.75, gamma=2.0,
                 deep_sup_weights=[0.4, 0.3, 0.2, 0.1]):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = SegFocalLoss(alpha=alpha, gamma=gamma)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.deep_sup_weights = deep_sup_weights
    
    def _seg_loss(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.dice_weight * dice + self.focal_weight * focal, dice, focal
    
    def forward(self, main_output, aux_outputs, target):
        main_loss, dice, focal = self._seg_loss(main_output, target)
        
        aux_loss = 0
        for aux_out, weight in zip(aux_outputs, self.deep_sup_weights):
            aux_seg_loss, _, _ = self._seg_loss(aux_out, target)
            aux_loss += weight * aux_seg_loss
        
        total_loss = main_loss + aux_loss
        return total_loss, dice, focal


# =============================================================================
# Metrics
# =============================================================================
def soft_dice_score(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1).float()
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dice.item()


def hard_dice_score(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.float()
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + 1) / (pred_flat.sum() + target_flat.sum() + 1)
    return dice.item()


def iou_score(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.float()
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = (intersection + 1) / (union + 1)
    return iou.item()


# =============================================================================
# Training
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device, grad_accum):
    model.train()
    total_loss = 0
    total_dice_loss = 0
    total_focal_loss = 0
    step_count = 0
    
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="Training")
    
    for i, (features, masks) in enumerate(pbar):
        features = [f.to(device) for f in features]
        masks = masks.to(device)
        
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        
        main_out, aux_outs = model(features)
        loss, dice_loss, focal_loss = criterion(main_out, aux_outs, masks)
        scaled_loss = loss / grad_accum
        
        scaled_loss.backward()
        
        if (i + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item()
        total_dice_loss += dice_loss.item()
        total_focal_loss += focal_loss.item()
        step_count += 1
        
        pbar.set_postfix({'loss': f'{total_loss/step_count:.4f}'})
    
    return {
        'loss': total_loss / step_count,
        'dice_loss': total_dice_loss / step_count,
        'focal_loss': total_focal_loss / step_count
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_soft_dice = 0
    total_hard_dice = 0
    total_iou = 0
    num_batches = 0
    
    positive_soft_dice = 0
    positive_hard_dice = 0
    positive_iou = 0
    positive_count = 0
    
    for features, masks in tqdm(loader, desc="Validating"):
        features = [f.to(device) for f in features]
        masks = masks.to(device)
        
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        
        main_out, aux_outs = model(features)
        loss, _, _ = criterion(main_out, aux_outs, masks)
        
        total_loss += loss.item()
        
        for b in range(masks.size(0)):
            p, m = main_out[b:b+1], masks[b:b+1]
            
            sd = soft_dice_score(p, m)
            hd = hard_dice_score(p, m)
            iou_val = iou_score(p, m)
            
            total_soft_dice += sd
            total_hard_dice += hd
            total_iou += iou_val
            
            if m.sum() > 0:
                positive_soft_dice += sd
                positive_hard_dice += hd
                positive_iou += iou_val
                positive_count += 1
        
        num_batches += 1
    
    n_samples = len(loader.dataset)
    
    return {
        'loss': total_loss / num_batches,
        'soft_dice': total_soft_dice / n_samples,
        'hard_dice': total_hard_dice / n_samples,
        'iou': total_iou / n_samples,
        'positive_soft_dice': positive_soft_dice / max(positive_count, 1),
        'positive_hard_dice': positive_hard_dice / max(positive_count, 1),
        'positive_iou': positive_iou / max(positive_count, 1),
        'positive_samples': positive_count,
    }


def main():
    print("=" * 70)
    print("Segmentation Training (Cached Features)")
    print("=" * 70)
    
    device = CONFIG['device']
    
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    os.makedirs(CONFIG['log_dir'], exist_ok=True)
    
    # Load datasets
    print("\nLoading cached features...")
    train_dataset = CachedFeatureDataset(CONFIG['cache_file'], split='train')
    val_dataset = CachedFeatureDataset(CONFIG['cache_file'], split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                             shuffle=True, num_workers=CONFIG['num_workers'],
                             collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                           shuffle=False, num_workers=CONFIG['num_workers'],
                           collate_fn=collate_fn, pin_memory=True)
    
    # Create model
    print("\nCreating SegFormer decoder...")
    model = SegFormerHead(
        in_channels_list=CONFIG['in_channels'],
        embed_dim=CONFIG['embed_dim'],
        num_classes=1,
        dropout=CONFIG['dropout']
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Decoder parameters: {n_params/1e6:.2f}M")
    
    # Loss
    criterion = CombinedLoss(
        dice_weight=CONFIG['dice_weight'],
        focal_weight=CONFIG['focal_weight'],
        alpha=CONFIG['focal_alpha'],
        gamma=CONFIG['focal_gamma']
    )
    
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=CONFIG['min_lr'])
    
    # Training log
    log_file = os.path.join(CONFIG['log_dir'], 'segmentation_log.csv')
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_loss', 'dice_loss', 'focal_loss',
            'val_loss', 'soft_dice', 'hard_dice', 'iou',
            'pos_soft_dice', 'pos_hard_dice', 'pos_iou', 'lr'
        ])
    
    print(f"\nTraining config:")
    print(f"  Epochs: {CONFIG['epochs']}")
    print(f"  Batch: {CONFIG['batch_size']}x{CONFIG['grad_accum']}={CONFIG['batch_size']*CONFIG['grad_accum']}")
    print(f"  LR: {CONFIG['learning_rate']}")
    print(f"  Loss: Dice={CONFIG['dice_weight']}, Focal={CONFIG['focal_weight']}")
    
    best_dice = 0
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{CONFIG['epochs']} | LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"{'='*70}")
        
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, CONFIG['grad_accum'])
        val_metrics = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_metrics['loss']:.4f} (dice={train_metrics['dice_loss']:.4f}, focal={train_metrics['focal_loss']:.4f})")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Soft Dice: {val_metrics['soft_dice']:.4f} | Hard Dice: {val_metrics['hard_dice']:.4f} | IoU: {val_metrics['iou']:.4f}")
        print(f"  Positive-only ({val_metrics['positive_samples']} samples):")
        print(f"    Soft Dice: {val_metrics['positive_soft_dice']:.4f} | Hard Dice: {val_metrics['positive_hard_dice']:.4f} | IoU: {val_metrics['positive_iou']:.4f}")
        
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_metrics['loss'], train_metrics['dice_loss'], train_metrics['focal_loss'],
                val_metrics['loss'], val_metrics['soft_dice'], val_metrics['hard_dice'], val_metrics['iou'],
                val_metrics['positive_soft_dice'], val_metrics['positive_hard_dice'], val_metrics['positive_iou'],
                scheduler.get_last_lr()[0]
            ])
        
        if val_metrics['positive_soft_dice'] > best_dice:
            best_dice = val_metrics['positive_soft_dice']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
                'config': CONFIG,
            }
            torch.save(checkpoint, os.path.join(CONFIG['checkpoint_dir'], 'best_segmentation.pth'))
            print(f"  New best model! Positive Soft Dice = {best_dice:.4f}")
    
    print(f"\n{'='*70}")
    print(f"Training complete! Best Positive Soft Dice: {best_dice:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
