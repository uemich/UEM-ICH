"""
Joint Multi-Task Training: Classification + Segmentation

Architecture:
- ConvNeXtV2 Encoder (unfrozen, differential LR)
- Classification Head (trained RSNA weights)
- Segmentation Head (SegFormerHead)

Training Strategy:
- Alternating batches (50/50 probability)
- Loss weight annealing: seg 0.8 -> 0.5, cls 0.2 -> 0.5
- Differential LR: encoder lower than heads

Data:
- RSNA: ~340k slices for classification (supervised split)
- MBH: ~6k slices for segmentation
"""
import os
import sys
import random
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
import pandas as pd
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms

# Add repo root for model imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models import ConvNeXtV2Encoder, SegFormerHead, FocalLoss, CombinedSegLoss
from training.ssl.ssl_config import SSLConfig

# =============================================================================
# Configuration
# =============================================================================
CONFIG = {
    # Paths - Classification (Relative to REPO_ROOT)
    'rsna_images_dir': str(REPO_ROOT / 'preprocessed_data' / 'rsna' / 'images'),
    'rsna_labels': str(REPO_ROOT / 'preprocessed_data' / 'rsna' / 'labels' / 'slice_labels.csv'),
    'rsna_metadata': str(REPO_ROOT / 'preprocessed_data' / 'rsna' / 'metadata' / 'slice_metadata.csv'),
    'rsna_supervised_split': str(REPO_ROOT / 'preprocessed_data' / 'rsna' / 'splits' / 'supervised_train.csv'),
    
    # Paths - Segmentation
    'mbh_data_dir': str(REPO_ROOT / 'preprocessed_data' / 'mbh_seg'),
    
    # Model checkpoints
    'encoder_checkpoint': str(REPO_ROOT / 'weights' / 'best_rsna_slice_encoder.pth'),
    
    # Output
    'checkpoint_dir': str(REPO_ROOT / 'weights' / 'multitask'),
    'log_dir': str(REPO_ROOT / 'training' / 'supervised' / 'logs' / 'multitask'),
    
    # Training
    'epochs': 20,
    'batch_size': 16,
    'grad_accum': 4,
    
    # Learning rates (differential)
    'encoder_lr': 5e-5,
    'head_lr': 1e-4,
    'min_lr': 1e-7,
    'weight_decay': 1e-4,
    
    # Loss weights (start -> end)
    'seg_weight_start': 0.8,
    'seg_weight_end': 0.5,
    'cls_weight_start': 0.2,
    'cls_weight_end': 0.5,
    
    # Segmentation loss
    'dice_weight': 0.8,
    'focal_weight': 0.2,
    
    # Classification focal loss
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    
    # Model
    'embed_dim': 256,
    'dropout': 0.1,
    'num_seg_classes': 6,
    'num_cls_classes': 6,
    
    # Data split
    'seg_train_ratio': 0.85,
    'cls_val_ratio': 0.1,
    'seed': 42,
    
    # Sampling
    'seg_sample_prob': 0.5,  # 50% seg, 50% cls batches
    
    # Device
    'device': 'cuda',
    'num_workers': 4,
}


# =============================================================================
# Classification Dataset (RSNA)
# =============================================================================
class RSNAClassificationDataset(Dataset):
    """RSNA dataset for slice-level classification."""
    
    LABEL_COLS = ['epidural', 'intraparenchymal', 'intraventricular', 
                  'subarachnoid', 'subdural', 'any']
    
    def __init__(self, images_dir, labels_path, metadata_path, supervised_split_csv,
                 split='train', val_ratio=0.1, image_size=384, seed=42):
        self.images_dir = images_dir
        self.split = split
        self.image_size = image_size
        
        print(f"[CLS] Loading {split} data...")
        
        # Load supervised split
        if not os.path.exists(supervised_split_csv):
            raise FileNotFoundError(f"Supervised split CSV not found: {supervised_split_csv}. "
                                    "Run SSL pretraining or split generation first.")
            
        supervised_df = pd.read_csv(supervised_split_csv)
        supervised_ids = set(f.replace('.png', '') for f in supervised_df['image_filename'])
        
        # Load metadata and labels
        meta_df = pd.read_csv(metadata_path)
        labels_df = pd.read_csv(labels_path)
        
        # Merge on slice_id
        # meta_df has original_filename (ID_xxx) and image_filename (volume_xxx_slice_yyy.png)
        # labels_df has slice_id (ID_xxx)
        df = pd.merge(meta_df, labels_df, 
                     left_on='original_filename', right_on='slice_id', how='inner')
        
        # Filter to only supervised split images
        df = df[df['original_filename'].isin(supervised_ids)].reset_index(drop=True)
        
        # Split by volume_id for train/val
        volume_ids = sorted(df['volume_id_x'].unique())
        np.random.seed(seed)
        np.random.shuffle(volume_ids)
        
        val_count = int(val_ratio * len(volume_ids))
        if split == 'train':
            selected_vols = set(volume_ids[val_count:])
        else:
            selected_vols = set(volume_ids[:val_count])
        
        df = df[df['volume_id_x'].isin(selected_vols)].reset_index(drop=True)
        
        self.image_filenames = df['image_filename'].tolist()
        self.labels_array = df[self.LABEL_COLS].values.astype(np.float32)
        
        print(f"[CLS {split}] {len(self)} samples")
        
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
# Segmentation Dataset (MBH)
# =============================================================================
class MBHSegmentationDataset(Dataset):
    """MBH dataset for segmentation."""
    
    def __init__(self, data_dir, split='train', train_ratio=0.85, seed=42, transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        metadata_path = self.data_dir / "metadata" / "slice_metadata.csv"
        metadata = pd.read_csv(metadata_path)
        
        scan_ids = sorted(metadata['scan_id'].unique())
        np.random.seed(seed)
        np.random.shuffle(scan_ids)
        
        split_idx = int(len(scan_ids) * train_ratio)
        if split == 'train':
            selected_scans = set(scan_ids[:split_idx])
        else:
            selected_scans = set(scan_ids[split_idx:])
        
        metadata = metadata[metadata['scan_id'].isin(selected_scans)]
        
        self.samples = []
        for _, row in metadata.iterrows():
            filename = row['image_filename']
            mask_path = self.data_dir / "masks_combined" / filename
            if mask_path.exists():
                self.samples.append(filename)
        
        print(f"[SEG {split}] {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filename = self.samples[idx]
        image = np.array(Image.open(self.data_dir / "images" / filename).convert('RGB'))
        mask = np.array(Image.open(self.data_dir / "masks_combined" / filename))
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask.long()


def get_seg_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])
    return A.Compose([
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])


# =============================================================================
# Model Loading
# =============================================================================
def load_encoder_with_classifier(checkpoint_path, device):
    """Load encoder with trained classification head."""
    config = SSLConfig()
    config.encoder_pretrained = False
    encoder = ConvNeXtV2Encoder(config)
    
    cls_head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(CONFIG['dropout']),
        nn.Linear(768, 256),
        nn.GELU(),
        nn.Dropout(CONFIG['dropout']),
        nn.Linear(256, CONFIG['num_cls_classes'])
    )
    
    print(f"Loading checkpoint: {checkpoint_path}")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # 1. Load encoder
        if 'encoder_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            print("✓ Loaded encoder from 'encoder_state_dict'")
        elif 'model_state_dict' in checkpoint:
            # Extract encoder from full model (SSL/Classification)
            full_state = checkpoint['model_state_dict']
            encoder_state = {k[8:]: v for k, v in full_state.items() if k.startswith('encoder.')}
            if encoder_state:
                encoder.load_state_dict(encoder_state)
                print(f"✓ Extracted encoder from 'model_state_dict' ({len(encoder_state)} keys)")
        
        # 2. Load classifier
        if 'classifier_state_dict' in checkpoint:
            cls_head.load_state_dict(checkpoint['classifier_state_dict'])
            print("✓ Loaded classifier from 'classifier_state_dict'")
        elif 'model_state_dict' in checkpoint:
            full_state = checkpoint['model_state_dict']
            cls_state = {k[11:]: v for k, v in full_state.items() if k.startswith('classifier.')}
            if cls_state:
                cls_head.load_state_dict(cls_state)
                print(f"✓ Extracted classifier from 'model_state_dict' ({len(cls_state)} keys)")
    else:
        print(f"WARNING: Checkpoint {checkpoint_path} not found. Starting from scratch.")
    
    encoder = encoder.to(device)
    cls_head = cls_head.to(device)
    
    for param in encoder.parameters():
        param.requires_grad = True
    
    return encoder, cls_head


# =============================================================================
# Metrics & Helper Functions
# =============================================================================
def compute_seg_dice(pred_logits, target, num_classes=6):
    pred = pred_logits.argmax(dim=1)
    dices = []
    for c in range(1, num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        if union > 0:
            dices.append((2. * intersection / union).item())
    return np.mean(dices) if dices else 0.0


def get_loss_weights(epoch, total_epochs):
    progress = epoch / total_epochs
    seg_weight = CONFIG['seg_weight_start'] - (CONFIG['seg_weight_start'] - CONFIG['seg_weight_end']) * progress
    cls_weight = CONFIG['cls_weight_start'] + (CONFIG['cls_weight_end'] - CONFIG['cls_weight_start']) * progress
    return seg_weight, cls_weight


# =============================================================================
# Training Functions
# =============================================================================
def train_one_epoch(encoder, cls_head, seg_head, cls_loader, seg_loader,
                    cls_criterion, seg_criterion, optimizer, device, 
                    grad_accum, seg_weight, cls_weight):
    encoder.train()
    cls_head.train()
    seg_head.train()
    
    cls_iter = iter(cls_loader)
    seg_iter = iter(seg_loader)
    
    total_cls_loss, total_seg_loss = 0, 0
    cls_batches, seg_batches = 0, 0
    
    total_steps = len(cls_loader) + len(seg_loader)
    optimizer.zero_grad()
    pbar = tqdm(range(total_steps), desc="Training")
    
    for step in pbar:
        do_seg = random.random() < CONFIG['seg_sample_prob']
        
        if do_seg:
            try: images, masks = next(seg_iter)
            except StopIteration:
                seg_iter = iter(seg_loader)
                images, masks = next(seg_iter)
            
            images, masks = images.to(device), masks.to(device)
            features = encoder(images)
            main_out, aux_outs = seg_head(features)
            loss, _, _ = seg_criterion(main_out, aux_outs, masks)
            weighted_loss = seg_weight * loss
            total_seg_loss += loss.item()
            seg_batches += 1
        else:
            try: images, labels = next(cls_iter)
            except StopIteration:
                cls_iter = iter(cls_loader)
                images, labels = next(cls_iter)
            
            images, labels = images.to(device), labels.to(device)
            features = encoder(images)
            logits = cls_head(features[-1])
            loss = cls_criterion(logits, labels)
            weighted_loss = cls_weight * loss
            total_cls_loss += loss.item()
            cls_batches += 1
        
        (weighted_loss / grad_accum).backward()
        
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(cls_head.parameters()) + list(seg_head.parameters()), 1.0
            )
            optimizer.step()
            optimizer.zero_grad()
        
        pbar.set_postfix({
            'cls': f"{total_cls_loss/(cls_batches+1e-6):.4f}",
            'seg': f"{total_seg_loss/(seg_batches+1e-6):.4f}"
        })
    
    return {'cls_loss': total_cls_loss / max(1, cls_batches), 'seg_loss': total_seg_loss / max(1, seg_batches)}


@torch.no_grad()
def validate_segmentation(encoder, seg_head, loader, criterion, device):
    encoder.eval()
    seg_head.eval()
    total_loss, total_dice, num_batches = 0, 0, 0
    for images, masks in tqdm(loader, desc="Val Seg"):
        images, masks = images.to(device), masks.to(device)
        features = encoder(images)
        main_out, aux_outs = seg_head(features)
        loss, _, _ = criterion(main_out, aux_outs, masks)
        total_loss += loss.item()
        total_dice += compute_seg_dice(main_out, masks)
        num_batches += 1
    return {'loss': total_loss / num_batches, 'dice': total_dice / num_batches}


# =============================================================================
# Main
# =============================================================================
def main():
    device = CONFIG['device']
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    os.makedirs(CONFIG['log_dir'], exist_ok=True)
    
    print("=" * 70)
    print("Joint Multi-Task Training: Classification + Segmentation")
    print("=" * 70)
    
    encoder, cls_head = load_encoder_with_classifier(CONFIG['encoder_checkpoint'], device)
    
    with torch.no_grad():
        dummy = torch.randn(1, 3, 384, 384).to(device)
        features = encoder(dummy)
        in_channels = [f.shape[1] for f in features]
    
    seg_head = SegFormerHead(in_channels, CONFIG['embed_dim'], CONFIG['num_seg_classes'], CONFIG['dropout']).to(device)
    
    # Datasets
    cls_train = RSNAClassificationDataset(CONFIG['rsna_images_dir'], CONFIG['rsna_labels'], CONFIG['rsna_metadata'],
                                          CONFIG['rsna_supervised_split'], 'train', CONFIG['cls_val_ratio'])
    seg_train = MBHSegmentationDataset(CONFIG['mbh_data_dir'], 'train', CONFIG['seg_train_ratio'], CONFIG['seed'],
                                        get_seg_transforms(True))
    seg_val = MBHSegmentationDataset(CONFIG['mbh_data_dir'], 'val', CONFIG['seg_train_ratio'], CONFIG['seed'],
                                      get_seg_transforms(False))
    
    # Loaders
    cls_train_loader = DataLoader(cls_train, CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
    seg_train_loader = DataLoader(seg_train, CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
    seg_val_loader = DataLoader(seg_val, CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)
    
    # Losses
    cls_criterion = FocalLoss(CONFIG['focal_alpha'], CONFIG['focal_gamma'])
    seg_criterion = CombinedSegLoss(CONFIG['num_seg_classes'], CONFIG['focal_weight'], CONFIG['dice_weight'])
    
    # Optimizer
    optimizer = AdamW([
        {'params': encoder.parameters(), 'lr': CONFIG['encoder_lr']},
        {'params': cls_head.parameters(), 'lr': CONFIG['head_lr']},
        {'params': seg_head.parameters(), 'lr': CONFIG['head_lr']}
    ], weight_decay=CONFIG['weight_decay'])
    
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=CONFIG['min_lr'])
    
    log_file = Path(CONFIG['log_dir']) / f"multitask_{datetime.now():%Y%m%d_%H%M}.csv"
    with open(log_file, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'seg_weight', 'cls_weight', 'train_cls', 'train_seg', 'val_seg_dice'])
    
    best_dice = 0.0
    for epoch in range(1, CONFIG['epochs'] + 1):
        seg_w, cls_w = get_loss_weights(epoch - 1, CONFIG['epochs'])
        print(f"\nEpoch {epoch}/{CONFIG['epochs']} | Seg: {seg_w:.2f}, Cls: {cls_w:.2f}")
        
        train_metrics = train_one_epoch(encoder, cls_head, seg_head, cls_train_loader, seg_train_loader,
                                        cls_criterion, seg_criterion, optimizer, device, CONFIG['grad_accum'],
                                        seg_w, cls_w)
        val_metrics = validate_segmentation(encoder, seg_head, seg_val_loader, seg_criterion, device)
        scheduler.step()
        
        print(f"Train: CLS={train_metrics['cls_loss']:.4f}, SEG={train_metrics['seg_loss']:.4f}")
        print(f"Val:   Dice={val_metrics['dice']:.4f}")
        
        with open(log_file, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, seg_w, cls_w, train_metrics['cls_loss'], train_metrics['seg_loss'], val_metrics['dice']])
        
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            torch.save({'epoch': epoch, 'encoder_state_dict': encoder.state_dict(),
                        'classifier_state_dict': cls_head.state_dict(), 'seg_head_state_dict': seg_head.state_dict(),
                        'val_dice': val_metrics['dice']}, Path(CONFIG['checkpoint_dir']) / 'best_multitask.pth')
            print(f"✓ New best Dice! {best_dice:.4f}")
            
    print(f"\nTraining complete! Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
