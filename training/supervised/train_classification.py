"""
Supervised Training: Slice-Level Classification

Trains a ConvNeXt encoder (initialized from SSL pretrained weights) on
the 50% of RSNA data not used for SSL pretraining.

Task: 6-class multi-label hemorrhage classification
- epidural, intraparenchymal, intraventricular, subarachnoid, subdural, any

Features:
- Focal Loss
- Early stopping
- Cosine annealing with ReduceLROnPlateau backup
- Gradient accumulation + mixed precision

Usage:
    python train_classification.py
"""
import os
import sys
import csv
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Add repo root to path for model imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)

from models import ConvNeXtV2Encoder

# Also need SSLConfig for encoder architecture params
sys.path.insert(0, os.path.join(REPO_ROOT, 'training', 'ssl'))
from ssl_config import SSLConfig


# ==============================================================================
# PATHS (relative to repo root)
# ==============================================================================

SSL_ENCODER_CHECKPOINT = os.path.join(REPO_ROOT, 'weights', 'ssl', 'best_model.pth')

RSNA_IMAGES_DIR = os.path.join(REPO_ROOT, 'preprocessed_data', 'rsna', 'images')
RSNA_SLICE_LABELS = os.path.join(REPO_ROOT, 'preprocessed_data', 'rsna', 'labels', 'slice_labels.csv')
RSNA_METADATA = os.path.join(REPO_ROOT, 'preprocessed_data', 'rsna', 'metadata', 'slice_metadata.csv')

SSL_SPLITS_DIR = os.path.join(REPO_ROOT, 'training', 'ssl', 'splits')

OUTPUT_DIR = os.path.join(REPO_ROOT, 'weights', 'supervised')
LOG_DIR = os.path.join(REPO_ROOT, 'training', 'supervised', 'logs')


# ==============================================================================
# TRAINING CONFIG
# ==============================================================================

EPOCHS = 10
BATCH_SIZE = 16
GRAD_ACCUM = 4        # Effective batch size = 64
ENCODER_LR = 5e-5     # Fine-tuning LR for encoder
HEAD_LR = 1e-4        # Higher LR for classification head
MIN_LR = 1e-7
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 384
DROPOUT = 0.3

EARLY_STOPPING_PATIENCE = 7
MIN_DELTA = 0.001

FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0


# ==============================================================================
# EARLY STOPPING
# ==============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=7, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


# ==============================================================================
# FOCAL LOSS
# ==============================================================================

class FocalLoss(nn.Module):
    """Multi-label Focal Loss with optional label smoothing."""
    
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight
        
        return (focal_weight * bce_loss).mean()


# ==============================================================================
# DATASET
# ==============================================================================

class SupervisedRSNADataset(Dataset):
    """
    RSNA dataset using only the 50% not used for SSL pretraining.
    
    Uses the supervised_train.csv from SSL splits to determine which
    images to include in training.
    """
    
    LABEL_COLS = ['epidural', 'intraparenchymal', 'intraventricular', 
                  'subarachnoid', 'subdural', 'any']
    
    def __init__(self, images_dir, labels_path, metadata_path,
                 supervised_split_csv, split='train', val_ratio=0.1,
                 image_size=384):
        self.images_dir = images_dir
        self.split = split
        self.image_size = image_size
        
        print(f"[SupervisedRSNA] Loading data...")
        
        # Load supervised split (50% not used in SSL)
        supervised_df = pd.read_csv(supervised_split_csv)
        supervised_images = set(supervised_df['image_path'].tolist())
        print(f"[SupervisedRSNA] Supervised split: {len(supervised_images)} images")
        
        # Load metadata and labels
        meta_df = pd.read_csv(metadata_path)
        labels_df = pd.read_csv(labels_path)
        
        # Merge on slice_id
        df = pd.merge(meta_df, labels_df, 
                     left_on='original_filename', right_on='slice_id', how='inner')
        
        # Filter to only supervised split images
        df['image_basename'] = df['image_filename'].apply(lambda x: os.path.basename(x))
        supervised_basenames = set(os.path.basename(p) for p in supervised_images)
        df = df[df['image_basename'].isin(supervised_basenames)].reset_index(drop=True)
        
        print(f"[SupervisedRSNA] After filtering: {len(df)} slices")
        
        # Split by volume_id for train/val (no leakage)
        volume_ids = df['volume_id_x'].unique()
        volume_ids = sorted(volume_ids)
        np.random.seed(42)
        np.random.shuffle(volume_ids)
        
        val_count = int(val_ratio * len(volume_ids))
        if split == 'train':
            selected_vols = set(volume_ids[val_count:])
        else:
            selected_vols = set(volume_ids[:val_count])
        
        df = df[df['volume_id_x'].isin(selected_vols)].reset_index(drop=True)
        
        self.image_filenames = df['image_filename'].tolist()
        self.labels_array = df[self.LABEL_COLS].values.astype(np.float32)
        
        print(f"[SupervisedRSNA] {split} set: {len(self)} slices")
        
        # Transforms
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        label = self.labels_array[idx]
        
        img_path = os.path.join(self.images_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
            image = Image.new('RGB', (self.image_size, self.image_size))
        
        image = self.transform(image)
        
        return {
            'image': image,
            'labels': torch.from_numpy(label)
        }


# ==============================================================================
# MODEL
# ==============================================================================

class SliceClassificationModel(nn.Module):
    """
    ConvNeXt encoder + classification head for 6-class hemorrhage classification.
    
    The encoder is initialized from SSL pretrained weights.
    """
    
    def __init__(self, encoder, num_classes=6, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        
        embed_dim = encoder.out_channels[-1]  # 768 for ConvNeXtV2-Tiny
        self.embed_dim = embed_dim
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
    
    def forward(self, x):
        features = self.encoder.get_final_features(x)
        logits = self.classifier(features)
        return logits
    
    def save_encoder(self, path):
        """Save only the encoder weights for downstream tasks."""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'embed_dim': self.embed_dim,
            'out_channels': self.encoder.out_channels,
        }, path)
        print(f"[Model] Encoder saved to {path}")


def load_ssl_encoder(checkpoint_path, device):
    """Load ConvNeXt encoder with SSL pretrained weights."""
    config = SSLConfig()
    config.encoder_pretrained = False  # Load our own weights, not ImageNet
    
    encoder = ConvNeXtV2Encoder(config)
    
    print(f"[Model] Loading SSL encoder from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'encoder_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        print(f"[Model] Loaded encoder from 'encoder_state_dict'")
    elif 'model_state_dict' in checkpoint:
        full_state = checkpoint['model_state_dict']
        encoder_state = {}
        for key, value in full_state.items():
            if key.startswith('encoder.'):
                new_key = key[8:]
                encoder_state[new_key] = value
        
        if encoder_state:
            encoder.load_state_dict(encoder_state)
            print(f"[Model] Extracted encoder from 'model_state_dict' ({len(encoder_state)} keys)")
        else:
            raise ValueError("No encoder keys found in model_state_dict!")
    else:
        encoder.load_state_dict(checkpoint)
        print(f"[Model] Loaded raw encoder state dict")
    
    print(f"[Model] SSL encoder loaded successfully")
    return encoder


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def setup_logging(log_dir):
    """Setup logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'train_classification_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, logger):
    """Train for one epoch with mixed precision and gradient accumulation."""
    model.train()
    
    total_loss = 0.0
    step_count = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()
    
    for i, batch in enumerate(pbar):
        images = batch['image'].to(device)
        labels = batch['labels'].to(device)
        
        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)
            scaled_loss = loss / GRAD_ACCUM
        
        scaler.scale(scaled_loss).backward()
        
        if (i + 1) % GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item()
        step_count += 1
        
        pbar.set_postfix({'loss': f'{total_loss / step_count:.4f}'})
    
    avg_loss = total_loss / step_count
    logger.info(f"Epoch {epoch} Train Loss: {avg_loss:.6f}")
    
    return avg_loss


@torch.no_grad()
def validate(model, loader, criterion, device, logger):
    """Validate model and compute per-class AUCs."""
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    for batch in tqdm(loader, desc="Validating", leave=False):
        images = batch['image'].to(device)
        labels = batch['labels'].to(device)
        
        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)
        
        total_loss += loss.item()
        
        probs = torch.sigmoid(logits)
        all_preds.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    
    preds = np.vstack(all_preds)
    labels = np.vstack(all_labels)
    
    class_names = ['EDH', 'IPH', 'IVH', 'SAH', 'SDH', 'Any']
    aucs = []
    
    logger.info("Per-class AUCs:")
    for i, name in enumerate(class_names):
        if labels[:, i].sum() > 0:
            try:
                auc = roc_auc_score(labels[:, i], preds[:, i])
                aucs.append(auc)
                logger.info(f"  {name:<6} AUC: {auc:.4f}  (n+={int(labels[:, i].sum())})")
            except Exception as e:
                logger.warning(f"  {name:<6} AUC: N/A ({e})")
    
    macro_auc = np.mean(aucs) if aucs else 0.0
    avg_loss = total_loss / len(loader)
    
    logger.info(f"Macro AUC: {macro_auc:.4f}, Val Loss: {avg_loss:.6f}")
    
    return avg_loss, macro_auc, aucs


def save_training_log(log_path, epoch, train_loss, val_loss, val_auc, lr):
    """Save training metrics to CSV."""
    file_exists = os.path.exists(log_path)
    
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_auc', 'lr', 'timestamp'])
        writer.writerow([
            epoch, f'{train_loss:.6f}', f'{val_loss:.6f}', 
            f'{val_auc:.4f}', f'{lr:.2e}', datetime.now().isoformat()
        ])


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = setup_logging(LOG_DIR)
    
    logger.info("=" * 70)
    logger.info("SUPERVISED TRAINING: SLICE-LEVEL CLASSIFICATION")
    logger.info("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Load datasets
    logger.info("\nLoading datasets...")
    
    supervised_split_csv = os.path.join(SSL_SPLITS_DIR, 'supervised_train.csv')
    
    train_dataset = SupervisedRSNADataset(
        images_dir=RSNA_IMAGES_DIR,
        labels_path=RSNA_SLICE_LABELS,
        metadata_path=RSNA_METADATA,
        supervised_split_csv=supervised_split_csv,
        split='train', val_ratio=0.1, image_size=IMAGE_SIZE
    )
    
    val_dataset = SupervisedRSNADataset(
        images_dir=RSNA_IMAGES_DIR,
        labels_path=RSNA_SLICE_LABELS,
        metadata_path=RSNA_METADATA,
        supervised_split_csv=supervised_split_csv,
        split='val', val_ratio=0.1, image_size=IMAGE_SIZE
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
    logger.info(f"Train: {len(train_dataset)} slices")
    logger.info(f"Val: {len(val_dataset)} slices")
    
    # Build model
    logger.info("\nBuilding model...")
    
    encoder = load_ssl_encoder(SSL_ENCODER_CHECKPOINT, device)
    model = SliceClassificationModel(encoder, num_classes=6, dropout=DROPOUT)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,}")
    
    # Training setup
    param_groups = [
        {'params': model.encoder.parameters(), 'lr': ENCODER_LR},
        {'params': model.classifier.parameters(), 'lr': HEAD_LR}
    ]
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=MIN_LR)
    
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
    scaler = GradScaler()
    
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, min_delta=MIN_DELTA, mode='max')
    
    logger.info(f"\nTraining config:")
    logger.info(f"  Epochs: {EPOCHS}, Batch: {BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}")
    logger.info(f"  Encoder LR: {ENCODER_LR}, Head LR: {HEAD_LR}")
    logger.info(f"  Focal Loss: alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA}")
    
    # Training loop
    logger.info("\nStarting training...")
    
    best_auc = 0.0
    csv_log_path = os.path.join(LOG_DIR, 'training_metrics.csv')
    
    for epoch in range(1, EPOCHS + 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"Epoch {epoch}/{EPOCHS}")
        logger.info(f"{'='*70}")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, logger)
        
        torch.cuda.empty_cache()
        val_loss, val_auc, class_aucs = validate(model, val_loader, criterion, device, logger)
        
        cosine_scheduler.step()
        plateau_scheduler.step(val_auc)
        
        save_training_log(csv_log_path, epoch, train_loss, val_loss, val_auc, optimizer.param_groups[0]['lr'])
        
        # Save checkpoints
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'encoder_state_dict': model.encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_auc': val_auc,
            'val_loss': val_loss,
        }
        
        torch.save(checkpoint, os.path.join(OUTPUT_DIR, 'latest_checkpoint.pth'))
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(checkpoint, os.path.join(OUTPUT_DIR, 'best_classification.pth'))
            model.save_encoder(os.path.join(OUTPUT_DIR, 'best_classification_encoder.pth'))
            logger.info(f"New best model! Val AUC = {val_auc:.4f}")
        
        if early_stopping(val_auc, epoch):
            logger.info(f"\nEarly stopping at epoch {epoch}")
            logger.info(f"Best epoch: {early_stopping.best_epoch}, Best AUC: {early_stopping.best_score:.4f}")
            break
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best Val AUC: {best_auc:.4f}")
    logger.info(f"Checkpoints saved to: {OUTPUT_DIR}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
