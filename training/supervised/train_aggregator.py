"""
Supervised Training: Scan-Level Aggregator

Uses:
1. Frozen ConvNeXtV2 Encoder (loaded from best classification checkpoint)
2. Spatial Transformer Aggregator
3. Focal Loss + Cosine Annealing

Data:
- MBH-Seg scan-level preprocessed images
- Reads PNG slices, applies normalization on-the-fly

Usage:
    python train_aggregator.py
"""
import os
import sys
import logging
import random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from pathlib import Path

# Add repo root to path for model imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)

from models import ConvNeXtV2Encoder, SpatialTransformerAggregator

# SSLConfig needed for encoder architecture
sys.path.insert(0, os.path.join(REPO_ROOT, 'training', 'ssl'))
from ssl_config import SSLConfig


# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    # Paths (relative to repo root)
    'image_dir': os.path.join(REPO_ROOT, 'preprocessed_data', 'MBH_Seg_scan', 'images'),
    'metadata_csv': os.path.join(REPO_ROOT, 'preprocessed_data', 'MBH_Seg_scan', 'metadata', 'slice_metadata.csv'),
    'splits_dir': os.path.join(REPO_ROOT, 'training', 'supervised', 'aggregator_splits'),
    'checkpoint_path': os.path.join(REPO_ROOT, 'weights', 'supervised', 'best_classification_encoder.pth'),
    'output_dir': os.path.join(REPO_ROOT, 'weights', 'supervised'),
    'log_dir': os.path.join(REPO_ROOT, 'training', 'supervised', 'logs'),
    
    # Model
    'feature_dim': 768,
    'spatial_size': 12,
    'num_classes': 6,   # any + 5 subtypes
    'freeze_encoder': True,
    
    # Training
    'epochs': 30,
    'batch_size': 1,       # 1 scan per batch (memory constraint)
    'grad_accum': 8,       # Effective batch size = 8
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'num_workers': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Loss
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    
    # Preprocessing
    'image_size': 384,
}

LABELS = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']


# ==============================================================================
# UTILS
# ==============================================================================
def setup_logging(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/train_aggregator.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


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


# ==============================================================================
# DATASET
# ==============================================================================
class ScanAggregatorDataset(Dataset):
    def __init__(self, split_csv, image_dir, metadata_csv):
        self.split_df = pd.read_csv(split_csv)
        self.meta_df = pd.read_csv(metadata_csv)
        self.image_dir = Path(image_dir)
        
        self.scan_ids = self.split_df['patientID_studyID'].values
        self.labels = self.split_df[LABELS].values.astype(np.float32)
        
        # Group metadata by scan_id for faster access
        self.scan_to_slices = self.meta_df.groupby('scan_id')['image_filename'].apply(list).to_dict()
        
    def __len__(self):
        return len(self.split_df)
        
    def __getitem__(self, idx):
        scan_id = self.scan_ids[idx]
        image_filenames = self.scan_to_slices.get(scan_id, [])
        
        image_filenames.sort()
        
        slices = []
        if not image_filenames:
            print(f"Warning: No slices found for {scan_id}")
            slices.append(torch.zeros((3, 384, 384)))
        else:
            for fname in image_filenames:
                img_path = self.image_dir / fname
                try:
                    img = cv2.imread(str(img_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype(np.float32) / 255.0
                    img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1)
                    slices.append(img_tensor)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
            
        if not slices:
            slices.append(torch.zeros((3, 384, 384)))
            
        scan_tensor = torch.stack(slices)
        
        # Max slices clipping
        if scan_tensor.shape[0] > 64:
            indices = np.linspace(0, scan_tensor.shape[0]-1, 64).astype(int)
            scan_tensor = scan_tensor[indices]
            
        return {
            'scan': scan_tensor,
            'label': torch.tensor(self.labels[idx]),
            'id': scan_id
        }


def collate_fn_scan(batch):
    max_slices = max(item['scan'].shape[0] for item in batch)
    
    padded_scans = []
    masks = []
    labels = []
    ids = []
    
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
        'scan': torch.stack(padded_scans),
        'mask': torch.stack(masks),
        'label': torch.stack(labels),
        'id': ids
    }


# ==============================================================================
# MODEL
# ==============================================================================
class ScanAggregatorModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # 1. SSL Encoder
        ssl_config = SSLConfig()
        ssl_config.encoder_pretrained = False
        self.encoder = ConvNeXtV2Encoder(ssl_config)
        self._load_encoder(config['checkpoint_path'])
        
        if config['freeze_encoder']:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder Frozen")
        
        # 2. Aggregator
        self.aggregator = SpatialTransformerAggregator(
            feature_dim=config['feature_dim'],
            spatial_size=config['spatial_size'],
            max_slices=64,
            num_heads=8,
            num_layers=4,
            dropout=0.1
        )
        
        # 3. Classification head
        self.head = nn.Sequential(
            nn.Linear(config['feature_dim'], config['feature_dim'] // 2),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(config['feature_dim'] // 2, config['num_classes'])
        )
        
    def _load_encoder(self, path):
        print(f"Loading encoder from {path}")
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('encoder_state_dict', ckpt)
        state_dict = {k.replace('module.', ''): v for k,v in state_dict.items()}
        self.encoder.load_state_dict(state_dict, strict=False)
        
    def forward(self, scans, mask):
        """
        scans: (B, D, 3, H, W)
        mask: (B, D)
        """
        B, D, C, H, W = scans.shape
        
        scans_flat = scans.view(B*D, C, H, W)
        
        if not any(p.requires_grad for p in self.encoder.parameters()):
            with torch.no_grad():
                features_flat = self.encoder.get_final_features(scans_flat)
        else:
            features_flat = self.encoder.get_final_features(scans_flat)
             
        features = features_flat.view(B, D, 768, 12, 12)
        
        pooled = self.aggregator(features, mask, return_spatial=False)
        
        logits = self.head(pooled)
        
        return logits


# ==============================================================================
# TRAINING
# ==============================================================================
def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    if CONFIG['freeze_encoder']:
        model.encoder.eval()
        
    total_loss = 0
    steps = 0
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for i, batch in enumerate(pbar):
        scans = batch['scan'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['label'].to(device)
        
        with autocast():
            logits = model(scans, mask)
            loss = criterion(logits, labels)
            loss = loss / CONFIG['grad_accum']
            
        scaler.scale(loss).backward()
        
        if (i + 1) % CONFIG['grad_accum'] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        total_loss += loss.item() * CONFIG['grad_accum']
        steps += 1
        pbar.set_postfix({'loss': total_loss/steps})
        
    return total_loss / steps


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in loader:
        scans = batch['scan'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['label'].to(device)
        
        with autocast():
            logits = model(scans, mask)
            loss = criterion(logits, labels)
        
        total_loss += loss.item()
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        
    preds = np.vstack(all_preds)
    targs = np.vstack(all_labels)
    
    aucs = []
    for i in range(CONFIG['num_classes']):
        try:
            auc = roc_auc_score(targs[:, i], preds[:, i])
            aucs.append(auc)
        except:
            pass
            
    macro_auc = np.mean(aucs) if aucs else 0
    
    return total_loss / len(loader), macro_auc


def main():
    logger = setup_logging(CONFIG['log_dir'])
    logger.info("Starting Scan-Level Aggregator Training")
    
    device = torch.device(CONFIG['device'])
    
    # Datasets
    train_ds = ScanAggregatorDataset(
        f"{CONFIG['splits_dir']}/aggregator_train.csv", 
        CONFIG['image_dir'],
        CONFIG['metadata_csv']
    )
    val_ds = ScanAggregatorDataset(
        f"{CONFIG['splits_dir']}/aggregator_val.csv", 
        CONFIG['image_dir'],
        CONFIG['metadata_csv']
    )
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, 
                             num_workers=CONFIG['num_workers'], collate_fn=collate_fn_scan)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, 
                           num_workers=CONFIG['num_workers'], collate_fn=collate_fn_scan)
    
    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Model
    model = ScanAggregatorModel(CONFIG).to(device)
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    criterion = FocalLoss(alpha=CONFIG['focal_alpha'], gamma=CONFIG['focal_gamma'])
    scaler = GradScaler()
    
    best_auc = 0
    save_path = Path(CONFIG['output_dir'])
    save_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_loss, val_auc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val AUC={val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path / "best_aggregator.pth")
            logger.info(f"Saved best model (AUC {best_auc:.4f})")
            
    logger.info("Training Complete.")


if __name__ == "__main__":
    main()
