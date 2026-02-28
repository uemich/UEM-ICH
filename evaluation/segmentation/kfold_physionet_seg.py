"""
PhysioNet K-Fold Cross-Validation — Segmentation

Evaluates encoder + pre-trained SegFormer decoder on PhysioNet binary
segmentation using 3-fold stratified cross-validation.

V2: Loads both encoder AND pre-trained SegFormer decoder from best_multitask.pth.
Selection criterion: best positive hard dice.

Metrics: Dice (soft/hard), IoU, pixel accuracy, slice classification, subtype dice.
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
from tqdm import tqdm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add repo root for model imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# Configuration
# =============================================================================
CONFIG = {
    # Paths (relative to repo root)
    'images_dir': str(REPO_ROOT / 'preprocessed_data' / 'physionet' / 'images'),
    'masks_dir': str(REPO_ROOT / 'preprocessed_data' / 'physionet' / 'masks'),
    'splits_dir': str(REPO_ROOT / 'preprocessed_data' / 'physionet' / 'splits'),

    # Checkpoint
    'multitask_checkpoint': str(REPO_ROOT / 'weights' / 'best_multitask.pth'),

    # Output
    'output_dir': str(REPO_ROOT / 'evaluation' / 'segmentation' / 'results_physionet'),

    # K-Fold settings
    'n_folds': 3,
    'epochs_per_fold': 60,
    'seed': 42,

    # Training
    'batch_size': 16,
    'encoder_lr': 3e-5,
    'head_lr': 5e-5,
    'min_lr': 1e-7,
    'weight_decay': 1e-4,

    # Freeze/unfreeze encoder
    'freeze_encoder': False,

    # Loss
    'loss_weights': {'dice': 0.7, 'focal': 0.3},
    'focal_alpha': 0.75,

    # Device
    'device': 'cuda',
    'num_workers': 4,
}


# =============================================================================
# Dataset
# =============================================================================
class PhysioNetSegDataset(Dataset):
    """PhysioNet Segmentation Dataset."""

    def __init__(self, images_dir, masks_dir, df, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.filenames = df['filename'].tolist()
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = self.images_dir / filename
        mask_path = self.masks_dir / filename

        image = np.array(Image.open(image_path).convert('RGB'))

        mask = np.array(Image.open(mask_path))
        mask = (mask > 127).astype(np.float32)

        # Rotate 90° CCW to align PhysioNet with RSNA orientation
        image = np.rot90(image, k=1).copy()
        mask = np.rot90(mask, k=1).copy()

        # Get subtype labels
        row = self.df.iloc[idx]
        labels = row[['Intraventricular', 'Intraparenchymal', 'Subarachnoid', 'Epidural', 'Subdural']].values.astype(np.float32)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return image, mask.unsqueeze(0), torch.tensor(labels)


def get_train_transforms():
    return A.Compose([
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])


# =============================================================================
# Model Loading
# =============================================================================
def load_encoder_and_decoder(checkpoint_path, device, freeze_encoder=False):
    """Load multitask encoder + pre-trained SegFormer decoder from best_multitask.pth."""
    from training.ssl.ssl_config import SSLConfig
    from models import ConvNeXtV2Encoder
    from models.segmentation_head import SegFormerHead

    config = SSLConfig()
    config.encoder_pretrained = False
    encoder = ConvNeXtV2Encoder(config)

    # Binary segmentation head
    decoder = SegFormerHead(num_classes=1)

    print(f"Loading encoder + decoder from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load encoder
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    print("  Encoder weights loaded.")

    # Load SegFormer decoder (handle num_classes mismatch: checkpoint=6, model=1)
    if 'seg_head_state_dict' in checkpoint:
        ckpt_sd = checkpoint['seg_head_state_dict']
        model_sd = decoder.state_dict()

        compatible = {}
        skipped = []
        for k, v in ckpt_sd.items():
            if k in model_sd and v.shape == model_sd[k].shape:
                compatible[k] = v
            else:
                skipped.append(k)

        decoder.load_state_dict(compatible, strict=False)
        print(f"  SegFormer decoder: loaded {len(compatible)}/{len(ckpt_sd)} keys.")
        if skipped:
            print(f"  Skipped {len(skipped)} keys (shape mismatch): {skipped}")
    else:
        print("  WARNING: No 'seg_head_state_dict' in checkpoint! Decoder initialized fresh.")

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    if freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
        print("  Encoder FROZEN")
    else:
        print("  Encoder UNFROZEN")

    return encoder, decoder


# =============================================================================
# Loss Functions
# =============================================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pt = torch.exp(-bce_loss)
        focal_weight = (1 - pt) ** self.gamma
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * focal_weight * bce_loss
        return loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = BinaryFocalLoss(alpha=CONFIG['focal_alpha'])

    def forward(self, main_out, aux_outs, target):
        loss = CONFIG['loss_weights']['dice'] * self.dice_loss(main_out, target) + \
               CONFIG['loss_weights']['focal'] * self.focal_loss(main_out, target)

        aux_weights = [0.4, 0.3, 0.2, 0.1]
        for i, aux in enumerate(aux_outs):
            if i < len(aux_weights):
                loss += aux_weights[i] * (
                    CONFIG['loss_weights']['dice'] * self.dice_loss(aux, target) +
                    CONFIG['loss_weights']['focal'] * self.focal_loss(aux, target)
                )
        return loss


# =============================================================================
# Metrics
# =============================================================================
def compute_metrics(preds, targets):
    """Compute Dice, IoU, pixel accuracy, and raw counts for micro averaging."""
    preds_flat = torch.sigmoid(preds).view(-1)
    targets_flat = targets.view(-1)

    # Soft Dice
    intersection_soft = (preds_flat * targets_flat).sum()
    union_soft = preds_flat.sum() + targets_flat.sum()
    soft_dice = (2. * intersection_soft + 1e-6) / (union_soft + 1e-6)

    # Hard Dice
    preds_hard = (preds_flat > 0.5).float()
    intersection_hard = (preds_hard * targets_flat).sum()
    union_hard = preds_hard.sum() + targets_flat.sum()
    hard_dice = (2. * intersection_hard + 1e-6) / (union_hard + 1e-6)

    # IoU
    union = preds_hard.sum() + targets_flat.sum() - intersection_hard
    if union == 0:
        iou = 1.0 if intersection_hard == 0 else 0.0
    else:
        iou = intersection_hard / (union + 1e-6)

    # Pixel accuracy
    correct_pixels = (preds_hard == targets_flat).sum()
    pixel_acc = correct_pixels / (preds_hard.numel() + 1e-6)

    tp = intersection_hard
    fp = preds_hard.sum() - tp
    fn = targets_flat.sum() - tp
    tn = preds_hard.numel() - (tp + fp + fn)

    return {
        'soft_dice': soft_dice.item() if isinstance(soft_dice, torch.Tensor) else soft_dice,
        'hard_dice': hard_dice.item() if isinstance(hard_dice, torch.Tensor) else hard_dice,
        'iou': iou.item() if isinstance(iou, torch.Tensor) else iou,
        'pixel_acc': pixel_acc.item() if isinstance(pixel_acc, torch.Tensor) else pixel_acc,
        'has_lesion': targets_flat.sum().item() > 0,
        'intersection_soft': intersection_soft.item(),
        'union_soft': union_soft.item(),
        'intersection_hard': intersection_hard.item(),
        'union_hard': union_hard.item(),
        'tp': tp.item(), 'fp': fp.item(), 'fn': fn.item(), 'tn': tn.item()
    }


# =============================================================================
# Training & Validation
# =============================================================================
def train_one_epoch(encoder, decoder, loader, criterion, optimizer, device, freeze_encoder):
    if not freeze_encoder:
        encoder.train()
    else:
        encoder.eval()
    decoder.train()

    total_loss = 0
    batches = 0

    for images, masks, _ in tqdm(loader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)

        if freeze_encoder:
            with torch.no_grad():
                features = encoder(images)
        else:
            features = encoder(images)

        main_out, aux_outs = decoder(features)
        loss = criterion(main_out, aux_outs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batches += 1

    return total_loss / batches


@torch.no_grad()
def validate(encoder, decoder, loader, device):
    encoder.eval()
    decoder.eval()

    all_soft_dice, all_hard_dice, all_iou, all_pixel_acc = [], [], [], []
    pos_soft_dice, pos_hard_dice, pos_iou = [], [], []

    micro_stats = {
        'int_soft': 0.0, 'uni_soft': 0.0,
        'int_hard': 0.0, 'uni_hard': 0.0,
        'tp': 0.0, 'fp': 0.0, 'fn': 0.0, 'tn': 0.0
    }

    subtype_names = ['Intraventricular', 'Intraparenchymal', 'Subarachnoid', 'Epidural', 'Subdural']
    subtype_dice = {name: [] for name in subtype_names}
    slice_preds, slice_targets = [], []

    for images, masks, labels in tqdm(loader, desc="Validating", leave=False):
        images, masks, labels = images.to(device), masks.to(device), labels.to(device)
        features = encoder(images)
        main_out, _ = decoder(features)

        for i in range(main_out.size(0)):
            m = compute_metrics(main_out[i:i+1], masks[i:i+1])

            all_soft_dice.append(m['soft_dice'])
            all_hard_dice.append(m['hard_dice'])
            all_iou.append(m['iou'])
            all_pixel_acc.append(m['pixel_acc'])

            micro_stats['int_soft'] += m['intersection_soft']
            micro_stats['uni_soft'] += m['union_soft']
            micro_stats['int_hard'] += m['intersection_hard']
            micro_stats['uni_hard'] += m['union_hard']
            micro_stats['tp'] += m['tp']
            micro_stats['fp'] += m['fp']
            micro_stats['fn'] += m['fn']
            micro_stats['tn'] += m['tn']

            if m['has_lesion']:
                pos_soft_dice.append(m['soft_dice'])
                pos_hard_dice.append(m['hard_dice'])
                pos_iou.append(m['iou'])

                for j, name in enumerate(subtype_names):
                    if labels[i, j] == 1:
                        subtype_dice[name].append(m['hard_dice'])

            probs = torch.sigmoid(main_out[i]).view(-1)
            pred_has_lesion = (probs > 0.5).sum() > 0
            slice_preds.append(1 if pred_has_lesion else 0)
            slice_targets.append(1 if m['has_lesion'] else 0)

    # Aggregate metrics
    metrics = {
        'all_soft_dice': np.mean(all_soft_dice),
        'all_hard_dice': np.mean(all_hard_dice),
        'all_iou': np.mean(all_iou),
        'all_pixel_acc': np.mean(all_pixel_acc),
        'pos_soft_dice': np.mean(pos_soft_dice) if pos_soft_dice else 0.0,
        'pos_hard_dice': np.mean(pos_hard_dice) if pos_hard_dice else 0.0,
        'pos_iou': np.mean(pos_iou) if pos_iou else 0.0,
        'micro_soft_dice': (2.0 * micro_stats['int_soft'] + 1e-6) / (micro_stats['uni_soft'] + 1e-6),
        'micro_hard_dice': (2.0 * micro_stats['int_hard'] + 1e-6) / (micro_stats['uni_hard'] + 1e-6),
        'micro_iou': micro_stats['int_hard'] / (micro_stats['uni_hard'] + 1e-6),
        'micro_pixel_acc': (micro_stats['tp'] + micro_stats['tn']) / (micro_stats['tp'] + micro_stats['tn'] + micro_stats['fp'] + micro_stats['fn'] + 1e-6),
    }

    # Subtype dice
    subtype_means = []
    for name in subtype_names:
        score = np.mean(subtype_dice[name]) if subtype_dice[name] else 0.0
        metrics[f'{name}_dice'] = score
        if subtype_dice[name]:
            subtype_means.append(score)
    metrics['macro_subtype_dice'] = np.mean(subtype_means) if subtype_means else 0.0

    # Slice classification
    sp = np.array(slice_preds)
    st = np.array(slice_targets)
    tp = ((sp == 1) & (st == 1)).sum()
    tn = ((sp == 0) & (st == 0)).sum()
    fp = ((sp == 1) & (st == 0)).sum()
    fn = ((sp == 0) & (st == 1)).sum()
    metrics['slice_acc'] = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    metrics['slice_sens'] = tp / (tp + fn + 1e-6)
    metrics['slice_spec'] = tn / (tn + fp + 1e-6)

    return metrics


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("PhysioNet Segmentation K-Fold")
    print("=" * 60)

    device = torch.device(CONFIG['device'])

    output_dir = Path(CONFIG['output_dir'])
    if CONFIG['freeze_encoder']:
        output_dir = output_dir.parent / 'results_physionet_seg_frozen'
    else:
        output_dir = output_dir.parent / 'results_physionet_seg_unfrozen'
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []

    for fold_idx in range(CONFIG['n_folds']):
        print(f"\nFold {fold_idx}")

        # PhysioNet splits: train_fold{N}.csv / val_fold{N}.csv
        train_df = pd.read_csv(Path(CONFIG['splits_dir']) / f'train_fold{fold_idx}.csv')
        val_df = pd.read_csv(Path(CONFIG['splits_dir']) / f'val_fold{fold_idx}.csv')

        train_loader = DataLoader(
            PhysioNetSegDataset(CONFIG['images_dir'], CONFIG['masks_dir'], train_df, get_train_transforms()),
            batch_size=CONFIG['batch_size'], shuffle=True,
            num_workers=CONFIG['num_workers'], pin_memory=True
        )
        val_loader = DataLoader(
            PhysioNetSegDataset(CONFIG['images_dir'], CONFIG['masks_dir'], val_df, get_val_transforms()),
            batch_size=CONFIG['batch_size'], shuffle=False,
            num_workers=CONFIG['num_workers'], pin_memory=True
        )

        encoder, decoder = load_encoder_and_decoder(
            CONFIG['multitask_checkpoint'], device, CONFIG['freeze_encoder']
        )

        optimizer = AdamW(
            list(decoder.parameters()) + (list(encoder.parameters()) if not CONFIG['freeze_encoder'] else []),
            lr=CONFIG['head_lr'] if CONFIG['freeze_encoder'] else CONFIG['encoder_lr']
        )

        criterion = CombinedLoss()
        best_pos_hard_dice = 0
        best_metrics = {}

        for epoch in range(1, CONFIG['epochs_per_fold'] + 1):
            train_loss = train_one_epoch(encoder, decoder, train_loader, criterion, optimizer, device, CONFIG['freeze_encoder'])
            metrics = validate(encoder, decoder, val_loader, device)

            current_pos_hard_dice = metrics['pos_hard_dice']

            print(f"  Epoch {epoch}: Loss={train_loss:.4f} | "
                  f"All Dice(S/H)={metrics['all_soft_dice']:.3f}/{metrics['all_hard_dice']:.3f} | "
                  f"Pos Dice(S/H)={metrics['pos_soft_dice']:.3f}/{metrics['pos_hard_dice']:.3f} | "
                  f"IoU(All/Pos)={metrics['all_iou']:.3f}/{metrics['pos_iou']:.3f} | "
                  f"PixAcc={metrics['all_pixel_acc']:.4f} | SliceAcc={metrics['slice_acc']:.3f}")

            if current_pos_hard_dice > best_pos_hard_dice:
                best_pos_hard_dice = current_pos_hard_dice
                best_metrics = metrics.copy()

                torch.save({
                    'fold': fold_idx,
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'metrics': best_metrics,
                    'pos_hard_dice': best_pos_hard_dice,
                }, output_dir / f'fold{fold_idx}_best.pth')
                print(f"    Saved best model: pos_hard_dice={best_pos_hard_dice:.4f}")

        print(f"  Fold {fold_idx} Best pos_hard_dice: {best_pos_hard_dice:.4f}")
        fold_results.append({
            'fold': fold_idx,
            'selection_score': best_pos_hard_dice,
            **best_metrics
        })

    # Save results
    df = pd.DataFrame(fold_results)
    df.to_csv(output_dir / 'kfold_results.csv', index=False)

    print("\nOverall Metrics (Mean ± Std):")

    print("Per-Slice Averages:")
    for metric in ['selection_score', 'all_soft_dice', 'all_hard_dice', 'pos_soft_dice', 'pos_hard_dice', 'all_iou', 'pos_iou', 'all_pixel_acc']:
        mean = df[metric].mean()
        std = df[metric].std()
        print(f"  {metric:<20}: {mean:.4f} ± {std:.4f}")

    print("\nMicro Averages:")
    for metric in ['micro_soft_dice', 'micro_hard_dice', 'micro_iou', 'micro_pixel_acc']:
        if metric in df.columns:
            mean = df[metric].mean()
            std = df[metric].std()
            print(f"  {metric:<20}: {mean:.4f} ± {std:.4f}")

    print("\nSlice Classification:")
    for metric in ['slice_acc', 'slice_sens', 'slice_spec']:
        mean = df[metric].mean()
        std = df[metric].std()
        print(f"  {metric:<20}: {mean:.4f} ± {std:.4f}")

    print("\nSubtype Dice:")
    for name in ['Intraventricular', 'Intraparenchymal', 'Subarachnoid', 'Epidural', 'Subdural']:
        metric = f'{name}_dice'
        if metric in df.columns:
            mean = df[metric].mean()
            std = df[metric].std()
            print(f"  {metric:<30}: {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
