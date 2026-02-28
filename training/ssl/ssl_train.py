"""
SSL Training Script - FCMAE Pretraining (RSNA Only)

Main training loop for self-supervised pretraining using only 50% of RSNA data.
Uses GLCM-MAE with MSE + GLCM loss.

Usage:
    python ssl_train.py [--resume PATH] [--debug]
"""
import argparse
import os
import sys
import random
import csv
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Add repo root to path for model imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)

from models import GLCM_MAE
from ssl_config import SSLConfig
from ssl_dataset import SSLDataset, collate_fn


# =============================================================================
# Training Utilities
# =============================================================================

class CheckpointManager:
    """Manages saving and loading of training checkpoints."""
    
    def __init__(self, save_dir, max_checkpoints=3):
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        os.makedirs(save_dir, exist_ok=True)
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_loss': val_loss,
        }
        
        # Save periodic checkpoint
        path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"  Best model saved: {best_path}")
        
        # Cleanup old checkpoints (keep max_checkpoints most recent)
        self._cleanup()
    
    def load_checkpoint(self, model, optimizer, scheduler, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"  Resumed from epoch {epoch}, val_loss={val_loss:.6f}")
        return epoch, val_loss
    
    def save_encoder_only(self, model, epoch):
        """Save only the encoder weights for downstream tasks."""
        encoder_path = os.path.join(self.save_dir, 'pretrained_encoder.pth')
        torch.save({
            'encoder_state_dict': model.encoder.state_dict(),
            'epoch': epoch,
        }, encoder_path)
        print(f"  Encoder saved: {encoder_path}")
    
    def _cleanup(self):
        import glob
        checkpoints = sorted(glob.glob(os.path.join(self.save_dir, 'checkpoint_epoch_*.pth')))
        while len(checkpoints) > self.max_checkpoints:
            os.remove(checkpoints.pop(0))


# =============================================================================
# Core Functions
# =============================================================================

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    """Cosine learning rate schedule with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        
        base_lr = optimizer.param_groups[0]['lr']
        min_lr_ratio = min_lr / base_lr
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, epoch, config, global_step):
    """Train for one epoch."""
    model.train()
    
    epoch_loss = 0.0
    epoch_mse = 0.0
    epoch_glcm = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{config.num_epochs}')
    
    for batch_idx, (images, _) in enumerate(pbar):
        images = images.to(config.device)
        
        with autocast(enabled=config.use_amp):
            loss, loss_dict = model(images, return_reconstruction=False)
            loss = loss / config.gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            if scheduler is not None:
                scheduler.step()
        
        epoch_loss += loss_dict['total_loss']
        epoch_mse += loss_dict['mse_loss']
        epoch_glcm += loss_dict['glcm_loss']
        
        pbar.set_postfix({
            'mse': f"{loss_dict['mse_loss']:.6f}",
            'glcm': f"{loss_dict['glcm_loss']:.6f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.1e}"
        })
        
        global_step += 1
    
    num_batches = len(dataloader)
    avg_loss = epoch_loss / num_batches
    avg_mse = epoch_mse / num_batches
    avg_glcm = epoch_glcm / num_batches
    
    return avg_loss, avg_mse, avg_glcm, global_step


def validate(model, dataloader, config):
    """Validation loop."""
    model.eval()
    
    total_loss = 0.0
    total_mse = 0.0
    total_glcm = 0.0
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc='Validation'):
            images = images.to(config.device)
            loss, loss_dict = model(images, return_reconstruction=False)
            
            total_loss += loss_dict['total_loss']
            total_mse += loss_dict['mse_loss']
            total_glcm += loss_dict['glcm_loss']
    
    num_batches = len(dataloader)
    return total_loss / num_batches, total_mse / num_batches, total_glcm / num_batches


def main(args):
    """Main training function."""
    config = SSLConfig()
    
    if args.debug:
        config.debug_mode = True
        config.num_epochs = 5
    
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*60}")
    print(f"SSL Pretraining - RSNA Only")
    print(f"{'='*60}")
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Effective batch size: {config.get_effective_batch_size()}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"GLCM weight: {config.glcm_weight}")
    print(f"SSL train ratio: {config.ssl_train_ratio}")
    print(f"{'='*60}\n")
    
    set_seed(config.seed)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = SSLDataset(config, split='train', is_train=True)
    val_dataset = SSLDataset(config, split='val', is_train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}\n")
    
    # Create model
    print("Creating model...")
    model = GLCM_MAE(config).to(config.device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    num_training_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    num_warmup_steps = len(train_loader) * config.warmup_epochs // config.gradient_accumulation_steps
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr=config.min_lr
    )
    
    print(f"Total training steps: {num_training_steps}")
    print(f"Warmup steps: {num_warmup_steps}\n")
    
    scaler = GradScaler(enabled=config.use_amp)
    checkpoint_mgr = CheckpointManager(config.output_dir, max_checkpoints=3)
    
    # CSV loss logger
    csv_log_path = os.path.join(config.log_dir, 'training_losses.csv')
    csv_file = open(csv_log_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'train_mse', 'train_glcm', 'val_loss', 'val_mse', 'val_glcm', 'lr', 'timestamp'])
    print(f"Loss log: {csv_log_path}")
    
    # Resume if specified
    start_epoch = 0
    global_step = 0
    if args.resume:
        start_epoch, _ = checkpoint_mgr.load_checkpoint(model, optimizer, scheduler, args.resume)
        start_epoch += 1
        global_step = start_epoch * len(train_loader)
    
    # Training loop
    print("Starting training...\n")
    best_val_loss = float('inf')
    patience = 10
    no_improve_count = 0
    
    for epoch in range(start_epoch, config.num_epochs):
        train_loss, train_mse, train_glcm, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            epoch, config, global_step
        )
        
        print(f"\nEpoch {epoch}/{config.num_epochs} - Train Loss: {train_loss:.6f} (MSE: {train_mse:.6f}, GLCM: {train_glcm:.6f})")
        
        val_loss, val_mse, val_glcm = validate(model, val_loader, config)
        print(f"Epoch {epoch}/{config.num_epochs} - Val Loss: {val_loss:.6f} (MSE: {val_mse:.6f}, GLCM: {val_glcm:.6f})")
        
        # Log to CSV
        current_lr = optimizer.param_groups[0]['lr']
        csv_writer.writerow([epoch, train_loss, train_mse, train_glcm, val_loss, val_mse, val_glcm, current_lr, datetime.now().isoformat()])
        csv_file.flush()
        
        # Check for improvement
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            no_improve_count = 0
            print(f"  -> New best! Saving checkpoint...")
        else:
            no_improve_count += 1
            print(f"  -> No improvement for {no_improve_count}/{patience} epochs")
        
        # Save checkpoint
        if (epoch + 1) % config.save_freq == 0 or is_best:
            checkpoint_mgr.save_checkpoint(model, optimizer, scheduler, epoch, val_loss, is_best=is_best)
        
        # Early stopping
        if no_improve_count >= patience:
            print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
            print(f"Stopping at epoch {epoch}. Best val loss: {best_val_loss:.6f}")
            break
    
    # Save final encoder
    print("\nTraining complete! Saving final pretrained encoder...")
    checkpoint_mgr.save_encoder_only(model, config.num_epochs - 1)
    
    csv_file.close()
    print(f"\n{'='*60}")
    print(f"Training finished!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {config.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SSL Pretraining - RSNA Only')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true', help='Debug mode with small dataset')
    
    args = parser.parse_args()
    main(args)
