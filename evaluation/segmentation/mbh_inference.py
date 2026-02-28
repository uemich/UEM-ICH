"""
Multi-class Segmentation Inference for MBH Segmentation Challenge

Generates 3D NIfTI predictions with 6 classes from trained encoder + decoder.
Output format compatible with official MBH evaluation toolkit.

Classes: 0=BG, 1=SDH, 2=EDH, 3=IPH, 4=SAH, 5=IVH

Evaluation:
  The official MBH evaluation toolkit should be used to compute metrics.
  See: https://github.com/MBH-Seg/metrics-mbhseg2025

  Usage:
    python evaluate.py \\
      --path_gts <path_to_ground_truth_labels> \\
      --path_preds <output_dir_from_this_script> \\
      --path_out <results_output_dir>
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import nibabel as nib
import pandas as pd
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import defaultdict
from scipy.ndimage import zoom

# Add repo root for model imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# Configuration
# =============================================================================
CONFIG = {
    # Input data (relative to repo root)
    'data_dir': str(REPO_ROOT / 'preprocessed_data' / 'mbh_seg'),
    'nifti_dir': str(REPO_ROOT / 'raw_data' / 'MBH' / 'mbh_seg_val'),

    # Model checkpoints
    'encoder_checkpoint': str(REPO_ROOT / 'weights' / 'best_multitask.pth'),
    'decoder_checkpoint': str(REPO_ROOT / 'weights' / 'best_multiclass_decoder.pth'),

    # Output
    'output_dir': str(REPO_ROOT / 'evaluation' / 'segmentation' / 'predictions_mbh'),

    # Processing
    'batch_size': 8,
    'device': 'cuda',
    'num_classes': 6,

    # Split
    'split': 'official_val',
}


# =============================================================================
# Model Loading
# =============================================================================
def load_encoder(checkpoint_path, device):
    """Load the ConvNeXtV2Encoder."""
    from training.ssl.ssl_config import SSLConfig
    from models import ConvNeXtV2Encoder

    config = SSLConfig()
    config.encoder_pretrained = False
    encoder = ConvNeXtV2Encoder(config)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'encoder_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
    else:
        encoder.load_state_dict(checkpoint)

    encoder = encoder.to(device)
    encoder.eval()
    print(f"Encoder loaded: {sum(p.numel() for p in encoder.parameters())/1e6:.2f}M params")

    return encoder


def load_decoder(checkpoint_path, device, in_channels_list):
    """Load the SegFormerHead decoder from models/ package."""
    from models import SegFormerHead

    decoder = SegFormerHead(
        in_channels_list=in_channels_list,
        embed_dim=256,
        num_classes=CONFIG['num_classes']
    )

    print(f"Loading decoder from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        decoder.load_state_dict(checkpoint['model_state_dict'])
    else:
        decoder.load_state_dict(checkpoint)

    decoder = decoder.to(device)
    decoder.eval()
    print(f"Decoder loaded: {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M params")

    return decoder


# =============================================================================
# Dataset
# =============================================================================
class SliceDataset(Dataset):
    """Dataset for processing individual slices."""

    def __init__(self, data_dir, scan_ids, metadata):
        self.data_dir = Path(data_dir)
        self.scan_ids = scan_ids
        self.metadata = metadata
        self.transform = A.Compose([
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])

        # Build ordered slice list for each scan
        self.samples = []
        for scan_id in scan_ids:
            scan_meta = metadata[metadata['scan_id'] == scan_id].sort_values('slice_idx')
            for _, row in scan_meta.iterrows():
                self.samples.append({
                    'scan_id': scan_id,
                    'slice_idx': row['slice_idx'],
                    'image_filename': row['image_filename']
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = self.data_dir / "images" / sample['image_filename']
        image = np.array(Image.open(image_path).convert('RGB'))

        transformed = self.transform(image=image)
        image = transformed['image']

        return image, sample['scan_id'], sample['slice_idx']


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    scan_ids = [b[1] for b in batch]
    slice_idxs = [b[2] for b in batch]
    return images, scan_ids, slice_idxs


# =============================================================================
# Inference
# =============================================================================
@torch.no_grad()
def run_inference(encoder, decoder, dataloader, device):
    """Run inference and collect predictions per scan."""
    predictions = defaultdict(dict)

    for images, scan_ids, slice_idxs in tqdm(dataloader, desc="Running inference"):
        images = images.to(device)

        features = encoder(images)
        logits, _ = decoder(features)  # (B, 6, 384, 384)

        # Get class predictions
        preds = logits.argmax(dim=1).cpu().numpy()  # (B, 384, 384)

        for pred, scan_id, slice_idx in zip(preds, scan_ids, slice_idxs):
            predictions[scan_id][slice_idx] = pred

    return predictions


def save_predictions(predictions, output_dir, nifti_dir):
    """Save predictions as NIfTI files matching original resolution."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nifti_dir = Path(nifti_dir)

    saved_count = 0

    for scan_id, slices_dict in tqdm(predictions.items(), desc="Saving NIfTI"):
        sorted_indices = sorted(slices_dict.keys())

        # Stack into volume (384, 384, num_slices)
        volume_384 = np.stack([slices_dict[i] for i in sorted_indices], axis=-1)

        # Undo the rot90 applied during preprocessing
        volume_384 = np.rot90(volume_384, k=-1, axes=(0, 1))

        # Load reference NIfTI to get original shape
        ref_path_dir = nifti_dir / scan_id / "image.nii.gz"
        ref_path_flat = nifti_dir / f"{scan_id}.nii.gz"

        if ref_path_dir.exists():
            ref_path = ref_path_dir
        elif ref_path_flat.exists():
            ref_path = ref_path_flat
        else:
            ref_path = None

        if ref_path:
            ref_nii = nib.load(str(ref_path))
            orig_shape = ref_nii.shape
            affine = ref_nii.affine

            # Resize to original resolution
            if orig_shape[:2] != (384, 384):
                zoom_factors = (orig_shape[0] / 384, orig_shape[1] / 384, 1.0)
                volume = zoom(volume_384, zoom_factors, order=0)
            else:
                volume = volume_384

            if volume.shape[2] != orig_shape[2]:
                print(f"  Warning: {scan_id} depth mismatch: {volume.shape[2]} vs {orig_shape[2]}")
        else:
            volume = volume_384
            affine = np.eye(4)
            print(f"  No reference NIfTI for {scan_id}, using 384x384")

        # Create and save NIfTI
        volume = volume.astype(np.uint8)
        nii = nib.Nifti1Image(volume, affine)

        output_path = output_dir / f"{scan_id}.nii.gz"
        nib.save(nii, str(output_path))
        saved_count += 1

    print(f"\nSaved {saved_count} NIfTI predictions to {output_dir}")
    return saved_count


# =============================================================================
# Main
# =============================================================================
def main():
    device = CONFIG['device']
    data_dir = Path(CONFIG['data_dir'])

    print("=" * 70)
    print("Multi-class Segmentation Inference (MBH)")
    print("=" * 70)

    # Load metadata for official validation set
    if CONFIG['split'] == 'official_val':
        metadata_path = data_dir / "metadata" / "val_slice_metadata.csv"
        if not metadata_path.exists():
            print(f"Error: {metadata_path} not found")
            print("Run preprocessing/mbh_seg/preprocess_val.py first")
            return
        metadata = pd.read_csv(metadata_path)
        scan_ids = metadata['scan_id'].unique().tolist()
        print(f"Official validation set: {len(scan_ids)} scans, {len(metadata)} slices")
    else:
        raise ValueError(f"Unknown split: {CONFIG['split']}")

    # Load models
    encoder = load_encoder(CONFIG['encoder_checkpoint'], device)

    with torch.no_grad():
        dummy = torch.randn(1, 3, 384, 384).to(device)
        features = encoder(dummy)
        in_channels = [f.shape[1] for f in features]

    decoder = load_decoder(CONFIG['decoder_checkpoint'], device, in_channels)

    # Create dataset and dataloader
    dataset = SliceDataset(data_dir, scan_ids, metadata)
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Run inference
    predictions = run_inference(encoder, decoder, dataloader, device)

    # Save predictions
    save_predictions(predictions, CONFIG['output_dir'], CONFIG['nifti_dir'])

    print("\n" + "=" * 70)
    print("Inference Complete!")
    print("=" * 70)
    print(f"Predictions saved to: {CONFIG['output_dir']}")
    print(f"\nTo evaluate, use the official MBH evaluation toolkit:")
    print(f"  git clone https://github.com/MBH-Seg/metrics-mbhseg2025")
    print(f"  python evaluate.py \\")
    print(f"    --path_gts <path_to_ground_truth_labels> \\")
    print(f"    --path_preds {CONFIG['output_dir']} \\")
    print(f"    --path_out evaluation/segmentation/eval_results_mbh")


if __name__ == "__main__":
    main()
