"""
SSL Dataset - RSNA Only with 50:50 Split

Loads only RSNA slices for SSL pretraining.
Splits the original train set into SSL and supervised portions.
"""
import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ssl_transforms import SSLTransforms


class SSLDataset(Dataset):
    """
    RSNA-only SSL dataset with 50:50 train split.
    
    Uses half of RSNA train set for SSL, reserves other half for supervised.
    """
    
    def __init__(
        self,
        config,
        split: str = 'train',
        is_train: bool = True,
    ):
        """
        Args:
            config: SSLConfig instance
            split: 'train' (SSL portion) or 'val' (subset for validation)
            is_train: whether to apply training augmentations
        """
        self.config = config
        self.split = split
        self.is_train = is_train
        
        self.transforms = SSLTransforms(config)
        
        # Load or create splits
        self.image_paths = self._get_split_images()
        
        if config.debug_mode:
            self.image_paths = self.image_paths[:config.debug_samples]
        
        print(f"[SSL] {split} set: {len(self.image_paths)} images")
    
    def _filter_blank_slices(self, image_paths: List[str], 
                             min_std: float = 0.02, 
                             min_mean: float = 0.05) -> List[str]:
        """
        Filter out blank/air-only slices.
        
        Args:
            image_paths: list of image paths
            min_std: minimum standard deviation (blank images have very low std)
            min_mean: minimum mean intensity (air is mostly black/dark)
            
        Returns:
            Filtered list of image paths
        """
        cache_file = os.path.join(self.config.splits_dir, f'{self.split}_filtered.csv')
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file)
            print(f"[SSL] Loaded {len(df)} pre-filtered images from cache")
            return df['image_path'].tolist()
        
        print(f"[SSL] Filtering blank slices from {len(image_paths)} images...")
        filtered = []
        skipped = 0
        
        for img_path in tqdm(image_paths, desc="Filtering blanks"):
            try:
                img = np.array(Image.open(img_path).convert('L'), dtype=np.float32) / 255.0
                
                img_std = img.std()
                img_mean = img.mean()
                
                if img_std >= min_std and img_mean >= min_mean:
                    filtered.append(img_path)
                else:
                    skipped += 1
            except Exception as e:
                skipped += 1
                continue
        
        print(f"[SSL] Filtered: kept {len(filtered)}, skipped {skipped} blank slices")
        
        pd.DataFrame({'image_path': filtered}).to_csv(cache_file, index=False)
        
        return filtered
    
    def _get_split_images(self) -> List[str]:
        """Get image paths for this split."""
        ssl_train_file = os.path.join(self.config.splits_dir, 'ssl_train.csv')
        
        if os.path.exists(ssl_train_file):
            return self._load_existing_splits()
        else:
            return self._create_and_save_splits()
    
    def _load_existing_splits(self) -> List[str]:
        """Load pre-computed splits."""
        if self.split == 'train':
            split_file = os.path.join(self.config.splits_dir, 'ssl_train.csv')
        else:
            split_file = os.path.join(self.config.splits_dir, 'ssl_val.csv')
        
        df = pd.read_csv(split_file)
        return df['image_path'].tolist()
    
    def _create_and_save_splits(self) -> List[str]:
        """Create 50:50 split and save to disk."""
        print("[SSL] Creating data splits...")
        
        # Load RSNA metadata to get volume IDs
        meta_df = pd.read_csv(self.config.rsna_metadata)
        
        # Get all image files
        rsna_dir = Path(self.config.rsna_images)
        all_images = sorted([str(p) for p in rsna_dir.glob("*.png")])
        print(f"[SSL] Found {len(all_images)} total RSNA images")
        
        # Get unique volume IDs for proper splitting
        # (split by volume to avoid data leakage)
        meta_df['image_path'] = meta_df['image_filename'].apply(
            lambda x: str(rsna_dir / x)
        )
        
        # Filter to existing images
        existing_set = set(all_images)
        meta_df = meta_df[meta_df['image_path'].isin(existing_set)]
        
        # Get unique volumes
        if 'volume_id' in meta_df.columns:
            volume_col = 'volume_id'
        else:
            meta_df['volume_id'] = meta_df['image_filename'].apply(
                lambda x: '_'.join(x.split('_')[:-1])
            )
            volume_col = 'volume_id'
        
        unique_volumes = meta_df[volume_col].unique()
        np.random.seed(self.config.seed)
        np.random.shuffle(unique_volumes)
        
        # First split: 80% train, 20% val (original split)
        train_val_split = int(0.8 * len(unique_volumes))
        train_volumes = set(unique_volumes[:train_val_split])
        
        # Get train set images
        train_df = meta_df[meta_df[volume_col].isin(train_volumes)]
        train_images = train_df['image_path'].tolist()
        
        print(f"[SSL] Train set (80%): {len(train_images)} images from {len(train_volumes)} volumes")
        
        # Second split: 50% SSL, 50% supervised (within train set)
        train_volumes_list = list(train_volumes)
        np.random.shuffle(train_volumes_list)
        ssl_split_idx = int(self.config.ssl_train_ratio * len(train_volumes_list))
        
        ssl_volumes = set(train_volumes_list[:ssl_split_idx])
        supervised_volumes = set(train_volumes_list[ssl_split_idx:])
        
        ssl_df = train_df[train_df[volume_col].isin(ssl_volumes)]
        supervised_df = train_df[train_df[volume_col].isin(supervised_volumes)]
        
        ssl_images = ssl_df['image_path'].tolist()
        supervised_images = supervised_df['image_path'].tolist()
        
        print(f"[SSL] SSL split (50%): {len(ssl_images)} images from {len(ssl_volumes)} volumes")
        print(f"[SSL] Supervised split (50%): {len(supervised_images)} images from {len(supervised_volumes)} volumes")
        
        # Create SSL validation subset (10% of SSL split)
        np.random.shuffle(ssl_images)
        val_size = int(self.config.val_subset_ratio * len(ssl_images))
        ssl_val_images = ssl_images[:val_size]
        ssl_train_images = ssl_images[val_size:]
        
        print(f"[SSL] SSL train: {len(ssl_train_images)} images")
        print(f"[SSL] SSL val: {len(ssl_val_images)} images")
        
        # Save splits
        pd.DataFrame({'image_path': ssl_train_images}).to_csv(
            os.path.join(self.config.splits_dir, 'ssl_train.csv'), index=False
        )
        pd.DataFrame({'image_path': ssl_val_images}).to_csv(
            os.path.join(self.config.splits_dir, 'ssl_val.csv'), index=False
        )
        pd.DataFrame({'image_path': supervised_images}).to_csv(
            os.path.join(self.config.splits_dir, 'supervised_train.csv'), index=False
        )
        
        # Save volume mappings for reference
        pd.DataFrame({'volume_id': list(ssl_volumes)}).to_csv(
            os.path.join(self.config.splits_dir, 'ssl_volumes.csv'), index=False
        )
        pd.DataFrame({'volume_id': list(supervised_volumes)}).to_csv(
            os.path.join(self.config.splits_dir, 'supervised_volumes.csv'), index=False
        )
        
        print(f"[SSL] Splits saved to {self.config.splits_dir}")
        
        if self.split == 'train':
            return ssl_train_images
        else:
            return ssl_val_images
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load image and generate mask."""
        image_path = self.image_paths[idx]
        image = self._load_image(image_path)
        image = self.transforms(image, is_train=self.is_train)
        mask = self._generate_mask()
        return image, mask
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and normalize image."""
        image = Image.open(image_path).convert('RGB')
        image = np.array(image, dtype=np.float32) / 255.0
        return image
    
    def _generate_mask(self) -> torch.Tensor:
        """Generate random patch mask."""
        num_patches_per_side = self.config.image_size // self.config.patch_size
        num_patches = num_patches_per_side ** 2
        num_masked = int(num_patches * self.config.mask_ratio)
        
        mask = torch.zeros(num_patches, dtype=torch.float32)
        masked_indices = torch.randperm(num_patches)[:num_masked]
        mask[masked_indices] = 1.0
        
        return mask


def collate_fn(batch):
    """Collate function for batching."""
    images = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])
    return images, masks
