"""
SSL Configuration - RSNA Only Pretraining

Uses 50% of RSNA train data for SSL pretraining.
The other 50% is reserved for supervised training.
"""
import os
from dataclasses import dataclass, field
from typing import List

# Base directory: UEM-ICH repo root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


@dataclass
class SSLConfig:
    """Configuration for SSL pretraining - RSNA only"""
    
    # ===== Data Configuration =====
    rsna_images: str = os.path.join(BASE_DIR, 'preprocessed_data', 'rsna', 'images')
    rsna_metadata: str = os.path.join(BASE_DIR, 'preprocessed_data', 'rsna', 'metadata', 'slice_metadata.csv')
    rsna_labels: str = os.path.join(BASE_DIR, 'preprocessed_data', 'rsna', 'labels', 'slice_labels.csv')
    
    # Output directories
    output_dir: str = os.path.join(BASE_DIR, 'weights', 'ssl')
    log_dir: str = os.path.join(BASE_DIR, 'training', 'ssl', 'logs')
    splits_dir: str = os.path.join(BASE_DIR, 'training', 'ssl', 'splits')
    
    # ===== Data Split Configuration =====
    # Split the original train set (80% of total) into:
    # - 50% for SSL pretraining (train_ssl)
    # - 50% for supervised training later (train_supervised)
    ssl_train_ratio: float = 0.5  # 50% of train set for SSL
    val_subset_ratio: float = 0.1  # Use 10% of SSL split for validation
    
    # ===== Image Configuration =====
    image_size: int = 384
    num_channels: int = 3  # Causal windowing: brain, subdural, bone
    patch_size: int = 16  # 384/16 = 24 patches per side
    mask_ratio: float = 0.75  # 75% masking (harder task)
    
    # ===== Model Configuration =====
    encoder_name: str = "convnextv2_tiny"
    encoder_pretrained: bool = True  # Start from ImageNet
    encoder_out_channels: List[int] = field(default_factory=lambda: [96, 192, 384, 768])
    use_coordinate_attention: bool = True
    ca_stages: List[int] = field(default_factory=lambda: [2, 3])
    
    # Decoder
    decoder_dim: int = 512
    decoder_depth: int = 3
    decoder_num_heads: int = 8
    decoder_mlp_ratio: float = 4.0
    
    # ===== GLCM Configuration =====
    glcm_distances: List[int] = field(default_factory=lambda: [1, 2, 4])  # Multi-scale
    glcm_angles: List[float] = field(default_factory=lambda: [0, 45, 90, 135])
    glcm_levels: int = 256
    glcm_features: List[str] = field(default_factory=lambda: [
        'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation'
    ])
    
    # ===== Training Configuration =====
    num_epochs: int = 100
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    
    # Learning rate
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Loss weights
    mse_weight: float = 1.0
    glcm_weight: float = 0.1
    
    # Mixed precision and gradient accumulation
    use_amp: bool = True
    gradient_accumulation_steps: int = 16  # Effective batch = 256
    max_grad_norm: float = 1.0
    
    # ===== Augmentation =====
    aug_rotation_limit: int = 30
    aug_horizontal_flip_prob: float = 0.5
    aug_brightness_contrast_prob: float = 0.5
    aug_brightness_limit: float = 0.2
    aug_contrast_limit: float = 0.2
    aug_cutout_prob: float = 0.3
    aug_cutout_num_holes: int = 4
    aug_cutout_max_size: int = 48
    
    # ===== Logging =====
    log_freq: int = 100
    vis_freq: int = 500
    save_freq: int = 10
    log_precision: int = 6
    
    # ===== Device =====
    device: str = "cuda"
    seed: int = 42
    
    # ===== Debugging =====
    debug_mode: bool = False
    debug_samples: int = 100
    
    def __post_init__(self):
        """Create directories"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.splits_dir, exist_ok=True)
    
    def get_effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps
