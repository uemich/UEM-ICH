"""
Data augmentation transforms for SSL pretraining

Designed to preserve medical image characteristics while providing variance.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class SSLTransforms:
    """Augmentation pipeline for SSL pretraining"""
    
    def __init__(self, config):
        """
        Initialize SSL augmentation pipeline
        
        Args:
            config: SSLConfig instance
        """
        self.config = config
        
        # Training transforms - stronger for harder pretext task
        self.train_transform = A.Compose([
            # Geometric transforms
            A.HorizontalFlip(p=config.aug_horizontal_flip_prob),
            A.Rotate(
                limit=config.aug_rotation_limit,
                interpolation=cv2.INTER_LANCZOS4,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            
            # Intensity transforms
            A.RandomBrightnessContrast(
                brightness_limit=config.aug_brightness_limit,
                contrast_limit=config.aug_contrast_limit,
                p=config.aug_brightness_contrast_prob
            ),
            
            # Final preprocessing
            A.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                max_pixel_value=1.0
            ),
            ToTensorV2()
        ])
        
        # Validation transforms - minimal preprocessing
        self.val_transform = A.Compose([
            A.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                max_pixel_value=1.0
            ),
            ToTensorV2()
        ])
    
    def __call__(self, image, is_train=True):
        """
        Apply transforms to image
        
        Args:
            image: numpy array (H, W, C) in range [0, 1]
            is_train: whether to use training augmentations
            
        Returns:
            Transformed tensor (C, H, W)
        """
        transform = self.train_transform if is_train else self.val_transform
        transformed = transform(image=image)
        return transformed['image']
