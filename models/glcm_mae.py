"""
Complete GLCM-MAE Model

Combines:
- ConvNeXt v2 Encoder with Coordinate Attention
- SparK Hierarchical Decoder
- GLCM Texture-aware Loss
"""
import torch
import torch.nn as nn

from .encoder import ConvNeXtV2Encoder
from .decoder import SparKDecoder, MaskGenerator
from .glcm import FastGLCMFeatureExtractor
from .losses import SSLLoss


class GLCM_MAE(nn.Module):
    """
    Complete GLCM-MAE model for self-supervised pretraining
    
    Architecture:
    1. Mask random patches in input image
    2. Encode visible patches with ConvNeXt v2 + Coordinate Attention
    3. Decode with hierarchical UNet to reconstruct original image
    4. Loss = MSE(reconstruction, original) + λ * GLCM(reconstruction, original)
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Encoder: ConvNeXt v2 Tiny with Coordinate Attention
        self.encoder = ConvNeXtV2Encoder(config)
        
        # Decoder: Hierarchical UNet
        self.decoder = SparKDecoder(config, encoder_channels=self.encoder.out_channels)
        
        # Mask generator
        self.mask_generator = MaskGenerator(config)
        
        # GLCM feature extractor for loss
        self.glcm_extractor = FastGLCMFeatureExtractor(config)
        
        # Loss function
        self.criterion = SSLLoss(config, self.glcm_extractor)
        
        # Count parameters
        self.encoder_params = sum(p.numel() for p in self.encoder.parameters()) / 1e6
        self.decoder_params = sum(p.numel() for p in self.decoder.parameters()) / 1e6
        self.total_params = self.encoder_params + self.decoder_params
        
        print(f"\nGLCM-MAE Model Summary:")
        print(f"  Encoder: {self.encoder_params:.2f}M parameters")
        print(f"  Decoder: {self.decoder_params:.2f}M parameters")
        print(f"  Total:   {self.total_params:.2f}M parameters")
    
    def forward(self, images, return_reconstruction=False):
        # Generate random mask
        mask = self.mask_generator(images)
        
        # Encode
        encoder_features = self.encoder(images)
        
        # Decode
        reconstruction = self.decoder(encoder_features)
        
        # Compute loss (only on masked patches)
        loss, loss_dict = self.criterion(reconstruction, images, mask)
        
        if return_reconstruction:
            return loss, loss_dict, reconstruction, mask
        else:
            return loss, loss_dict
    
    def get_encoder(self):
        """Get the encoder for downstream tasks."""
        return self.encoder
    
    def save_encoder(self, path):
        """Save only the encoder weights for downstream tasks."""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'config': self.config,
        }, path)
        print(f"Encoder saved to {path}")
    
    @torch.no_grad()
    def visualize(self, images):
        """Generate visualization of masked input and reconstruction."""
        self.eval()
        
        mask = self.mask_generator(images)
        masked_images = self.mask_generator.apply_mask(images, mask)
        
        encoder_features = self.encoder(images)
        reconstruction = self.decoder(encoder_features)
        
        return {
            'original': images,
            'masked': masked_images,
            'reconstruction': reconstruction,
            'mask': mask
        }
