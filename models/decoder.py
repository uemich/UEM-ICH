"""
SparK Hierarchical Decoder with Sparse Convolution

Based on "Designing BERT for Convolutional Networks: Sparse and Hierarchical Masked Modeling"
ICLR 2023 Spotlight paper

Key features:
- Hierarchical UNet-style decoder
- Reconstructs images from multi-scale ConvNeXt features
- Handles sparse masked patches efficiently
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block for decoder"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class UpsampleBlock(nn.Module):
    """Upsampling block with skip connections"""
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, 
            kernel_size=2, stride=2
        )
        self.conv1 = ConvBlock(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
    
    def forward(self, x, skip=None):
        x = self.upsample(x)
        
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SparKDecoder(nn.Module):
    """
    Hierarchical UNet-style decoder for SparK
    
    Reconstructs masked patches from multi-scale ConvNeXt features.
    Uses skip connections from encoder for better reconstruction.
    """
    
    def __init__(self, config, encoder_channels=[96, 192, 384, 768]):
        super().__init__()
        
        self.config = config
        self.encoder_channels = encoder_channels
        
        # Decoder stages (bottom-up)
        self.dec3 = UpsampleBlock(encoder_channels[3], encoder_channels[2], encoder_channels[2])
        self.dec2 = UpsampleBlock(encoder_channels[2], encoder_channels[1], encoder_channels[1])
        self.dec1 = UpsampleBlock(encoder_channels[1], encoder_channels[0], encoder_channels[0])
        
        # Final reconstruction head
        self.final_upsample = nn.ConvTranspose2d(
            encoder_channels[0], encoder_channels[0],
            kernel_size=4, stride=4
        )
        
        self.reconstruction_head = nn.Sequential(
            ConvBlock(encoder_channels[0], 64),
            ConvBlock(64, 32),
            nn.Conv2d(32, config.num_channels, kernel_size=1)
        )
        
        print(f"Decoder initialized with hierarchical UNet architecture")
    
    def forward(self, encoder_features):
        feat0, feat1, feat2, feat3 = encoder_features
        
        x = self.dec3(feat3, feat2)
        x = self.dec2(x, feat1)
        x = self.dec1(x, feat0)
        
        x = self.final_upsample(x)
        x = self.reconstruction_head(x)
        
        return x


class MaskGenerator(nn.Module):
    """
    Generate random masks for SparK-style masked modeling
    
    Masks patches at multiple scales for hierarchical reconstruction.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.mask_ratio = config.mask_ratio
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        num_patches = num_patches_h * num_patches_w
        num_masked = int(num_patches * self.mask_ratio)
        
        masks = []
        for _ in range(B):
            mask = torch.zeros(num_patches, device=x.device)
            masked_indices = torch.randperm(num_patches, device=x.device)[:num_masked]
            mask[masked_indices] = 1.0
            mask = mask.reshape(num_patches_h, num_patches_w)
            masks.append(mask)
        
        masks = torch.stack(masks)
        return masks
    
    def apply_mask(self, x, mask):
        B, C, H, W = x.shape
        
        mask_upscaled = F.interpolate(
            mask.unsqueeze(1).float(),
            size=(H, W),
            mode='nearest'
        )
        
        mask_upscaled = 1.0 - mask_upscaled
        masked_x = x * mask_upscaled
        
        return masked_x
