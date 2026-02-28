"""
ConvNeXt v2 Tiny Encoder with Coordinate Attention

Backbone encoder for SSL pretraining and downstream supervised tasks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention module
    
    Factorizes global pooling into H and W vectors to preserve spatial information.
    Critical for distinguishing hemorrhage subtypes by location.
    
    Paper: "Coordinate Attention for Efficient Mobile Network Design"
    """
    
    def __init__(self, inp, oup, reduction=32):
        """
        Args:
            inp: input channels
            oup: output channels
            reduction: channel reduction ratio
        """
        super().__init__()
        
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, inp // reduction)
        
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            Attention-weighted features (B, C, H, W)
        """
        identity = x
        
        b, c, h, w = x.size()
        
        # Factorized pooling
        x_h = self.pool_h(x)  # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, W, 1)
        
        # Concatenate and process
        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # Split back
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        # Generate attention weights
        a_h = self.conv_h(x_h).sigmoid()  # (B, C, H, 1)
        a_w = self.conv_w(x_w).sigmoid()  # (B, C, 1, W)
        
        # Apply attention
        out = identity * a_h * a_w
        
        return out


class ConvNeXtV2Encoder(nn.Module):
    """
    ConvNeXt v2 Tiny encoder with Coordinate Attention
    
    Serves as the feature extractor for SSL pretraining.
    In supervised training, this is frozen or fine-tuned.
    """
    
    def __init__(self, config):
        """
        Args:
            config: SSLConfig instance (or any object with encoder_name,
                    encoder_pretrained, num_channels, use_coordinate_attention,
                    ca_stages, encoder_out_channels)
        """
        super().__init__()
        
        self.config = config
        
        # Load pretrained ConvNeXt v2 Tiny
        self.backbone = timm.create_model(
            config.encoder_name,
            pretrained=config.encoder_pretrained,
            features_only=True,
            in_chans=config.num_channels
        )
        
        # Get feature dimensions for each stage
        self.feature_info = self.backbone.feature_info
        self.out_channels = [info['num_chs'] for info in self.feature_info]
        
        # Add Coordinate Attention after specified stages
        self.ca_blocks = nn.ModuleDict()
        if config.use_coordinate_attention:
            for stage_idx in config.ca_stages:
                channels = self.out_channels[stage_idx]
                self.ca_blocks[f'ca_{stage_idx}'] = CoordinateAttention(
                    channels, channels, reduction=32
                )
        
        print(f"Encoder initialized with {sum(p.numel() for p in self.parameters())/1e6:.2f}M parameters")
        print(f"Feature channels per stage: {self.out_channels}")
    
    def forward(self, x):
        """
        Forward pass through encoder
        
        Args:
            x: Input images (B, 3, H, W)
            
        Returns:
            List of feature maps from all stages
        """
        # Extract features from all stages
        features = self.backbone(x)
        
        # Apply Coordinate Attention to selected stages
        enhanced_features = []
        for idx, feat in enumerate(features):
            if f'ca_{idx}' in self.ca_blocks:
                feat = self.ca_blocks[f'ca_{idx}'](feat)
            enhanced_features.append(feat)
        
        return enhanced_features
    
    def get_final_features(self, x):
        """
        Get only the final stage features (for downstream tasks)
        
        Args:
            x: Input images (B, 3, H, W)
            
        Returns:
            Final feature map (B, C, H', W')
        """
        features = self.forward(x)
        return features[-1]  # Last stage
