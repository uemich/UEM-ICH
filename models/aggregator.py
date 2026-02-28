"""
Spatial Transformer Aggregator for Scan-Level Classification

Processes spatial features (B, N, C, H, W) across slices with self-attention,
incorporating 3D positional encoding (slice_idx + spatial x + spatial y).
"""
import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    """3D positional encoding for (slice, height, width)."""
    
    def __init__(self, d_model, max_slices=48, spatial_size=12):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_slices, spatial_size, spatial_size, d_model)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        # Slice dimension encoding (use 1/3 of the embedding)
        for pos in range(max_slices):
            pe[pos, :, :, 0::6] = torch.sin(pos * div_term[::3])
            pe[pos, :, :, 1::6] = torch.cos(pos * div_term[::3])
        
        # Height dimension encoding (use 1/3 of the embedding)
        for pos in range(spatial_size):
            pe[:, pos, :, 2::6] = torch.sin(pos * div_term[::3])
            pe[:, pos, :, 3::6] = torch.cos(pos * div_term[::3])
        
        # Width dimension encoding (use 1/3 of the embedding)
        for pos in range(spatial_size):
            pe[:, :, pos, 4::6] = torch.sin(pos * div_term[::3])
            pe[:, :, pos, 5::6] = torch.cos(pos * div_term[::3])
        
        self.register_buffer('pe', pe)
    
    def forward(self, num_slices):
        """Get positional encoding for given number of slices."""
        return self.pe[:num_slices]


class SpatialTransformerAggregator(nn.Module):
    """
    Transformer aggregator for spatial features.
    
    Processes (B, num_slices, C, H, W) spatial features with self-attention
    across all spatial tokens from all slices.
    """
    
    def __init__(
        self,
        feature_dim=768,
        num_heads=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        max_slices=48,
        spatial_size=12,
        use_grad_checkpoint=False
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_slices = max_slices
        self.spatial_size = spatial_size
        self.num_spatial_tokens = spatial_size * spatial_size
        self.use_grad_checkpoint = use_grad_checkpoint
        
        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(
            feature_dim, max_slices, spatial_size
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        if use_grad_checkpoint:
            for layer in self.transformer.layers:
                layer._checkpoint_forward = True
        
        # Output projection
        self.output_dim = feature_dim
        self.norm = nn.LayerNorm(feature_dim)
    
    def forward(self, spatial_features, mask=None, return_spatial=False):
        """
        Forward pass.
        
        Args:
            spatial_features: (B, num_slices, C, H, W)
            mask: (B, num_slices) - 1 for valid slices, 0 for padding
            return_spatial: If True, return (B, num_slices, C, H, W)
                          If False, return (B, C) for classification
        """
        B, N, C, H, W = spatial_features.shape
        
        # Reshape to token sequence: (B, N*H*W, C)
        tokens = spatial_features.permute(0, 1, 3, 4, 2)
        tokens = tokens.reshape(B, N * H * W, C)
        
        # Add positional encoding
        pos_enc = self.pos_encoding(N)
        pos_enc = pos_enc.reshape(N * H * W, C)
        tokens = tokens + pos_enc.unsqueeze(0)
        
        # Create attention mask for padding
        if mask is not None:
            attn_mask = mask.unsqueeze(-1).unsqueeze(-1)
            attn_mask = attn_mask.expand(-1, -1, H, W)
            attn_mask = attn_mask.reshape(B, N * H * W)
            attn_mask = ~attn_mask.bool()
            attn_mask = attn_mask.unsqueeze(1).expand(-1, N * H * W, -1)
        else:
            attn_mask = None
        
        # Apply transformer
        tokens = self.transformer(
            tokens, 
            src_key_padding_mask=attn_mask[:, 0] if attn_mask is not None else None
        )
        
        tokens = self.norm(tokens)
        
        if return_spatial:
            spatial_out = tokens.reshape(B, N, H, W, C)
            spatial_out = spatial_out.permute(0, 1, 4, 2, 3)
            return spatial_out
        else:
            # Global average pooling for classification
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
                mask_expanded = mask_expanded.reshape(B, N * H * W, 1)
                masked_tokens = tokens * mask_expanded
                pooled = masked_tokens.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                pooled = tokens.mean(dim=1)
            
            return pooled
