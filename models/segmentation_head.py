"""
SegFormer-based Segmentation Head with Coordinate Attention and Deep Supervision

Decodes multi-scale encoder features into segmentation masks.
Used for ICH segmentation on PhysioNet and MBH-Seg datasets.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MixFFN(nn.Module):
    """Mix-FFN with depthwise conv for positional encoding."""
    
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1)
        self.dwconv = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels)
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, 1)
        self.act = nn.GELU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class DepthwiseSeparableFusion(nn.Module):
    """Fusion with depthwise separable conv."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.pwconv = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        
    def forward(self, x):
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class BoundaryRefinement(nn.Module):
    """Boundary refinement with residual connection."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        
    def forward(self, x):
        residual = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)


class CoordinateAttentionSeg(nn.Module):
    """
    Coordinate Attention Module for segmentation decoder.
    
    Preserves spatial position information, critical for precise
    hemorrhage localization.
    """
    
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mid_channels = max(8, in_channels // reduction)
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.SiLU()
        
        self.conv_h = nn.Conv2d(mid_channels, in_channels, 1)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, 1)
    
    def forward(self, x):
        identity = x
        b, c, h, w = x.size()
        
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        return identity * a_h * a_w


class SegFormerHead(nn.Module):
    """
    SegFormer decoder with:
    - Coordinate Attention after each scale projection
    - Deep Supervision (auxiliary heads)
    """
    
    def __init__(
        self,
        in_channels_list: list = [96, 192, 384, 768],
        embed_dim: int = 256,
        num_classes: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.in_channels_list = in_channels_list
        
        # Mix-FFN for each scale
        self.mix_ffn = nn.ModuleList([
            MixFFN(in_ch, in_ch * 2, embed_dim) for in_ch in in_channels_list
        ])
        
        # Coordinate Attention for each scale
        self.coord_attn = nn.ModuleList([
            CoordinateAttentionSeg(embed_dim) for _ in in_channels_list
        ])
        
        # Deep Supervision auxiliary heads
        self.aux_heads = nn.ModuleList([
            nn.Conv2d(embed_dim, num_classes, 1) for _ in in_channels_list
        ])
        
        # Fusion
        self.fusion = DepthwiseSeparableFusion(
            embed_dim * len(in_channels_list), 
            embed_dim
        )
        
        # Boundary refinement
        self.boundary_refine1 = BoundaryRefinement(embed_dim)
        self.boundary_refine2 = BoundaryRefinement(embed_dim)
        
        self.dropout = nn.Dropout2d(dropout)
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, num_classes, 1)
        )
        
    def forward(self, features: list, target_size: tuple = (384, 384)):
        stage1_size = features[0].shape[2:]
        
        projected = []
        aux_outputs = []
        
        for feat, mix_ffn, coord_attn, aux_head in zip(
            features, self.mix_ffn, self.coord_attn, self.aux_heads
        ):
            # Project + Coordinate Attention
            x = mix_ffn(feat)
            x = coord_attn(x)
            
            # Auxiliary output for deep supervision
            aux = aux_head(x)
            aux = F.interpolate(aux, size=target_size, mode='bilinear', align_corners=False)
            aux_outputs.append(aux)
            
            # Upsample to stage1 size
            x = F.interpolate(x, size=stage1_size, mode='bilinear', align_corners=False)
            projected.append(x)
        
        # Concatenate and fuse
        x = torch.cat(projected, dim=1)
        x = self.fusion(x)
        
        # Boundary refinement
        x = self.boundary_refine1(x)
        x = self.dropout(x)
        x = self.boundary_refine2(x)
        
        # Upsample to target size
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        # Segmentation
        main_output = self.seg_head(x)
        
        return main_output, aux_outputs
