"""
FCOS-style Bounding Box Detection Head

Built on top of the SegFormer neck (shared with segmentation decoder).
Uses the same MixFFN → CoordinateAttention → Fusion → BoundaryRefinement
pipeline, with FCOS detection heads (classification, regression, centerness).

Architecture:
    Encoder features → SegFormer Neck → 3 parallel FCOS heads:
        - Classification: [B, num_classes, H, W] per-pixel class scores
        - Regression:     [B, 4, H, W] FCOS bbox offsets (l, t, r, b)
        - Centerness:     [B, 1, H, W] center quality score
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .segmentation_head import (
    MixFFN, DepthwiseSeparableFusion, BoundaryRefinement, CoordinateAttentionSeg
)


class FCOSHead(nn.Module):
    """
    FCOS-style detection head.

    Three parallel sub-heads on top of dense feature maps:
        - cls_head: per-pixel classification [B, num_classes, H, W]
        - reg_head: per-pixel bbox regression [B, 4, H, W] (l, t, r, b)
        - ctr_head: per-pixel centerness [B, 1, H, W]
    """

    def __init__(self, in_channels=256, num_classes=6, num_convs=4):
        super().__init__()

        # Shared conv tower for classification
        cls_tower = []
        for _ in range(num_convs):
            cls_tower.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU(inplace=True))
        self.cls_tower = nn.Sequential(*cls_tower)

        # Shared conv tower for regression
        reg_tower = []
        for _ in range(num_convs):
            reg_tower.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
            reg_tower.append(nn.GroupNorm(32, in_channels))
            reg_tower.append(nn.ReLU(inplace=True))
        self.reg_tower = nn.Sequential(*reg_tower)

        # Final prediction layers
        self.cls_logits = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in [self.cls_logits, self.bbox_pred, self.centerness]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
        # Initialize cls bias for focal loss (prior probability)
        prior_prob = 0.01
        nn.init.constant_(self.cls_logits.bias, -np.log((1 - prior_prob) / prior_prob))

    def forward(self, features):
        """
        Args:
            features: [B, C, H, W] dense feature map from neck
        Returns:
            cls_logits: [B, num_classes, H, W]
            bbox_pred:  [B, 4, H, W] (l, t, r, b) — always positive via exp
            centerness: [B, 1, H, W]
        """
        cls_feat = self.cls_tower(features)
        reg_feat = self.reg_tower(features)

        cls_logits = self.cls_logits(cls_feat)
        bbox_pred = torch.exp(self.bbox_pred(reg_feat))
        centerness = self.centerness(cls_feat)

        return cls_logits, bbox_pred, centerness


class BBoxDetectionModel(nn.Module):
    """
    Full detection model: Encoder → SegFormer Neck → FCOS Head

    The neck uses the same architecture as the segmentation decoder
    (MixFFN → CoordinateAttention → Fusion → BoundaryRefinement),
    allowing shared pretrained weights.
    """

    def __init__(self, encoder, in_channels_list=[96, 192, 384, 768],
                 embed_dim=256, num_classes=6, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder

        if freeze_encoder:
            for param in encoder.parameters():
                param.requires_grad = False

        # SegFormer Neck (shared architecture with segmentation)
        self.mix_ffn = nn.ModuleList([
            MixFFN(in_ch, in_ch * 2, embed_dim) for in_ch in in_channels_list
        ])
        self.coord_attn = nn.ModuleList([
            CoordinateAttentionSeg(embed_dim) for _ in in_channels_list
        ])
        self.fusion = DepthwiseSeparableFusion(
            embed_dim * len(in_channels_list), embed_dim
        )
        self.boundary_refine1 = BoundaryRefinement(embed_dim)
        self.boundary_refine2 = BoundaryRefinement(embed_dim)
        self.dropout = nn.Dropout2d(0.1)

        # FCOS Detection Head
        self.fcos_head = FCOSHead(
            in_channels=embed_dim, num_classes=num_classes
        )

    def forward_neck(self, features):
        """Shared neck producing dense feature map [B, embed_dim, H, W]."""
        stage1_size = features[0].shape[-2:]

        projected = []
        for feat, mix_ffn, coord_attn in zip(
            features, self.mix_ffn, self.coord_attn
        ):
            x = mix_ffn(feat)
            x = coord_attn(x)
            x = F.interpolate(x, size=stage1_size, mode='bilinear', align_corners=False)
            projected.append(x)

        x = self.fusion(torch.cat(projected, dim=1))
        x = self.boundary_refine1(x)
        x = self.dropout(x)
        x = self.boundary_refine2(x)
        return x  # [B, embed_dim, H/4, W/4]

    def forward(self, images):
        """
        Args:
            images: [B, 3, 384, 384]
        Returns:
            cls_logits, bbox_pred, centerness
        """
        if not any(p.requires_grad for p in self.encoder.parameters()):
            with torch.no_grad():
                features = self.encoder(images)
        else:
            features = self.encoder(images)

        neck_features = self.forward_neck(features)
        cls_logits, bbox_pred, centerness = self.fcos_head(neck_features)

        return cls_logits, bbox_pred, centerness

    def load_neck_weights(self, seg_head_state_dict):
        """
        Load pre-trained SegFormer neck weights from a segmentation checkpoint.

        Maps checkpoint keys to model keys. Skips aux_heads and seg_head
        (output layers) since we use FCOS heads instead.
        """
        if seg_head_state_dict is None:
            print("  No neck weights to load.")
            return

        model_sd = self.state_dict()
        loaded = 0
        skipped = []

        for ckpt_key, ckpt_val in seg_head_state_dict.items():
            # Skip segmentation-specific output layers
            if ckpt_key.startswith('aux_heads') or ckpt_key.startswith('seg_head'):
                skipped.append(ckpt_key)
                continue

            if ckpt_key in model_sd and ckpt_val.shape == model_sd[ckpt_key].shape:
                model_sd[ckpt_key] = ckpt_val
                loaded += 1
            else:
                skipped.append(ckpt_key)

        self.load_state_dict(model_sd)
        print(f"  Neck: loaded {loaded} keys, skipped {len(skipped)} (output layers / mismatched).")
