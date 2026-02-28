"""
Loss functions for SSL pretraining

Combines:
1. MSE loss for pixel-level reconstruction
2. GLCM loss for texture preservation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSLLoss(nn.Module):
    """
    Combined loss for GLCM-MAE
    
    L_total = L_MSE + λ * L_GLCM
    """
    
    def __init__(self, config, glcm_extractor):
        super().__init__()
        
        self.config = config
        self.glcm_extractor = glcm_extractor
        
        self.mse_weight = config.mse_weight
        self.glcm_weight = config.glcm_weight
        
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred, target, mask=None):
        # MSE loss
        if mask is not None:
            mse_loss = self._masked_mse_loss(pred, target, mask)
        else:
            mse_loss = self.mse_loss(pred, target)
        
        # GLCM texture loss
        glcm_loss = self._glcm_texture_loss(pred, target)
        
        # Combined loss
        total_loss = self.mse_weight * mse_loss + self.glcm_weight * glcm_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'glcm_loss': glcm_loss.item(),
        }
        
        return total_loss, loss_dict
    
    def _masked_mse_loss(self, pred, target, mask):
        B, C, H, W = pred.shape
        
        mask_upscaled = F.interpolate(
            mask.unsqueeze(1).float(),
            size=(H, W),
            mode='nearest'
        )
        
        squared_diff = (pred - target) ** 2
        masked_squared_diff = squared_diff * mask_upscaled
        
        num_masked_pixels = mask_upscaled.sum() * C
        mse_loss = masked_squared_diff.sum() / (num_masked_pixels + 1e-8)
        
        return mse_loss
    
    def _glcm_texture_loss(self, pred, target):
        pred_features = self.glcm_extractor(pred)
        target_features = self.glcm_extractor(target)
        glcm_loss = self.glcm_extractor.compute_similarity(pred_features, target_features)
        return glcm_loss


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in classification."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight
        
        focal_loss = focal_weight * bce_loss
        return focal_loss.mean()


class SegmentationDiceLoss(nn.Module):
    """
    Multi-class Dice Loss for segmentation.
    
    Computes Dice coefficient per class (excluding background) and returns 1 - mean_dice.
    """
    
    def __init__(self, num_classes=6, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, pred, target):
        # pred: (B, C, H, W), target: (B, H, W)
        pred = F.softmax(pred, dim=1)
        
        # OHE target
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        dice_per_class = []
        # Skip background class (index 0)
        for c in range(1, self.num_classes):
            pred_c = pred[:, c].contiguous().view(-1)
            target_c = target_onehot[:, c].contiguous().view(-1)
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_per_class.append(dice)
        
        if not dice_per_class:
            return torch.tensor(0.0, device=pred.device)
            
        return 1 - torch.stack(dice_per_class).mean()


class SegmentationFocalLoss(nn.Module):
    """
    Multi-class Focal Loss for segmentation.
    
    Handles class imbalance by down-weighting easy examples.
    """
    
    def __init__(self, num_classes=6, alpha=0.75, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        # pred: (B, C, H, W), target: (B, H, W)
        probs = F.softmax(pred, dim=1)
        target_flat = target.view(-1)
        
        # Flatten probs: (B*H*W, C)
        probs_flat = probs.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
        
        # Get probability of the foreground class
        pt = probs_flat.gather(1, target_flat.unsqueeze(1)).squeeze(1)
        
        focal_weight = (1 - pt) ** self.gamma
        
        # Cross entropy loss
        ce_loss = F.cross_entropy(pred, target, reduction='none').view(-1)
        
        return (self.alpha * focal_weight * ce_loss).mean()


class CombinedSegLoss(nn.Module):
    """
    Combined Segmentation Loss (Focal + Dice) with Deep Supervision support.
    
    Loss = λ_f * L_focal + λ_d * L_dice
    """
    
    def __init__(self, num_classes=6, focal_weight=0.2, dice_weight=0.8):
        super().__init__()
        self.focal_loss = SegmentationFocalLoss(num_classes)
        self.dice_loss = SegmentationDiceLoss(num_classes)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, main_out, aux_outs, target):
        """
        Args:
            main_out: Primary head output (B, C, H, W)
            aux_outs: List of auxiliary head outputs (for deep supervision)
            target: Ground truth masks (B, H, W)
        """
        if target.dim() == 4:
            target = target.squeeze(1)
            
        focal = self.focal_loss(main_out, target)
        dice = self.dice_loss(main_out, target)
        main_loss = self.focal_weight * focal + self.dice_weight * dice
        
        # Auxiliary losses for deep supervision
        aux_weights = [0.4, 0.3, 0.2, 0.1]
        for i, aux in enumerate(aux_outs):
            if i < len(aux_weights):
                a_focal = self.focal_loss(aux, target)
                a_dice = self.dice_loss(aux, target)
                main_loss += aux_weights[i] * (self.focal_weight * a_focal + self.dice_weight * a_dice)
                
        return main_loss, focal, dice
