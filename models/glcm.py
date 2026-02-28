"""
GLCM (Gray Level Co-occurrence Matrix) Feature Extractor

Computes texture features from images to prevent smoothing during reconstruction.
This is the key innovation of GLCM-MAE over standard MAE.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.feature import graycomatrix, graycoprops


class GLCMFeatureExtractor(nn.Module):
    """
    Differentiable GLCM texture feature extractor
    
    Computes Gray Level Co-occurrence Matrix features (contrast, entropy, etc.)
    to ensure the model learns texture-rich representations.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.distances = config.glcm_distances
        self.angles = [np.deg2rad(a) for a in config.glcm_angles]
        self.levels = config.glcm_levels
        self.features = config.glcm_features
    
    def forward(self, images):
        batch_size = images.shape[0]
        device = images.device
        
        images_np = self._prepare_images(images)
        
        feature_dict = {feat: [] for feat in self.features}
        
        for i in range(batch_size):
            img = images_np[i]
            glcm = self._compute_glcm(img)
            
            for feat_name in self.features:
                feat_val = self._extract_feature(glcm, feat_name)
                feature_dict[feat_name].append(feat_val)
        
        for feat_name in self.features:
            feature_dict[feat_name] = torch.tensor(
                feature_dict[feat_name],
                dtype=torch.float32,
                device=device
            )
        
        return feature_dict
    
    def _prepare_images(self, images):
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)
        
        images = (images + 1.0) / 2.0
        images = (images * (self.levels - 1)).clamp(0, self.levels - 1)
        images_np = images.squeeze(1).cpu().numpy().astype(np.uint8)
        
        return images_np
    
    def _compute_glcm(self, image):
        glcm = graycomatrix(
            image,
            distances=self.distances,
            angles=self.angles,
            levels=self.levels,
            symmetric=True,
            normed=True
        )
        return glcm
    
    def _extract_feature(self, glcm, feature_name):
        feature_values = graycoprops(glcm, feature_name)
        return feature_values.mean()
    
    def compute_similarity(self, features1, features2):
        total_loss = 0.0
        for feat_name in self.features:
            feat1 = features1[feat_name]
            feat2 = features2[feat_name]
            loss = F.mse_loss(feat1, feat2)
            total_loss += loss
        return total_loss / len(self.features)


class FastGLCMFeatureExtractor(nn.Module):
    """
    Fast GPU-friendly approximation of GLCM features using differentiable operations.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def forward(self, images):
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)
        
        batch_size = images.shape[0]
        
        features = {}
        features['contrast'] = images.view(batch_size, -1).var(dim=1)
        features['energy'] = (images ** 2).view(batch_size, -1).mean(dim=1)
        
        local_var = self._local_variance(images)
        features['homogeneity'] = 1.0 / (1.0 + local_var)
        features['correlation'] = self._spatial_autocorrelation(images)
        
        return features
    
    def _local_variance(self, images):
        avg_pool = F.avg_pool2d(images, kernel_size=3, stride=1, padding=1)
        sq_pool = F.avg_pool2d(images ** 2, kernel_size=3, stride=1, padding=1)
        local_var = sq_pool - avg_pool ** 2
        return local_var.view(images.shape[0], -1).mean(dim=1)
    
    def _spatial_autocorrelation(self, images):
        shifted = F.pad(images[:, :, :-1, :], (0, 0, 1, 0))
        correlation = (images * shifted).view(images.shape[0], -1).mean(dim=1)
        return correlation
    
    def compute_similarity(self, features1, features2):
        total_loss = 0.0
        for key in features1:
            total_loss += F.mse_loss(features1[key], features2[key])
        return total_loss / len(features1)
