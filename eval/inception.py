"""
InceptionV3 Feature Extractor
Extracts 2048-dimensional features for FID/MiFID calculation
"""

import torch
import torch.nn as nn
from torchvision.models import inception_v3


class InceptionV3Features(nn.Module):
    """
    Pretrained InceptionV3 for feature extraction
    - Input: Images (B, 3, H, W) in [0, 1] range
    - Output: Features (B, 2048)
    """
    
    def __init__(self):
        super().__init__()
        
        # Load pretrained InceptionV3
        inception = inception_v3(pretrained=True, transform_input=False)
        inception.eval()
        
        # Remove classification head
        inception.fc = nn.Identity()
        inception.aux_logits = False
        
        self.model = inception
        
        # Freeze weights
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        Extract features
        Args:
            x: Images (B, 3, H, W), range [0, 1]
        Returns:
            features: (B, 2048)
        """
        # Resize to 299x299 (InceptionV3 input size)
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = torch.nn.functional.interpolate(
                x, size=(299, 299), mode='bilinear', align_corners=False
            )
        
        # Normalize to [-1, 1] (InceptionV3 expects this)
        x = 2 * x - 1
        
        # Extract features
        with torch.no_grad():
            features = self.model(x)
        
        return features