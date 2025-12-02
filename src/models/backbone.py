"""Shared backbone architecture for feature extraction"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Any


class ResNetBackbone(nn.Module):
    """
    ResNet50 backbone for shared feature extraction
    
    Args:
        pretrained: Whether to use pretrained weights
        freeze_early_layers: Whether to freeze early convolutional layers
    """
    
    def __init__(self, pretrained: bool = True, freeze_early_layers: bool = False):
        super().__init__()
        
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final fully connected layer
        # ResNet50 architecture: conv layers + avgpool + fc
        # We want features before the final fc layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Feature dimension (ResNet50 outputs 2048-dim features)
        self.feature_dim = 2048
        
        # Optionally freeze early layers for faster training
        if freeze_early_layers:
            # Freeze first 6 layers (up to layer2)
            for name, param in self.features.named_parameters():
                if 'layer3' not in name and 'layer4' not in name:
                    param.requires_grad = False
            print("Frozen early layers (conv1, bn1, layer1, layer2)")
        
        print(f"ResNetBackbone initialized with {'pretrained' if pretrained else 'random'} weights")
        print(f"Output feature dimension: {self.feature_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            features: Feature maps [B, 2048, 1, 1] → squeezed to [B, 2048]
        """
        features = self.features(x)  # [B, 2048, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [B, 2048]
        return features
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get spatial feature maps before global avgpool (for Slot Attention)
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            feature_maps: Spatial features [B, 2048, H', W']
        """
        # Execute all layers except the final avgpool
        for module in list(self.features.children())[:-1]:
            x = module(x)
        return x

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ResNetBackbone":
        """Create backbone from config dict"""
        return cls(
            pretrained=config.get('pretrained', True),
            freeze_early_layers=config.get('freeze_early_layers', False)
        )
