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
    
    def __init__(self, pretrained: bool = True, freeze_early_layers: bool = False, use_layer4: bool = False):
        super().__init__()
        
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final fully connected layer
        # ResNet50 architecture: conv layers + avgpool + fc
        # We want features before the final fc layer
        self.use_layer4 = use_layer4
        
        if use_layer4:
            # Include layer4. Original resnet children:
            # [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]
            # [:-2] removes avgpool and fc.
            self.features = nn.Sequential(*list(resnet.children())[:-2])
            self.feature_dim = 2048
        else:
            # Stop at layer3.
            # [:-3] removes layer4, avgpool, fc.
            self.features = nn.Sequential(*list(resnet.children())[:-3])
            self.feature_dim = 1024 # ResNet50 layer3 output is 1024
            
        # Note: The original code used [:-1] which includes layer4 AND avgpool (if we consider children list).
        # But get_feature_maps used list(self.features.children())[:-1] which removed the last element.
        # If self.features was [:-1] of resnet, it ended with avgpool.
        # So get_feature_maps removed avgpool, leaving layer4.
        
        # So "use_layer4=True" corresponds to the original behavior of getting layer4 features.
        # But my new implementation of use_layer4=True explicitly creates features WITHOUT avgpool.
        # So get_feature_maps needs to be adjusted too.
        
        # Let's align:
        # If use_layer4=True: self.features contains up to layer4. get_feature_maps should return it as is.
        # If use_layer4=False: self.features contains up to layer3. get_feature_maps should return it as is.
        
        # Wait, get_feature_maps logic I wrote:
        # layers = list(self.features.children())[:-1]
        # This assumes self.features HAS avgpool at the end.
        
        # If I change self.features to NOT have avgpool, I need to change get_feature_maps.
        
        # Let's stick to the cleanest approach:
        # self.features should contain exactly what we want to run.
        
        # But for backward compatibility with 'pretrained' loading, we might need to be careful?
        # No, we are creating a new Sequential from resnet children, so weights are preserved.

        
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
        features = self.features(x)  # [B, C, H, W]
        
        # Global Average Pooling
        features = torch.mean(features, dim=[2, 3])  # [B, C]
        return features
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get spatial feature maps before global avgpool (for Slot Attention)
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            feature_maps: Spatial features [B, 2048, H', W']
        """
        # self.features now contains exactly the layers we want (up to layer3 or layer4)
        # UNLESS we kept the old structure in the else block above?
        
        # In my previous edit I made self.features = ...[:-2] or ...[:-3].
        # So it does NOT contain avgpool.
        
        # So we can just run self.features(x)
        return self.features(x)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ResNetBackbone":
        """Create backbone from config dict"""
        return cls(
            pretrained=config.get('pretrained', True),
            freeze_early_layers=config.get('freeze_early_layers', False),
            use_layer4=config.get('use_layer4', True) # Default to True (original behavior)
        )
