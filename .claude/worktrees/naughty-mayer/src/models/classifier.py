"""Classification head for supervised learning"""

import torch
import torch.nn as nn
from typing import Dict, Any


class Classifier(nn.Module):
    """
    Classification head on top of backbone features
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        num_classes: Number of output classes
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dims: list = [512, 256],
        num_classes: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        self.num_classes = num_classes
        
        print(f"Classifier initialized: {input_dim} -> {hidden_dims} -> {num_classes} classes")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            features: Backbone features [B, input_dim]
        
        Returns:
            logits: Class logits [B, num_classes]
        """
        return self.classifier(features)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Classifier":
        """Create classifier from config dict"""
        return cls(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            num_classes=config['num_classes'],
            dropout=config.get('dropout', 0.3)
        )
