"""Utility functions and helpers"""

import yaml
import torch
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        # For M-series Macs
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_name: str = "auto") -> torch.device:
    """
    Get torch device
    
    Args:
        device_name: "auto", "cuda", "mps", or "cpu"
    
    Returns:
        torch.device
    """
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Metal Performance Shaders) device")
        else:
            device = torch.device("cpu")
            print("Using CPU device")
    else:
        device = torch.device(device_name)
        print(f"Using {device_name} device")
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
    additional_info: Dict[str, Any] = None
):
    """Save model checkpoint"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: torch.device = None
) -> Dict[str, Any]:
    """Load model checkpoint"""
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found at {checkpoint_path}. Starting from scratch.")
        return None
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Handle potential shape mismatches (e.g. pos_grid when changing resolution)
    model_state = model.state_dict()
    keys_to_remove = []
    
    for k, v in state_dict.items():
        if k in model_state:
            if v.shape != model_state[k].shape:
                print(f"Shape mismatch for {k}: checkpoint {v.shape} != model {model_state[k].shape}. Ignoring.")
                keys_to_remove.append(k)
                
    for k in keys_to_remove:
        del state_dict[k]
        
    model.load_state_dict(state_dict, strict=False)
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    
    return checkpoint
