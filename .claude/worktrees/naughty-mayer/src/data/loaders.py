"""DataLoader creation utilities"""

import torch
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Dict, Any
import numpy as np

from .data_classes import CIFAR10Subset, CustomImageFolder
from .transforms import get_train_transforms, get_val_transforms


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloaders(
    data_config: Dict[str, Any],
    augmentation_config: Dict[str, Any],
    dataloader_config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders
    
    Args:
        data_config: Dataset configuration
        augmentation_config: Augmentation configuration
        dataloader_config: DataLoader configuration
    
    Returns:
        train_loader, val_loader
    """
    # Set seed for reproducibility
    if 'seed' in data_config:
        set_seed(data_config['seed'])
    
    # Get transforms
    train_transform = get_train_transforms(augmentation_config)
    val_transform = get_val_transforms(augmentation_config)
    
    # Create full training dataset
    if data_config['name'] == 'custom':
        import os
        full_train_dataset = CustomImageFolder(
            root=os.path.join(data_config['custom_path'], 'train'),
            transform=train_transform,
            classes=data_config.get('classes', None)
        )
    else:
        full_train_dataset = CIFAR10Subset(
            root=data_config['root'],
            train=True,
            transform=train_transform,
            download=data_config.get('download', True),
            classes=data_config.get('classes', None)
        )
    
    
    if data_config['name'] == 'custom':
        # For custom dataset with pre-split folders, we don't use random_split
        # We just use the datasets directly
        train_dataset = full_train_dataset
        
        import os
        val_dataset = CustomImageFolder(
            root=os.path.join(data_config['custom_path'], 'val'),
            transform=val_transform,
            classes=data_config.get('classes', None)
        )
    else:
        # Split into train and validation
        val_split = data_config.get('val_split', 0.2)
        num_val = int(len(full_train_dataset) * val_split)
        num_train = len(full_train_dataset) - num_val
        
        train_dataset, val_dataset_temp = random_split(
            full_train_dataset,
            [num_train, num_val],
            generator=torch.Generator().manual_seed(data_config.get('seed', 42))
        )
        
        val_dataset = CIFAR10Subset(
            root=data_config['root'],
            train=True,
            transform=val_transform,
            download=False,
            classes=data_config.get('classes', None)
        )
        # Use the same indices as val_dataset_temp
        val_dataset.indices = [val_dataset.indices[i] for i in val_dataset_temp.indices]
    
    print(f"\nDataset splits:")
    print(f"  Training: {len(train_dataset)} images")
    print(f"  Validation: {len(val_dataset)} images")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=dataloader_config['batch_size'],
        shuffle=dataloader_config.get('shuffle_train', True),
        num_workers=dataloader_config.get('num_workers', 4),
        pin_memory=dataloader_config.get('pin_memory', True),
        drop_last=dataloader_config.get('drop_last', True)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=dataloader_config['batch_size'],
        shuffle=dataloader_config.get('shuffle_val', False),
        num_workers=dataloader_config.get('num_workers', 4),
        pin_memory=dataloader_config.get('pin_memory', True),
        drop_last=False
    )
    
    return train_loader, val_loader
