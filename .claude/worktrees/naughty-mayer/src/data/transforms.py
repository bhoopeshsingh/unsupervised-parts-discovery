"""Data augmentation and preprocessing transforms"""

from torchvision import transforms
from typing import Dict, Any


def get_train_transforms(config: Dict[str, Any]) -> transforms.Compose:
    """
    Get training data transformations
    
    Args:
        config: Augmentation config from data_config.yaml
    
    Returns:
        Composed transforms for training
    """
    aug_config = config.get('train', {})
    
    transform_list = []
    
    # Random crop with padding
    if aug_config.get('random_crop', False):
        transform_list.append(
            transforms.RandomCrop(
                aug_config['crop_size'],
                padding=aug_config.get('padding', 4)
            )
        )
    
    # Random horizontal flip
    if aug_config.get('horizontal_flip', False):
        transform_list.append(
            transforms.RandomHorizontalFlip(p=aug_config.get('flip_prob', 0.5))
        )
    
    # Color jitter
    if 'color_jitter' in aug_config:
        jitter = aug_config['color_jitter']
        transform_list.append(
            transforms.ColorJitter(
                brightness=jitter.get('brightness', 0),
                contrast=jitter.get('contrast', 0),
                saturation=jitter.get('saturation', 0),
                hue=jitter.get('hue', 0)
            )
        )
    
    # Convert to tensor and normalize
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 mean
            std=[0.2470, 0.2435, 0.2616]    # CIFAR-10 std
        )
    ])
    
    return transforms.Compose(transform_list)


def get_val_transforms(config: Dict[str, Any]) -> transforms.Compose:
    """
    Get validation/test data transformations
    
    Args:
        config: Augmentation config from data_config.yaml
    
    Returns:
        Composed transforms for validation
    """
    aug_config = config.get('val', {})
    
    transform_list = []
    
    # Resize if specified (usually not needed for CIFAR-10)
    if 'resize' in aug_config:
        transform_list.append(transforms.Resize(aug_config['resize']))
    
    # Convert to tensor and normalize
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    
    return transforms.Compose(transform_list)


def get_inverse_normalize():
    """Get inverse normalization for visualization"""
    return transforms.Normalize(
        mean=[-0.4914/0.2470, -0.4822/0.2435, -0.4465/0.2616],
        std=[1/0.2470, 1/0.2435, 1/0.2616]
    )
