"""Test data loading and visualize samples"""

import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.loaders import create_dataloaders
from src.data.transforms import get_inverse_normalize
from src.utils import load_config, set_seed


def denormalize_image(img_tensor):
    """Convert normalized tensor to displayable image"""
    inv_normalize = get_inverse_normalize()
    img = inv_normalize(img_tensor)
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).cpu().numpy()


def main():
    # Load config
    data_config = load_config('configs/data_config.yaml')
    
    # Set seed
    set_seed(data_config.get('seed', 42))
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_config['dataset'],
        data_config['augmentation'],
        data_config['dataloader']
    )
    
    # Get class names
    class_names = data_config['dataset']['classes']
    
    # Sample batch from train loader
    images, labels = next(iter(train_loader))
    
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Visualize samples
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle('Sample CIFAR-10 Images (4 Classes)', fontsize=16)
    
    for idx in range(32):
        row = idx // 8
        col = idx % 8
        
        img = denormalize_image(images[idx])
        label = labels[idx].item()
        
        axes[row, col].imshow(img)
        axes[row, col].set_title(class_names[label], fontsize=10)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.tight_layout()
    # plt.savefig('data_samples.png', dpi=150, bbox_inches='tight')
    # print("\nSaved visualization to: data_samples.png")
    print("Skipping save to data_samples.png (Notebook mode)")
    plt.show()
    
    # Per-class distribution
    all_labels = []
    for _, labels_batch in train_loader:
        all_labels.extend(labels_batch.tolist())
    
    print("\nClass distribution in training set:")
    for class_idx, class_name in enumerate(class_names):
        count = all_labels.count(class_idx)
        print(f"  {class_name}: {count} images ({100*count/len(all_labels):.1f}%)")
    
    print("\n✓ Data loading test completed successfully!")


if __name__ == '__main__':
    main()
