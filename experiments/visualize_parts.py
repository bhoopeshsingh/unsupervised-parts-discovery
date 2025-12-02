"""
Visualize part consistency for a single class.
Loads a trained model and visualizes attention masks for a batch of images from a specific class.
"""

import sys
sys.path.append('.')

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm

from src.data.loaders import create_dataloaders
from src.models.backbone import ResNetBackbone
from src.models.part_discovery.slot_attention import SlotAttentionModel
from src.utils import load_config, get_device, set_seed, load_checkpoint

def visualize_class_consistency(
    backbone,
    slot_model,
    dataloader,
    device,
    target_class_idx,
    class_name,
    num_samples=20,
    save_dir='parts/consistency_check'
):
    """
    Visualize attention masks for multiple images of the same class to check consistency.
    """
    backbone.eval()
    slot_model.eval()
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Collecting {num_samples} samples for class '{class_name}' (index {target_class_idx})...")
    
    collected_images = []
    collected_recons = []
    collected_masks = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            # Filter for target class
            mask = (labels == target_class_idx)
            if not mask.any():
                continue
                
            class_images = images[mask]
            
            # Process batch
            class_images = class_images.to(device)
            features = backbone.get_feature_maps(class_images)
            recon, masks, _, _ = slot_model(features)
            
            # Store samples
            for img, rec, msk in zip(class_images.cpu(), recon.cpu(), masks.cpu()):
                collected_images.append(img)
                collected_recons.append(rec)
                collected_masks.append(msk)
                
                if len(collected_images) >= num_samples:
                    break
            
            if len(collected_images) >= num_samples:
                break
    
    if len(collected_images) < num_samples:
        print(f"Warning: Only found {len(collected_images)} samples for class {class_name}")
        num_samples = len(collected_images)
    
    # Calculate global stats
    all_masks_tensor = torch.stack(collected_masks)
    print(f"\nMask Statistics:")
    print(f"  Min: {all_masks_tensor.min():.4f}")
    print(f"  Max: {all_masks_tensor.max():.4f}")
    print(f"  Mean: {all_masks_tensor.mean():.4f}")
    print(f"  Std: {all_masks_tensor.std():.4f}")
    
    # Create grid visualization
    # Rows: Images, Cols: Original + Reconstruction + Slots
    num_slots = collected_masks[0].shape[0]
    cols = num_slots + 2
    rows = num_samples
    
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    
    # Get reconstructions
    with torch.no_grad():
        # We need to run forward pass again to get reconstruction
        # This is inefficient but fine for visualization script
        pass

    for idx in range(num_samples):
        # Original Image
        img = collected_images[idx].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_ylabel(f'Sample {idx+1}', fontsize=12)
        if idx == 0:
            axes[idx, 0].set_title("Original", fontsize=14)
        axes[idx, 0].set_xticks([])
        axes[idx, 0].set_yticks([])

        # Reconstruction
        rec = collected_recons[idx].permute(1, 2, 0).numpy()
        rec = (rec - rec.min()) / (rec.max() - rec.min())
        
        axes[idx, 1].imshow(rec)
        if idx == 0:
            axes[idx, 1].set_title("Reconstruction", fontsize=14)
        axes[idx, 1].axis('off')
        
        # Slot Masks
        masks = collected_masks[idx].numpy() # [num_slots, H, W]
        
        for slot_i in range(num_slots):
            mask = masks[slot_i]
            
            # Normalize for visualization to see structure
            mask_min, mask_max = mask.min(), mask.max()
            if mask_max - mask_min > 1e-6:
                mask_vis = (mask - mask_min) / (mask_max - mask_min)
            else:
                mask_vis = mask
                
            axes[idx, slot_i + 2].imshow(mask_vis, cmap='jet', vmin=0, vmax=1)
            if idx == 0:
                axes[idx, slot_i + 2].set_title(f"Slot {slot_i}\n(max: {mask_max:.2f})", fontsize=10)
            axes[idx, slot_i + 2].axis('off')
            
    plt.tight_layout()
    save_path = save_dir / f'consistency_{class_name}.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"Saved consistency visualization to: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class-name', type=str, default='cat', help='Class to visualize (e.g., cat, airplane)')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of images to visualize')
    args = parser.parse_args()
    
    # Load config
    data_config = load_config('configs/data_config.yaml')
    model_config = load_config('configs/model_config.yaml')
    
    # Setup
    device = get_device(model_config.get('device', 'auto'))
    set_seed(42)
    
    # Get class index
    classes = data_config['dataset']['classes']
    if args.class_name not in classes:
        print(f"Error: Class '{args.class_name}' not found in dataset classes: {classes}")
        return
    class_idx = classes.index(args.class_name)
    
    # Load Data
    train_loader, _ = create_dataloaders(
        data_config['dataset'],
        data_config['augmentation'],
        data_config['dataloader']
    )
    
    # Load Model
    backbone = ResNetBackbone.from_config(model_config['backbone'])
    slot_model = SlotAttentionModel.from_config(model_config)
    
    checkpoint_path = 'checkpoints/part_discovery/best_model.pt'
    model_dict = torch.nn.ModuleDict({
        'backbone': backbone, 
        'slot_model': slot_model
    })
    load_checkpoint(checkpoint_path, model_dict, device=device)
    
    backbone.to(device)
    slot_model.to(device)
    
    # Run Visualization
    visualize_class_consistency(
        backbone,
        slot_model,
        train_loader,
        device,
        class_idx,
        args.class_name,
        num_samples=args.num_samples
    )

if __name__ == '__main__':
    main()
