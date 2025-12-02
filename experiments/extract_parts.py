"""Extract parts from trained Slot Attention model"""

import sys
sys.path.append('.')

import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json

from src.data.loaders import create_dataloaders
from src.models.backbone import ResNetBackbone
from src.models.part_discovery.slot_attention import SlotAttentionModel
from src.utils import load_config, get_device, set_seed, load_checkpoint


def extract_parts(
    backbone,
    slot_model,
    dataloader,
    device,
    limit=None
):
    """
    Extract parts from all images in dataloader
    
    Args:
        backbone: ResNet backbone model
        slot_model: Slot Attention model
        dataloader: DataLoader to extract from
        device: torch.device
        limit: Optional limit on number of images to process
    
    Returns:
        parts_data: Dictionary containing all extracted information
    """
    backbone.eval()
    slot_model.eval()
    
    all_slots = []
    all_attn_weights = []
    all_masks = []
    all_image_ids = []
    all_labels = []
    
    print("Extracting parts from images...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader)):
            if limit and batch_idx * dataloader.batch_size >= limit:
                break
            
            images = images.to(device)
            
            # Forward pass through backbone
            features = backbone.get_feature_maps(images)
            
            # Forward pass through slot attention
            recon, masks, slots, attn = slot_model(features)
            
            # Move to CPU and store
            all_slots.append(slots.cpu().numpy())
            all_attn_weights.append(attn.cpu().numpy())
            all_masks.append(masks.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())
            
            # Track image IDs
            start_id = batch_idx * dataloader.batch_size
            batch_size = images.size(0)
            all_image_ids.extend(range(start_id, start_id + batch_size))
    
    # Concatenate all batches
    parts_data = {
        'slots': np.concatenate(all_slots, axis=0),  # [N, num_slots, slot_dim]
        'attention_weights': np.concatenate(all_attn_weights, axis=0),  # [N, num_slots, spatial_elements]
        'masks': np.concatenate(all_masks, axis=0),  # [N, num_slots, H, W]
        'image_ids': np.array(all_image_ids),
        'labels': np.array(all_labels)
    }
    
    print(f"\nExtracted parts from {len(all_image_ids)} images")
    print(f"Slots shape: {parts_data['slots'].shape}")
    print(f"Masks shape: {parts_data['masks'].shape}")
    
    return parts_data


def save_parts_data(parts_data, save_dir, metadata):
    """Save extracted parts data to disk"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arrays
    np.save(save_dir / 'slots.npy', parts_data['slots'])
    np.save(save_dir / 'attention_weights.npy', parts_data['attention_weights'])
    np.save(save_dir / 'masks.npy', parts_data['masks'])
    np.save(save_dir / 'image_ids.npy', parts_data['image_ids'])
    np.save(save_dir / 'labels.npy', parts_data['labels'])
    
    # Save metadata
    metadata_path = save_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nParts data saved to: {save_dir}")
    print(f"  - slots.npy: {parts_data['slots'].nbytes / 1e6:.2f} MB")
    print(f"  - masks.npy: {parts_data['masks'].nbytes / 1e6:.2f} MB")
    print(f"  - metadata.json")


def visualize_sample_parts(
    parts_data,
    dataloader,
    num_samples=5,
    save_dir=None
):
    """
    Create visualizations of extracted parts
    
    Shows: original image | attention maps (one per slot) | reconstructed masks
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get some sample images
    images_iter = iter(dataloader)
    images, labels = next(images_iter)
    
    num_slots = parts_data['masks'].shape[1]
    
    for sample_idx in range(min(num_samples, len(images))):
        fig = plt.figure(figsize=(20, 4))
        gs = GridSpec(1, num_slots + 2, figure=fig)
        
        # Original image
        ax = fig.add_subplot(gs[0, 0])
        img = images[sample_idx].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())  # Normalize
        ax.imshow(img)
        ax.set_title(f'Original\nLabel: {labels[sample_idx]}')
        ax.axis('off')
        
        # Attention masks for each slot
        masks = parts_data['masks'][sample_idx]  # [num_slots, H, W]
        
        for slot_idx in range(num_slots):
            ax = fig.add_subplot(gs[0, slot_idx + 1])
            ax.imshow(masks[slot_idx], cmap='hot', interpolation='bilinear')
            ax.set_title(f'Slot {slot_idx}')
            ax.axis('off')
        
        # Composite (all masks summed)
        ax = fig.add_subplot(gs[0, -1])
        composite = masks.sum(axis=0)
        ax.imshow(composite, cmap='viridis', interpolation='bilinear')
        ax.set_title('Composite')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(save_dir / f'sample_{sample_idx}.png', dpi=150, bbox_inches='tight')
            print(f"Saved visualization: sample_{sample_idx}.png")
        else:
            plt.show()
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Extract parts from trained Slot Attention model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/part_discovery/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./parts/extracted',
                        help='Directory to save extracted parts')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of images to process (for testing)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization samples')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of visualization samples to generate')
    args = parser.parse_args()
    
    # Load configurations
    data_config = load_config('configs/data_config.yaml')
    model_config = load_config('configs/model_config.yaml')
    
    # Set seed
    set_seed(data_config.get('seed', 42))
    
    # Device
    device = get_device(model_config.get('device', 'auto'))
    
    # Create dataloaders (use train set for extraction)
    print("\nPreparing datasets...")
    train_loader, _ = create_dataloaders(
        data_config['dataset'],
        data_config['augmentation'],
        data_config['dataloader']
    )
    
    # Create models
    print("\nInitializing models...")
    backbone = ResNetBackbone.from_config(model_config['backbone'])
    slot_model = SlotAttentionModel.from_config(model_config)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from: {args.checkpoint}")
    model_dict = torch.nn.ModuleDict({
        'backbone': backbone,
        'slot_model': slot_model
    })
    checkpoint = load_checkpoint(args.checkpoint, model_dict, device=device)
    
    # Move models to device
    backbone.to(device)
    slot_model.to(device)
    
    # Extract parts
    parts_data = extract_parts(
        backbone=backbone,
        slot_model=slot_model,
        dataloader=train_loader,
        device=device,
        limit=args.limit
    )
    
    # Prepare metadata
    metadata = {
        'num_images': len(parts_data['image_ids']),
        'num_slots': parts_data['slots'].shape[1],
        'slot_dim': parts_data['slots'].shape[2],
        'mask_size': list(parts_data['masks'].shape[2:]),
        'checkpoint_epoch': checkpoint.get('epoch', 'unknown'),
        'checkpoint_loss': float(checkpoint.get('loss', 0)),
        'classes': data_config['dataset']['classes'],
        'model_config': model_config
    }
    
    # Save extracted data
    save_parts_data(parts_data, args.output_dir, metadata)
    
    # Generate visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        vis_dir = Path(args.output_dir) / 'visualizations'
        visualize_sample_parts(
            parts_data=parts_data,
            dataloader=train_loader,
            num_samples=args.num_samples,
            save_dir=vis_dir
        )
    
    print("\n✓ Part extraction completed successfully!")
    print(f"\nNext steps:")
    print(f"  1. Review visualizations in: {Path(args.output_dir) / 'visualizations'}")
    print(f"  2. Run clustering: python experiments/cluster_parts.py")


if __name__ == '__main__':
    main()
