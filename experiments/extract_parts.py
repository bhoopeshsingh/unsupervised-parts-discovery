"""Extract parts with rich descriptors from trained Slot Attention model"""

import sys
sys.path.append('.')

import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import pickle

from src.data.loaders import create_dataloaders
from src.models.backbone import ResNetBackbone
from src.models.part_discovery.slot_attention import SlotAttentionModel
from src.extraction.part_extractor import PartExtractor
from src.extraction.part_filter import PartFilter, FilterConfig
from src.utils import load_config, get_device, set_seed, load_checkpoint


def extract_parts(
    backbone,
    slot_model,
    part_extractor,
    dataloader,
    device,
    limit=None
):
    """
    Extract parts from all images in dataloader using PartExtractor.
    Also collects source images for visualization overlay.
    """
    backbone.eval()
    slot_model.eval()
    
    all_descriptors = []
    all_images = []  # Store source images for visualization

    print("Extracting rich part descriptors...")
    
    with torch.no_grad():
        image_offset = 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader)):
            if limit and batch_idx * dataloader.batch_size >= limit:
                break
            
            images = images.to(device)
            # Labels: move to cpu for storage in descriptor
            labels_cpu = labels.cpu()
            
            # Store images for visualization (denormalize and convert to uint8)
            images_np = images.cpu().numpy()
            for img in images_np:
                # img is [C, H, W], convert to [H, W, C]
                img_hwc = np.transpose(img, (1, 2, 0))
                # Denormalize (assuming ImageNet normalization)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_denorm = img_hwc * std + mean
                img_denorm = np.clip(img_denorm * 255, 0, 255).astype(np.uint8)
                all_images.append(img_denorm)

            # Forward pass
            features = backbone.get_feature_maps(images)
            recon, masks, slots, attn = slot_model(features)
            
            # Extract rich parts
            batch_parts = part_extractor.extract_parts_from_batch(
                images=images,
                slots=slots,
                masks=masks,
                labels=labels_cpu,
                image_offset=image_offset
            )
            
            all_descriptors.extend(batch_parts)
            
            image_offset += images.size(0)
            
    print(f"\nExtracted {len(all_descriptors)} parts from {image_offset} images")
    print(f"Collected {len(all_images)} source images for visualization")
    return all_descriptors, all_images


def save_parts_data(descriptors, images, save_dir, metadata):
    """Save extracted parts data to disk including source images for visualization"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create combined feature matrix [N_parts, D_total]
    print("Creating feature matrix...")
    features = np.array([d.to_combined_vector() for d in descriptors], dtype=np.float32)
    
    # 2. Extract components for backward compatibility/analysis
    # Note: These are flattened, unlike original [N_images, N_slots, D]
    slots = np.array([d.slot_features for d in descriptors], dtype=np.float32)
    masks = np.array([d.mask for d in descriptors], dtype=np.float32)
    
    # 3. Extract metadata arrays
    image_ids = np.array([d.image_idx for d in descriptors], dtype=int)
    slot_ids = np.array([d.slot_idx for d in descriptors], dtype=int)
    labels = np.array([d.class_label for d in descriptors], dtype=int)
    bboxes = np.array([d.bbox for d in descriptors], dtype=int)
    
    # 4. Save source images for visualization overlay
    print("Saving source images for visualization...")
    images_array = np.array(images, dtype=np.uint8)

    # 5. Save arrays
    np.save(save_dir / 'features.npy', features)
    np.save(save_dir / 'slots.npy', slots)
    np.save(save_dir / 'masks.npy', masks)
    np.save(save_dir / 'part_to_image.npy', image_ids)
    np.save(save_dir / 'part_to_slot.npy', slot_ids)
    np.save(save_dir / 'part_to_class.npy', labels)
    np.save(save_dir / 'bboxes.npy', bboxes)
    np.save(save_dir / 'images.npy', images_array)  # Source images for overlay

    # 5. Save metadata
    metadata_path = save_dir / 'metadata.json'
    # Add feature dimensions to metadata
    if descriptors:
        d = descriptors[0]
        metadata['feature_dims'] = {
            'total': int(d.total_dim),
            'slot': len(d.slot_features),
            'visual': len(d.visual_features),
            'spatial': len(d.spatial_features),
            'shape': len(d.shape_features)
        }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\nParts data saved to: {save_dir}")
    print(f"  - features.npy: {features.nbytes / 1e6:.2f} MB")
    print(f"  - slots.npy: {slots.nbytes / 1e6:.2f} MB")
    print(f"  - masks.npy: {masks.nbytes / 1e6:.2f} MB")
    print(f"  - images.npy: {images_array.nbytes / 1e6:.2f} MB")
    print(f"  - metadata.json")


def main():
    # Load config for defaults
    try:
        unified_config = load_config('configs/unified_config.yaml')
        paths_config = unified_config.get('paths', {})
        default_checkpoint = paths_config.get('best_model', 'checkpoints/part_discovery/best_model.pt')
        default_output = paths_config.get('extracted_parts', './parts/extracted')
    except:
        default_checkpoint = 'checkpoints/part_discovery/best_model.pt'
        default_output = './parts/extracted'

    parser = argparse.ArgumentParser(description='Extract parts from trained Slot Attention model')
    parser.add_argument('--checkpoint', type=str, default=default_checkpoint,
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default=default_output,
                        help='Directory to save extracted parts')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of images to process (for testing)')
    args = parser.parse_args()
    
    # Load configurations
    # Load configurations
    config = load_config('configs/unified_config.yaml')
    data_config = config
    model_config = config
    
    # Ensure classes are propagated from paths to dataset config if missing
    # This fixes the KeyError and ensures data loader uses the correct classes
    if 'classes' not in data_config['dataset'] and 'classes' in data_config.get('paths', {}):
        data_config['dataset']['classes'] = data_config['paths']['classes']
    
    # Set seed
    set_seed(data_config.get('seed', 42))
    
    # Device
    device = get_device(model_config.get('device', 'auto'))
    
    # Create dataloaders
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
    
    # Initialize Part Extractor
    print("Initializing Part Extractor...")
    part_extractor = PartExtractor(device=device)
    
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
    
    # Extract parts (now also returns source images for visualization)
    descriptors, source_images = extract_parts(
        backbone=backbone,
        slot_model=slot_model,
        part_extractor=part_extractor,
        dataloader=train_loader,
        device=device,
        limit=args.limit
    )
    
    # Load configuration for filtering
    unified_config = load_config('configs/unified_config.yaml')
    filter_params = unified_config.get('filtering', {})

    # Filter out low-quality parts before clustering
    print("\n=== Filtering Low-Quality Parts ===")
    filter_config = FilterConfig(
        min_coverage=filter_params.get('min_coverage', 0.02),
        max_coverage=filter_params.get('max_coverage', 0.85),
        max_connected_components=filter_params.get('max_connected_components', 2),
        min_edge_density=filter_params.get('min_edge_density', 0.05),
        min_std=filter_params.get('min_std', 0.1),
        min_bbox_size=filter_params.get('min_bbox_size', 4)
    )
    part_filter = PartFilter(config=filter_config)

    # Filter and get statistics
    filtered_descriptors, filter_stats = part_filter.filter_valid_parts(
        descriptors, return_stats=True
    )

    print(f"Total input parts: {filter_stats['total_input']}")
    print(f"Valid parts after filtering: {filter_stats['total_valid']}")
    print(f"Rejected parts: {filter_stats['total_rejected']} ({filter_stats['rejection_rate']:.1%})")
    print(f"Rejection breakdown:")
    for reason, count in filter_stats['rejection_reasons'].items():
        if count > 0:
            print(f"  - {reason}: {count}")

    # Use filtered descriptors
    descriptors = filtered_descriptors

    # Prepare metadata
    metadata = {
        'num_parts': len(descriptors),
        'num_parts_before_filtering': filter_stats['total_input'],
        'filtering_stats': filter_stats,
        'num_images': len(source_images),
        'checkpoint_epoch': checkpoint.get('epoch', 'unknown') if checkpoint else 'none',
        'checkpoint_loss': float(checkpoint.get('loss', 0)) if checkpoint else 0.0,
        'classes': data_config['dataset']['classes'],
        'model_config': model_config
    }
    
    # Save extracted data (including source images for visualization)
    save_parts_data(descriptors, source_images, args.output_dir, metadata)

    print("\n✓ Part extraction completed successfully!")
    print(f"\nNext step: Run clustering: python experiments/cluster_parts.py")


if __name__ == '__main__':
    main()
