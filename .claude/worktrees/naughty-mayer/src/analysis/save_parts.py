import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from src.data.transforms import get_inverse_normalize

def save_parts_data(
    parts_data: Dict[str, Any],
    output_dir: str,
    checkpoint: Dict[str, Any],
    data_config: Dict[str, Any],
    model_config: Dict[str, Any],
    all_images: Optional[List[torch.Tensor]] = None
):
    """
    Save extracted parts data and metadata to disk.
    
    Args:
        parts_data: Dictionary containing extracted parts (slots, masks, etc.)
        output_dir: Directory to save the results
        checkpoint: Model checkpoint dictionary
        data_config: Data configuration dictionary
        model_config: Model configuration dictionary
        all_images: Optional list of image tensors to process and save
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process images if provided
    if all_images is not None and len(all_images) > 0:
        images_tensor = torch.cat(all_images, dim=0)
        inv_norm = get_inverse_normalize()
        images_denorm = []
        for i in range(len(images_tensor)):
            img = inv_norm(images_tensor[i])
            img = img.permute(1, 2, 0).cpu().numpy()
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            images_denorm.append(img)
        parts_data['images'] = np.array(images_denorm)

    # Save numpy arrays
    np.save(output_path / 'slots.npy', parts_data['slots'])
    np.save(output_path / 'attention_weights.npy', parts_data['attention_weights'])
    np.save(output_path / 'masks.npy', parts_data['masks'])
    np.save(output_path / 'image_ids.npy', parts_data['image_ids'])
    np.save(output_path / 'labels.npy', parts_data['labels'])
    
    if 'images' in parts_data:
        np.save(output_path / 'images.npy', parts_data['images'])

    # Save metadata
    metadata = {
        'num_images': len(parts_data['image_ids']),
        'num_slots': parts_data['slots'].shape[1],
        'slot_dim': parts_data['slots'].shape[2],
        'mask_size': list(parts_data['masks'].shape[2:]),
        'checkpoint_epoch': checkpoint.get('epoch', 'unknown') if checkpoint else 'unknown',
        'checkpoint_loss': float(checkpoint.get('loss', 0)) if checkpoint else 0.0,
        'classes': data_config['dataset']['classes'],
        'model_config': model_config
    }

    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Parts data saved to: {output_path}")
