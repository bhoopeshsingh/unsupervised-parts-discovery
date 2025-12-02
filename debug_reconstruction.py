import torch
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from src.utils import load_config, get_device, load_checkpoint
from src.models.backbone import ResNetBackbone
from src.models.part_discovery.slot_attention import SlotAttentionModel
from src.data.loaders import create_dataloaders

def visualize_reconstruction():
    # Load config
    data_config = load_config('configs/data_config.yaml')
    model_config = load_config('configs/model_config.yaml')
    device = get_device('auto')
    
    # Load model
    backbone = ResNetBackbone.from_config(model_config['backbone'])
    slot_model = SlotAttentionModel.from_config(model_config)
    
    checkpoint_path = 'checkpoints/part_discovery/best_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # The checkpoint contains a single 'model_state_dict' which has keys for both backbone and slot_model
    # We need to split them or load them carefully
    state_dict = checkpoint['model_state_dict']
    
    # Split state dict
    backbone_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')}
    slot_dict = {k.replace('slot_model.', ''): v for k, v in state_dict.items() if k.startswith('slot_model.')}
    
    backbone.load_state_dict(backbone_dict)
    slot_model.load_state_dict(slot_dict)
    
    backbone.to(device).eval()
    slot_model.to(device).eval()
    
    # Get data
    train_loader, _ = create_dataloaders(data_config['dataset'], data_config['augmentation'], data_config['dataloader'])
    images, labels = next(iter(train_loader))
    images = images[:5].to(device)
    
    # Forward pass
    with torch.no_grad():
        features = backbone.get_feature_maps(images)
        recon, masks, slots, attn = slot_model(features)
    
    # Plot
    fig, axes = plt.subplots(5, 2, figsize=(5, 12))
    for i in range(5):
        # Original
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')
        
        # Recon
        rec = recon[i].permute(1, 2, 0).cpu().numpy()
        rec = (rec - rec.min()) / (rec.max() - rec.min())
        axes[i, 1].imshow(rec)
        axes[i, 1].set_title("Reconstruction")
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    plt.savefig('debug_reconstruction.png')
    print("Saved debug_reconstruction.png")

if __name__ == '__main__':
    visualize_reconstruction()
