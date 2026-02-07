"""
Verify model shapes for high resolution input.
"""

import torch
import yaml
from src.models.backbone import ResNetBackbone
from src.models.part_discovery.slot_attention import SlotAttentionModel

def verify_shapes():
    # Load config
    with open('configs/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Override for 128x128 if not already set (it should be set)
    print(f"Config output size: {config['slot_attention']['decoder']['output_size']}")
    
    # Create model
    # We need to manually construct the full model or just test components
    # Let's test components first
    
    # 1. Backbone
    print("\nTesting Backbone...")
    backbone = ResNetBackbone(pretrained=False, use_layer4=True)
    x = torch.randn(2, 3, 128, 128)
    features = backbone.get_feature_maps(x)
    print(f"Input: {x.shape}")
    print(f"Backbone features: {features.shape}")
    
    # Check feature spatial size
    # 128 -> layer1(64) -> layer2(32) -> layer3(16) -> layer4(8)
    # So we expect [2, 2048, 8, 8]
    assert features.shape[2:] == (4, 4) or features.shape[2:] == (8, 8), \
        f"Unexpected spatial size: {features.shape[2:]}"
    
    # 2. Slot Attention Model
    print("\nTesting Slot Attention Model...")
    model = SlotAttentionModel(config)
    
    # Forward pass
    recon, masks, slots, attn = model(features)
    
    print(f"Reconstruction: {recon.shape}")
    print(f"Masks: {masks.shape}")
    print(f"Slots: {slots.shape}")
    
    # Verify outputs
    assert recon.shape == (2, 3, 128, 128), f"Wrong recon shape: {recon.shape}"
    assert masks.shape == (2, config['slot_attention']['num_slots'], 128, 128), \
        f"Wrong mask shape: {masks.shape}"
        
    print("\n✅ Verification Successful!")

if __name__ == "__main__":
    verify_shapes()
