import torch
import sys
sys.path.append('.')

checkpoint_path = 'checkpoints/part_discovery/best_model.pt'
try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Checkpoint Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint Loss: {checkpoint.get('loss', 'unknown')}")
    if 'model_state_dict' in checkpoint:
        print("Model state dict keys:", list(checkpoint['model_state_dict'].keys())[:5])
except Exception as e:
    print(f"Error loading checkpoint: {e}")
