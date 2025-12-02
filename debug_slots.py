import numpy as np
import matplotlib.pyplot as plt

def analyze_slots():
    print("Loading slots...")
    slots = np.load('parts/extracted/slots.npy')
    print(f"Slots shape: {slots.shape}")
    
    # Check for NaNs or Infs
    if np.isnan(slots).any():
        print("WARNING: Slots contain NaNs!")
    if np.isinf(slots).any():
        print("WARNING: Slots contain Infs!")
        
    # Basic stats
    print(f"Mean: {np.mean(slots):.4f}")
    print(f"Std: {np.std(slots):.4f}")
    print(f"Min: {np.min(slots):.4f}")
    print(f"Max: {np.max(slots):.4f}")
    
    # Check if slots are identical across images
    # Take first slot of first image
    ref_slot = slots[0, 0]
    # Check distance to other slots
    diffs = np.linalg.norm(slots.reshape(-1, slots.shape[-1]) - ref_slot, axis=1)
    print(f"Mean distance from first slot: {np.mean(diffs):.4f}")
    print(f"Min distance from first slot: {np.min(diffs):.4f}")
    
    # Check variance per dimension
    var_per_dim = np.var(slots.reshape(-1, slots.shape[-1]), axis=0)
    print(f"Mean variance per dimension: {np.mean(var_per_dim):.4f}")
    print(f"Min variance per dimension: {np.min(var_per_dim):.4f}")
    print(f"Max variance per dimension: {np.max(var_per_dim):.4f}")
    
    # Check if slots within same image are identical (collapse)
    intra_image_diffs = []
    for i in range(min(100, len(slots))):
        s = slots[i] # [K, D]
        # pairwise distances
        dists = []
        for j in range(len(s)):
            for k in range(j+1, len(s)):
                dists.append(np.linalg.norm(s[j] - s[k]))
        intra_image_diffs.append(np.mean(dists))
    
    print(f"Mean intra-image slot distance: {np.mean(intra_image_diffs):.4f}")

if __name__ == "__main__":
    analyze_slots()
