# visualize_utils.py
import os
import json

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import colors

DATA_DIR = "./data/cats"
FEATURE_DIR = "./data/features"
CLUSTER_DIR = "./data/clusters"
N_CLUSTERS = 20

with open(os.path.join(FEATURE_DIR, "image_paths.json"), "r") as f:
    image_paths = json.load(f)
with open(os.path.join(CLUSTER_DIR, "cluster_maps.json"), "r") as f:
    cluster_maps = json.load(f)

# create a qualitative color map with N_CLUSTERS distinct colors
cmap = plt.cm.get_cmap("tab20", N_CLUSTERS)

def overlay_clusters(img_path, cluster_map, alpha=0.55):
    img = Image.open(img_path).convert("RGBA").resize((224, 224))
    cluster_arr = np.array(cluster_map)
    # scale cluster map to image size if needed (cluster_map is already same spatial size as conv output -> 224 depends on backbone)
    # Create colored overlay
    overlay = np.zeros((cluster_arr.shape[0], cluster_arr.shape[1], 4), dtype=np.uint8)
    for k in range(N_CLUSTERS):
        mask = (cluster_arr == k)
        if mask.sum() == 0:
            continue
        color = tuple((np.array(cmap(k)[:3]) * 255).astype(np.uint8))
        overlay[mask] = np.array([color[0], color[1], color[2], int(255*alpha)])
    overlay_img = Image.fromarray(overlay, mode="RGBA").resize(img.size, resample=Image.NEAREST)
    blended = Image.alpha_composite(img, overlay_img)
    return blended

# Visualize the first few images
for idx in range(min(20, len(image_paths))):
    path = image_paths[idx]
    cluster_map = np.array(cluster_maps[str(idx)])
    # If cluster_map spatial size != image resize used earlier, you might need to upsample cluster_map to 224x224.
    # Here, we assume the cluster_map is already the spatial size from the conv feature map; we resize overlay to match image.
    blended = overlay_clusters(path, cluster_map)
    plt.figure(figsize=(4,4)); plt.imshow(blended); plt.axis("off")
plt.show()


# -------------------------------------------------------------
# COLOR MAP APPLICATION
# -------------------------------------------------------------
def apply_colormap(label_map, cmap="tab10"):
    """
    Convert a label mask (0..1 or integer clusters) into a colorized RGB map.

    Args:
        label_map: 2D numpy array (H, W) with cluster IDs or 0–1 floats
        cmap: any matplotlib colormap

    Returns:
        RGB color image (H, W, 3) as uint8
    """
    # Convert to int cluster labels if in 0–1 range
    if label_map.max() <= 1.0:
        # assume normalized mask (0–1), convert to 0–255 ID space
        label_map_int = (label_map * 255).astype(np.uint8)
    else:
        label_map_int = label_map.astype(np.uint8)

    # Determine how many classes to map
    n_classes = int(label_map_int.max()) + 1
    if n_classes < 1:
        n_classes = 1

    colormap = matplotlib.cm.get_cmap(cmap, n_classes)

    # Normalize IDs to colormap range
    out = colormap(label_map_int / max(1, n_classes - 1))[..., :3]  # strip alpha
    out = (out * 255).astype(np.uint8)
    return out


# -------------------------------------------------------------
# ALPHA BLENDING
# -------------------------------------------------------------
def blend_overlay(image, overlay, alpha=0.5):
    """
    Blend original image and overlay using alpha.

    Args:
        image (H,W,3) uint8
        overlay (H,W,3) uint8
        alpha (float): overlay strength

    Returns:
        blended uint8 RGB image
    """

    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    image = image.astype(np.float32)
    overlay = overlay.astype(np.float32)

    blended = (1 - alpha) * image + alpha * overlay
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended


# -------------------------------------------------------------
# SAVE OVERLAY IMAGE
# -------------------------------------------------------------
def save_overlay(image_arr, out_path):
    """
    Save RGB array as a PNG.

    Args:
        image_arr: uint8 RGB array (H,W,3)
        out_path: destination file path
    """
    Image.fromarray(image_arr).save(out_path)