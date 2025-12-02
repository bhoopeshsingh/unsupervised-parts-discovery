"""
patch_activation.py

Purpose:
- Run images through a pretrained convolutional backbone (ResNet50) and extract spatial
  feature maps. Each spatial cell in the conv map is considered a 'patch' and its
  feature vector is saved for later clustering.

Notes:
- The script saves three artifacts under `data/features/`:
  - patch_features.npy : stacked (total_patches, C) array
  - patch_shapes.npy   : list of (h, w) per image indicating the conv-grid size
  - image_paths.json   : list of original image file paths
- For large datasets this approach may require a lot of RAM; consider writing chunks to disk
  or using memory-mapped arrays if you hit memory limits.
"""

# patch_activation.py
# Extract patch-level features by running images through a pretrained ResNet50
# and saving the spatial feature maps. Each spatial cell in the conv map is one "patch".

import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing import image as kimage
import os

# CONFIG - adjust these based on your dataset and model
DATA_DIR = "./data/train/cat"           # folder containing images (flat or class subfolders)
SAVE_DIR = "./data/features"
IMG_SIZE = 224                     # adjust to what your model expects
BATCH_SIZE = 8

os.makedirs(SAVE_DIR, exist_ok=True)

# Build a ResNet50 backbone that outputs conv feature maps (no pooling)
backbone = resnet50.ResNet50(include_top=False, weights="imagenet", pooling=None, input_shape=(IMG_SIZE, IMG_SIZE, 3))
# backbone.output is a 4D tensor (B, H, W, C)
backbone.trainable = False

# Preprocessing function (ResNet50-specific)
preprocess = resnet50.preprocess_input

# Collect image filepaths (flatten)
image_paths = []
for root, dirs, files in os.walk(DATA_DIR):
    for fname in files:
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            image_paths.append(os.path.join(root, fname))
image_paths = sorted(image_paths)
print(f"Found {len(image_paths)} images.")

# Helper function - loads an image and gets it ready for ResNet50
def load_and_preprocess(path, img_size=IMG_SIZE):
    img = Image.open(path).convert("RGB").resize((img_size, img_size))
    arr = np.asarray(img).astype("float32")
    arr = preprocess(arr)   # in-place normalization expected by ResNet50
    return arr

# Run inference in batches and collect patch features
all_patches = []
shapes = []
saved_paths = []

for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Extracting"):
    batch_paths = image_paths[i:i+BATCH_SIZE]
    batch_imgs = np.stack([load_and_preprocess(p) for p in batch_paths], axis=0)  # shape [B, H, W, 3]
    # Obtain conv feature maps: [B, h, w, C]
    feat_maps = backbone.predict(batch_imgs, verbose=0)
    # For each image in batch, flatten spatial dims to patches
    for b in range(feat_maps.shape[0]):
        fm = feat_maps[b]             # [h, w, C]
        h, w, c = fm.shape
        patches = fm.reshape(-1, c)   # [h*w, C]
        all_patches.append(patches)
        shapes.append((h, w))
        saved_paths.append(batch_paths[b])

# Stack everything and save
all_patches_np = np.vstack(all_patches)  # (total_patches, C)
np.save(os.path.join(SAVE_DIR, "patch_features.npy"), all_patches_np)
np.save(os.path.join(SAVE_DIR, "patch_shapes.npy"), np.array(shapes, dtype=object))
with open(os.path.join(SAVE_DIR, "image_paths.json"), "w") as f:
    json.dump(saved_paths, f)

print("Saved patch_features.npy with shape:", all_patches_np.shape)
print("Saved patch_shapes.npy for", len(shapes), "images")