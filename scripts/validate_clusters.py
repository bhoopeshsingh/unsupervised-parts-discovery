# scripts/validate_clusters.py — run after clustering to check quality
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from pathlib import Path

import numpy as np
import torch
import yaml

from src.pipeline.patch_clusterer import PatchClusterer, visualise_part_map

cfg = yaml.safe_load(open("configs/config.yaml"))
data = torch.load(cfg["dino"]["features_cache"], weights_only=False)
n_clusters = cfg["clustering"]["n_clusters"]
clusterer_path = cfg["dino"].get("clusterer_path", "cache/kmeans.pkl")
cluster_labels_path = cfg["dino"].get("cluster_labels_path", "cache/cluster_labels.pt")

use_spatial = cfg["clustering"].get("use_spatial_features", False)
spatial_weight = cfg["clustering"].get("spatial_weight", 0.15)
clusterer = PatchClusterer(
    n_clusters=n_clusters,
    random_seed=cfg["clustering"].get("random_seed", 42),
    use_spatial_features=use_spatial,
    spatial_weight=spatial_weight,
)
labels = clusterer.fit(
    data["features"],
    patch_ids=data["patch_ids"] if use_spatial else None,
)

# Save clusterer and labels
clusterer.save(clusterer_path)
torch.save(torch.tensor(labels), cluster_labels_path)

# Validate: cluster sizes
sizes = np.bincount(labels)
print("Cluster sizes:", sizes)
print("Min cluster size:", sizes.min(), "(should be > 50)")

# Visualise part maps for 3 random images
cache_dir = Path(cfg["dino"].get("cache_dir", "cache/")).resolve()
n_images = len(data["image_paths"])
for img_idx in random.sample(
    range(n_images), min(3, n_images)
):
    part_map = clusterer.get_part_map(img_idx, data["image_ids"])
    visualise_part_map(
        data["image_paths"][img_idx],
        part_map,
        n_clusters=n_clusters,
        save_path=str(cache_dir / f"part_map_img{img_idx}.png"),
    )
print(f"Done — check {cache_dir}/part_map_img*.png for visual validation")
