# visualize_cluster_samples.py
"""
Create visualization montages for clusters.

Given:
  - clusters.json (with sample_files per cluster)
  - saved sample images under cluster_samples/cluster_<id>/

This file builds:
  - <out_dir>/cluster_<id>_montage.png

This is purely diagnostic + presentation utility.
References methodology inspiration:
  "/mnt/data/PDiscoFormer-Relaxing Part Discovery.pdf"
"""

import os
import json
from typing import List, Dict
import numpy as np
from PIL import Image, ImageDraw


def make_montage(image_paths: List[str], cols: int = 5, cell_size=(128,128)) -> Image.Image:
    """
    Creates a montage PIL image of sample files.
    """
    if len(image_paths) == 0:
        # empty placeholder
        img = Image.new("RGB", (cell_size[0], cell_size[1]), color=(200,200,200))
        draw = ImageDraw.Draw(img)
        draw.text((10,10), "No samples", fill=(0,0,0))
        return img

    rows = int(np.ceil(len(image_paths) / cols))
    W, H = cell_size

    montage = Image.new("RGB", (cols*W, rows*H), color=(255,255,255))

    for idx, path in enumerate(image_paths):
        r = idx // cols
        c = idx % cols
        tile_x = c * W
        tile_y = r * H

        try:
            im = Image.open(path).convert("RGB")
            im = im.resize(cell_size)
            montage.paste(im, (tile_x, tile_y))
        except Exception:
            # draw placeholder if missing
            placeholder = Image.new("RGB", cell_size, color=(190,190,190))
            d = ImageDraw.Draw(placeholder)
            d.text((10,10), f"Missing\n{path}")
            montage.paste(placeholder, (tile_x, tile_y))

    return montage


def visualize_cluster_samples(
    clusters_json: str,
    out_dir: str = None,
    cols: int = 5,
    cell_size=(128,128),
) -> Dict[int, str]:
    """
    Builds montage PNG for each cluster.

    Args:
      clusters_json: path to clusters.json
      out_dir: directory for montage PNGs (default: same as clusters.json directory)
      cols: images per row
      cell_size: (width, height) per tile

    Returns:
      dict cluster_id -> saved montage path
    """
    if not os.path.exists(clusters_json):
        raise FileNotFoundError(f"clusters.json not found: {clusters_json}")

    with open(clusters_json, "r") as f:
        blob = json.load(f)

    clusters = blob.get("clusters", {})
    sample_files = blob.get("sample_files", {})

    base_dir = os.path.dirname(clusters_json)
    if out_dir is None:
        out_dir = os.path.join(base_dir, "montages")

    os.makedirs(out_dir, exist_ok=True)

    saved = {}
    for cid, samples in sample_files.items():
        montage = make_montage(samples, cols=cols, cell_size=cell_size)
        save_path = os.path.join(out_dir, f"cluster_{cid}_montage.png")
        montage.save(save_path)
        saved[cid] = save_path

    retur
