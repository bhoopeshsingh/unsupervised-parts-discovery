# experiments/extract_dino_features.py
"""
One-time DINO feature extraction for all images in data/.
Saves to cache/dino_features.pt — all subsequent steps load from here.
Runtime: ~15-20 min for 1000 images on M4 Mac.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from pathlib import Path
from tqdm import tqdm

from src.models.dino_extractor import DinoExtractor

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(data_dir: str, classes_filter=None):
    """
    Walk data_dir, collect image paths and class labels.
    Expects structure: {data_dir}/{class_name}/*.jpg
    If classes_filter is set (e.g. ["cat"]), only those class subdirs are used.
    """
    paths, labels, class_names = [], [], []
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    for class_dir in sorted(data_path.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        if classes_filter is not None and class_name not in classes_filter:
            continue
        if class_name not in class_names:
            class_names.append(class_name)
        class_id = class_names.index(class_name)
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in SUPPORTED:
                paths.append(str(img_path))
                labels.append(class_id)
    return paths, labels, class_names


def extract_all(config_path: str = "configs/unified_config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    dcfg = cfg["dino"]
    cache_path = dcfg["features_cache"]

    if Path(cache_path).exists() and not dcfg.get("force_recompute", False):
        print(f"Cache exists at {cache_path}. Set force_recompute: true to re-run.")
        data = torch.load(cache_path, weights_only=False)
        print(
            f"Loaded: {data['features'].shape[0]} patches from "
            f"{len(data['image_paths'])} images"
        )
        return data

    extractor = DinoExtractor(
        model_name=dcfg["model"],
        device=dcfg["device"],
        image_size=dcfg["image_size"],
    )

    # Input images: dino.data_root overrides dataset.root; dino.classes restricts to those classes
    data_dir = dcfg.get("data_root") or cfg.get("dataset", {}).get("root", "data")
    classes_filter = dcfg.get("classes")  # e.g. ["cat"] for cat-only; None = all
    image_paths, image_labels, class_names = collect_images(data_dir, classes_filter=classes_filter)
    if not image_paths:
        raise RuntimeError(
            f"No images found under {data_dir}"
            + (f" for classes {classes_filter}" if classes_filter else "")
        )
    print(f"Found {len(image_paths)} images across classes: {class_names}")

    all_features = []
    all_image_ids = []
    all_patch_ids = []
    all_labels = []
    failed = []

    for img_idx, (img_path, label) in enumerate(
        tqdm(zip(image_paths, image_labels), total=len(image_paths), desc="Extracting")
    ):
        try:
            feats = extractor.extract_from_path(img_path)  # [784, 384]
            all_features.append(feats)
            all_image_ids.extend([img_idx] * 784)
            all_patch_ids.extend(list(range(784)))
            all_labels.extend([label] * 784)
        except Exception as e:
            print(f" Warning: failed on {img_path}: {e}")
            failed.append(img_path)
            continue

    if not all_features:
        raise RuntimeError(
            "No features extracted — check your data/ directory structure"
        )

    features_tensor = torch.cat(all_features, dim=0)  # [N*784, 384]
    cache = {
        "features": features_tensor,
        "image_ids": torch.tensor(all_image_ids, dtype=torch.long),
        "patch_ids": torch.tensor(all_patch_ids, dtype=torch.long),
        "patch_labels": torch.tensor(all_labels, dtype=torch.long),
        "image_paths": image_paths,
        "class_names": class_names,
        "image_labels": image_labels,
        "grid_size": 28,
        "feat_dim": 384,
    }
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    print(f"Saved {features_tensor.shape[0]:,} patch features → {cache_path}")
    if failed:
        print(f"Failed images ({len(failed)}): {failed[:5]}")
    return cache


if __name__ == "__main__":
    extract_all()
