# experiments/extract_features.py
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


def extract_all(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    dcfg = cfg["dino"]
    cache_path = dcfg["features_cache"]

    if Path(cache_path).exists() and not dcfg.get("force_recompute", False):
        print(f"Cache exists at {cache_path}. Set force_recompute: true to re-run.")
        data = torch.load(cache_path, weights_only=False)
        fg_threshold = data.get("fg_threshold", None)
        fg_note = f", fg_threshold={fg_threshold}" if fg_threshold is not None else " (no foreground masking)"
        print(
            f"Loaded: {data['features'].shape[0]:,} patches from "
            f"{len(data['image_paths'])} images{fg_note}"
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

    # Foreground masking: use DINO attention to drop background patches before clustering.
    # The CLS token naturally attends more to foreground (object) than background.
    # fg_threshold=0.5 means: keep only the top 50% most-attended patches per image.
    # Set fg_threshold: null in config to disable masking and keep all patches.
    fg_threshold = dcfg.get("fg_threshold", 0.5)
    use_fg_mask = fg_threshold is not None
    if use_fg_mask:
        print(f"Foreground masking enabled (fg_threshold={fg_threshold}) — "
              f"keeping top {(1 - fg_threshold) * 100:.0f}% attended patches per image.")
    else:
        print("Foreground masking disabled — all 784 patches per image will be kept.")

    all_features = []
    all_image_ids = []
    all_patch_ids = []
    all_labels = []
    failed = []
    total_patches_before = 0
    total_patches_after = 0

    for img_idx, (img_path, label) in enumerate(
        tqdm(zip(image_paths, image_labels), total=len(image_paths), desc="Extracting")
    ):
        try:
            img_tensor = extractor.load_image(img_path)          # [1, 3, 224, 224]
            feats = extractor.extract_patches(img_tensor).squeeze(0).cpu()  # [784, 384]
            total_patches_before += feats.shape[0]

            if use_fg_mask:
                # Attention shape: [1, num_heads, 785, 785]
                # Index 0 is CLS token; its attention to the 784 patch tokens:
                attn = extractor.extract_attention(img_tensor)   # [1, heads, 785, 785]
                cls_attn = attn[0, :, 0, 1:].mean(dim=0).cpu()  # [784] mean across heads
                threshold = cls_attn.quantile(fg_threshold)
                fg_mask = cls_attn > threshold                   # [784] boolean
                fg_indices = fg_mask.nonzero(as_tuple=True)[0]  # indices of foreground patches

                # Safety: if masking is too aggressive, fall back to top 25%
                if len(fg_indices) < 10:
                    fg_indices = cls_attn.topk(max(10, feats.shape[0] // 4)).indices

                kept_feats = feats[fg_indices]                   # [K, 384]
                kept_patch_ids = fg_indices.tolist()
            else:
                kept_feats = feats
                kept_patch_ids = list(range(784))

            n_kept = kept_feats.shape[0]
            total_patches_after += n_kept
            all_features.append(kept_feats)
            all_image_ids.extend([img_idx] * n_kept)
            all_patch_ids.extend(kept_patch_ids)
            all_labels.extend([label] * n_kept)

        except Exception as e:
            print(f" Warning: failed on {img_path}: {e}")
            failed.append(img_path)
            continue

    if not all_features:
        raise RuntimeError(
            "No features extracted — check your data/ directory structure"
        )

    features_tensor = torch.cat(all_features, dim=0)
    if use_fg_mask:
        reduction = (1 - total_patches_after / total_patches_before) * 100
        print(f"Foreground masking: {total_patches_before:,} → {total_patches_after:,} patches "
              f"({reduction:.1f}% background removed)")

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
        "fg_threshold": fg_threshold,   # record masking config used
    }
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    print(f"Saved {features_tensor.shape[0]:,} patch features → {cache_path}")
    if failed:
        print(f"Failed images ({len(failed)}): {failed[:5]}")
    return cache


if __name__ == "__main__":
    extract_all()
