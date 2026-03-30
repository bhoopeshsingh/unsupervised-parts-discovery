"""
Download and prepare clean curated image datasets — no background removal needed.

Sources:
  cat  → Oxford-IIIT Pet Dataset   (pcuenq/oxford-pets)
           ~3,600 cat images, subjects fill frame, ~400×400px originals
  car  → Stanford Cars Dataset      (tanganke/stanford_cars)
           ~8,000 images, 196 fine-grained models, ~600×400px originals
  bird → CUB-200-2011               (bentrevett/caltech-ucsd-birds-200-2011)
           ~6,000 images, 200 species, bounding boxes provided → crop to bird first

Pipeline per image:
  cat/car  → resize directly to TARGET_SIZE × TARGET_SIZE (Lanczos)
  bird     → crop to bbox, then resize to TARGET_SIZE × TARGET_SIZE

Output layout:
    output_dir/train/<class>/
    output_dir/val/<class>/

Switch between v1 (data/masked) and v2 (data/v2/images) via config.yaml:
    dataset.custom_path: "data/v2/images/train"
    dino.data_root:      "data/v2/images/train"

Usage:
    python src/data/prepare_data.py --output_dir data/v2/images --num_images 5000
    python src/data/prepare_data.py --output_dir data/v2/images --classes cat --num_images 3000
"""

import argparse
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm

TARGET_SIZE = 128
_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------

DATASET_CONFIG = {
    "cat": {
        "hf_path":    "pcuenq/oxford-pets",
        "split":      "train",
        "filter":     lambda item: not item["dog"],   # dog=False → cat
        "get_image":  lambda item: item["image"],
        "get_bbox":   None,                           # no crop needed
        "description": "Oxford-IIIT Pet (cats only)",
    },
    "car": {
        "hf_path":    "tanganke/stanford_cars",
        "split":      "train",
        "filter":     lambda item: True,              # all cars
        "get_image":  lambda item: item["image"],
        "get_bbox":   None,
        "description": "Stanford Cars (196 models)",
    },
    "bird": {
        "hf_path":    "bentrevett/caltech-ucsd-birds-200-2011",
        "split":      "train",
        "filter":     lambda item: True,
        "get_image":  lambda item: item["image"],
        "get_bbox":   lambda item: item["bbox"],      # [x, y, w, h]
        "description": "CUB-200-2011 (bbox-cropped)",
    },
}


def _process_image(image: Image.Image, bbox=None) -> Image.Image:
    """
    Resize to TARGET_SIZE × TARGET_SIZE.
    For birds: crop to bounding box first so the bird fills the frame.
    """
    img = image.convert("RGB")

    if bbox is not None:
        x, y, w, h = bbox
        # Add 5% padding around bbox so we don't clip feathers at the edges
        pad_x = int(w * 0.05)
        pad_y = int(h * 0.05)
        W, H = img.size
        left   = max(0, int(x) - pad_x)
        top    = max(0, int(y) - pad_y)
        right  = min(W, int(x + w) + pad_x)
        bottom = min(H, int(y + h) + pad_y)
        img = img.crop((left, top, right, bottom))

    return img.resize((TARGET_SIZE, TARGET_SIZE), _LANCZOS)


def prepare_dataset(
    output_dir: str,
    num_images_per_class: int = 5000,
    target_classes: list = None,
    val_every: int = 5,
    seed: int = 42,
):
    """
    Stream each curated dataset from HuggingFace, process each image,
    and save to output_dir/{train,val}/{class_name}/.

    Args:
        output_dir:           Root folder (e.g. "data/v2/images")
        num_images_per_class: How many images per class
        target_classes:       Subset of ["cat", "car", "bird"] — None = all
        val_every:            Every N-th image → val (~20% split)
    """
    from datasets import load_dataset

    output_path = Path(output_dir)
    classes = target_classes or list(DATASET_CONFIG.keys())

    unknown = set(classes) - set(DATASET_CONFIG.keys())
    if unknown:
        raise ValueError(f"Unknown classes: {unknown}. Available: {list(DATASET_CONFIG.keys())}")

    print(f"Classes     : {classes}")
    print(f"Images/class: {num_images_per_class}")
    print(f"Output      : {output_path}")
    print(f"Output size : {TARGET_SIZE}×{TARGET_SIZE}")
    print()

    for cls in classes:
        (output_path / "train" / cls).mkdir(parents=True, exist_ok=True)
        (output_path / "val"   / cls).mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")

    for cls in classes:
        cfg = DATASET_CONFIG[cls]
        print(f"── {cls.upper()} ─── {cfg['description']} ─────────────────────")

        ds = load_dataset(
            cfg["hf_path"],
            split=cfg["split"],
            streaming=True,
            token=token,
        )

        count    = 0
        failures = 0
        pbar     = tqdm(total=num_images_per_class, unit="img", desc=cls)

        for item in ds:
            if count >= num_images_per_class:
                break

            if not cfg["filter"](item):
                continue

            try:
                image = cfg["get_image"](item)
                bbox  = cfg["get_bbox"](item) if cfg["get_bbox"] else None
                processed = _process_image(image, bbox)

                split     = "val" if (count % val_every == 0) else "train"
                save_path = output_path / split / cls / f"{count:05d}.png"
                processed.save(save_path, format="PNG")

                count += 1
                pbar.update(1)

            except Exception as exc:
                failures += 1
                if failures <= 3:
                    print(f"\n  Warning: skipped — {exc}")

        pbar.close()

        train_n = len(list((output_path / "train" / cls).glob("*.png")))
        val_n   = len(list((output_path / "val"   / cls).glob("*.png")))
        print(f"  ✓ {cls}: train={train_n}  val={val_n}  total={train_n+val_n}")
        if failures:
            print(f"  (skipped {failures} images due to errors)")
        print()

    print("── Download complete ──────────────────────────────────────────")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download curated image datasets for parts discovery")
    parser.add_argument("--output_dir", default="data/v2/images",
                        help="Root output directory (default: data/v2/images)")
    parser.add_argument("--num_images", type=int, default=5000,
                        help="Images per class (default: 5000)")
    parser.add_argument("--classes", nargs="+", default=None,
                        choices=["cat", "car", "bird"],
                        help="Classes to download (default: all)")
    args = parser.parse_args()

    prepare_dataset(
        output_dir=args.output_dir,
        num_images_per_class=args.num_images,
        target_classes=args.classes,
    )
