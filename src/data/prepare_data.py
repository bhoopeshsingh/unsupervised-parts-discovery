"""
Download and prepare ImageNet subset with background removal + subject centering.

Pipeline per image:
  1. Remove background with BiRefNet (higher quality than U2Net default)
  2. Crop tight to subject bounding box using alpha mask
  3. Pad symmetrically so subject fills SUBJECT_FILL (70%) of the canvas
  4. Composite on white, resize to TARGET_SIZE × TARGET_SIZE (Lanczos)

This ensures DINO's 28×28 patch grid lands on the actual object, not empty borders.

Usage:
    # All three classes (default)
    python src/data/prepare_data.py --output_dir data/masked --num_images 500

    # Specific classes only
    python src/data/prepare_data.py --output_dir data/masked --classes cat bird --num_images 300

    # Skip background removal (faster, raw images)
    python src/data/prepare_data.py --output_dir data/raw --num_images 500 --no_bg_removal
"""

import argparse
import os
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# ImageNet-1k index → our class name
# ---------------------------------------------------------------------------
#
# Cat: DOMESTIC CATS ONLY (281-285).
#   Excluded big/wild cats: cougar(286), lynx(287), leopard(288),
#   snow leopard(289), jaguar(290), lion(291), tiger(292), cheetah(293).
#
# Bird: FLYING BIRDS ONLY (10 sub-types).
#   Explicitly excluded non-flying birds:
#     ostrich(9), penguin/king penguin(96), peacock(84),
#     black grouse(80), ptarmigan(81), prairie chicken(83).
#
FULL_CLASS_MAPPING = {
    # Cat — domestic only (5 sub-types, ~6500 images available)
    281: "cat",   # tabby cat
    282: "cat",   # tiger cat  (domestic tabby-striped, NOT a tiger)
    283: "cat",   # Persian cat
    284: "cat",   # Siamese cat
    285: "cat",   # Egyptian cat

    # Car (7 sub-types, ~9100 images available)
    511: "car",   # convertible
    817: "car",   # sports car
    627: "car",   # limousine
    468: "car",   # cab / taxi
    609: "car",   # jeep
    436: "car",   # beach wagon / station wagon
    751: "car",   # race car / racer

    # Bird — flying birds only (10 sub-types, ~13000 images available)
    # Excluded: ostrich(9), peacock(84), penguin(96), grouse(80,81,83)
    11:  "bird",  # goldfinch         — small, vivid yellow, clear subject
    12:  "bird",  # house finch        — common songbird
    13:  "bird",  # junco              — dark-eyed junco, flying songbird
    14:  "bird",  # indigo bunting     — striking blue plumage
    15:  "bird",  # robin              — very recognisable, red breast
    16:  "bird",  # bulbul             — tropical songbird
    17:  "bird",  # jay                — colourful, clear silhouette
    18:  "bird",  # magpie             — high contrast, strong features
    19:  "bird",  # chickadee          — small acrobatic flyer
    20:  "bird",  # water ouzel/dipper — active flyer over streams
}

TARGET_SIZE   = 128    # final output resolution
SUBJECT_FILL  = 0.70   # subject occupies this fraction of the canvas (each axis)

# Pillow 9 moved resampling filters under Image.Resampling; support both
_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS

# ---------------------------------------------------------------------------
# rembg session — created once, reused for all images (expensive to init)
# ---------------------------------------------------------------------------
_rembg_session = None


def _get_rembg_session():
    """
    Return a cached rembg session using BiRefNet-general.
    BiRefNet produces significantly sharper subject boundaries than the default
    U2Net, especially for fine details (fur, feathers, wheel spokes).
    Falls back to the rembg default model if BiRefNet weights are unavailable.
    """
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session
        try:
            _rembg_session = new_session("birefnet-general")
            print("  rembg: using BiRefNet-general (high quality)")
        except Exception:
            _rembg_session = new_session()   # default U2Net fallback
            print("  rembg: BiRefNet unavailable, falling back to U2Net")
    return _rembg_session


# ---------------------------------------------------------------------------
# Core image processing
# ---------------------------------------------------------------------------

def _process_subject(image: Image.Image) -> Image.Image:
    """
    Full subject isolation and centering pipeline:

      1. Remove background → RGBA via BiRefNet
      2. Tight crop using alpha bounding box
      3. Symmetric padding → subject = SUBJECT_FILL of canvas
      4. Composite on white, Lanczos resize → TARGET_SIZE × TARGET_SIZE RGB

    Returns a plain RGB image ready to save.
    """
    from rembg import remove

    # Step 1 — background removal (RGBA output)
    rgba = remove(image, session=_get_rembg_session())

    # Step 2 — tight crop to subject using alpha channel
    alpha = rgba.split()[3]          # PIL alpha channel (L mode)
    bbox  = alpha.getbbox()          # (left, top, right, bottom) of non-zero alpha

    if bbox is None:
        # No subject detected — return plain white frame (will be filtered by
        # brightness/variance thresholds in the DINO pipeline)
        return Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (255, 255, 255))

    subject = rgba.crop(bbox)        # RGBA, tightly cropped
    sw, sh  = subject.size

    # Step 3 — symmetric padding so subject covers SUBJECT_FILL of canvas
    # Canvas side = max subject dimension / fill fraction → subject never clipped
    canvas_px = max(int(max(sw, sh) / SUBJECT_FILL), TARGET_SIZE)
    pad_left  = (canvas_px - sw) // 2
    pad_top   = (canvas_px - sh) // 2

    canvas = Image.new("RGBA", (canvas_px, canvas_px), (255, 255, 255, 255))
    canvas.paste(subject, (pad_left, pad_top), mask=subject.split()[3])

    # Step 4 — convert to RGB and resize (Lanczos = best downscale quality)
    return canvas.convert("RGB").resize((TARGET_SIZE, TARGET_SIZE), _LANCZOS)


def prepare_dataset(
    output_dir: str,
    num_images_per_class: int = 500,
    target_classes: list = None,
    remove_bg: bool = True,
    val_every: int = 5,           # every N-th image goes to val (→ ~20% val split)
    seed: int = 42,
):
    """
    Stream ImageNet-1k-128x128 from HuggingFace, process each image through the
    background-removal + subject-centering pipeline, and save to:
        output_dir/{train,val}/{class_name}/

    Args:
        output_dir:           Root folder for saved images (e.g. "data/masked")
        num_images_per_class: How many images to collect per class
        target_classes:       Subset of ["cat", "car", "bird"] — None = all three
        remove_bg:            Run full subject isolation pipeline (recommended)
        val_every:            1 out of every N images → validation set (~20% split)
        seed:                 Kept for documentation; streaming order is fixed by HF
    """
    output_path = Path(output_dir)

    # Build filtered class mapping
    if target_classes:
        unknown = set(target_classes) - set(FULL_CLASS_MAPPING.values())
        if unknown:
            raise ValueError(f"Unknown classes: {unknown}. Available: {set(FULL_CLASS_MAPPING.values())}")
        class_mapping = {k: v for k, v in FULL_CLASS_MAPPING.items() if v in target_classes}
    else:
        class_mapping = FULL_CLASS_MAPPING

    active_classes = sorted(set(class_mapping.values()))
    target_indices = set(class_mapping.keys())

    print(f"Classes     : {active_classes}")
    print(f"Images/class: {num_images_per_class}")
    print(f"Output      : {output_path}")
    print(f"Pipeline    : {'BiRefNet bg removal + subject centering (70% fill)' if remove_bg else 'resize only'}")
    print(f"Output size : {TARGET_SIZE}×{TARGET_SIZE}")
    print()

    # Create output directories
    for cls in active_classes:
        (output_path / "train" / cls).mkdir(parents=True, exist_ok=True)
        (output_path / "val"   / cls).mkdir(parents=True, exist_ok=True)

    print("Streaming dataset from HuggingFace (benjamin-paine/imagenet-1k-128x128)…")
    dataset = load_dataset(
        "benjamin-paine/imagenet-1k-128x128",
        split="train",
        streaming=True,
        token=os.environ.get("HF_TOKEN"),   # set via: export HF_TOKEN=your_token
    )

    counts   = {cls: 0 for cls in active_classes}
    failures = 0
    total_needed = num_images_per_class * len(active_classes)

    pbar = tqdm(total=total_needed, unit="img")

    for item in dataset:
        label = item["label"]
        if label not in target_indices:
            continue

        cls = class_mapping[label]
        if counts[cls] >= num_images_per_class:
            continue

        try:
            image: Image.Image = item["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")

            if remove_bg:
                # Full pipeline: bg removal → tight crop → center → 70% fill → resize
                image = _process_subject(image)
            else:
                # Raw path: just resize cleanly
                image = image.resize((TARGET_SIZE, TARGET_SIZE), _LANCZOS)

            # Train / val split
            split = "val" if (counts[cls] % val_every == 0) else "train"
            save_path = output_path / split / cls / f"{counts[cls]:05d}.png"
            image.save(save_path, format="PNG")

            counts[cls] += 1
            pbar.update(1)
            pbar.set_postfix({c: counts[c] for c in active_classes})

        except Exception as exc:
            failures += 1
            if failures <= 5:
                print(f"\n  Warning: skipped an image — {exc}")

        if all(counts[c] >= num_images_per_class for c in active_classes):
            break

    pbar.close()

    print("\n── Download complete ─────────────────────────────────")
    for cls in active_classes:
        train_n = len(list((output_path / "train" / cls).glob("*.png")))
        val_n   = len(list((output_path / "val"   / cls).glob("*.png")))
        print(f"  {cls:6s}  train={train_n:4d}  val={val_n:4d}  total={train_n+val_n:4d}")
    if failures:
        print(f"  (skipped {failures} images due to errors)")
    print("─────────────────────────────────────────────────────")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ImageNet subset with subject isolation")
    parser.add_argument("--output_dir",  default="data/masked",
                        help="Root output directory (default: data/masked)")
    parser.add_argument("--num_images",  type=int, default=500,
                        help="Images per class (default: 500)")
    parser.add_argument("--classes",     nargs="+", default=None,
                        choices=["cat", "car", "bird"],
                        help="Classes to download (default: all — cat car bird)")
    parser.add_argument("--no_bg_removal", action="store_true",
                        help="Skip background removal and centering (faster, raw images)")
    args = parser.parse_args()

    prepare_dataset(
        output_dir=args.output_dir,
        num_images_per_class=args.num_images,
        target_classes=args.classes,
        remove_bg=not args.no_bg_removal,
    )
