#!/usr/bin/env python3
"""
Create N augmented copies per image (default 3) while preserving train/val/class layout.

Typical use after select_top_quality_images.py:
  1) Filter → data/v2/images
  2) Augment → data/v2/images_aug  (4× file count if --variants 3: original + 3)

Examples
--------
  python scripts/augment_dataset_x3.py \\
      --input data/v2/images --output data/v2/images_aug --variants 3

  # add aug next to originals (no --output)
  python scripts/augment_dataset_x3.py --input data/v2/images --in-place
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
import torchvision.transforms as T

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val")


def list_images(class_dir: Path) -> list[Path]:
    return [
        p
        for p in sorted(class_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED
    ]


def augment_pil(img: Image.Image, seed: int) -> Image.Image:
    """Deterministic aug for one (image, seed) pair."""
    random.seed(seed)
    torch.manual_seed(seed)
    pipe = T.Compose(
        [
            T.RandomResizedCrop(size=128, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
            T.RandomRotation(degrees=12),
        ]
    )
    out = pipe(img)
    if isinstance(out, torch.Tensor):
        return T.ToPILImage()(out)
    return out


def main():
    ap = argparse.ArgumentParser(description="Add N random augmentations per image.")
    ap.add_argument("--input", type=Path, required=True, help="Root with train/ and val/.")
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output root (created). Not required with --in-place.",
    )
    ap.add_argument("--variants", type=int, default=3, help="Number of extra aug images per file.")
    ap.add_argument("--seed", type=int, default=42, help="Base RNG seed (offset per image index).")
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="Write augmented files into the same class folder as --input (still reads from --input).",
    )
    args = ap.parse_args()
    if args.variants < 1:
        raise SystemExit("--variants must be >= 1")
    if not args.in_place and args.output is None:
        raise SystemExit("Need --output unless using --in-place")

    input_root = args.input.resolve()
    if args.in_place:
        output_root = input_root
    else:
        output_root = args.output.resolve()

    for sp in SPLITS:
        if not (input_root / sp).is_dir():
            raise SystemExit(f"Missing {input_root / sp}")

    if not args.in_place:
        if output_root == input_root:
            raise SystemExit("Refusing to use same path without --in-place; pick a different --output.")
        for sp in SPLITS:
            p = output_root / sp
            if p.is_dir():
                shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)

    n_in = n_out = 0

    for split in SPLITS:
        base_in = input_root / split
        base_out = output_root / split
        if not args.in_place:
            base_out.mkdir(parents=True, exist_ok=True)

        for class_dir in sorted(base_in.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            out_class = base_out / class_name
            if not args.in_place:
                out_class.mkdir(parents=True, exist_ok=True)

            for img_path in list_images(class_dir):
                n_in += 1
                stem = img_path.stem
                suf = img_path.suffix.lower()
                if suf not in SUPPORTED:
                    suf = ".png"

                img = Image.open(img_path).convert("RGB")
                if not args.in_place:
                    # Copy original into mirrored layout
                    shutil.copy2(img_path, out_class / img_path.name)
                else:
                    # Original stays; only add aug files
                    pass

                if not args.in_place:
                    n_out += 1

                for v in range(args.variants):
                    seed = args.seed + n_in * 1000 + v * 17
                    aug = augment_pil(img, seed)
                    out_name = f"{stem}_aug{v + 1}{suf}"
                    dest = out_class / out_name
                    if dest.suffix.lower() in (".jpg", ".jpeg"):
                        aug.save(dest, quality=95)
                    else:
                        aug.save(dest)
                    n_out += 1

    print(f"Read ~{n_in} source images; wrote {n_out} augmented files under {output_root}")
    if args.in_place:
        print("Original files unchanged; new files are *_aug1, *_aug2, ...")


if __name__ == "__main__":
    main()
