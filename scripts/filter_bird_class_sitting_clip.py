#!/usr/bin/env python3
"""
Keep only **bird** images that look *perched / sitting*, not *flying*.

Uses OpenAI CLIP (zero-shot): compares two text prompts against each image.
**cat** / **car** (and any non-``bird`` folder) are copied unchanged.

Requires:
  pip install transformers

Example
-------
  python scripts/filter_bird_class_sitting_clip.py \\
      --input data/v2/images_centered_128 \\
      --output data/v2/images_sitting_birds_only \\
      --device mps
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val")
BIRD_DIR = "bird"

# Single contrastive pair (CLIP logits: index 0 = sitting, 1 = flying)
PROMPT_SITTING = (
    "a photo of a bird perched on a branch or sitting on the ground, not flying"
)
PROMPT_FLYING = "a photo of a bird flying in the sky with wings spread"


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )


def load_clip(device: str):
    from transformers import CLIPModel, CLIPProcessor

    mid = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(mid).to(device)
    processor = CLIPProcessor.from_pretrained(mid)
    model.eval()
    return model, processor


def score_p_sitting(model, processor, image_path: Path, device: str) -> float:
    """Probability of sitting prompt vs flying (softmax over 2 logits)."""
    img = Image.open(image_path).convert("RGB")
    inputs = processor(
        text=[PROMPT_SITTING, PROMPT_FLYING],
        images=img,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits_per_image[0]
    return float(F.softmax(logits, dim=0)[0].item())


def main():
    ap = argparse.ArgumentParser(
        description="Copy dataset; filter class 'bird' with CLIP (sitting vs flying)."
    )
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--min-p-sitting",
        type=float,
        default=0.55,
        help="Keep bird image if P(sitting) >= this. Raise to 0.6–0.7 if too many flyers remain.",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="cuda | mps | cpu (default: auto)",
    )
    args = ap.parse_args()

    input_root = args.input.resolve()
    output_root = args.output.resolve()
    if not input_root.is_dir():
        raise SystemExit(f"Missing {input_root}")

    for sp in SPLITS:
        if not (input_root / sp).is_dir():
            raise SystemExit(f"Expected {input_root}/train/ and {input_root}/val/")

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    try:
        model, processor = load_clip(device)
    except ImportError as e:
        raise SystemExit(
            "Install: pip install transformers\n"
            f"Import error: {e}"
        ) from e

    bird_kept = bird_drop = 0
    other_n = 0

    for split in SPLITS:
        base_in = input_root / split
        base_out = output_root / split
        for class_dir in sorted(base_in.iterdir()):
            if not class_dir.is_dir():
                continue
            cname = class_dir.name
            imgs = list_images(class_dir)
            if not imgs:
                continue

            is_bird = cname == BIRD_DIR
            for img_path in tqdm(imgs, desc=f"{split}/{cname}"):
                dest = base_out / cname / img_path.name
                dest.parent.mkdir(parents=True, exist_ok=True)

                if not is_bird:
                    shutil.copy2(img_path, dest)
                    other_n += 1
                    continue

                try:
                    p_sit = score_p_sitting(model, processor, img_path, device)
                    if p_sit >= args.min_p_sitting:
                        shutil.copy2(img_path, dest)
                        bird_kept += 1
                    else:
                        bird_drop += 1
                except Exception as e:
                    print(f"\n  skip {img_path}: {e}")
                    bird_drop += 1

    print(
        f"\nDone. bird: kept={bird_kept}  dropped={bird_drop}  "
        f"other classes copied={other_n}  → {output_root}"
    )


if __name__ == "__main__":
    main()
