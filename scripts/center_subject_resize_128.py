#!/usr/bin/env python3
"""
Center the DINO-attended subject in the frame, then resize to 128×128.

Uses the same CLS→patch attention as the rest of the project (no LLM / no API).
DINO already highlights the object; we take the tight bbox of high-attention
patches on the 28×28 grid, expand to a square, crop from the 224×224 view,
and Lanczos-resize to TARGET×TARGET (default 128).

Layout matches filter_foreground_dataset.py:
  INPUT_ROOT/train/<class>/*  and  INPUT_ROOT/val/<class>/*
  OUTPUT_ROOT/train/<class>/*  idem

Example
-------
  python scripts/center_subject_resize_128.py \\
      --input data/v2/images \\
      --output data/v2/images_centered_128 \\
      --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

from src.models.dino_extractor import DinoExtractor

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val")
TARGET = 128
DINO_SIZE = 224
_PATCH = 8
_GRID = 28


def cls_attn_28(extractor: DinoExtractor, tensor_224: torch.Tensor) -> np.ndarray:
    attn = extractor.extract_attention(tensor_224)
    v = attn[0, :, 0, 1:].mean(dim=0).cpu().numpy().reshape(_GRID, _GRID)
    return v.astype(np.float64)


def square_bbox_from_mask(mask: np.ndarray, h: int, w: int, margin: float = 1.08) -> tuple[int, int, int, int]:
    """mask [H,W] bool → square (l,t,r,b) in pixel coords, clamped."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        side = min(w, h)
        cx, cy = w // 2, h // 2
        half = side // 2
        l = max(0, cx - half)
        t = max(0, cy - half)
        r = min(w, l + side)
        b = min(h, t + side)
        return l, t, r, b

    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    # expand patch indices to pixel edges (each patch is 8px in 224 space)
    px0, px1 = x0 * _PATCH, x1 * _PATCH
    py0, py1 = y0 * _PATCH, y1 * _PATCH
    cw = px1 - px0
    ch = py1 - py0
    side = int(max(cw, ch) * margin)
    cx = (px0 + px1) // 2
    cy = (py0 + py1) // 2
    half = side // 2
    l = max(0, cx - half)
    t = max(0, cy - half)
    r = min(w, l + side)
    b = min(h, t + side)
    # if clamping shrunk square, shift back
    if r - l < side:
        l = max(0, r - side)
        r = min(w, l + side)
    if b - t < side:
        t = max(0, b - side)
        b = min(h, t + side)
    return l, t, r, b


def process_one_pil(pil_224: Image.Image, attn28: np.ndarray, fg_quantile: float) -> Image.Image:
    """pil_224: RGB 224×224. attn28: 28×28."""
    thr = np.quantile(attn28, fg_quantile)
    mask = attn28 >= thr
    l, t, r, b = square_bbox_from_mask(
        np.kron(mask, np.ones((_PATCH, _PATCH), dtype=bool)),
        DINO_SIZE,
        DINO_SIZE,
    )
    cropped = pil_224.crop((l, t, r, b))
    return cropped.resize((TARGET, TARGET), Image.Resampling.LANCZOS)


def load_extractor(config_path: Path | None):
    import yaml

    p = config_path or Path("configs/config.yaml")
    with open(p) as f:
        cfg = yaml.safe_load(f)
    d = cfg["dino"]
    return DinoExtractor(
        model_name=d["model"],
        device=d["device"],
        image_size=d["image_size"],
        use_multilayer=d.get("use_multilayer", False),
    )


def geometry_224_from_path(path: Path) -> Image.Image:
    """Same spatial pipeline as DinoExtractor: Resize(256) → CenterCrop(224)."""
    img = Image.open(path).convert("RGB")
    return T.Compose([T.Resize(256), T.CenterCrop(DINO_SIZE)])(img)


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )


def main():
    ap = argparse.ArgumentParser(
        description="DINO-guided square crop + 128×128 resize (train/ + val/)."
    )
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument(
        "--fg-quantile",
        type=float,
        default=0.75,
        help="Patches with attn ≥ this quantile define the subject bbox (aligns with fg_threshold style).",
    )
    args = ap.parse_args()

    input_root = args.input.resolve()
    output_root = args.output.resolve()
    if not input_root.is_dir():
        raise SystemExit(f"Missing {input_root}")

    for sp in SPLITS:
        if not (input_root / sp).is_dir():
            raise SystemExit(f"Expected {input_root}/train/ and {input_root}/val/")

    extractor = load_extractor(args.config)

    # Same normalization as extractor for attention
    tensorize = T.Compose([
        T.ToTensor(),
        T.Normalize(DinoExtractor.MEAN, DinoExtractor.STD),
    ])

    n_ok = n_fail = 0
    for split in SPLITS:
        base_in = input_root / split
        base_out = output_root / split
        for class_dir in sorted(base_in.iterdir()):
            if not class_dir.is_dir():
                continue
            cname = class_dir.name
            for img_path in tqdm(list_images(class_dir), desc=f"{split}/{cname}"):
                try:
                    pil_224 = geometry_224_from_path(img_path)
                    t = tensorize(pil_224).unsqueeze(0).to(extractor.device)
                    attn28 = cls_attn_28(extractor, t)
                    out_pil = process_one_pil(pil_224, attn28, args.fg_quantile)
                    dest = base_out / cname / img_path.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    out_pil.save(dest, format="PNG")
                    n_ok += 1
                except Exception as e:
                    print(f"\n  skip {img_path}: {e}")
                    n_fail += 1

    print(f"\nDone. saved={n_ok}  failed={n_fail}  → {output_root}")
    print(f"Point dataset paths at: {output_root / 'train'} (and use val/ for held-out if needed)")


if __name__ == "__main__":
    main()
