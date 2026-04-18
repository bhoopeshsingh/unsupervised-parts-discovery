#!/usr/bin/env python3
"""
Select top-fraction images per class (train + val) from a source tree.

**Default (`--method rembg`)** — what “70% foreground” usually means
---------------------------------------------------------------------
Uses **rembg** (U-Net segmentation) to build a **foreground mask**, then
``subject_fraction = (# pixels with opaque alpha) / (H×W)``. This is **pixel
area of the segmented subject**, not DINO attention.

DINO CLS attention counts “patches the model looked at”; a small bird on grass
or sky can still show many high-attention patches — **it is the wrong signal
for “subject fills the frame.”**

**Legacy (`--method dino`)** — old behaviour
--------------------------------------------
Patch attention thresholds; kept for experiments only, not subject area %.

Defaults
--------
  INPUT:  data/v2/images_core/{train,val}/{class}/...
  OUTPUT: data/v2/images/{train,val}/...   (cleared first)

Examples
--------
  python scripts/select_top_quality_images.py \\
      --input data/v2/images_core --output data/v2/images \\
      --min-subject-fraction 0.70 --top-fraction 0.30

  # keep every image that already has ≥70% subject pixels (no second top-k)
  python scripts/select_top_quality_images.py --top-fraction 1.0

Performance (rembg)
-------------------
Always pass a **reused session** (the script does this). Without it, each image
reloads the model and throughput collapses. Optional: ``--rembg-model u2netp``
(smaller/faster) or ``--rembg-max-side 512`` to downscale before segmentation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.models.dino_extractor import DinoExtractor

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val")
N_PATCHES = 784


# --- rembg: subject area in the image (pixel fraction) ------------------------
#
# IMPORTANT: pass the same `session` from `rembg.new_session()` for every image.
# Calling `remove(bytes)` without `session=` reloads the ONNX model **per image**
# and is unusable at scale (hours per thousand images).


def subject_fraction_rembg(
    path: Path,
    session,
    alpha_threshold: int = 128,
    max_side: int | None = None,
) -> float:
    """
    Fraction of pixels classified as foreground by rembg (opaque in output alpha).
    This approximates “how much of the frame is the subject / salient object.”

    Args:
        session:  Return value of ``rembg.new_session(model_name)`` — reuse one
                  session for all images in a run.
        max_side: If set, downscale so the longer edge is at most this (faster).
    """
    from rembg import remove

    img = Image.open(path).convert("RGB")
    if max_side is not None:
        w, h = img.size
        m = max(w, h)
        if m > max_side:
            s = max_side / float(m)
            img = img.resize((int(w * s), int(h * s)), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    out_bytes = remove(buf.getvalue(), session=session)
    out = Image.open(BytesIO(out_bytes)).convert("RGBA")
    alpha = np.asarray(out.split()[3], dtype=np.uint8)
    return float((alpha > alpha_threshold).mean())


# --- DINO attention (legacy; NOT subject area %) ----------------------------


def cls_attn_vector(extractor: DinoExtractor, img_tensor):
    attn = extractor.extract_attention(img_tensor)
    return attn[0, :, 0, 1:].mean(dim=0).cpu()


def foreground_patch_ratio_rel_to_max(cls_attn, min_attn_frac_of_max: float) -> float:
    mx = max(float(cls_attn.max()), 1e-12)
    thr = min_attn_frac_of_max * mx
    return float((cls_attn >= thr).float().mean())


def background_patch_ratio(cls_attn, min_attn_frac_of_max: float) -> float:
    return 1.0 - foreground_patch_ratio_rel_to_max(cls_attn, min_attn_frac_of_max)


def attention_peakiness(cls_attn) -> float:
    mx = float(cls_attn.max())
    if mx <= 1e-12:
        return 0.0
    med = float(cls_attn.median())
    return (mx - med) / mx


def softmax_entropy_norm(cls_attn) -> float:
    import math as _math

    p = F.softmax(cls_attn.float().flatten(), dim=0).clamp_min(1e-12)
    h = float(-(p * p.log()).sum())
    return h / _math.log(N_PATCHES)


def quality_score_dino(cls_attn, min_attn_frac_of_max: float) -> float:
    fg = foreground_patch_ratio_rel_to_max(cls_attn, min_attn_frac_of_max)
    peak = attention_peakiness(cls_attn)
    entn = softmax_entropy_norm(cls_attn)
    return float(fg * peak * max(0.0, 1.0 - entn))


def list_images(class_dir: Path) -> list[Path]:
    return [
        p
        for p in sorted(class_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED
    ]


def load_extractor_from_config(config_path: Path | None, finetune_path: Path | None):
    import yaml

    cfg_path = config_path or Path("configs/config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    d = cfg["dino"]
    ext = DinoExtractor(
        model_name=d["model"],
        device=d["device"],
        image_size=d["image_size"],
        use_multilayer=d.get("use_multilayer", False),
    )
    if finetune_path and finetune_path.is_file():
        from src.models.dino_finetuner import DinoSemanticFinetuner

        DinoSemanticFinetuner.load_weights_into_extractor(ext, str(finetune_path))
        print(f"Loaded fine-tuned weights: {finetune_path}")
    return ext


def clear_split_trees(output_root: Path, splits: tuple[str, ...]) -> None:
    for sp in splits:
        p = output_root / sp
        if p.is_dir():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(
        description="Filter + rank images per class. Default: rembg subject-area fraction; "
        "optional legacy DINO attention mode."
    )
    ap.add_argument(
        "--method",
        choices=("rembg", "dino"),
        default="rembg",
        help="rembg = pixel fraction of segmented subject (default). "
        "dino = CLS attention patches (NOT subject area %%).",
    )
    ap.add_argument("--input", type=Path, default=Path("data/v2/images_core"))
    ap.add_argument("--output", type=Path, default=Path("data/v2/images"))
    ap.add_argument(
        "--top-fraction",
        type=float,
        default=0.30,
        help="After gating, keep this fraction per class per split (1.0 = keep all that pass).",
    )
    # rembg
    ap.add_argument(
        "--min-subject-fraction",
        type=float,
        default=0.70,
        help="[rembg] Require at least this fraction of pixels as foreground (0–1).",
    )
    ap.add_argument(
        "--alpha-threshold",
        type=int,
        default=128,
        help="[rembg] Alpha ≥ this counts as foreground pixel.",
    )
    ap.add_argument(
        "--rembg-model",
        type=str,
        default="u2net",
        metavar="NAME",
        help="[rembg] Model passed to new_session(), e.g. u2net (default) or u2netp (faster, lighter).",
    )
    ap.add_argument(
        "--rembg-max-side",
        type=int,
        default=None,
        metavar="PX",
        help="[rembg] If set, resize so max(width,height) ≤ this before segmentation (faster).",
    )
    # dino legacy
    ap.add_argument(
        "--max-bg-fraction",
        type=float,
        default=0.20,
        help="[dino only] Reject if low-attention patch fraction exceeds this.",
    )
    ap.add_argument(
        "--min-attn-frac-of-max",
        type=float,
        default=0.20,
        help="[dino only] Patch is high-attention if attn ≥ this × max.",
    )
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--finetune-weights", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()
    if not (0.0 < args.top_fraction <= 1.0):
        raise SystemExit("--top-fraction must be in (0, 1]")
    if args.method == "rembg":
        if not (0.0 <= args.min_subject_fraction <= 1.0):
            raise SystemExit("--min-subject-fraction must be in [0, 1]")
    else:
        if not (0.0 <= args.max_bg_fraction < 1.0):
            raise SystemExit("--max-bg-fraction must be in [0, 1)")

    input_root = args.input.resolve()
    output_root = args.output.resolve()
    if not input_root.is_dir():
        raise SystemExit(f"Input not found: {input_root}")

    for sp in SPLITS:
        p = input_root / sp
        if not p.is_dir():
            raise SystemExit(
                f"Missing {p}\nExpected {input_root}/train/ and {input_root}/val/ "
                "with class folders inside each."
            )

    extractor = None
    if args.method == "dino":
        finetune = args.finetune_weights
        if finetune is None and args.config:
            import yaml

            with open(args.config) as f:
                c = yaml.safe_load(f)
            cand = Path(c.get("finetune", {}).get("save_path", "cache/dino_finetuned.pt"))
            if cand.is_file():
                finetune = cand
        extractor = load_extractor_from_config(args.config, finetune)
        print(
            f"[dino] Gate: BG patch fraction ≤ {args.max_bg_fraction}; "
            f"then top {args.top_fraction:.0%} by attention-based score (not subject area %%)."
        )
    else:
        print(
            f"[rembg] Gate: subject pixel fraction ≥ {args.min_subject_fraction:.0%} "
            f"(segmentation mask); then top {args.top_fraction:.0%} per class by subject fraction."
        )

    rembg_session = None
    if args.method == "rembg":
        from rembg import new_session

        print(
            f"  Loading rembg once: new_session({args.rembg_model!r}) — "
            "reused for every image (do not call remove() without session)."
        )
        rembg_session = new_session(args.rembg_model)
        if args.rembg_max_side:
            print(f"  [rembg] max_side={args.rembg_max_side}px before segmentation")

    if not args.dry_run:
        clear_split_trees(output_root, SPLITS)
    else:
        print("(dry-run: would clear and recreate output train/ and val/)")

    summary: dict = {
        "method": args.method,
        "input": str(input_root),
        "output": str(output_root),
        "top_fraction": args.top_fraction,
        "per_split_class": [],
    }
    if args.method == "rembg":
        summary["min_subject_fraction"] = args.min_subject_fraction
        summary["alpha_threshold"] = args.alpha_threshold
        summary["rembg_model"] = args.rembg_model
        summary["rembg_max_side"] = args.rembg_max_side
    else:
        summary["max_bg_fraction"] = args.max_bg_fraction
        summary["min_attn_frac_of_max"] = args.min_attn_frac_of_max

    total_kept = total_pool = 0

    for split in SPLITS:
        base_in = input_root / split
        base_out = output_root / split

        for class_dir in sorted(base_in.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            imgs = list_images(class_dir)
            if not imgs:
                continue

            scored: list[tuple[float, Path]] = []
            rejected = 0

            for img_path in tqdm(imgs, desc=f"{split}/{class_name}"):
                try:
                    if args.method == "rembg":
                        assert rembg_session is not None
                        frac = subject_fraction_rembg(
                            img_path,
                            rembg_session,
                            alpha_threshold=args.alpha_threshold,
                            max_side=args.rembg_max_side,
                        )
                        if frac < args.min_subject_fraction - 1e-6:
                            rejected += 1
                            continue
                        scored.append((frac, img_path))
                    else:
                        assert extractor is not None
                        tensor = extractor.load_image(str(img_path))
                        cls_attn = cls_attn_vector(extractor, tensor)
                        bg = background_patch_ratio(cls_attn, args.min_attn_frac_of_max)
                        if bg > args.max_bg_fraction + 1e-6:
                            rejected += 1
                            continue
                        q = quality_score_dino(cls_attn, args.min_attn_frac_of_max)
                        scored.append((q, img_path))
                except Exception:
                    rejected += 1

            pool = len(scored)
            total_pool += pool
            if pool == 0:
                summary["per_split_class"].append(
                    {
                        "split": split,
                        "class": class_name,
                        "pool_after_gate": 0,
                        "kept": 0,
                        "rejected": rejected,
                    }
                )
                print(f"  Warning: no images passed gate for {split}/{class_name}")
                continue

            k = max(1, math.ceil(pool * args.top_fraction))
            k = min(k, pool)
            scored.sort(key=lambda t: t[0], reverse=True)
            chosen = scored[:k]

            row = {
                "split": split,
                "class": class_name,
                "raw_images": len(imgs),
                "pool_after_gate": pool,
                "rejected": rejected,
                "kept": len(chosen),
            }
            summary["per_split_class"].append(row)
            total_kept += len(chosen)

            if not args.dry_run:
                out_class = base_out / class_name
                out_class.mkdir(parents=True, exist_ok=True)
                for _, src in chosen:
                    dest = out_class / src.name
                    shutil.copy2(src, dest)

    reports_dir = output_root / "_reports"
    if not args.dry_run:
        reports_dir.mkdir(parents=True, exist_ok=True)
        with open(reports_dir / "top_quality_summary.json", "w") as f:
            json.dump(
                {**summary, "total_kept": total_kept, "total_pool_after_gate": total_pool},
                f,
                indent=2,
            )

    print(
        f"\nDone. Passed gate: {total_pool}  →  kept (top {args.top_fraction:.0%}): {total_kept}"
        + ("  (dry-run: no files written)" if args.dry_run else "")
    )
    if not args.dry_run:
        print(f"Wrote: {output_root / 'train'}  and  {output_root / 'val'}")
        print(f"Summary: {reports_dir / 'top_quality_summary.json'}")
        print(f"Set dino.data_root to: {output_root / 'train'}")


if __name__ == "__main__":
    main()
