#!/usr/bin/env python3
"""
Filter images by DINO CLS-attention — standalone (not the pipeline).

What you get (the “dataset”)
----------------------------
  OUTPUT/train/<class>/*.png
  OUTPUT/val/<class>/*.png

That is all the pipeline needs: set ``dino.data_root`` to ``OUTPUT/train``.
No JSON is required for training.

Optional diagnostics (off by default for huge runs)
-----------------------------------------------------
  OUTPUT/_reports/summary.json     — tiny file: counts + filter thresholds only (written by default)
  --full-report                     — also writes per-image details to OUTPUT/_reports/per_image.json

Why extra checks besides fg_ratio
---------------------------------
Near-uniform CLS attention makes “fraction ≥ k×max” pass for almost every patch.
The script also gates on entropy, max/mean, and peakiness (see --help).

Example
-------
  python scripts/filter_foreground_dataset.py \\
      --input data/v2/images \\
      --output data/v2/images_clean \\
      --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn.functional as F
from tqdm import tqdm

from src.models.dino_extractor import DinoExtractor

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SPLITS = ("train", "val")
N_PATCHES = 784


def cls_attn_vector(extractor: DinoExtractor, img_tensor):
    attn = extractor.extract_attention(img_tensor)
    return attn[0, :, 0, 1:].mean(dim=0).cpu()


def foreground_patch_ratio_rel_to_max(cls_attn, min_attn_frac_of_max: float) -> float:
    mx = max(float(cls_attn.max()), 1e-12)
    thr = min_attn_frac_of_max * mx
    return float((cls_attn >= thr).float().mean())


def attention_peakiness(cls_attn) -> float:
    mx = float(cls_attn.max())
    if mx <= 1e-12:
        return 0.0
    med = float(cls_attn.median())
    return (mx - med) / mx


def softmax_entropy_norm(cls_attn) -> float:
    """
    H(p) / log(N) with p = softmax(cls_attn). ~1.0 = nearly uniform (unreliable);
    lower = more peaked / structured attention.
    """
    p = F.softmax(cls_attn.float().flatten(), dim=0).clamp_min(1e-12)
    h = float(-(p * p.log()).sum())
    return h / math.log(N_PATCHES)


def max_over_mean(cls_attn) -> float:
    """max/mean; ~1.0 if all patches equal (uniform), larger if some patches dominate."""
    m = float(cls_attn.mean())
    if m <= 1e-12:
        return 0.0
    return float(cls_attn.max()) / m


def evaluate_image(
    cls_attn,
    min_fg_ratio: float,
    min_attn_frac_of_max: float,
    min_peakiness: float,
    max_entropy_norm: float,
    min_max_over_mean: float,
):
    """
    Returns (kept: bool, reason: str, metrics: dict).
    """
    fg_ratio = foreground_patch_ratio_rel_to_max(cls_attn, min_attn_frac_of_max)
    peak = attention_peakiness(cls_attn)
    entn = softmax_entropy_norm(cls_attn)
    mx_mn = max_over_mean(cls_attn)

    metrics = {
        "fg_ratio": round(fg_ratio, 4),
        "peakiness": round(peak, 4),
        "entropy_norm": round(entn, 4),
        "max_over_mean": round(mx_mn, 4),
    }

    if entn > max_entropy_norm:
        return False, "high_entropy_uniform", metrics
    if mx_mn < min_max_over_mean:
        return False, "low_max_over_mean", metrics
    if peak < min_peakiness:
        return False, "low_peakiness", metrics
    if fg_ratio < min_fg_ratio:
        return False, "low_fg_ratio", metrics
    return True, "ok", metrics


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


def main():
    ap = argparse.ArgumentParser(
        description="Copy images that pass the foreground-coverage score; "
        "both train/ and val/ are processed."
    )
    ap.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Dataset root containing train/ and val/ (each with class subfolders).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output root; will contain train/ and val/ mirroring --input.",
    )
    ap.add_argument("--min-fg-ratio", type=float, default=0.7)
    ap.add_argument(
        "--min-attn-frac-of-max",
        type=float,
        default=0.20,
        help="Patch counts as foreground-like if attn ≥ this × max(CLS attention).",
    )
    ap.add_argument(
        "--min-peakiness",
        type=float,
        default=0.08,
        help="Require (max−median)/max ≥ this; fails flat maps.",
    )
    ap.add_argument(
        "--max-entropy-norm",
        type=float,
        default=0.96,
        help="Reject if softmax entropy / log(784) exceeds this (≈1 = uniform noise).",
    )
    ap.add_argument(
        "--min-max-over-mean",
        type=float,
        default=1.35,
        help="Reject if max(attn)/mean(attn) below this (~1 = uniform).",
    )
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--finetune-weights", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true", help="Report only; no file copies.")
    ap.add_argument(
        "--full-report",
        action="store_true",
        help="Write per-image rows to OUTPUT/_reports/per_image.json (can be very large).",
    )
    ap.add_argument(
        "--no-summary-json",
        action="store_true",
        help="Do not write _reports/summary.json (only folders + terminal output).",
    )
    args = ap.parse_args()

    input_root = args.input.resolve()
    output_root = args.output.resolve()
    if not input_root.is_dir():
        raise SystemExit(f"Input not found: {input_root}")

    for sp in SPLITS:
        p = input_root / sp
        if not p.is_dir():
            raise SystemExit(
                f"Missing {p}\n"
                f"Expected both {input_root}/train/ and {input_root}/val/ "
                f"(class folders inside each)."
            )

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
        "Filters (all must pass): "
        f"fg_ratio≥{args.min_fg_ratio} @ {args.min_attn_frac_of_max}×max, "
        f"peakiness≥{args.min_peakiness}, "
        f"entropy_norm≤{args.max_entropy_norm}, "
        f"max/mean≥{args.min_max_over_mean}"
    )

    # Per-image rows are large; only collected when --full-report
    report_rows: list[dict] | None = [] if args.full_report else None
    n_kept = n_drop = 0

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

            for img_path in tqdm(imgs, desc=f"{split}/{class_name}"):
                try:
                    tensor = extractor.load_image(str(img_path))
                    cls_attn = cls_attn_vector(extractor, tensor)
                    ok, reason, metrics = evaluate_image(
                        cls_attn,
                        min_fg_ratio=args.min_fg_ratio,
                        min_attn_frac_of_max=args.min_attn_frac_of_max,
                        min_peakiness=args.min_peakiness,
                        max_entropy_norm=args.max_entropy_norm,
                        min_max_over_mean=args.min_max_over_mean,
                    )
                    if report_rows is not None:
                        report_rows.append(
                            {
                                "path": str(img_path),
                                "class": class_name,
                                "split": split,
                                "kept": ok,
                                "reason": reason,
                                **metrics,
                            }
                        )

                    if ok:
                        n_kept += 1
                        if not args.dry_run:
                            dest = base_out / class_name / img_path.name
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(img_path, dest)
                    else:
                        n_drop += 1
                except Exception as e:
                    if report_rows is not None:
                        report_rows.append(
                            {
                                "path": str(img_path),
                                "class": class_name,
                                "split": split,
                                "error": str(e),
                                "kept": False,
                            }
                        )
                    n_drop += 1

    reports_dir = output_root / "_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_meta = {
        "input": str(input_root),
        "output": str(output_root),
        "pipeline_hint": f"Set configs/config.yaml dino.data_root to: {output_root / 'train'}",
        "splits": list(SPLITS),
        "min_fg_ratio": args.min_fg_ratio,
        "min_attn_frac_of_max": args.min_attn_frac_of_max,
        "min_peakiness": args.min_peakiness,
        "max_entropy_norm": args.max_entropy_norm,
        "min_max_over_mean": args.min_max_over_mean,
        "dry_run": args.dry_run,
        "kept": n_kept,
        "dropped": n_drop,
    }

    if not args.no_summary_json:
        with open(reports_dir / "summary.json", "w") as f:
            json.dump(summary_meta, f, indent=2)

    if args.full_report and report_rows is not None:
        with open(reports_dir / "per_image.json", "w") as f:
            json.dump({**summary_meta, "images": report_rows}, f, indent=2)

    print(
        f"\nDone. kept={n_kept}  dropped={n_drop}"
        + ("  (dry-run: no copies)" if args.dry_run else "")
    )
    print(f"Clean image folders: {output_root / 'train'}  and  {output_root / 'val'}")
    if not args.no_summary_json:
        print(f"Summary (counts only): {reports_dir / 'summary.json'}")
    if args.full_report:
        print(f"Per-image report: {reports_dir / 'per_image.json'}")


if __name__ == "__main__":
    main()
