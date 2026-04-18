"""
Build a clean 128×128 dataset for cat / car / bird with ≥70% foreground.

Sources
-------
  cats  → Oxford IIIT Pet  (torchvision)  — uses provided trimap masks, no ML needed
  cars  → tanganke/stanford_cars          (HuggingFace datasets)
  birds → alkzar90/CUB-200-2011           (HuggingFace datasets)

Background removal
------------------
  cats  → trimap mask (fg pixel ratio check built-in)
  cars  → rembg u2netp model (fast CPU model, ~0.3 s/image)
  birds → rembg u2netp model

Output
------
  data_clean/
    train/ {cat, car, bird}/   up to MAX_TRAIN images each
    test/  {cat, car, bird}/   up to MAX_TEST  images each

  White background PNG at 128×128.
  Images that don't reach FG_THRESHOLD are silently skipped.

Install
-------
  pip install torch torchvision datasets rembg Pillow numpy onnxruntime
"""

import os
import sys
import io
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR    = "data_clean"
IMG_SIZE      = (128, 128)
MAX_TRAIN     = 4000
MAX_TEST      = 1000
FG_THRESHOLD  = 0.70          # minimum foreground fraction
REMBG_MODEL   = "u2netp"      # fastest CPU model; use "isnet-general-use" for better quality
NUM_WORKERS   = 4             # parallel workers for rembg

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def composite_on_white(rgba: Image.Image) -> Image.Image:
    """Flatten RGBA onto a white background."""
    bg = Image.new("RGB", rgba.size, (255, 255, 255))
    bg.paste(rgba, mask=rgba.split()[3])
    return bg


def foreground_ratio(rgba: Image.Image) -> float:
    """Fraction of pixels that are NOT background (alpha > 10)."""
    alpha = np.array(rgba.split()[3])
    return (alpha > 10).sum() / alpha.size


def save_image(img: Image.Image, out_path: str):
    img.resize(IMG_SIZE, Image.BICUBIC).save(out_path)


# ── CATS via Oxford IIIT Pet trimaps ─────────────────────────────────────────
def build_cats(max_train=MAX_TRAIN, max_test=MAX_TEST):
    from torchvision.datasets import OxfordIIITPet

    print("\n[cats] Downloading Oxford IIIT Pet…")
    for split, max_n in [("trainval", max_train), ("test", max_test)]:
        out_split = "train" if split == "trainval" else "test"
        ds = OxfordIIITPet(
            root="/tmp/oxford_pet",
            split=split,
            target_types=["category", "segmentation"],
            download=True,
        )

        out_dir = os.path.join(OUTPUT_DIR, out_split, "cat")
        os.makedirs(out_dir, exist_ok=True)
        saved = 0

        # Species: 0–18 are cats, 19–36 are dogs (alphabetical order in dataset)
        # OxfordIIITPet has 37 breeds; indices 0–11 are Cat breeds
        CAT_CLASSES = {
            i for i, name in enumerate(ds.classes)
            if any(c in name for c in [
                "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
                "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll",
                "Russian_Blue", "Siamese", "Sphynx"
            ])
        }

        for img_pil, (label, seg_pil) in ds:
            if saved >= max_n:
                break
            if label not in CAT_CLASSES:
                continue

            # Trimap: 1 = foreground, 2 = background, 3 = boundary
            seg = np.array(seg_pil)
            fg_mask = (seg == 1).astype(np.uint8) * 255
            ratio = fg_mask.sum() / (255 * fg_mask.size)
            if ratio < FG_THRESHOLD:
                continue

            # Apply mask
            img_np  = np.array(img_pil.convert("RGB"))
            alpha   = Image.fromarray(fg_mask)
            rgba    = Image.fromarray(img_np).convert("RGBA")
            rgba.putalpha(alpha.resize(img_pil.size, Image.NEAREST))
            out_img = composite_on_white(rgba)

            out_path = os.path.join(out_dir, f"cat_{saved:05d}.png")
            save_image(out_img, out_path)
            saved += 1

        print(f"  [cats/{out_split}] saved {saved}")


# ── Generic rembg builder ─────────────────────────────────────────────────────
def remove_bg_and_save(args):
    """Worker: remove background from one PIL image, return (rgba, ratio) or None."""
    from rembg import remove
    img_pil, out_path = args
    buf_in  = io.BytesIO()
    img_pil.save(buf_in, format="PNG")
    buf_out = remove(buf_in.getvalue(), model=REMBG_MODEL)
    rgba    = Image.open(io.BytesIO(buf_out)).convert("RGBA")
    ratio   = foreground_ratio(rgba)
    if ratio < FG_THRESHOLD:
        return None
    result = composite_on_white(rgba)
    result.resize(IMG_SIZE, Image.BICUBIC).save(out_path)
    return out_path


def build_hf_class(hf_dataset_name: str, cls_name: str,
                   image_col: str = "image",
                   split_map: dict = None,
                   filter_fn=None,
                   max_train=MAX_TRAIN, max_test=MAX_TEST):
    """
    Generic builder for a HuggingFace image dataset.
    split_map: {"train": hf_split_name, "test": hf_split_name}
    filter_fn: optional callable(example) -> bool
    """
    from datasets import load_dataset

    if split_map is None:
        split_map = {"train": "train", "test": "test"}

    print(f"\n[{cls_name}] Loading {hf_dataset_name}…")

    for out_split, hf_split in split_map.items():
        max_n   = max_train if out_split == "train" else max_test
        out_dir = os.path.join(OUTPUT_DIR, out_split, cls_name)
        os.makedirs(out_dir, exist_ok=True)

        try:
            ds = load_dataset(hf_dataset_name, split=hf_split, trust_remote_code=True)
        except Exception as e:
            print(f"  [{cls_name}/{out_split}] failed to load: {e}")
            continue

        # Build a work queue
        queue   = []
        counter = 0
        for ex in ds:
            if counter >= max_n * 3:   # oversample; rembg will filter some out
                break
            if filter_fn and not filter_fn(ex):
                continue
            img = ex[image_col]
            if not isinstance(img, Image.Image):
                try:
                    img = Image.fromarray(img)
                except Exception:
                    continue
            out_path = os.path.join(out_dir, f"{cls_name}_{counter:05d}.png")
            queue.append((img.convert("RGB"), out_path))
            counter += 1

        print(f"  [{cls_name}/{out_split}] processing {len(queue)} candidates with rembg…")

        saved = 0
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex_pool:
            futures = {ex_pool.submit(remove_bg_and_save, item): item for item in queue}
            for fut in as_completed(futures):
                result = fut.result()
                if result:
                    saved += 1
                if saved >= max_n:
                    ex_pool.shutdown(wait=False, cancel_futures=True)
                    break

        print(f"  [{cls_name}/{out_split}] saved {saved}")


# ── CARS ─────────────────────────────────────────────────────────────────────
def build_cars():
    build_hf_class(
        hf_dataset_name = "tanganke/stanford_cars",
        cls_name        = "car",
        image_col       = "image",
        split_map       = {"train": "train", "test": "test"},
    )


# ── BIRDS ─────────────────────────────────────────────────────────────────────
def build_birds():
    build_hf_class(
        hf_dataset_name = "alkzar90/CUB-200-2011",
        cls_name        = "bird",
        image_col       = "image",
        split_map       = {"train": "train", "test": "test"},
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Clean dataset builder  —  128×128 px, ≥70% foreground")
    print(f"Target: {MAX_TRAIN} train + {MAX_TEST} test per class")
    print("=" * 60)

    build_cats()
    build_cars()
    build_birds()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n── Final counts ──────────────────────────────────────────")
    total = 0
    for split in ("train", "test"):
        for cls in ("cat", "car", "bird"):
            path = os.path.join(OUTPUT_DIR, split, cls)
            n = len([f for f in os.listdir(path) if f.endswith(".png")]) if os.path.isdir(path) else 0
            print(f"  {split:5s}/{cls:4s}: {n:4d}")
            total += n
    print(f"  Total : {total}")
    print(f"\n✓ Dataset ready at ./{OUTPUT_DIR}/")
    print(f"  White background PNG, {IMG_SIZE[0]}×{IMG_SIZE[1]} px, ≥{FG_THRESHOLD*100:.0f}% foreground")


if __name__ == "__main__":
    main()
