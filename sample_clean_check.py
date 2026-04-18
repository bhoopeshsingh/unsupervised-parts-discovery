"""
Quick visual check — saves 5 clean-background samples per class
and produces a contact sheet: samples_clean/contact_sheet.png

Run this BEFORE the full build_clean_dataset.py to confirm quality.

    pip install torch torchvision datasets rembg Pillow numpy onnxruntime
    python sample_clean_check.py
"""

import os, io
import numpy as np
from PIL import Image, ImageDraw

IMG_SIZE     = (128, 128)
SAMPLES      = 5
FG_THRESHOLD = 0.70
REMBG_MODEL  = "u2netp"
OUT_DIR      = "samples_clean"

os.makedirs(OUT_DIR, exist_ok=True)


def composite_on_white(rgba):
    bg = Image.new("RGB", rgba.size, (255, 255, 255))
    bg.paste(rgba, mask=rgba.split()[3])
    return bg

def fg_ratio(rgba):
    alpha = np.array(rgba.split()[3])
    return (alpha > 10).sum() / alpha.size

def rembg_clean(img_pil):
    from rembg import remove
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    out = remove(buf.getvalue(), model=REMBG_MODEL)
    rgba = Image.open(io.BytesIO(out)).convert("RGBA")
    return rgba if fg_ratio(rgba) >= FG_THRESHOLD else None


# ── CATS ─────────────────────────────────────────────────────────────────────
print("\nSampling cats (Oxford IIIT Pet + trimap)…")
from torchvision.datasets import OxfordIIITPet
pet_ds = OxfordIIITPet(root="/tmp/oxford_pet", split="trainval",
                        target_types=["category", "segmentation"], download=True)
CAT_CLASSES = {
    i for i, name in enumerate(pet_ds.classes)
    if any(c in name for c in [
        "Abyssinian","Bengal","Birman","Bombay","British_Shorthair",
        "Egyptian_Mau","Maine_Coon","Persian","Ragdoll",
        "Russian_Blue","Siamese","Sphynx"
    ])
}
cat_dir = os.path.join(OUT_DIR, "cat"); os.makedirs(cat_dir, exist_ok=True)
saved = 0
for img_pil, (label, seg_pil) in pet_ds:
    if saved >= SAMPLES: break
    if label not in CAT_CLASSES: continue
    seg = np.array(seg_pil)
    fg_mask = (seg == 1).astype(np.uint8) * 255
    ratio = fg_mask.sum() / (255 * fg_mask.size)
    if ratio < FG_THRESHOLD: continue
    img_np = np.array(img_pil.convert("RGB"))
    alpha  = Image.fromarray(fg_mask).resize(img_pil.size, Image.NEAREST)
    rgba   = img_pil.convert("RGBA"); rgba.putalpha(alpha)
    out    = composite_on_white(rgba).resize(IMG_SIZE, Image.BICUBIC)
    out.save(os.path.join(cat_dir, f"cat_{saved}.png")); saved += 1
print(f"  cats: {saved} samples saved")


# ── CARS ─────────────────────────────────────────────────────────────────────
print("\nSampling cars (Stanford Cars + rembg)…")
from datasets import load_dataset
cars_ds = load_dataset("tanganke/stanford_cars", split="train", trust_remote_code=True)
car_dir = os.path.join(OUT_DIR, "car"); os.makedirs(car_dir, exist_ok=True)
saved = 0
for ex in cars_ds:
    if saved >= SAMPLES: break
    img = ex["image"].convert("RGB") if isinstance(ex["image"], Image.Image) else Image.fromarray(ex["image"]).convert("RGB")
    rgba = rembg_clean(img)
    if rgba is None: continue
    composite_on_white(rgba).resize(IMG_SIZE, Image.BICUBIC).save(
        os.path.join(car_dir, f"car_{saved}.png")); saved += 1
    print(f"  car sample {saved} done")
print(f"  cars: {saved} samples saved")


# ── BIRDS ─────────────────────────────────────────────────────────────────────
print("\nSampling birds (CUB-200-2011 + rembg)…")
birds_ds = load_dataset("alkzar90/CUB-200-2011", split="train", trust_remote_code=True)
bird_dir = os.path.join(OUT_DIR, "bird"); os.makedirs(bird_dir, exist_ok=True)
saved = 0
for ex in birds_ds:
    if saved >= SAMPLES: break
    img = ex["image"].convert("RGB") if isinstance(ex["image"], Image.Image) else Image.fromarray(ex["image"]).convert("RGB")
    rgba = rembg_clean(img)
    if rgba is None: continue
    composite_on_white(rgba).resize(IMG_SIZE, Image.BICUBIC).save(
        os.path.join(bird_dir, f"bird_{saved}.png")); saved += 1
    print(f"  bird sample {saved} done")
print(f"  birds: {saved} samples saved")


# ── Contact sheet ─────────────────────────────────────────────────────────────
print("\nBuilding contact sheet…")
PAD = 8; LABEL_H = 22
W = SAMPLES * (IMG_SIZE[0] + PAD) + PAD
H = 3 * (IMG_SIZE[1] + PAD + LABEL_H) + PAD
sheet = Image.new("RGB", (W, H), (240, 240, 240))
draw  = ImageDraw.Draw(sheet)

for row, cls in enumerate(["cat", "car", "bird"]):
    y_top = PAD + row * (IMG_SIZE[1] + PAD + LABEL_H)
    draw.text((PAD, y_top + 3), cls.upper(), fill=(40, 40, 40))
    for col in range(SAMPLES):
        p = os.path.join(OUT_DIR, cls, f"{cls}_{col}.png")
        if not os.path.exists(p): continue
        img = Image.open(p)
        x = PAD + col * (IMG_SIZE[0] + PAD)
        y = y_top + LABEL_H
        sheet.paste(img, (x, y))

sheet_path = os.path.join(OUT_DIR, "contact_sheet.png")
sheet.save(sheet_path)
print(f"\n✓ Contact sheet saved → {sheet_path}")
print("  Open it to confirm foreground quality before running build_clean_dataset.py")
