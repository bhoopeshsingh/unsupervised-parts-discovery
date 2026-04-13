"""
Ablation Study: Gap-Grounded Evaluation of the DINO Concept Discovery Pipeline
===============================================================================

Runs 4 controlled variants to generate evidence for 3 paper gap claims:

  Variant A — FULL    : multi-layer (L8+L10+L12) + FG masking + fine-tuned DINO  [baseline]
  Variant B — SINGLE  : single-layer (L12 only)  + FG masking + fine-tuned DINO  [vs A → DINO gap]
  Variant C — NO_MASK : multi-layer              + no FG mask  + fine-tuned DINO  [vs A → ACE gap]
  Variant D — PRETRAIN: multi-layer              + FG masking  + base DINO        [vs A → finetune value]

Paper gap linkage
-----------------
  A > B  →  DINO (Caron et al., 2021): "intermediate layer representations are not analyzed"
             Our multi-layer concat exploits what DINO left unexplored.

  A > C  →  ACE (Ghassemi et al., 2019): "relies on image segmentation preprocessing"
             Our attention-guided masking replaces segmentation — no annotation needed.

  A > D  →  Semantic fine-tuning effect: domain-informed loss improves concept cluster quality.

Metrics
-------
  Silhouette Score     : cluster compactness & separation [-1, 1] ↑
  Class Purity         : fraction of patches from dominant class per cluster [0, 1] ↑
  NMI                  : alignment of cluster assignments with true class labels [0, 1] ↑
  Intra/Inter ratio    : mean intra-cluster dist / mean inter-cluster dist [0, ∞] ↓

Usage
-----
  # Full run (extract + cluster + evaluate all 4 variants):
  python experiments/run_ablations.py

  # Skip re-extraction if features already cached:
  python experiments/run_ablations.py --skip_extract

  # Run specific variants only:
  python experiments/run_ablations.py --variants A B
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import normalize
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dino_extractor import DinoExtractor
from src.pipeline.patch_clusterer import PatchClusterer

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

VARIANTS = {
    "A": {
        "name":         "Full (multi-layer + FG mask + fine-tuned)",
        "use_multilayer": True,
        "fg_threshold":  0.75,
        "use_finetune":  True,
        "paper_gap":     "Baseline — all contributions active",
    },
    "B": {
        "name":         "Single-layer (L12 only)",
        "use_multilayer": False,
        "fg_threshold":  0.75,
        "use_finetune":  True,
        "paper_gap":     "DINO (Caron 2021): intermediate layers not analyzed",
    },
    "C": {
        "name":         "No FG masking (all patches)",
        "use_multilayer": True,
        "fg_threshold":  None,
        "use_finetune":  True,
        "paper_gap":     "ACE (Ghassemi 2019): relies on segmentation preprocessing",
    },
    "D": {
        "name":         "Pre-finetune DINO (base weights)",
        "use_multilayer": True,
        "fg_threshold":  0.75,
        "use_finetune":  False,
        "paper_gap":     "Domain-informed fine-tuning effect on cluster quality",
    },
}

# Subsample size for silhouette score (exact computation on 1M+ points is infeasible)
SILHOUETTE_SUBSAMPLE = 50_000
ABLATION_CACHE_DIR   = Path("cache/ablation")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_variant(variant_id: str, cfg: dict, device: str) -> dict:
    """
    Extract DINO patch features for one variant.
    Saves to cache/ablation/{variant_id}_features.pt and returns the cache dict.
    """
    vdef   = VARIANTS[variant_id]
    dcfg   = cfg["dino"]
    out_pt = ABLATION_CACHE_DIR / f"{variant_id}_features.pt"

    print(f"\n{'='*60}")
    print(f"  Extracting Variant {variant_id}: {vdef['name']}")
    print(f"  Gap addressed: {vdef['paper_gap']}")
    print(f"{'='*60}")

    extractor = DinoExtractor(
        model_name=dcfg["model"],
        device=device,
        image_size=dcfg["image_size"],
        use_multilayer=vdef["use_multilayer"],
    )

    # Load fine-tuned weights if requested and available
    ft_path = Path(dcfg.get("finetune_weights", "cache/dino_finetuned.pt"))
    if vdef["use_finetune"] and ft_path.exists():
        from src.models.dino_finetuner import DinoSemanticFinetuner
        DinoSemanticFinetuner.load_weights_into_extractor(extractor, str(ft_path))
        print(f"  Loaded fine-tuned weights: {ft_path}")
    elif vdef["use_finetune"]:
        print(f"  Warning: fine-tuned weights not found at {ft_path} — using base DINO")

    fg_threshold = vdef["fg_threshold"]
    if fg_threshold is not None:
        print(f"  FG masking: threshold={fg_threshold} (keeping top "
              f"{(1-fg_threshold)*100:.0f}% patches)")
    else:
        print("  FG masking: DISABLED — all 784 patches per image")

    # Collect images
    data_dir    = Path(dcfg.get("data_root", "data/masked/train"))
    classes     = dcfg.get("classes", ["cat", "car", "bird"])
    image_paths, image_labels, class_names = _collect_images(data_dir, classes)
    print(f"  Images: {len(image_paths)} across {class_names}")

    all_features, all_image_ids, all_patch_ids, all_patch_labels = [], [], [], []
    total_before = total_after = 0

    for img_idx, (img_path, label) in enumerate(
        tqdm(zip(image_paths, image_labels), total=len(image_paths),
             desc=f"  Variant {variant_id}")
    ):
        try:
            img_tensor = extractor.load_image(img_path)
            feats = extractor.extract_patches(img_tensor).squeeze(0).cpu()
            total_before += feats.shape[0]

            if fg_threshold is not None:
                attn    = extractor.extract_attention(img_tensor)
                cls_attn = attn[0, :, 0, 1:].mean(dim=0).cpu()
                thresh   = cls_attn.quantile(fg_threshold)
                fg_idx   = (cls_attn > thresh).nonzero(as_tuple=True)[0]
                if len(fg_idx) < 10:
                    fg_idx = cls_attn.topk(max(10, feats.shape[0] // 4)).indices
                kept_feats    = feats[fg_idx]
                kept_patch_ids = fg_idx.tolist()
            else:
                kept_feats    = feats
                kept_patch_ids = list(range(feats.shape[0]))

            total_after += kept_feats.shape[0]
            all_features.append(kept_feats)
            all_image_ids.extend([img_idx] * kept_feats.shape[0])
            all_patch_ids.extend(kept_patch_ids)
            all_patch_labels.extend([label] * kept_feats.shape[0])

        except Exception as e:
            print(f"  Warning: skipped {img_path}: {e}")

    if fg_threshold is not None:
        pct = (1 - total_after / max(total_before, 1)) * 100
        print(f"  Masking: {total_before:,} → {total_after:,} patches ({pct:.1f}% removed)")

    cache = {
        "features":     torch.cat(all_features, dim=0),
        "image_ids":    torch.tensor(all_image_ids,    dtype=torch.long),
        "patch_ids":    torch.tensor(all_patch_ids,    dtype=torch.long),
        "patch_labels": torch.tensor(all_patch_labels, dtype=torch.long),
        "image_paths":  image_paths,
        "class_names":  class_names,
        "image_labels": image_labels,
        "variant_id":   variant_id,
        "use_multilayer": vdef["use_multilayer"],
        "fg_threshold": fg_threshold,
    }
    torch.save(cache, out_pt)
    print(f"  Saved {cache['features'].shape[0]:,} patches → {out_pt}")
    return cache


def _collect_images(data_dir: Path, classes: list):
    """Walk data_dir/{class}/*.png and return (paths, labels, class_names)."""
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths, labels, class_names = [], [], []
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name not in classes:
            continue
        if class_dir.name not in class_names:
            class_names.append(class_dir.name)
        cid = class_names.index(class_dir.name)
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() in supported:
                paths.append(str(p))
                labels.append(cid)
    return paths, labels, class_names


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_variant(variant_id: str, cache: dict, cfg: dict) -> tuple:
    """
    Fit GMM on variant's features and return (clusterer, labels).
    Saves clusterer to cache/ablation/{variant_id}_kmeans.pkl
    """
    ccfg = cfg["clustering"]
    print(f"\n  Clustering Variant {variant_id}...")

    clusterer = PatchClusterer(
        n_clusters=ccfg["n_clusters"],
        random_seed=ccfg["random_seed"],
        use_pca=ccfg["use_pca"],
        pca_dims=ccfg["pca_dims"],
        method=ccfg["method"],
        use_spatial_features=ccfg["use_spatial_features"],
        spatial_weight=ccfg["spatial_weight"],
        gmm_max_fit_samples=ccfg.get("gmm_max_fit_samples", 300_000),
    )

    labels = clusterer.fit(cache["features"], cache["patch_ids"])

    pkl_path = str(ABLATION_CACHE_DIR / f"{variant_id}_kmeans.pkl")
    clusterer.save(pkl_path)
    return clusterer, labels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    variant_id: str,
    clusterer: PatchClusterer,
    labels: np.ndarray,
    cache: dict,
) -> dict:
    """
    Compute 4 cluster quality metrics for the dissertation ablation table.

    Returns dict with keys: silhouette, class_purity, nmi, intra_inter_ratio
    """
    print(f"\n  Computing metrics for Variant {variant_id}...")

    features     = cache["features"]
    patch_labels = cache["patch_labels"].numpy()  # true class labels per patch
    n_clusters   = clusterer.n_clusters

    # --- Prepare PCA-reduced + normalised features (same space as clustering) ---
    X = normalize(features.numpy(), norm="l2")
    if clusterer.use_pca and clusterer.pca is not None:
        X = clusterer.pca.transform(X)
        X = normalize(X, norm="l2")

    # ------------------------------------------------------------------
    # 1. Silhouette Score (subsampled for speed)
    # ------------------------------------------------------------------
    N = X.shape[0]
    if N > SILHOUETTE_SUBSAMPLE:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, SILHOUETTE_SUBSAMPLE, replace=False)
        sil = silhouette_score(X[idx], labels[idx], metric="euclidean",
                               sample_size=None)
    else:
        sil = silhouette_score(X, labels, metric="euclidean")
    print(f"    Silhouette score         : {sil:.4f}")

    # ------------------------------------------------------------------
    # 2. Class Purity (per cluster, weighted average)
    # ------------------------------------------------------------------
    n_classes    = len(set(patch_labels))
    purity_total = 0.0
    for k in range(n_clusters):
        mask   = labels == k
        if mask.sum() == 0:
            continue
        cls_counts = np.bincount(patch_labels[mask], minlength=n_classes)
        purity_total += cls_counts.max() / mask.sum()
    class_purity = purity_total / n_clusters
    print(f"    Class purity             : {class_purity:.4f}")

    # ------------------------------------------------------------------
    # 3. NMI (between cluster assignments and true class labels)
    # ------------------------------------------------------------------
    nmi = normalized_mutual_info_score(patch_labels, labels, average_method="arithmetic")
    print(f"    NMI (clusters vs classes): {nmi:.4f}")

    # ------------------------------------------------------------------
    # 4. Intra / Inter cluster distance ratio
    #    intra = mean distance from each point to its cluster center
    #    inter = mean pairwise distance between cluster centers
    # ------------------------------------------------------------------
    centers = clusterer.centers_.numpy()
    if centers.shape[1] != X.shape[1]:
        # Centers were stored before spatial augmentation — re-derive from means
        centers = np.array([X[labels == k].mean(axis=0) for k in range(n_clusters)
                            if (labels == k).sum() > 0])

    # Intra: mean distance from each patch to its assigned center
    intra_dists = np.linalg.norm(X - centers[labels], axis=1)
    intra_mean  = intra_dists.mean()

    # Inter: mean pairwise distance between all cluster centers
    n_c    = len(centers)
    diffs  = centers[:, None, :] - centers[None, :, :]          # [n_c, n_c, D]
    dist_m = np.linalg.norm(diffs, axis=-1)                     # [n_c, n_c]
    inter_mean = dist_m[np.triu_indices(n_c, k=1)].mean()

    intra_inter = intra_mean / max(inter_mean, 1e-9)
    print(f"    Intra-cluster dist       : {intra_mean:.4f}")
    print(f"    Inter-cluster dist       : {inter_mean:.4f}")
    print(f"    Intra/Inter ratio        : {intra_inter:.4f}  (lower = better separation)")

    return {
        "silhouette":      round(float(sil),           4),
        "class_purity":    round(float(class_purity),  4),
        "nmi":             round(float(nmi),            4),
        "intra_mean":      round(float(intra_mean),     4),
        "inter_mean":      round(float(inter_mean),     4),
        "intra_inter_ratio": round(float(intra_inter), 4),
        "n_patches":       int(N),
        "n_clusters":      int(n_clusters),
    }


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_results_table(results: dict):
    """Pretty-print the ablation comparison table for copy-paste into dissertation."""

    # Deltas vs baseline (Variant A)
    base = results.get("A", {})

    header = f"\n{'─'*100}"
    print(header)
    print(f"  ABLATION RESULTS — Unsupervised Parts Discovery")
    print(f"{'─'*100}")
    fmt = "  {:<4}  {:<42}  {:>11}  {:>12}  {:>6}  {:>13}  {:>9}"
    print(fmt.format(
        "ID", "Variant", "Silhouette↑", "ClassPurity↑", "NMI↑", "Intra/Inter↓", "Patches"
    ))
    print(f"{'─'*100}")

    for vid, vdef in VARIANTS.items():
        if vid not in results:
            continue
        r   = results[vid]
        sil = r["silhouette"]
        pur = r["class_purity"]
        nmi = r["nmi"]
        ii  = r["intra_inter_ratio"]
        n   = r["n_patches"]

        # Colour-code deltas vs baseline A
        def delta(val, base_val, higher_better=True):
            d = val - base_val
            if abs(d) < 1e-4:
                return ""
            sign = "+" if d > 0 else ""
            indicator = "▲" if (d > 0) == higher_better else "▼"
            return f" {indicator}{sign}{d:+.3f}"

        sil_d = delta(sil, base.get("silhouette", sil)) if vid != "A" else "  [baseline]"
        pur_d = delta(pur, base.get("class_purity", pur)) if vid != "A" else ""
        nmi_d = delta(nmi, base.get("nmi", nmi)) if vid != "A" else ""
        ii_d  = delta(ii,  base.get("intra_inter_ratio", ii), higher_better=False) if vid != "A" else ""

        print(fmt.format(
            vid,
            vdef["name"][:42],
            f"{sil:.4f}{sil_d}",
            f"{pur:.4f}{pur_d}",
            f"{nmi:.4f}{nmi_d}",
            f"{ii:.4f}{ii_d}",
            f"{n:,}",
        ))

    print(f"{'─'*100}")
    print("  ▲ = better than baseline A  |  ▼ = worse than baseline A")
    print(f"{'─'*100}\n")

    # Gap summary
    print("  PAPER GAP EVIDENCE:")
    if "A" in results and "B" in results:
        d_sil = results["A"]["silhouette"] - results["B"]["silhouette"]
        d_nmi = results["A"]["nmi"]        - results["B"]["nmi"]
        print(f"    A > B: Multi-layer vs single-layer  →  Δsilhouette={d_sil:+.4f}, ΔNMI={d_nmi:+.4f}")
        print(f"           (evidence for: DINO Caron 2021 — intermediate layers contain concept structure)")
    if "A" in results and "C" in results:
        d_sil = results["A"]["silhouette"] - results["C"]["silhouette"]
        d_nmi = results["A"]["nmi"]        - results["C"]["nmi"]
        print(f"    A > C: FG masking vs no masking     →  Δsilhouette={d_sil:+.4f}, ΔNMI={d_nmi:+.4f}")
        print(f"           (evidence for: ACE Ghassemi 2019 — segmentation-free masking improves purity)")
    if "A" in results and "D" in results:
        d_sil = results["A"]["silhouette"] - results["D"]["silhouette"]
        d_nmi = results["A"]["nmi"]        - results["D"]["nmi"]
        print(f"    A > D: Fine-tuned vs base DINO      →  Δsilhouette={d_sil:+.4f}, ΔNMI={d_nmi:+.4f}")
        print(f"           (evidence for: semantic consistency loss improves concept cluster quality)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run ablation study for DINO concept discovery pipeline"
    )
    parser.add_argument("--config",        default="configs/config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--skip_extract",  action="store_true",
                        help="Skip re-extraction if variant features already cached")
    parser.add_argument("--variants",      nargs="+", default=list(VARIANTS.keys()),
                        choices=list(VARIANTS.keys()),
                        help="Which variants to run (default: all A B C D)")
    parser.add_argument("--device",        default=None,
                        help="Override device (mps/cuda/cpu)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or cfg["dino"].get("device", "cpu")
    ABLATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    results      = {}
    timing       = {}
    results_path = ABLATION_CACHE_DIR / "ablation_results.json"

    # Load any previously saved results (allows partial re-runs)
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        print(f"Loaded existing results for variants: {list(results.keys())}")

    for vid in args.variants:
        vdef    = VARIANTS[vid]
        out_pt  = ABLATION_CACHE_DIR / f"{vid}_features.pt"

        print(f"\n{'#'*60}")
        print(f"  VARIANT {vid}: {vdef['name']}")
        print(f"  Gap: {vdef['paper_gap']}")
        print(f"{'#'*60}")

        t0 = time.time()

        # --- 1. Extract ---
        if args.skip_extract and out_pt.exists():
            print(f"  Loading cached features from {out_pt}")
            cache = torch.load(out_pt, map_location="cpu", weights_only=False)
            print(f"  Loaded {cache['features'].shape[0]:,} patches × {cache['features'].shape[1]} dims")
        else:
            cache = extract_variant(vid, cfg, device)

        # --- 2. Cluster ---
        clusterer, labels = cluster_variant(vid, cache, cfg)

        # --- 3. Metrics ---
        metrics = compute_metrics(vid, clusterer, labels, cache)

        elapsed = time.time() - t0
        timing[vid]  = round(elapsed, 1)
        results[vid] = metrics
        results[vid]["variant_name"] = vdef["name"]
        results[vid]["paper_gap"]    = vdef["paper_gap"]

        # Save incrementally so partial runs aren't lost
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Variant {vid} done in {elapsed/60:.1f} min. Results saved → {results_path}")

    # --- Final table ---
    print_results_table(results)

    # Timing summary
    print("  TIMING:")
    for vid, t in timing.items():
        print(f"    Variant {vid}: {t/60:.1f} min")
    print()


if __name__ == "__main__":
    main()
