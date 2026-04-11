"""
run_variant_C_subsampled.py
===========================
Memory-efficient evaluation of Variant C (no FG masking).

Variant C has ~9.4M patches × 1152 dims = 43GB — too large to load fully on 32GB RAM.
This script loads a random subsample of 2.4M patches (matching Variant A's count)
via memory-mapped I/O, then clusters and computes identical metrics.

The comparison is still valid: we're testing whether FG-filtered (A) patches cluster
better than a random sample of ALL patches (C), which is the ACE gap claim.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.patch_clusterer import PatchClusterer

CONFIG          = "configs/config.yaml"
ABLATION_DIR    = Path("cache/ablation")
RESULTS_PATH    = ABLATION_DIR / "ablation_results.json"
SUBSAMPLE_N     = 2_400_000   # match Variant A's patch count
SILHOUETTE_SUB  = 50_000
RANDOM_SEED     = 42


def main():
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    ccfg = cfg["clustering"]

    c_pt = ABLATION_DIR / "C_features.pt"
    if not c_pt.exists():
        print("ERROR: C_features.pt not found. Run extraction first.")
        sys.exit(1)

    print("Loading C_features.pt (memory-mapped)...")
    # mmap=True avoids reading the whole file into RAM; we subset below
    cache = torch.load(c_pt, map_location="cpu", weights_only=False, mmap=True)
    full_features     = cache["features"]       # shape: [N, D]  (memory-mapped)
    full_patch_labels = cache["patch_labels"]   # shape: [N]
    full_patch_ids    = cache["patch_ids"]      # shape: [N]
    N_total = full_features.shape[0]
    print(f"  Total patches in C: {N_total:,}  (will subsample to {SUBSAMPLE_N:,})")

    # Random subsample
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(N_total, SUBSAMPLE_N, replace=False)
    idx_sorted = np.sort(idx)  # sequential access is faster with mmap

    print("  Copying subsample to RAM...")
    features     = full_features[idx_sorted].clone()          # [sub, D]
    patch_labels = full_patch_labels[idx_sorted].numpy()
    patch_ids    = full_patch_ids[idx_sorted].numpy()

    print(f"  Subsampled: {features.shape[0]:,} patches × {features.shape[1]} dims")

    # ── Cluster ──────────────────────────────────────────────────────────────
    print("\n  Clustering Variant C (subsampled)...")
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
    labels = clusterer.fit(features, patch_ids)
    clusterer.save(str(ABLATION_DIR / "C_kmeans.pkl"))

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\n  Computing metrics for Variant C...")
    n_clusters = clusterer.n_clusters

    X = normalize(features.numpy(), norm="l2")
    if clusterer.use_pca and clusterer.pca is not None:
        X = clusterer.pca.transform(X)
        X = normalize(X, norm="l2")

    # Silhouette
    rng2 = np.random.default_rng(42)
    sub_idx = rng2.choice(len(X), SILHOUETTE_SUB, replace=False)
    sil = silhouette_score(X[sub_idx], labels[sub_idx], metric="euclidean")
    print(f"    Silhouette score         : {sil:.4f}")

    # Class purity
    n_classes = len(set(patch_labels))
    purity_total = 0.0
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() == 0:
            continue
        cls_counts = np.bincount(patch_labels[mask], minlength=n_classes)
        purity_total += cls_counts.max() / mask.sum()
    class_purity = purity_total / n_clusters
    print(f"    Class purity             : {class_purity:.4f}")

    # NMI
    nmi = normalized_mutual_info_score(patch_labels, labels, average_method="arithmetic")
    print(f"    NMI (clusters vs classes): {nmi:.4f}")

    # Intra/Inter
    centers = clusterer.centers_.numpy()
    if centers.shape[1] != X.shape[1]:
        centers = np.array([X[labels == k].mean(axis=0) for k in range(n_clusters)
                            if (labels == k).sum() > 0])
    intra_dists = np.linalg.norm(X - centers[labels], axis=1)
    intra_mean  = intra_dists.mean()
    n_c = len(centers)
    diffs    = centers[:, None, :] - centers[None, :, :]
    dist_m   = np.linalg.norm(diffs, axis=-1)
    inter_mean = dist_m[np.triu_indices(n_c, k=1)].mean()
    intra_inter = intra_mean / max(inter_mean, 1e-9)
    print(f"    Intra-cluster dist       : {intra_mean:.4f}")
    print(f"    Inter-cluster dist       : {inter_mean:.4f}")
    print(f"    Intra/Inter ratio        : {intra_inter:.4f}")

    result = {
        "silhouette":        round(float(sil),           4),
        "class_purity":      round(float(class_purity),  4),
        "nmi":               round(float(nmi),            4),
        "intra_mean":        round(float(intra_mean),     4),
        "inter_mean":        round(float(inter_mean),     4),
        "intra_inter_ratio": round(float(intra_inter),   4),
        "n_patches":         int(SUBSAMPLE_N),
        "n_clusters":        int(n_clusters),
        "variant_name":      "No FG masking (all patches, subsampled to 2.4M)",
        "paper_gap":         "ACE (Ghassemi 2019): relies on segmentation preprocessing",
        "note":              f"Subsampled {SUBSAMPLE_N:,} of {N_total:,} total patches (memory constraint)",
    }

    # Merge into results file
    results = {}
    if RESULTS_PATH.exists():
        results = json.load(open(RESULTS_PATH))
    results["C"] = result
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  ✓  Variant C results saved → {RESULTS_PATH}")
    print(f"     (note: {SUBSAMPLE_N:,}/{N_total:,} patches used due to memory constraint)")


if __name__ == "__main__":
    main()
