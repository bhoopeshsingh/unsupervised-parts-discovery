# experiments/find_optimal_k.py
"""
Elbow + Silhouette analysis to find the optimal number of clusters (K)
for the DINO patch feature space.

DESIGN NOTE — Single-Class Pilot
---------------------------------
This script was run on a single-class feature cache (cat-only pilot) to
determine K per class.  Result: K=10, silhouette=0.61 (Table 6 in dissertation).

The three-class full pipeline uses K = K_per_class × n_classes = 10 × 3 = 30,
set in configs/config.yaml (clustering.n_clusters: 30).

The script uses MiniBatchKMeans for the sweep because GMM fitting across
25 values of K on millions of patches is prohibitively slow.  The selected K
is then used to configure the GMM in the main pipeline.  GMM convergence
(converged_=True, stable lower_bound_) confirmed K=30 is valid.

Usage:
    python experiments/find_optimal_k.py
    python experiments/find_optimal_k.py --k_min 5 --k_max 30 --config configs/config.yaml

Output:
    - Prints silhouette scores and inertia for each K
    - Saves a plot to cache/optimal_k_analysis.png
    - Prints a recommendation for the best K
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA


def run_analysis(
    config_path: str = "configs/config.yaml",
    k_min: int = 5,
    k_max: int = 30,
    k_step: int = 1,
    n_sample: int = 50_000,
    use_pca: bool = True,
    pca_dims: int = 64,
    random_seed: int = 42,
):
    """
    Args:
        n_sample:  Max patches to use for silhouette (full set is slow; 50k is representative).
        use_pca:   Reduce 384 → pca_dims before clustering. Speeds up and often improves quality.
        pca_dims:  PCA output dimensions (64 is a good default).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cache_path = cfg["dino"]["features_cache"]
    if not Path(cache_path).exists():
        print(f"ERROR: Feature cache not found at {cache_path}")
        print("Run extraction first: python experiments/run_pipeline.py --stage extract")
        sys.exit(1)

    print(f"Loading features from {cache_path} ...")
    data = torch.load(cache_path, weights_only=False)
    features = data["features"]  # [N, 384]
    fg_threshold = data.get("fg_threshold", None)
    print(f"  {features.shape[0]:,} patches × {features.shape[1]} dims")
    if fg_threshold is not None:
        print(f"  (extracted with fg_threshold={fg_threshold} — foreground only)")
    else:
        print("  (no foreground masking applied — consider re-extracting with fg_threshold)")

    # Normalize
    X = normalize(features.numpy(), norm="l2")

    # Optional PCA
    if use_pca and X.shape[1] > pca_dims:
        print(f"\nApplying PCA: {X.shape[1]} → {pca_dims} dims ...")
        pca = PCA(n_components=pca_dims, random_state=random_seed)
        X = pca.fit_transform(X)
        explained = pca.explained_variance_ratio_.sum() * 100
        print(f"  Explained variance: {explained:.1f}%")

    # Sample for silhouette (it's O(n²) so we cap it)
    rng = np.random.default_rng(random_seed)
    n_total = X.shape[0]
    if n_total > n_sample:
        sample_idx = rng.choice(n_total, n_sample, replace=False)
        X_sample = X[sample_idx]
        print(f"\nUsing {n_sample:,} / {n_total:,} patches for silhouette scoring.")
    else:
        X_sample = X
        print(f"\nUsing all {n_total:,} patches.")

    k_values = list(range(k_min, k_max + 1, k_step))
    inertias = []
    silhouette_scores = []

    print(f"\nRunning KMeans for K = {k_values[0]} to {k_values[-1]} ...")
    print(f"{'K':>4}  {'Inertia':>12}  {'Silhouette':>12}  {'Quality':>10}")
    print("-" * 44)

    best_k = k_values[0]
    best_silhouette = -1.0

    for k in k_values:
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            n_init=10,
            max_iter=300,
            batch_size=min(8192, n_total),
            random_state=random_seed,
        )
        kmeans.fit(X)
        inertia = kmeans.inertia_

        # Silhouette on the sample
        labels_sample = kmeans.predict(X_sample)
        # Need at least 2 clusters with >1 member in the sample
        unique_in_sample = len(np.unique(labels_sample))
        if unique_in_sample < 2:
            sil = -1.0
        else:
            sil = silhouette_score(X_sample, labels_sample, sample_size=min(10_000, len(X_sample)))

        inertias.append(inertia)
        silhouette_scores.append(sil)

        quality = "★★★" if sil > 0.25 else ("★★ " if sil > 0.15 else "★  ")
        print(f"{k:>4}  {inertia:>12.1f}  {sil:>12.4f}  {quality}")

        if sil > best_silhouette:
            best_silhouette = sil
            best_k = k

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Inertia (elbow)
    ax1.plot(k_values, inertias, "b-o", linewidth=2, markersize=5)
    ax1.axvline(best_k, color="red", linestyle="--", alpha=0.7, label=f"Best K={best_k}")
    ax1.set_xlabel("Number of clusters (K)", fontsize=12)
    ax1.set_ylabel("Inertia (within-cluster sum of squares)", fontsize=12)
    ax1.set_title("Elbow Method", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Silhouette
    colors = ["#2196F3" if k != best_k else "#F44336" for k in k_values]
    ax2.bar(k_values, silhouette_scores, color=colors, edgecolor="white")
    ax2.axhline(0.25, color="green", linestyle="--", alpha=0.6, label="Good threshold (0.25)")
    ax2.axhline(0.15, color="orange", linestyle="--", alpha=0.6, label="Acceptable (0.15)")
    ax2.set_xlabel("Number of clusters (K)", fontsize=12)
    ax2.set_ylabel("Silhouette Score", fontsize=12)
    ax2.set_title("Silhouette Score (higher = better separated clusters)", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    pca_note = f" | PCA {features.shape[1]}→{pca_dims}d" if use_pca else ""
    fg_note = f" | fg_threshold={fg_threshold}" if fg_threshold is not None else " | no fg mask"
    fig.suptitle(
        f"Optimal K Analysis — {features.shape[0]:,} patches{pca_note}{fg_note}",
        fontsize=11, y=1.01
    )
    plt.tight_layout()

    out_path = Path(cfg["dino"]["cache_dir"]) / "optimal_k_analysis.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")

    # ── Recommendation ───────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"  RECOMMENDATION: K = {best_k}")
    print(f"  Silhouette score: {best_silhouette:.4f}", end="  ")
    if best_silhouette > 0.25:
        print("(good separation ★★★)")
    elif best_silhouette > 0.15:
        print("(acceptable ★★)")
    else:
        print("(weak — consider foreground masking or PCA if not used ★)")
    print(f"\n  Update configs/config.yaml:")
    print(f"    clustering:")
    print(f"      n_clusters: {best_k}")
    print("=" * 50)

    return best_k, silhouette_scores, inertias


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--k_min", type=int, default=5)
    parser.add_argument("--k_max", type=int, default=30)
    parser.add_argument("--k_step", type=int, default=1)
    parser.add_argument("--n_sample", type=int, default=50_000,
                        help="Max patches used for silhouette scoring")
    parser.add_argument("--no_pca", action="store_true",
                        help="Disable PCA pre-processing")
    parser.add_argument("--pca_dims", type=int, default=64)
    args = parser.parse_args()

    run_analysis(
        config_path=args.config,
        k_min=args.k_min,
        k_max=args.k_max,
        k_step=args.k_step,
        n_sample=args.n_sample,
        use_pca=not args.no_pca,
        pca_dims=args.pca_dims,
    )
