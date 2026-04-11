"""
compute_lab_cluster_metrics.py
==============================
Compute per-panel cluster quality metrics from the existing lab cache.
Used to compare structured (shuffle_panels=false) vs shuffled (shuffle_panels=true) runs.

Metrics computed per panel:
  - Silhouette score     (cluster compactness/separation, no ground truth needed)
  - NMI vs diabetes      (DIQ010: cluster alignment with unseen clinical label)
  - NMI vs hypertension  (BPQ020: cluster alignment with unseen clinical label)
  - Intra/Inter ratio    (lower = better separation)

Usage:
  python experiments/compute_lab_cluster_metrics.py                 # uses default cache paths
  python experiments/compute_lab_cluster_metrics.py --tag shuffled  # label the run in output
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).parent.parent))

SILHOUETTE_SUBSAMPLE = 10_000   # subsample for speed (silhouette is O(n²))
CONFIG_LAB = "configs/config_lab.yaml"


def load_cache(cfg):
    """Load panel patches and cluster labels from disk."""
    import yaml
    cfg = yaml.safe_load(open(cfg))

    patches_path  = cfg["transformer"]["panel_patches_path"]
    clusters_path = Path(cfg["lab_data"]["cache_dir"]) / "panel_clusters.pt"
    demo_path     = Path(cfg["lab_data"]["cache_dir"]) / "records_with_demo.csv"

    if not Path(patches_path).exists():
        print(f"ERROR: {patches_path} not found. Run --stage encode first.")
        sys.exit(1)
    if not clusters_path.exists():
        print(f"ERROR: {clusters_path} not found. Run --stage cluster first.")
        sys.exit(1)

    patches_data  = torch.load(patches_path,  weights_only=False)
    clusters_data = torch.load(clusters_path, weights_only=False)

    demo_df = pd.read_csv(demo_path) if demo_path.exists() else None

    return patches_data, clusters_data, demo_df


def compute_panel_metrics(patches_data, clusters_data, demo_df):
    """
    Returns dict: {panel_name: {silhouette, nmi_diabetes, nmi_hypert, intra_inter}}
    """
    patches    = patches_data["patches"]       # [N*P, D]
    panel_idx  = patches_data["panel_idx"]     # [N*P]
    panel_names = clusters_data["panel_names"]
    per_panel_labels = clusters_data["per_panel_labels"]

    # Ground truth labels (aligned by record order)
    diab_labels  = None
    hypert_labels = None
    if demo_df is not None:
        if "DIQ010" in demo_df.columns:
            # 1=Yes diabetes, 2=No, 7/9=refused/unknown → map to binary
            raw = demo_df["DIQ010"].values
            diab_labels = np.where(raw == 1, 1, np.where(raw == 2, 0, -1))
        if "BPQ020" in demo_df.columns:
            raw = demo_df["BPQ020"].values
            hypert_labels = np.where(raw == 1, 1, np.where(raw == 2, 0, -1))

    results = {}

    for p_idx, name in enumerate(panel_names):
        mask = (panel_idx == p_idx)
        X = patches[mask].numpy()                         # [N, D]
        cluster_labels = per_panel_labels[name]           # [N] int array
        if not isinstance(cluster_labels, np.ndarray):
            cluster_labels = np.array(cluster_labels)

        N = X.shape[0]
        print(f"\n  Panel '{name}'  ({N:,} patches × {X.shape[1]} dims,  "
              f"{len(np.unique(cluster_labels))} clusters)")

        # Normalize to unit sphere (same as clustering)
        X_norm = normalize(X, norm="l2")

        # ── 1. Silhouette score ──────────────────────────────────────────────
        if N > SILHOUETTE_SUBSAMPLE:
            rng = np.random.default_rng(42)
            idx = rng.choice(N, SILHOUETTE_SUBSAMPLE, replace=False)
            sil = silhouette_score(X_norm[idx], cluster_labels[idx], metric="euclidean")
        else:
            sil = silhouette_score(X_norm, cluster_labels, metric="euclidean")
        print(f"    Silhouette           : {sil:.4f}")

        # ── 2. NMI vs diabetes ───────────────────────────────────────────────
        nmi_diab = None
        if diab_labels is not None:
            valid = diab_labels >= 0
            if valid.sum() > 100:
                nmi_diab = normalized_mutual_info_score(
                    diab_labels[valid], cluster_labels[valid],
                    average_method="arithmetic"
                )
                print(f"    NMI vs diabetes      : {nmi_diab:.4f}")

        # ── 3. NMI vs hypertension ───────────────────────────────────────────
        nmi_hypert = None
        if hypert_labels is not None:
            valid = hypert_labels >= 0
            if valid.sum() > 100:
                nmi_hypert = normalized_mutual_info_score(
                    hypert_labels[valid], cluster_labels[valid],
                    average_method="arithmetic"
                )
                print(f"    NMI vs hypertension  : {nmi_hypert:.4f}")

        # ── 4. Intra / Inter distance ratio ──────────────────────────────────
        n_clusters = len(np.unique(cluster_labels))
        centers = np.array([
            X_norm[cluster_labels == k].mean(axis=0)
            for k in range(n_clusters)
            if (cluster_labels == k).sum() > 0
        ])
        intra_dists = np.linalg.norm(X_norm - centers[cluster_labels], axis=1)
        intra_mean  = intra_dists.mean()

        diffs     = centers[:, None, :] - centers[None, :, :]
        dist_m    = np.linalg.norm(diffs, axis=-1)
        inter_mean = dist_m[np.triu_indices(len(centers), k=1)].mean()
        intra_inter = intra_mean / max(inter_mean, 1e-9)
        print(f"    Intra/Inter ratio    : {intra_inter:.4f}")

        results[name] = {
            "silhouette":    round(float(sil),          4),
            "nmi_diabetes":  round(float(nmi_diab),     4) if nmi_diab  is not None else None,
            "nmi_hypertens": round(float(nmi_hypert),   4) if nmi_hypert is not None else None,
            "intra_inter":   round(float(intra_inter),  4),
            "n_patches":     int(N),
            "n_clusters":    int(n_clusters),
        }

    return results


def print_table(structured: dict, shuffled: dict):
    """Print side-by-side comparison table."""
    panels = list(structured.keys())
    sep = "─" * 105

    print(f"\n{sep}")
    print("  LAB PANEL PE ABLATION — Structured vs Shuffled Panel Order")
    print(sep)
    hdr = f"  {'Panel':<12}  {'Metric':<22}  {'Structured':>12}  {'Shuffled':>12}  {'Delta':>10}  {'Better':>8}"
    print(hdr)
    print(sep)

    for panel in panels:
        s = structured.get(panel, {})
        h = shuffled.get(panel, {})

        metrics_to_show = [
            ("Silhouette↑",   "silhouette",    True),
            ("NMI-Diabetes↑", "nmi_diabetes",  True),
            ("NMI-Hypert↑",   "nmi_hypertens", True),
            ("Intra/Inter↓",  "intra_inter",   False),   # lower is better
        ]

        for i, (label, key, higher_better) in enumerate(metrics_to_show):
            sv = s.get(key)
            hv = h.get(key)

            if sv is None or hv is None:
                continue

            delta = sv - hv
            if higher_better:
                better = "Struct ✓" if delta > 0.001 else ("Shuffle ✓" if delta < -0.001 else "≈ tie")
            else:
                better = "Struct ✓" if delta < -0.001 else ("Shuffle ✓" if delta > 0.001 else "≈ tie")

            panel_col = panel if i == 0 else ""
            print(f"  {panel_col:<12}  {label:<22}  {sv:>12.4f}  {hv:>12.4f}  {delta:>+10.4f}  {better:>8}")

        print()

    print(sep)
    print("  Panel PE matters if Structured consistently outperforms Shuffled.")
    print(sep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_LAB)
    parser.add_argument("--tag",    default="structured",
                        help="Label for this run (structured / shuffled)")
    parser.add_argument("--save",   default=None,
                        help="Save results to this JSON path")
    parser.add_argument("--compare", default=None,
                        help="Compare against previously saved JSON (structured run)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Lab Cluster Metrics — {args.tag.upper()}")
    print(f"{'='*60}")

    patches_data, clusters_data, demo_df = load_cache(args.config)
    results = compute_panel_metrics(patches_data, clusters_data, demo_df)

    # Save if requested
    if args.save:
        import json
        with open(args.save, "w") as f:
            json.dump({"tag": args.tag, "results": results}, f, indent=2)
        print(f"\n  Results saved → {args.save}")

    # Compare if requested
    if args.compare and Path(args.compare).exists():
        import json
        structured_data = json.load(open(args.compare))
        print(f"\n  Comparing {args.tag} vs {structured_data['tag']}:")
        print_table(structured_data["results"], results)

    # Always print a per-panel summary
    print(f"\n  {'─'*60}")
    print(f"  SUMMARY — {args.tag.upper()}")
    print(f"  {'─'*60}")
    for panel, m in results.items():
        print(f"  {panel:<12}  sil={m['silhouette']:.4f}  "
              f"nmi_diab={m['nmi_diabetes'] or 'n/a'!s:>8}  "
              f"intra/inter={m['intra_inter']:.4f}")
    print()


if __name__ == "__main__":
    main()
