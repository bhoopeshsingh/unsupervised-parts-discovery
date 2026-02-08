#!/usr/bin/env python
"""Visualize parts from clusters - saves images to disk for viewing"""

import sys
sys.path.append('.')

import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def main():
    print("Loading data...")
    parts_dir = Path('./parts/extracted')
    clusters_dir = Path('./parts/clusters')
    output_dir = Path('./parts/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load arrays
    masks = np.load(parts_dir / 'masks.npy')
    cluster_labels = np.load(clusters_dir / 'cluster_labels.npy')
    part_to_class = np.load(clusters_dir / 'part_to_class.npy')

    with open(clusters_dir / 'cluster_metadata.json', 'r') as f:
        cluster_meta = json.load(f)

    print(f"Loaded {len(masks)} parts across {len(cluster_meta)} clusters")

    # Sort clusters by size
    sorted_clusters = sorted(cluster_meta.items(), key=lambda x: x[1]['size'], reverse=True)

    # Create visualization for top 20 clusters
    n_clusters = min(20, len(sorted_clusters))
    n_samples = 8

    print(f"\nGenerating visualization for top {n_clusters} clusters...")

    fig, axes = plt.subplots(n_clusters, n_samples + 1, figsize=(2*(n_samples+1), 2*n_clusters))
    fig.suptitle('Top 20 Clusters - Sample Attention Masks', fontsize=16, y=1.01)

    for i, (cid, meta) in enumerate(sorted_clusters[:n_clusters]):
        cid_int = int(cid)

        # Cluster info label
        axes[i, 0].text(0.5, 0.5,
                       f"Cluster {cid}\n{meta['dominant_class']}\nN={meta['size']}",
                       ha='center', va='center', fontsize=10,
                       transform=axes[i, 0].transAxes,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        axes[i, 0].axis('off')

        # Get parts in this cluster
        part_indices = np.where(cluster_labels == cid_int)[0]

        # Sample parts
        np.random.seed(42)  # Reproducible
        if len(part_indices) > n_samples:
            sampled = np.random.choice(part_indices, n_samples, replace=False)
        else:
            sampled = part_indices

        for j, pidx in enumerate(sampled):
            if j >= n_samples:
                break
            mask = masks[pidx]
            axes[i, j+1].imshow(mask, cmap='hot', vmin=0, vmax=1)
            axes[i, j+1].axis('off')

        # Hide unused axes
        for j in range(len(sampled), n_samples):
            axes[i, j+1].axis('off')

    plt.tight_layout()
    save_path = output_dir / 'top20_clusters.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")

    # Create individual cluster visualizations for detailed inspection
    print("\nGenerating individual cluster views...")

    for i, (cid, meta) in enumerate(sorted_clusters[:10]):  # Top 10
        cid_int = int(cid)
        part_indices = np.where(cluster_labels == cid_int)[0]

        # Show more samples for individual clusters
        n_show = min(24, len(part_indices))
        rows = (n_show + 5) // 6

        fig, axes = plt.subplots(rows, 6, figsize=(12, 2*rows))
        fig.suptitle(f"Cluster {cid} - {meta['dominant_class']} ({meta['size']} parts)", fontsize=14)

        if rows == 1:
            axes = axes.reshape(1, -1)

        np.random.seed(42)
        sampled = np.random.choice(part_indices, n_show, replace=False) if len(part_indices) > n_show else part_indices

        for j, pidx in enumerate(sampled):
            row, col = j // 6, j % 6
            axes[row, col].imshow(masks[pidx], cmap='hot', vmin=0, vmax=1)
            axes[row, col].axis('off')

        # Hide unused
        for j in range(len(sampled), rows * 6):
            row, col = j // 6, j % 6
            axes[row, col].axis('off')

        plt.tight_layout()
        save_path = output_dir / f'cluster_{cid}_detail.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

    print(f"Saved individual cluster views to: {output_dir}/")

    # Create a summary grid of cluster diversity
    print("\nGenerating cluster diversity overview...")

    fig, axes = plt.subplots(9, 10, figsize=(20, 18))
    fig.suptitle('All 90 Clusters - One Sample Each', fontsize=16)

    for i, (cid, meta) in enumerate(sorted_clusters[:90]):
        if i >= 90:
            break
        row, col = i // 10, i % 10
        cid_int = int(cid)

        part_indices = np.where(cluster_labels == cid_int)[0]
        if len(part_indices) > 0:
            # Pick middle sample for consistency
            sample_idx = part_indices[len(part_indices) // 2]
            axes[row, col].imshow(masks[sample_idx], cmap='hot', vmin=0, vmax=1)
        axes[row, col].set_title(f"C{cid}\n({meta['size']})", fontsize=8)
        axes[row, col].axis('off')

    plt.tight_layout()
    save_path = output_dir / 'all_clusters_overview.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print(f"\nOutput files saved to: {output_dir.absolute()}")
    print("\nFiles created:")
    print("  - top20_clusters.png      : Top 20 clusters with 8 samples each")
    print("  - all_clusters_overview.png: All 90 clusters, 1 sample each")
    print("  - cluster_X_detail.png    : Detailed view of top 10 clusters")
    print("\nOpen these images to inspect the discovered parts!")

if __name__ == '__main__':
    main()

