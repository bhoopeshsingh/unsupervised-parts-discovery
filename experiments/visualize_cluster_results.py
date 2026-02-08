#!/usr/bin/env python
"""Visualize clustering results - show sample parts from each cluster"""

import sys
sys.path.append('.')

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def load_data():
    """Load all clustering data"""
    parts_dir = Path('./parts/extracted')
    clusters_dir = Path('./parts/clusters')

    # Load arrays
    masks = np.load(parts_dir / 'masks.npy')
    bboxes = np.load(parts_dir / 'bboxes.npy')
    cluster_labels = np.load(clusters_dir / 'cluster_labels.npy')
    part_to_class = np.load(clusters_dir / 'part_to_class.npy')
    part_to_image = np.load(clusters_dir / 'part_to_image.npy')

    with open(clusters_dir / 'cluster_metadata.json', 'r') as f:
        cluster_meta = json.load(f)

    with open(parts_dir / 'metadata.json', 'r') as f:
        parts_meta = json.load(f)

    return masks, bboxes, cluster_labels, part_to_class, part_to_image, cluster_meta, parts_meta

def compute_cluster_stats(cluster_meta):
    """Compute summary statistics"""
    purities = [c['purity'] for c in cluster_meta.values()]
    sizes = [c['size'] for c in cluster_meta.values()]

    # Count by class
    class_counts = {}
    for c in cluster_meta.values():
        cls = c['dominant_class']
        class_counts[cls] = class_counts.get(cls, 0) + 1

    print("\n" + "="*60)
    print("CLUSTERING RESULTS SUMMARY")
    print("="*60)
    print(f"\nTotal clusters: {len(cluster_meta)}")
    print(f"Total parts: {sum(sizes)}")
    print(f"\nPurity Statistics:")
    print(f"  Mean purity: {np.mean(purities):.1%}")
    print(f"  Min purity:  {np.min(purities):.1%}")
    print(f"  Max purity:  {np.max(purities):.1%}")
    print(f"  100% pure clusters: {sum(1 for p in purities if p == 1.0)} / {len(purities)}")

    print(f"\nCluster Size Statistics:")
    print(f"  Mean size: {np.mean(sizes):.1f}")
    print(f"  Min size:  {np.min(sizes)}")
    print(f"  Max size:  {np.max(sizes)}")

    print(f"\nClusters per class:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count} clusters")

    return purities, sizes

def plot_cluster_samples(masks, cluster_labels, part_to_class, cluster_meta, n_clusters=20, n_samples=5):
    """Plot sample masks from top clusters"""

    # Sort clusters by size
    sorted_clusters = sorted(cluster_meta.items(), key=lambda x: x[1]['size'], reverse=True)
    top_clusters = sorted_clusters[:n_clusters]

    fig, axes = plt.subplots(n_clusters, n_samples + 1, figsize=(2*(n_samples+1), 2*n_clusters))
    fig.suptitle('Top Clusters by Size (with sample attention masks)', fontsize=14, y=1.02)

    for i, (cid, meta) in enumerate(top_clusters):
        cid = int(cid)
        # Cluster info
        axes[i, 0].text(0.5, 0.5,
                       f"C{cid}\n{meta['dominant_class']}\n{meta['purity']:.0%} pure\nN={meta['size']}",
                       ha='center', va='center', fontsize=9,
                       transform=axes[i, 0].transAxes)
        axes[i, 0].axis('off')

        # Get parts in this cluster
        part_indices = np.where(cluster_labels == cid)[0]

        # Sample parts
        if len(part_indices) > n_samples:
            sampled = np.random.choice(part_indices, n_samples, replace=False)
        else:
            sampled = part_indices

        for j, pidx in enumerate(sampled):
            if j >= n_samples:
                break
            mask = masks[pidx]
            axes[i, j+1].imshow(mask, cmap='hot')
            axes[i, j+1].axis('off')

        # Hide unused axes
        for j in range(len(sampled), n_samples):
            axes[i, j+1].axis('off')

    plt.tight_layout()
    save_path = './parts/clusters/cluster_samples.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved cluster samples to: {save_path}")
    plt.show()

def plot_purity_histogram(cluster_meta):
    """Plot histogram of cluster purities"""
    purities = [c['purity'] for c in cluster_meta.values()]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(purities, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(purities), color='red', linestyle='--', label=f'Mean: {np.mean(purities):.1%}')
    ax.set_xlabel('Cluster Purity')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Distribution of Cluster Purities')
    ax.legend()

    save_path = './parts/clusters/purity_histogram.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved purity histogram to: {save_path}")
    plt.show()

def main():
    print("Loading data...")
    masks, bboxes, cluster_labels, part_to_class, part_to_image, cluster_meta, parts_meta = load_data()

    # Compute and print stats
    purities, sizes = compute_cluster_stats(cluster_meta)

    # Plot visualizations
    print("\nGenerating visualizations...")
    plot_purity_histogram(cluster_meta)
    plot_cluster_samples(masks, cluster_labels, part_to_class, cluster_meta, n_clusters=15, n_samples=6)

    print("\n✓ Visualization complete!")

if __name__ == '__main__':
    main()

