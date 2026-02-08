#!/usr/bin/env python
"""
Analyze clusters and find the best ones based on quality metrics.

Quality metrics:
1. Size - not too small (noise) or too large (background)
2. Spatial compactness - masks should be concentrated, not scattered
3. Consistency - masks within cluster should look similar
"""

import sys
sys.path.append('.')

import numpy as np
import json
from pathlib import Path
from scipy import ndimage
import cv2


def compute_mask_compactness(mask):
    """
    Compute how compact/focused a mask is.
    Higher = more concentrated attention (better for parts)
    """
    # Threshold mask
    binary = (mask > 0.3).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0

    # Get largest contour
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)

    if perimeter == 0:
        return 0.0

    # Circularity: 4*pi*area / perimeter^2 (1.0 = perfect circle)
    circularity = 4 * np.pi * area / (perimeter ** 2)

    return min(circularity, 1.0)


def compute_mask_coverage(mask):
    """Compute what fraction of the image the mask covers."""
    return (mask > 0.3).sum() / mask.size


def compute_mask_entropy(mask):
    """
    Compute entropy of mask - lower entropy = more confident/focused attention.
    """
    # Normalize to probability distribution
    mask_norm = mask / (mask.sum() + 1e-8)
    mask_norm = mask_norm.flatten()
    mask_norm = mask_norm[mask_norm > 1e-10]  # Remove zeros

    entropy = -np.sum(mask_norm * np.log(mask_norm + 1e-10))
    max_entropy = np.log(mask.size)

    # Return normalized entropy (0 = very focused, 1 = uniform)
    return entropy / max_entropy


def compute_cluster_quality(cluster_masks):
    """
    Compute overall quality score for a cluster.

    Returns dict with:
    - compactness: average mask compactness
    - coverage_mean: average coverage
    - coverage_std: coverage consistency
    - entropy_mean: average entropy (lower = better)
    - quality_score: combined score (higher = better)
    """
    compactness_scores = []
    coverage_scores = []
    entropy_scores = []

    for mask in cluster_masks:
        compactness_scores.append(compute_mask_compactness(mask))
        coverage_scores.append(compute_mask_coverage(mask))
        entropy_scores.append(compute_mask_entropy(mask))

    compactness_mean = np.mean(compactness_scores)
    coverage_mean = np.mean(coverage_scores)
    coverage_std = np.std(coverage_scores)
    entropy_mean = np.mean(entropy_scores)

    # Quality score: high compactness, moderate coverage (5-40%), low entropy, consistent coverage
    # Penalize very small (<5%) or very large (>50%) coverage
    coverage_penalty = 0
    if coverage_mean < 0.05:
        coverage_penalty = (0.05 - coverage_mean) * 10
    elif coverage_mean > 0.50:
        coverage_penalty = (coverage_mean - 0.50) * 5

    quality_score = (
        compactness_mean * 0.3 +           # Reward compact shapes
        (1 - entropy_mean) * 0.3 +         # Reward focused attention
        (1 - coverage_std) * 0.2 +         # Reward consistency
        (1 - coverage_penalty) * 0.2       # Penalize extreme coverage
    )

    return {
        'compactness': float(compactness_mean),
        'coverage_mean': float(coverage_mean),
        'coverage_std': float(coverage_std),
        'entropy': float(entropy_mean),
        'quality_score': float(quality_score)
    }


def main():
    parts_dir = Path('./parts/extracted')
    clusters_dir = Path('./parts/clusters')

    print("Loading data...")
    masks = np.load(parts_dir / 'masks.npy')
    cluster_labels = np.load(clusters_dir / 'cluster_labels.npy')

    with open(clusters_dir / 'cluster_metadata.json', 'r') as f:
        cluster_meta = json.load(f)

    n_clusters = len(cluster_meta)
    print(f"Analyzing {n_clusters} clusters...")

    # Analyze each cluster
    cluster_quality = {}

    for cid in range(n_clusters):
        cid_str = str(cid)
        if cid_str not in cluster_meta:
            continue

        # Get masks for this cluster
        cluster_mask = cluster_labels == cid
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) < 5:
            # Too few samples to analyze
            cluster_quality[cid] = {
                'size': len(cluster_indices),
                'quality_score': 0.0,
                'reason': 'too_few_samples'
            }
            continue

        # Sample up to 50 masks for analysis
        if len(cluster_indices) > 50:
            sample_indices = np.random.choice(cluster_indices, 50, replace=False)
        else:
            sample_indices = cluster_indices

        cluster_masks = masks[sample_indices]

        # Compute quality metrics
        quality = compute_cluster_quality(cluster_masks)
        quality['size'] = int(cluster_meta[cid_str]['size'])

        cluster_quality[cid] = quality

    # Rank clusters by quality score
    ranked = sorted(cluster_quality.items(), key=lambda x: x[1].get('quality_score', 0), reverse=True)

    print("\n" + "="*80)
    print("CLUSTER QUALITY RANKING")
    print("="*80)
    print(f"{'Rank':<6}{'Cluster':<10}{'Score':<10}{'Size':<10}{'Compact':<12}{'Coverage':<12}{'Entropy':<10}")
    print("-"*80)

    # Show top 20
    for rank, (cid, quality) in enumerate(ranked[:20], 1):
        if 'reason' in quality:
            print(f"{rank:<6}{cid:<10}{quality['quality_score']:<10.3f}{quality['size']:<10}{'N/A':<12}{'N/A':<12}{'N/A':<10}")
        else:
            print(f"{rank:<6}{cid:<10}{quality['quality_score']:<10.3f}{quality['size']:<10}{quality['compactness']:<12.3f}{quality['coverage_mean']:<12.3f}{quality['entropy']:<10.3f}")

    print("\n" + "="*80)
    print("TOP 10 BEST CLUSTERS (Recommended for labeling)")
    print("="*80)

    top_10 = [cid for cid, _ in ranked[:10]]
    print(f"\nCluster IDs: {top_10}")

    print("\nThese clusters have:")
    print("  - High compactness (focused attention)")
    print("  - Moderate coverage (not too small/large)")
    print("  - Low entropy (confident attention)")
    print("  - Consistent masks within cluster")

    # Show bottom 10 (worst clusters - likely noise/background)
    print("\n" + "="*80)
    print("BOTTOM 10 WORST CLUSTERS (Likely noise/background)")
    print("="*80)

    for rank, (cid, quality) in enumerate(ranked[-10:], len(ranked)-9):
        if 'reason' in quality:
            print(f"{rank:<6}{cid:<10}{quality['quality_score']:<10.3f}{quality['size']:<10}{'N/A':<12}{'N/A':<12}{'N/A':<10}")
        else:
            print(f"{rank:<6}{cid:<10}{quality['quality_score']:<10.3f}{quality['size']:<10}{quality['compactness']:<12.3f}{quality['coverage_mean']:<12.3f}{quality['entropy']:<10.3f}")

    # Save results
    output_path = clusters_dir / 'cluster_quality_ranking.json'
    with open(output_path, 'w') as f:
        json.dump({
            'ranking': [{'cluster_id': cid, **quality} for cid, quality in ranked],
            'top_10': top_10,
            'bottom_10': [cid for cid, _ in ranked[-10:]]
        }, f, indent=2)

    print(f"\n✓ Quality ranking saved to: {output_path}")

    # Update Streamlit tip
    print("\n" + "="*80)
    print("TIP: In Streamlit, focus on these top clusters first:")
    print(f"  {top_10}")
    print("="*80)


if __name__ == '__main__':
    main()

