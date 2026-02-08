"""Cluster extracted parts using spatially-aware clustering"""

import sys
sys.path.append('.')

import numpy as np
from pathlib import Path
import argparse
import json

from src.clustering.cluster import (
    cluster_parts_with_spatial_awareness,
    FeatureWeights,
    analyze_cluster_class_correlation,
    visualize_clusters_tsne,
    visualize_clusters_by_class,
    split_rich_descriptors,
    combine_weighted_features,
    refine_clusters
)
import warnings
warnings.filterwarnings('ignore')

def load_rich_parts_data(parts_dir):
    """Load extracted rich parts data"""
    parts_dir = Path(parts_dir)
    print(f"Loading parts data from: {parts_dir}")
    
    try:
        features = np.load(parts_dir / 'features.npy')
        part_to_image = np.load(parts_dir / 'part_to_image.npy')
        part_to_slot = np.load(parts_dir / 'part_to_slot.npy')
        part_to_class = np.load(parts_dir / 'part_to_class.npy')
        
        with open(parts_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
            
        print(f"  Loaded {len(features)} parts")
        print(f"  Feature dim: {features.shape[1]}")
        
        return features, part_to_image, part_to_slot, part_to_class, metadata
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Make sure you have run extract_parts.py with the new PartExtractor.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Cluster extracted parts')
    parser.add_argument('--parts-dir', type=str, default='./parts/extracted',
                        help='Directory containing extracted parts')
    parser.add_argument('--output-dir', type=str, default='./parts/clusters',
                        help='Directory to save clustering results')
    parser.add_argument('--n-clusters', type=int, default=100,
                        help='Number of clusters (recommended: 100-200 for better granularity)')
    parser.add_argument('--method', type=str, default='agglomerative',
                        choices=['kmeans', 'agglomerative'],
                        help='Clustering method')
    parser.add_argument('--visual-weight', type=float, default=1.5,
                        help='Weight for ResNet visual features (primary signal)')
    parser.add_argument('--spatial-weight', type=float, default=0.3)
    parser.add_argument('--shape-weight', type=float, default=0.5)
    parser.add_argument('--slot-weight', type=float, default=0.3,
                        help='Weight for slot features (reduced - less discriminative alone)')
    parser.add_argument('--refine', action='store_true', default=True,
                        help='Apply post-processing to merge small clusters')
    parser.add_argument('--min-cluster-size', type=int, default=10,
                        help='Minimum samples per cluster (smaller clusters get merged)')

    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    features, part_to_image, part_to_slot, part_to_class, metadata = load_rich_parts_data(args.parts_dir)
    
    # Feature dims from metadata (with fallback defaults)
    dims = metadata.get('feature_dims', {
        'slot': 128, 'visual': 2048, 'spatial': 5, 'shape': 3
    })
    
    # Define weights
    weights = FeatureWeights(
        visual=args.visual_weight,
        spatial=args.spatial_weight,
        shape=args.shape_weight,
        slot=args.slot_weight
    )
    
    # Cluster
    cluster_labels, metrics = cluster_parts_with_spatial_awareness(
        part_descriptors=features,
        n_clusters=args.n_clusters,
        weights=weights,
        method=args.method,
        slot_dim=dims['slot'],
        visual_dim=dims['visual'],
        spatial_dim=dims['spatial'],
        shape_dim=dims['shape']
    )
    
    # Prepare weighted features (needed for refinement and visualization)
    components = split_rich_descriptors(
        features,
        dims['slot'], dims['visual'], dims['spatial'], dims['shape']
    )
    weighted_features = combine_weighted_features(components, weights)

    # Post-process: refine clusters by merging small ones
    if args.refine:
        print(f"\n=== Post-Processing: Refining Clusters ===")
        cluster_labels, refine_stats = refine_clusters(
            cluster_labels=cluster_labels,
            features=weighted_features,
            min_cluster_size=args.min_cluster_size
        )
        metrics['refinement'] = refine_stats
        print(f"Final cluster count: {refine_stats['final_clusters']}")

    # Save results
    print(f"\nSaving results to: {output_dir}")
    np.save(output_dir / 'cluster_labels.npy', cluster_labels)
    np.save(output_dir / 'part_to_image.npy', part_to_image)
    np.save(output_dir / 'part_to_slot.npy', part_to_slot)
    np.save(output_dir / 'part_to_class.npy', part_to_class)
    
    with open(output_dir / 'clustering_metrics.json', 'w') as f:
        # Helper for JSON serialization
        def convert(o):
            if isinstance(o, np.generic): return o.item()
            return o
        json.dump(metrics, f, indent=2, default=convert)
        
    # Analysis
    print("\nAnalyzing class correlation...")
    cooccurrence = analyze_cluster_class_correlation(
        cluster_labels,
        part_to_class, 
        metadata['classes']
    )
    
    # Save co-occurrence matrix
    np.save(output_dir / 'cluster_class_cooccurrence.npy', cooccurrence)

    # Create detailed cluster metadata
    print("\nGenerating cluster metadata...")
    cluster_meta = {}
    unique_clusters = np.unique(cluster_labels)
    for cid in unique_clusters:
        mask = cluster_labels == cid
        cluster_classes = part_to_class[mask]
        class_counts = np.bincount(cluster_classes, minlength=len(metadata['classes']))
        dominant_idx = int(np.argmax(class_counts))
        cluster_meta[int(cid)] = {
            'size': int(mask.sum()),
            'dominant_class': metadata['classes'][dominant_idx],
            'dominant_class_idx': dominant_idx,
            'purity': float(class_counts[dominant_idx] / mask.sum()),
            'class_distribution': {metadata['classes'][i]: int(c) for i, c in enumerate(class_counts)}
        }

    with open(output_dir / 'cluster_metadata.json', 'w') as f:
        json.dump(cluster_meta, f, indent=2)

    # t-SNE visualization (weighted_features already computed above)
    print("\nRunning t-SNE visualization...")
    try:
        visualize_clusters_tsne(
            weighted_features, 
            cluster_labels, 
            part_to_class, 
            save_path=output_dir / 'tsne.png'
        )
        visualize_clusters_by_class(
            weighted_features,
            cluster_labels,
            part_to_class,
            metadata['classes'],
            save_path=output_dir / 'tsne_by_class.png'
        )
    except Exception as e:
        print(f"t-SNE visualization skipped due to error: {e}")

    print("\n✓ Clustering complete!")

if __name__ == '__main__':
    main()
