"""Cluster extracted parts"""

import sys
sys.path.append('.')

import numpy as np
from pathlib import Path
import argparse
import json
import pickle

from src.clustering.cluster import (
    determine_optimal_k,
    plot_clustering_metrics,
    cluster_parts,
    visualize_clusters_tsne,
    analyze_cluster_class_correlation,
    visualize_clusters_by_class,
    visualize_clusters_per_class_separate_files
)


def load_parts_data(parts_dir):
    """Load extracted parts data"""
    parts_dir = Path(parts_dir)
    
    print(f"Loading parts data from: {parts_dir}")
    
    # Load arrays
    slots = np.load(parts_dir / 'slots.npy')
    masks = np.load(parts_dir / 'masks.npy')
    image_ids = np.load(parts_dir / 'image_ids.npy')
    labels = np.load(parts_dir / 'labels.npy')
    
    # Load metadata
    with open(parts_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"  Loaded {len(image_ids)} images")
    print(f"  Slots shape: {slots.shape}")
    print(f"  Masks shape: {masks.shape}")
    
    return {
        'slots': slots,
        'masks': masks,
        'image_ids': image_ids,
        'labels': labels,
        'metadata': metadata
    }


def flatten_slots_for_clustering(slots):
    """
    Flatten slots from [N_images, N_slots, D] to [N_images*N_slots, D]
    
    This treats each slot from each image as a separate part to cluster
    """
    N_images, N_slots, D = slots.shape
    flattened = slots.reshape(-1, D)
    
    print(f"\nFlattened slots for clustering:")
    print(f"  Original: {N_images} images × {N_slots} slots × {D} dims")
    print(f"  Flattened: {flattened.shape[0]} total parts × {D} dims")
    
    return flattened


def main():
    parser = argparse.ArgumentParser(description='Cluster extracted parts')
    parser.add_argument('--parts-dir', type=str, default='./parts/extracted',
                        help='Directory containing extracted parts')
    parser.add_argument('--output-dir', type=str, default='./parts/clusters',
                        help='Directory to save clustering results')
    # Removed manual K argument to enforce self-determination
    parser.add_argument('--k-min', type=int, default=10,
                        help='Minimum K to test for optimal K selection')
    parser.add_argument('--k-max', type=int, default=40,
                        help='Maximum K to test for optimal K selection')
    parser.add_argument('--skip-tsne', action='store_true',
                        help='Skip t-SNE visualization (faster)')
    parser.add_argument('--visualize-only', action='store_true',
                        help='Skip clustering and only run visualization (requires existing clustering results)')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load extracted parts
    parts_data = load_parts_data(args.parts_dir)
    
    # Flatten slots for clustering
    # Each slot from each image becomes a separate data point
    features = flatten_slots_for_clustering(parts_data['slots'])
    
    # Normalize features (L2 normalization)
    # This is crucial for clustering embeddings, effectively using cosine similarity
    print("Normalizing features (L2)...")
    from sklearn.preprocessing import normalize
    features = normalize(features, norm='l2', axis=1)
    
    # Also create mapping from part index to (image_id, slot_id)
    N_images, N_slots = parts_data['slots'].shape[:2]
    part_to_image = np.repeat(np.arange(N_images), N_slots)
    part_to_slot = np.tile(np.arange(N_slots), N_images)
    part_to_class = np.repeat(parts_data['labels'], N_slots)
    
    if args.visualize_only:
        print("\n" + "="*70)
        print("VISUALIZATION ONLY MODE")
        print("="*70)
        
        # Load existing clustering results
        try:
            cluster_labels = np.load(output_dir / 'cluster_labels.npy')
            print(f"Loaded existing cluster labels from {output_dir / 'cluster_labels.npy'}")
            
            with open(output_dir / 'cluster_metadata.json', 'r') as f:
                cluster_metadata = json.load(f)
            n_clusters = cluster_metadata['n_clusters']
            print(f"Loaded metadata (K={n_clusters})")
            
        except FileNotFoundError:
            print("Error: Could not load existing clustering results. Run without --visualize-only first.")
            return
            
    else:
        # Per-Class Clustering Strategy
        print("\n" + "="*70)
        print("PER-CLASS CLUSTERING (SELF-DETERMINATION MODE)")
        print("="*70)
        
        from src.clustering.cluster import cluster_parts_per_class
        
        cluster_labels, cluster_metadata, metrics = cluster_parts_per_class(
            features=features,
            class_labels=part_to_class,
            class_names=parts_data['metadata']['classes'],
            k_range=(args.k_min, args.k_max),
            n_init=20,
            random_state=42
        )
        
        # Save clustering results
        print(f"\nSaving clustering results to: {output_dir}")
        
        np.save(output_dir / 'cluster_labels.npy', cluster_labels)
        np.save(output_dir / 'part_to_image.npy', part_to_image)
        np.save(output_dir / 'part_to_slot.npy', part_to_slot)
        np.save(output_dir / 'part_to_class.npy', part_to_class)
        
        # Save metrics
        with open(output_dir / 'clustering_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
            
        # Save cluster metadata (mapping global ID to class/local ID)
        # Convert int64 to int for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(output_dir / 'cluster_metadata.json', 'w') as f:
            json.dump(cluster_metadata, f, indent=2, default=convert_numpy)
        
        # Analyze cluster-class correlation (should be diagonal-ish now)
        print("\n" + "="*70)
        print("ANALYZING CLUSTER-CLASS CORRELATION")
        print("="*70)
        
        cooccurrence_matrix = analyze_cluster_class_correlation(
            cluster_labels=cluster_labels,
            class_labels=part_to_class,
            class_names=parts_data['metadata']['classes']
        )
        
        np.save(output_dir / 'cluster_class_cooccurrence.npy', cooccurrence_matrix)
    
    # Visualize with t-SNE
    if not args.skip_tsne:
        print("\n" + "="*70)
        print("CREATING t-SNE VISUALIZATIONS")
        print("="*70)
        
        # Global visualization
        visualize_clusters_tsne(
            features=features,
            labels=cluster_labels,
            class_labels=part_to_class,
            perplexity=30,
            save_path=output_dir / 'tsne_projection.png'
        )
        
        # Per-class visualization (subplots)
        visualize_clusters_by_class(
            features=features,
            labels=cluster_labels,
            class_labels=part_to_class,
            class_names=parts_data['metadata']['classes'],
            perplexity=30,
            save_path=output_dir / 'tsne_by_class.png'
        )
        
        # Per-class visualization (separate files)
        visualize_clusters_per_class_separate_files(
            features=features,
            labels=cluster_labels,
            class_labels=part_to_class,
            class_names=parts_data['metadata']['classes'],
            perplexity=30,
            output_dir=output_dir
        )
    
    print("\n" + "="*70)
    print("✓ CLUSTERING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review global t-SNE visualization: {output_dir / 'tsne_projection.png'}")
    print(f"  2. Review per-class t-SNE visualization: {output_dir / 'tsne_by_class.png'}")
    print(f"  3. Review individual class plots in: {output_dir}")
    print(f"  4. Review clustering metrics: {output_dir / 'clustering_metrics.png'}")
    print(f"  5. Launch labeling interface: streamlit run src/clustering/streamlit_labeler.py")


if __name__ == '__main__':
    main()
