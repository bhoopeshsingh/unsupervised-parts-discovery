import numpy as np
import sys
from pathlib import Path

clusters_dir = Path('./parts/clusters')
cooc_path = clusters_dir / 'cluster_class_cooccurrence.npy'

try:
    matrix = np.load(cooc_path)
    print("Cluster-Class Co-occurrence Matrix:")
    print(matrix)
    
    # Calculate Purity
    # For each cluster, what is the dominant class percentage?
    cluster_sums = matrix.sum(axis=1)
    # Avoid division by zero
    valid_clusters = cluster_sums > 0
    
    max_counts = matrix[valid_clusters].max(axis=1)
    purities = max_counts / cluster_sums[valid_clusters]
    
    print(f"\nAverage Cluster Purity: {purities.mean():.2%}")
    print(f"Min Cluster Purity: {purities.min():.2%}")
    print(f"Max Cluster Purity: {purities.max():.2%}")
    
    # Check if any cluster is "discriminative" (> 50% purity for 3 classes)
    discriminative_clusters = np.sum(purities > 0.5)
    print(f"\nNumber of Discriminative Clusters (>50% purity): {discriminative_clusters} / {len(purities)}")
    
except FileNotFoundError:
    print(f"Could not find {cooc_path}. Please run clustering first.")
except Exception as e:
    print(f"Error: {e}")
