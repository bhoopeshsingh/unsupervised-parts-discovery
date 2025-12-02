import numpy as np
import json
from pathlib import Path
import pandas as pd

def analyze_distribution():
    clusters_dir = Path('parts/clusters')
    
    print("Loading clustering results...")
    try:
        cluster_labels = np.load(clusters_dir / 'cluster_labels.npy')
        part_to_class = np.load(clusters_dir / 'part_to_class.npy')
        
        with open(clusters_dir / 'cluster_metadata.json', 'r') as f:
            metadata = json.load(f)
            class_names = metadata['classes']
            
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    n_clusters = len(np.unique(cluster_labels))
    print(f"\nAnalyzing {n_clusters} clusters across {len(class_names)} classes: {class_names}")
    
    # Create DataFrame
    df = pd.DataFrame({
        'cluster': cluster_labels,
        'class_idx': part_to_class
    })
    df['class_name'] = df['class_idx'].apply(lambda x: class_names[x])
    
    # Calculate distribution
    print("\nCluster Composition (Top 3 Classes per Cluster):")
    print("-" * 60)
    print(f"{'Cluster':<8} {'Size':<8} {'Dominant Class':<15} {'%':<6} {'2nd Class':<15} {'%':<6}")
    print("-" * 60)
    
    mixed_clusters = 0
    class_specific_clusters = 0
    
    for k in range(n_clusters):
        cluster_data = df[df['cluster'] == k]
        size = len(cluster_data)
        if size == 0: continue
        
        counts = cluster_data['class_name'].value_counts()
        total = len(cluster_data)
        
        top1_name = counts.index[0]
        top1_pct = (counts.iloc[0] / total) * 100
        
        top2_name = counts.index[1] if len(counts) > 1 else "-"
        top2_pct = (counts.iloc[1] / total) * 100 if len(counts) > 1 else 0
        
        print(f"{k:<8} {size:<8} {top1_name:<15} {top1_pct:5.1f}  {top2_name:<15} {top2_pct:5.1f}")
        
        if top1_pct > 50:
            class_specific_clusters += 1
        else:
            mixed_clusters += 1
            
    print("-" * 60)
    print(f"\nSummary:")
    print(f"  Class-Specific Clusters (>50% one class): {class_specific_clusters}")
    print(f"  Mixed/Generic Clusters: {mixed_clusters}")

if __name__ == "__main__":
    analyze_distribution()
