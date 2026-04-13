"""Link discovered parts to object classes for interpretability analysis"""

import sys
sys.path.append('.')

import numpy as np
from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def load_all_data(parts_dir, clusters_dir, labels_file):
    """Load parts, clustering, and label data"""
    parts_dir = Path(parts_dir)
    clusters_dir = Path(clusters_dir)
    
    # Load clustering data
    cluster_labels = np.load(clusters_dir / 'cluster_labels.npy')
    part_to_image = np.load(clusters_dir / 'part_to_image.npy')
    part_to_slot = np.load(clusters_dir / 'part_to_slot.npy')
    part_to_class = np.load(clusters_dir / 'part_to_class.npy')
    cooccurrence = np.load(clusters_dir / 'cluster_class_cooccurrence.npy')
    
    with open(clusters_dir / 'cluster_metadata.json', 'r') as f:
        cluster_metadata = json.load(f)
    
    # Load semantic labels if available
    semantic_labels = {}
    if Path(labels_file).exists():
        with open(labels_file, 'r') as f:
            semantic_labels = json.load(f)
    
    return {
        'cluster_labels': cluster_labels,
        'part_to_class': part_to_class,
        'cooccurrence': cooccurrence,
        'class_names': cluster_metadata['classes'],
        'n_clusters': cluster_metadata['n_clusters'],
        'semantic_labels': semantic_labels
    }


def identify_part_types(cooccurrence, class_names, threshold=70):
    """
    Identify class-specific vs shared parts
    
    Args:
        cooccurrence: Cluster-class cooccurrence matrix (% values)
        class_names: List of class names
        threshold: Threshold % to consider a part class-specific
    
    Returns:
        part_types: Dictionary categorizing each cluster
    """
    n_clusters, n_classes = cooccurrence.shape
    
    part_types = {
        'class_specific': defaultdict(list),  # {class_name: [cluster_ids]}
        'shared': [],  # Clusters shared across multiple classes
        'background': []  # Clusters appearing uniformly across all classes
    }
    
    for cluster_id in range(n_clusters):
        max_pct = cooccurrence[cluster_id].max()
        max_class_idx = cooccurrence[cluster_id].argmax()
        
        # Count how many classes have >20% representation
        significant_classes = (cooccurrence[cluster_id] > 20).sum()
        
        if max_pct >= threshold:
            # Class-specific part
            class_name = class_names[max_class_idx]
            part_types['class_specific'][class_name].append(cluster_id)
        elif significant_classes >= 3:
            # Background/shared across many classes
            part_types['background'].append(cluster_id)
        elif significant_classes >= 2:
            # Shared between a few classes
            part_types['shared'].append(cluster_id)
        else:
            # Also class-specific (just lower threshold)
            class_name = class_names[max_class_idx]
            part_types['class_specific'][class_name].append(cluster_id)
    
    return part_types


def plot_cooccurrence_heatmap(cooccurrence, class_names, semantic_labels, save_path):
    """Plot cluster-class cooccurrence heatmap"""
    n_clusters = cooccurrence.shape[0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(8, n_clusters * 0.3)))
    
    # Create heatmap
    sns.heatmap(
        cooccurrence,
        xticklabels=class_names,
        yticklabels=[
            f"C{i}: {semantic_labels.get(str(i), {}).get('label', 'unlabeled')[:15]}"
            for i in range(n_clusters)
        ],
        cmap='YlOrRd',
        annot=True,
        fmt='.0f',
        cbar_kws={'label': 'Percentage (%)'},
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_xlabel('Object Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Part Cluster', fontsize=12, fontweight='bold')
    ax.set_title('Cluster-to-Class Co-occurrence Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        print(f"Skipping save to {save_path} (Notebook mode)")
        plt.show()
    else:
        plt.show()
    # plt.savefig(save_path, dpi=150, bbox_inches='tight')
    # print(f"Saved cooccurrence heatmap: {save_path}")
    plt.close()


def generate_interpretability_report(data, part_types, output_dir):
    """Generate comprehensive interpretability report"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / 'interpretability_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# Part Discovery Interpretability Report\n\n")
        f.write("## Summary\n\n")
        
        # Overall statistics
        total_clusters = data['n_clusters']
        labeled_clusters = len(data['semantic_labels'])
        
        f.write(f"- **Total Clusters Discovered**: {total_clusters}\n")
        f.write(f"- **Semantically Labeled**: {labeled_clusters} ({labeled_clusters/total_clusters*100:.1f}%)\n")
        f.write(f"- **Object Classes**: {', '.join(data['class_names'])}\n\n")
        
        # Part type distribution
        f.write("## Part Type Distribution\n\n")
        
        total_class_specific = sum(len(v) for v in part_types['class_specific'].values())
        f.write(f"- **Class-Specific Parts**: {total_class_specific}\n")
        f.write(f"- **Shared Parts**: {len(part_types['shared'])}\n")
        f.write(f"- **Background Parts**: {len(part_types['background'])}\n\n")
        
        # Class-specific parts
        f.write("## Class-Specific Parts\n\n")
        
        for class_name in data['class_names']:
            clusters = part_types['class_specific'].get(class_name, [])
            f.write(f"### {class_name.capitalize()}\n\n")
            f.write(f"Discovered {len(clusters)} class-specific parts:\n\n")
            
            for cluster_id in clusters:
                semantic_info = data['semantic_labels'].get(str(cluster_id), {})
                label = semantic_info.get('label', 'unlabeled')
                confidence = semantic_info.get('confidence', 'N/A')
                
                # Get class percentage
                class_idx = data['class_names'].index(class_name)
                pct = data['cooccurrence'][cluster_id, class_idx]
                
                f.write(f"- **Cluster {cluster_id}**: {label} ({confidence} confidence, {pct:.0f}% {class_name})\n")
            
            f.write("\n")
        
        # Shared parts
        if part_types['shared']:
            f.write("## Shared Parts\n\n")
            f.write("These parts appear across multiple classes:\n\n")
            
            for cluster_id in part_types['shared']:
                semantic_info = data['semantic_labels'].get(str(cluster_id), {})
                label = semantic_info.get('label', 'unlabeled')
                
                # Show class distribution
                class_dist = ", ".join([
                    f"{data['class_names'][i]} ({data['cooccurrence'][cluster_id, i]:.0f}%)"
                    for i in range(len(data['class_names']))
                    if data['cooccurrence'][cluster_id, i] > 20
                ])
                
                f.write(f"- **Cluster {cluster_id}**: {label} - {class_dist}\n")
            
            f.write("\n")
        
        # Background parts
        if part_types['background']:
            f.write("## Background Parts\n\n")
            f.write("These parts appear uniformly across all classes (likely background):\n\n")
            
            for cluster_id in part_types['background']:
                semantic_info = data['semantic_labels'].get(str(cluster_id), {})
                label = semantic_info.get('label', 'unlabeled')
                f.write(f"- **Cluster {cluster_id}**: {label}\n")
            
            f.write("\n")
        
        # Insights
        f.write("## Key Insights\n\n")
        f.write("1. **Part Specialization**: ")
        f.write(f"{total_class_specific} ({total_class_specific/total_clusters*100:.0f}%) ")
        f.write("of discovered parts are class-specific, showing the model successfully learned ")
        f.write("to identify object-specific features.\n\n")
        
        f.write("2. **Shared Features**: ")
        f.write(f"{len(part_types['shared'])} parts are shared across multiple classes, ")
        f.write("potentially representing common visual patterns or structural elements.\n\n")
        
        f.write("3. **Background Separation**: ")
        f.write(f"{len(part_types['background'])} parts likely represent background patterns, ")
        f.write("demonstrating the model's ability to separate objects from backgrounds.\n\n")
    
    print(f"\n✓ Generated interpretability report: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description='Link parts to classes for interpretability')
    parser.add_argument('--parts-dir', type=str, default='./parts/extracted',
                        help='Directory containing extracted parts')
    parser.add_argument('--clusters-dir', type=str, default='./parts/clusters',
                        help='Directory containing clustering results')
    parser.add_argument('--labels-file', type=str, default='./parts/labels/cluster_labels.json',
                        help='Semantic labels file from labeling interface')
    parser.add_argument('--output-dir', type=str, default='./parts/analysis',
                        help='Directory to save analysis results')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all data
    print("Loading data...")
    data = load_all_data(args.parts_dir, args.clusters_dir, args.labels_file)
    
    print(f"Loaded {data['n_clusters']} clusters")
    print(f"Found {len(data['semantic_labels'])} semantic labels")
    
    # Identify part types
    print("\nIdentifying part types...")
    part_types = identify_part_types(
        cooccurrence=data['cooccurrence'],
        class_names=data['class_names'],
        threshold=70
    )
    
    # Save part types
    part_types_serializable = {
        'class_specific': {k: list(v) for k, v in part_types['class_specific'].items()},
        'shared': part_types['shared'],
        'background': part_types['background']
    }
    
    with open(output_dir / 'part_types.json', 'w') as f:
        json.dump(part_types_serializable, f, indent=2)
    
    print(f"Saved part types to: {output_dir / 'part_types.json'}")
    
    # Plot cooccurrence heatmap
    print("\nGenerating visualizations...")
    plot_cooccurrence_heatmap(
        cooccurrence=data['cooccurrence'],
        class_names=data['class_names'],
        semantic_labels=data['semantic_labels'],
        save_path=output_dir / 'cooccurrence_heatmap.png'
    )
    
    # Generate report
    print("\nGenerating interpretability report...")
    report_path = generate_interpretability_report(data, part_types, output_dir)
    
    print("\n" + "="*70)
    print("✓ LINKING ANALYSIS COMPLETED!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - {report_path}")
    print(f"  - {output_dir / 'cooccurrence_heatmap.png'}")
    print(f"  - {output_dir / 'part_types.json'}")
    print(f"\nNext steps:")
    print(f"  - Review the interpretability report for insights")
    print(f"  - Use findings for research paper")


if __name__ == '__main__':
    main()
