import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

def determine_optimal_k(features, k_range=(10, 40), n_init=10, random_state=42):
    """
    Determine optimal number of clusters using elbow method and silhouette score.
    
    Args:
        features: Input features [N, D]
        k_range: Tuple of (min_k, max_k)
        n_init: Number of initializations for KMeans
        random_state: Random seed
        
    Returns:
        optimal_k: Best K value
        metrics: Dictionary containing inertia and silhouette scores for each K
    """
    print(f"Determining optimal K in range {k_range}...")
    
    inertias = []
    silhouette_scores = []
    k_values = range(k_range[0], k_range[1] + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        kmeans.fit(features)
        
        inertias.append(kmeans.inertia_)
        
        # Silhouette score (can be expensive for large N, sample if needed)
        if len(features) > 10000:
            indices = np.random.choice(len(features), 10000, replace=False)
            score = silhouette_score(features[indices], kmeans.labels_[indices])
        else:
            score = silhouette_score(features, kmeans.labels_)
            
        silhouette_scores.append(score)
        print(f"  K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={score:.4f}")
        
    # Simple heuristic: Maximize silhouette score
    best_idx = np.argmax(silhouette_scores)
    optimal_k = k_values[best_idx]
    
    print(f"Optimal K determined: {optimal_k} (Silhouette Score: {silhouette_scores[best_idx]:.4f})")
    
    return optimal_k, {
        'k_values': list(k_values),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores
    }

def plot_clustering_metrics(metrics, optimal_k, save_path):
    """Plot elbow curve and silhouette scores"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    k_values = metrics['k_values']
    
    # Elbow Curve
    ax1.plot(k_values, metrics['inertias'], 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    ax1.legend()
    
    # Silhouette Score
    ax2.plot(k_values, metrics['silhouette_scores'], 'go-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        print(f"Skipping save to {save_path} (Notebook mode)")
        plt.show()
    else:
        plt.show()

def cluster_parts(features, n_clusters, n_init=20, random_state=42):
    """
    Cluster extracted parts using K-Means.
    
    Args:
        features: Input features [N, D]
        n_clusters: Number of clusters
        
    Returns:
        labels: Cluster assignments [N]
        kmeans_model: Fitted KMeans object
    """
    print(f"Clustering {len(features)} parts into {n_clusters} clusters...")
    
    # Normalize features (L2 normalization)
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    norm[norm == 0] = 1e-10
    features = features / norm
    features = np.nan_to_num(features)
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=random_state
    )
    labels = kmeans.fit_predict(features)
    
    # Calculate quality metrics
    if len(features) > 10000:
        indices = np.random.choice(len(features), 10000, replace=False)
        sil_score = silhouette_score(features[indices], labels[indices])
    else:
        sil_score = silhouette_score(features, labels)
        
    db_score = davies_bouldin_score(features, labels)
    
    print(f"Clustering completed.")
    print(f"  Silhouette Score: {sil_score:.4f}")
    print(f"  Davies-Bouldin Score: {db_score:.4f}")
    
    return labels, kmeans

def cluster_parts_per_class(features, class_labels, class_names, k_range=(5, 20), n_init=20, random_state=42):
    """
    Cluster parts independently for each object class.
    
    Args:
        features: Input features [N, D]
        class_labels: Class index for each feature [N]
        class_names: List of class names
        k_range: Tuple of (min_k, max_k) for optimal K search
        
    Returns:
        global_cluster_labels: Cluster assignments [N] (unique across classes)
        cluster_metadata: Dictionary mapping global cluster ID to (class_name, local_cluster_id)
        metrics: Dictionary of metrics per class
    """
    print(f"\nStarting Per-Class Clustering for {len(class_names)} classes...")
    
    N = len(features)
    global_cluster_labels = np.zeros(N, dtype=int) - 1 # Initialize with -1
    cluster_metadata = {} # global_id -> {'class': name, 'local_id': id}
    metrics = {}
    
    # Normalize features (L2 normalization) to ensure cosine similarity behavior
    print("Normalizing features (L2) before clustering...")
    from sklearn.preprocessing import normalize
    # Add small epsilon to avoid division by zero for zero vectors
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    norm[norm == 0] = 1e-10
    features = features / norm
    features = np.nan_to_num(features)
    
    current_global_id_offset = 0
    
    for class_idx, class_name in enumerate(class_names):
        print(f"\n--- Processing Class: {class_name} ({class_idx}) ---")
        
        # Filter features for this class
        mask = (class_labels == class_idx)
        class_features = features[mask]
        indices = np.where(mask)[0]
        
        if len(class_features) < k_range[0]:
            print(f"Warning: Not enough samples for class {class_name} ({len(class_features)}). Skipping.")
            continue
            
        # Determine optimal K for this class
        # Adjust K range if we have few samples
        effective_max_k = min(k_range[1], len(class_features) - 1)
        effective_min_k = min(k_range[0], effective_max_k)
        
        if effective_max_k < effective_min_k:
             effective_max_k = effective_min_k
             
        optimal_k, class_metrics = determine_optimal_k(
            class_features, 
            k_range=(effective_min_k, effective_max_k),
            n_init=3,
            random_state=random_state
        )
        
        metrics[class_name] = {
            'optimal_k': int(optimal_k),
            'metrics': class_metrics
        }
        
        # Perform clustering
        print(f"Clustering {class_name} parts into {optimal_k} clusters...")
        kmeans = KMeans(
            n_clusters=optimal_k, 
            n_init=n_init, 
            random_state=random_state
        )
        local_labels = kmeans.fit_predict(class_features)
        
        # Assign global cluster IDs
        # global_id = offset + local_id
        for local_id in range(optimal_k):
            global_id = current_global_id_offset + local_id
            cluster_metadata[global_id] = {
                'class_name': class_name,
                'class_idx': class_idx,
                'local_cluster_id': int(local_id)
            }
            
        # Update global labels array
        global_cluster_labels[indices] = local_labels + current_global_id_offset
        
        # Store centroids in metadata
        # We need to store them as a list to be JSON serializable
        for local_id, centroid in enumerate(kmeans.cluster_centers_):
            global_id = current_global_id_offset + local_id
            cluster_metadata[global_id]['centroid'] = centroid.tolist()
            
        current_global_id_offset += optimal_k
        
    return global_cluster_labels, cluster_metadata, metrics


# =============================================================================
# Enhanced Spatially-Aware Clustering
# =============================================================================

@dataclass
class FeatureWeights:
    """Weights for combining different feature types."""
    visual: float = 1.0      # ResNet visual features
    spatial: float = 0.3     # Position, size, coverage
    shape: float = 0.5       # Aspect ratio, compactness, edge density
    slot: float = 0.5        # Slot attention learned features
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'visual': self.visual,
            'spatial': self.spatial,
            'shape': self.shape,
            'slot': self.slot
        }


def split_rich_descriptors(
    combined_features: np.ndarray,
    slot_dim: int = 128,
    visual_dim: int = 2048,
    spatial_dim: int = 5,
    shape_dim: int = 3
) -> Dict[str, np.ndarray]:
    """
    Split combined feature vectors into component types.
    
    Args:
        combined_features: Combined features [N, total_dim]
        slot_dim: Dimension of slot features
        visual_dim: Dimension of visual features
        spatial_dim: Dimension of spatial features
        shape_dim: Dimension of shape features
        
    Returns:
        Dictionary with 'slot', 'visual', 'spatial', 'shape' arrays
    """
    idx = 0
    result = {}
    
    result['slot'] = combined_features[:, idx:idx+slot_dim]
    idx += slot_dim
    
    result['visual'] = combined_features[:, idx:idx+visual_dim]
    idx += visual_dim
    
    result['spatial'] = combined_features[:, idx:idx+spatial_dim]
    idx += spatial_dim
    
    result['shape'] = combined_features[:, idx:idx+shape_dim]
    
    return result


def combine_weighted_features(
    feature_components: Dict[str, np.ndarray],
    weights: FeatureWeights = None
) -> np.ndarray:
    """
    Combine feature components with weights for clustering.
    
    Args:
        feature_components: Dict with 'slot', 'visual', 'spatial', 'shape' arrays
        weights: Feature weights
        
    Returns:
        Combined weighted features [N, total_weighted_dim]
    """
    weights = weights or FeatureWeights()
    
    # Normalize each component separately
    scaler = StandardScaler()
    
    weighted_parts = []
    
    for name, weight in weights.to_dict().items():
        if name in feature_components and weight > 0:
            features = feature_components[name]
            # Standardize
            normalized = scaler.fit_transform(features)
            # Apply weight
            weighted = normalized * weight
            weighted_parts.append(weighted)
    
    return np.concatenate(weighted_parts, axis=1)


def cluster_parts_with_spatial_awareness(
    part_descriptors: np.ndarray,
    n_clusters: int = 50,
    weights: FeatureWeights = None,
    method: str = 'agglomerative',
    slot_dim: int = 128,
    visual_dim: int = 2048,
    spatial_dim: int = 5,
    shape_dim: int = 3,
    random_state: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Cluster parts using visual + spatial similarity.
    
    This improved clustering:
    1. Weights feature types separately (visual most important)
    2. Uses hierarchical clustering for varying cluster sizes
    3. Returns quality metrics
    
    Args:
        part_descriptors: Combined part descriptors [N, D]
        n_clusters: Target number of clusters
        weights: Feature type weights
        method: 'agglomerative' or 'kmeans'
        slot_dim: Dimension of slot features
        visual_dim: Dimension of visual features
        spatial_dim: Dimension of spatial features
        shape_dim: Dimension of shape features
        random_state: Random seed
        
    Returns:
        cluster_labels: Cluster assignments [N]
        metrics: Dictionary with clustering quality metrics
    """
    weights = weights or FeatureWeights()
    
    print(f"\n=== Spatially-Aware Clustering ===")
    print(f"Input parts: {len(part_descriptors)}")
    print(f"Target clusters: {n_clusters}")
    print(f"Method: {method}")
    print(f"Weights: {weights.to_dict()}")
    
    # Split into components
    components = split_rich_descriptors(
        part_descriptors,
        slot_dim=slot_dim,
        visual_dim=visual_dim,
        spatial_dim=spatial_dim,
        shape_dim=shape_dim
    )
    
    # Combine with weights
    combined = combine_weighted_features(components, weights)
    print(f"Combined feature dim: {combined.shape[1]}")
    
    # Handle NaN values
    combined = np.nan_to_num(combined)
    
    # Perform clustering
    if method == 'agglomerative':
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        cluster_labels = clustering.fit_predict(combined)
    else:  # kmeans
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=20,
            random_state=random_state
        )
        cluster_labels = kmeans.fit_predict(combined)
    
    # Calculate quality metrics
    if len(part_descriptors) > 10000:
        indices = np.random.choice(len(combined), 10000, replace=False)
        sil_score = silhouette_score(combined[indices], cluster_labels[indices])
    else:
        sil_score = silhouette_score(combined, cluster_labels)
    
    db_score = davies_bouldin_score(combined, cluster_labels)
    
    # Cluster size distribution
    unique, counts = np.unique(cluster_labels, return_counts=True)
    size_stats = {
        'min': int(counts.min()),
        'max': int(counts.max()),
        'mean': float(counts.mean()),
        'std': float(counts.std())
    }
    
    metrics = {
        'silhouette_score': float(sil_score),
        'davies_bouldin_score': float(db_score),
        'n_clusters': int(n_clusters),
        'n_samples': int(len(part_descriptors)),
        'cluster_size_stats': size_stats,
        'weights': weights.to_dict(),
        'method': method
    }
    
    print(f"\nClustering Quality:")
    print(f"  Silhouette Score: {sil_score:.4f}")
    print(f"  Davies-Bouldin Score: {db_score:.4f}")
    print(f"  Cluster sizes: min={size_stats['min']}, max={size_stats['max']}, mean={size_stats['mean']:.1f}")
    
    return cluster_labels, metrics


def refine_clusters(
    cluster_labels: np.ndarray,
    features: np.ndarray,
    min_cluster_size: int = 5,
    max_intra_cluster_distance: float = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Post-process clusters: merge small clusters, optionally split diverse ones.
    
    Args:
        cluster_labels: Initial cluster assignments [N]
        features: Feature vectors [N, D]
        min_cluster_size: Minimum samples per cluster
        max_intra_cluster_distance: If set, split clusters exceeding this threshold
        
    Returns:
        refined_labels: Refined cluster assignments [N]
        stats: Refinement statistics
    """
    print(f"\n=== Cluster Refinement ===")
    print(f"Min cluster size: {min_cluster_size}")
    
    unique_clusters = np.unique(cluster_labels)
    n_original = len(unique_clusters)
    
    # Identify small clusters
    small_clusters = []
    valid_clusters = []
    
    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        if mask.sum() < min_cluster_size:
            small_clusters.append(cluster_id)
        else:
            valid_clusters.append(cluster_id)
    
    print(f"Found {len(small_clusters)} small clusters to merge")
    
    # For small clusters, assign to nearest valid cluster
    refined_labels = cluster_labels.copy()
    
    if len(small_clusters) > 0 and len(valid_clusters) > 0:
        # Compute cluster centroids for valid clusters
        centroids = {}
        for cluster_id in valid_clusters:
            mask = cluster_labels == cluster_id
            centroids[cluster_id] = features[mask].mean(axis=0)
        
        # Reassign small cluster members
        for cluster_id in small_clusters:
            mask = cluster_labels == cluster_id
            members = features[mask]
            
            # Find nearest valid centroid
            best_cluster = valid_clusters[0]
            best_dist = float('inf')
            
            for valid_id, centroid in centroids.items():
                dist = np.linalg.norm(members.mean(axis=0) - centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_cluster = valid_id
            
            refined_labels[mask] = best_cluster
    
    # Renumber clusters to be contiguous
    unique_refined = np.unique(refined_labels)
    mapping = {old: new for new, old in enumerate(unique_refined)}
    refined_labels = np.array([mapping[l] for l in refined_labels])
    
    n_final = len(np.unique(refined_labels))
    
    stats = {
        'original_clusters': n_original,
        'small_clusters_merged': len(small_clusters),
        'final_clusters': n_final
    }
    
    print(f"Clusters: {n_original} -> {n_final} (merged {len(small_clusters)})")
    
    return refined_labels, stats


def analyze_cluster_class_correlation(cluster_labels, class_labels, class_names):
    """
    Analyze how clusters distribute across object classes.
    
    Returns:
        cooccurrence_matrix: [n_clusters, n_classes] count matrix
    """
    n_clusters = len(np.unique(cluster_labels))
    n_classes = len(class_names)
    
    matrix = np.zeros((n_clusters, n_classes), dtype=int)
    
    for cluster_id in range(n_clusters):
        cluster_mask = (cluster_labels == cluster_id)
        cluster_classes = class_labels[cluster_mask]
        
        for class_idx in range(n_classes):
            count = np.sum(cluster_classes == class_idx)
            matrix[cluster_id, class_idx] = count
            
    return matrix

def visualize_cluster_samples_high_res(
    parts_data, 
    cluster_labels, 
    part_to_image, 
    part_to_slot, 
    part_to_class,
    class_names,
    n_samples=5,
    save_path=None
):
    """
    Visualize high-resolution samples from each cluster with mask overlays.
    
    Args:
        parts_data: Dictionary containing 'images' and 'masks'
        cluster_labels: Cluster assignments for each part
        part_to_image: Mapping from part index to image index
        part_to_slot: Mapping from part index to slot index
        part_to_class: Mapping from part index to class label
        class_names: List of class names
        n_samples: Number of samples to show per cluster
        save_path: Path to save the visualization
    """
    import cv2
    
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    
    # Limit to top 10 clusters for static visualization if too many
    if n_clusters > 10:
        print(f"Showing top 10 clusters out of {n_clusters}")
        # Sort by cluster size
        cluster_counts = np.bincount(cluster_labels)
        top_clusters = np.argsort(cluster_counts)[::-1][:10]
        unique_clusters = top_clusters
        n_clusters = 10
        
    fig, axes = plt.subplots(n_clusters, n_samples + 1, figsize=(2 * (n_samples + 1), 2 * n_clusters))
    if n_clusters == 1:
        axes = axes[np.newaxis, :]
        
    for i, cluster_id in enumerate(unique_clusters):
        # Find parts in this cluster
        cluster_mask = (cluster_labels == cluster_id)
        part_indices = np.where(cluster_mask)[0]
        
        # Calculate class distribution
        cluster_classes = part_to_class[cluster_mask]
        class_counts = np.bincount(cluster_classes, minlength=len(class_names))
        dominant_class_idx = np.argmax(class_counts)
        dominant_class = class_names[dominant_class_idx]
        purity = class_counts[dominant_class_idx] / len(cluster_classes)
        
        # Label the row
        axes[i, 0].text(0.5, 0.5, f"Cluster {cluster_id}\n{dominant_class}\n({purity:.1%} pure)\nN={len(part_indices)}", 
                       ha='center', va='center', fontsize=12)
        axes[i, 0].axis('off')
        
        # Sample parts
        if len(part_indices) > n_samples:
            sampled_indices = np.random.choice(part_indices, n_samples, replace=False)
        else:
            sampled_indices = part_indices
            
        for j, part_idx in enumerate(sampled_indices):
            if j >= n_samples: break
            
            img_idx = part_to_image[part_idx]
            slot_idx = part_to_slot[part_idx]
            
            # Get image and mask
            # Assuming images are [H, W, 3] and masks are [H, W] or [Slots, H, W]
            # We need to handle the data format carefully
            
            # Check if we have raw images available
            if 'images' in parts_data:
                img = parts_data['images'][img_idx]
                # Denormalize if necessary (assuming standard ImageNet normalization)
                if img.min() < 0:
                    img = (img * 0.5 + 0.5) # Approximate denorm
                img = np.clip(img, 0, 1)
            else:
                # Placeholder if no images
                img = np.zeros((32, 32, 3))
                
            # Get mask
            if 'masks' in parts_data:
                # Masks might be [N, S, H, W]
                mask = parts_data['masks'][img_idx, slot_idx]
            else:
                mask = np.zeros((32, 32))
                
            # Resize mask to match image if needed
            if mask.shape != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
                
            # Create heatmap overlay
            # Normalize mask to 0-1 for visualization
            mask_norm = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
            
            # Threshold for cleaner look
            mask_binary = mask_norm > 0.2
            
            # Overlay: Red color for the part
            overlay = img.copy()
            overlay[mask_binary] = overlay[mask_binary] * 0.5 + np.array([1, 0, 0]) * 0.5
            
            axes[i, j+1].imshow(overlay)
            axes[i, j+1].axis('off')
            axes[i, j+1].set_title(f"{class_names[part_to_class[part_idx]]}", fontsize=8)
            
    plt.tight_layout()
    if save_path:
        print(f"Skipping save to {save_path} (Notebook mode)")
        plt.show()
    else:
        plt.show()

def visualize_clusters_tsne(features, labels, class_labels=None, perplexity=30, save_path=None):
    """
    Visualize clusters using t-SNE.
    """
    from sklearn.manifold import TSNE
    
    print("\nComputing t-SNE projection (this may take a moment) ...")
    
    # Subsample if too many points for t-SNE
    max_samples = 5000
    if len(features) > max_samples:
        print(f"Subsampling to {max_samples} points for visualization...")
        indices = np.random.choice(len(features), max_samples, replace=False)
        features_vis = features[indices]
        labels_vis = labels[indices]
        if class_labels is not None:
            class_labels_vis = class_labels[indices]
    else:
        features_vis = features
        labels_vis = labels
        class_labels_vis = class_labels

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42
    )
    
    features_2d = tsne.fit_transform(features_vis)
    
    # Plot colored by Cluster ID
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        features_2d[:, 0], 
        features_2d[:, 1], 
        c=labels_vis, 
        cmap='tab20', 
        alpha=0.6,
        s=10
    )
    plt.colorbar(scatter, label='Cluster ID')
    plt.title('t-SNE Projection of Extracted Parts (Colored by Cluster)')
    
    if save_path:
        print(f"Skipping save to {save_path} (Notebook mode)")
        plt.show()
    else:
        plt.show()

def visualize_clusters_by_class(features, labels, class_labels, class_names, perplexity=30, save_path=None):
    """
    Visualize clusters using t-SNE, with separate plots for each class.
    """
    from sklearn.manifold import TSNE
    
    print("\nComputing t-SNE projection for per-class visualization...")
    
    # Subsample if too many points for t-SNE
    max_samples = 5000
    if len(features) > max_samples:
        print(f"Subsampling to {max_samples} points for visualization...")
        indices = np.random.choice(len(features), max_samples, replace=False)
        features_vis = features[indices]
        labels_vis = labels[indices]
        class_labels_vis = class_labels[indices]
    else:
        features_vis = features
        labels_vis = labels
        class_labels_vis = class_labels

    # Compute t-SNE globally so the space is consistent
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42
    )
    features_2d = tsne.fit_transform(features_vis)
    
    n_classes = len(class_names)
    n_cols = 2
    n_rows = (n_classes + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 7 * n_rows))
    axes = axes.flatten()
    
    # Get unique clusters to ensure consistent colormap
    unique_clusters = np.unique(labels_vis)
    n_clusters = len(unique_clusters)
    cmap = plt.get_cmap('tab20')
    
    for i, class_name in enumerate(class_names):
        ax = axes[i]
        
        # Filter points for this class
        class_mask = (class_labels_vis == i)
        class_points = features_2d[class_mask]
        class_cluster_labels = labels_vis[class_mask]
        
        if len(class_points) == 0:
            ax.text(0.5, 0.5, "No samples", ha='center', va='center')
            continue
            
        scatter = ax.scatter(
            class_points[:, 0],
            class_points[:, 1],
            c=class_cluster_labels,
            cmap=cmap,
            vmin=0,
            vmax=n_clusters-1, # Ensure consistent color mapping
            alpha=0.7,
            s=15
        )
        
        ax.set_title(f"Class: {class_name}")
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    # Add a single colorbar for the whole figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(scatter, cax=cbar_ax, label='Cluster ID')
    
    plt.suptitle('t-SNE Projection by Class (Consistent Cluster Colors)', fontsize=16)
    
    if save_path:
        print(f"Skipping save to {save_path} (Notebook mode)")
        plt.show()
    else:
        plt.show()

def visualize_clusters_per_class_separate_files(features, labels, class_labels, class_names, perplexity=30, output_dir=None):
    """
    Visualize clusters using t-SNE, saving separate image files for each class.
    """
    from sklearn.manifold import TSNE
    
    print("\nComputing t-SNE projection for individual class plots...")
    
    # Subsample if too many points for t-SNE
    max_samples = 5000
    if len(features) > max_samples:
        print(f"Subsampling to {max_samples} points for visualization...")
        indices = np.random.choice(len(features), max_samples, replace=False)
        features_vis = features[indices]
        labels_vis = labels[indices]
        class_labels_vis = class_labels[indices]
    else:
        features_vis = features
        labels_vis = labels
        class_labels_vis = class_labels

    # Compute t-SNE globally so the space is consistent
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42
    )
    features_2d = tsne.fit_transform(features_vis)
    
    # Get unique clusters to ensure consistent colormap
    unique_clusters = np.unique(labels_vis)
    n_clusters = len(unique_clusters)
    cmap = plt.get_cmap('tab20')
    
    output_dir = Path(output_dir) if output_dir else Path('.')
    
    for i, class_name in enumerate(class_names):
        # Filter points for this class
        class_mask = (class_labels_vis == i)
        class_points = features_2d[class_mask]
        class_cluster_labels = labels_vis[class_mask]
        
        if len(class_points) == 0:
            print(f"Skipping class {class_name} (no samples)")
            continue
            
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            class_points[:, 0],
            class_points[:, 1],
            c=class_cluster_labels,
            cmap=cmap,
            vmin=0,
            vmax=n_clusters-1, # Ensure consistent color mapping
            alpha=0.7,
            s=20
        )
        
        plt.colorbar(scatter, label='Cluster ID')
        plt.title(f't-SNE Projection: {class_name}')
        plt.axis('off') # Clean look
        
        save_path = output_dir / f'tsne_class_{class_name.lower()}.png'
        print(f"Skipping save to {save_path} (Notebook mode)")
        plt.show()
