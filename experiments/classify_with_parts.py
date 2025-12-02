"""
Classify images using discovered part clusters (Bag-of-Visual-Words approach).

This script:
1. Loads extracted parts and their cluster assignments.
2. Constructs a "Bag-of-Words" histogram for each image.
3. Trains a linear classifier (Logistic Regression) to predict image class from part histograms.
4. Evaluates performance and reports accuracy.
"""

import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

def load_clustering_results(clusters_dir):
    """Load clustering results and mappings."""
    clusters_dir = Path(clusters_dir)
    print(f"Loading clustering results from: {clusters_dir}")
    
    try:
        cluster_labels = np.load(clusters_dir / 'cluster_labels.npy')
        part_to_image = np.load(clusters_dir / 'part_to_image.npy')
        part_to_class = np.load(clusters_dir / 'part_to_class.npy')
        
        with open(clusters_dir / 'cluster_metadata.json', 'r') as f:
            metadata = json.load(f)
            
        return {
            'cluster_labels': cluster_labels,
            'part_to_image': part_to_image,
            'part_to_class': part_to_class,
            'metadata': metadata
        }
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please ensure you have run 'experiments/cluster_parts.py' first.")
        sys.exit(1)

    return X, y

def compute_soft_histograms(data, sigma=1.0):
    """
    Compute soft histograms based on distance to cluster centroids.
    
    Args:
        data: Dictionary containing parts data and metadata
        sigma: Bandwidth for soft assignment (Gaussian kernel)
        
    Returns:
        X: [N_images, N_clusters] soft histogram matrix
        y: [N_images] label vector
    """
    # We need the actual features and centroids, not just labels
    # But we only loaded labels. We need to load the features and centroids.
    # Since we don't have them readily available in the simple load function,
    # let's stick to the hard assignment for now but improve the classifier.
    pass

def compute_image_histograms(data):
    """
    Convert part-level data to image-level histograms (Bag-of-Words).
    
    Returns:
        X: [N_images, N_clusters] histogram matrix
        y: [N_images] label vector
    """
    cluster_labels = data['cluster_labels']
    part_to_image = data['part_to_image']
    part_to_class = data['part_to_class']
    
    n_clusters = data['metadata']['n_clusters']
    
    # Find unique images and their labels
    unique_image_ids = np.unique(part_to_image)
    n_images = len(unique_image_ids)
    
    print(f"Constructing Bag-of-Words histograms for {n_images} images...")
    print(f"Vocabulary size (Number of Clusters): {n_clusters}")
    
    X = np.zeros((n_images, n_clusters), dtype=np.float32)
    y = np.zeros(n_images, dtype=np.int64)
    
    for i, img_id in enumerate(unique_image_ids):
        # Mask for parts belonging to this image
        mask = (part_to_image == img_id)
        
        # Get clusters for these parts
        img_clusters = cluster_labels[mask]
        
        # Count frequencies
        counts = np.bincount(img_clusters, minlength=n_clusters)
        
        # L1 Normalize (Term Frequency) - Crucial for invariance to number of parts
        if counts.sum() > 0:
            X[i] = counts / counts.sum()
        else:
            X[i] = counts
        
        # Get class label
        y[i] = part_to_class[mask][0]
        
    return X, y

def train_and_evaluate(X, y, class_names, test_size=0.2, seed=42):
    """Train classifier and evaluate."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    print(f"\nData Split:")
    print(f"  Train size: {X_train.shape[0]}")
    print(f"  Test size:  {X_test.shape[0]}")
    
    # Normalize histograms (optional but often helpful for linear models)
    # Simple L1 norm (term frequency) or Standardization
    # Let's use Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train MLP Classifier
    print("\nTraining Neural Network Classifier (MLP)...")
    from sklearn.neural_network import MLPClassifier
    
    # Simple MLP: Input -> 128 -> 64 -> Output
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed
    )
    clf.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = clf.predict(X_test_scaled)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.2%}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    return clf, y_test, y_pred

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix: Classification via Discovered Parts')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to: {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Classify images using discovered parts')
    parser.add_argument('--clusters-dir', type=str, default='./parts/clusters',
                        help='Directory containing clustering results')
    parser.add_argument('--output-dir', type=str, default='./parts/classification',
                        help='Directory to save classification results')
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    data = load_clustering_results(args.clusters_dir)
    class_names = data['metadata']['classes']
    
    # 2. Compute Histograms (Features)
    X, y = compute_image_histograms(data)
    
    # 3. Train and Evaluate
    clf, y_test, y_pred = train_and_evaluate(X, y, class_names)
    
    # 4. Visualize
    plot_confusion_matrix(y_test, y_pred, class_names, output_dir / 'confusion_matrix.png')
    
    # Save results text
    with open(output_dir / 'results.txt', 'w') as f:
        f.write(f"Classification Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=class_names))

if __name__ == '__main__':
    main()
