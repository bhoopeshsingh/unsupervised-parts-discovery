"""Streamlit interface for labeling discovered part clusters and running inference"""

import streamlit as st
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import sys

# Add project root to path
sys.path.append('.')

from src.inference.pipeline import InferencePipeline

# Page configuration
st.set_page_config(
    page_title="Part Discovery & Labeling",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_clustering_data(parts_dir, clusters_dir):
    """Load all necessary data"""
    parts_dir = Path(parts_dir)
    clusters_dir = Path(clusters_dir)
    
    # Load parts data
    masks = np.load(parts_dir / 'masks.npy')
    labels = np.load(parts_dir / 'labels.npy')
    
    with open(parts_dir / 'metadata.json', 'r') as f:
        parts_metadata = json.load(f)
    
    # Load clustering data
    cluster_labels = np.load(clusters_dir / 'cluster_labels.npy')
    part_to_image = np.load(clusters_dir / 'part_to_image.npy')
    part_to_slot = np.load(clusters_dir / 'part_to_slot.npy')
    part_to_class = np.load(clusters_dir / 'part_to_class.npy')
    
    # Try to load images if available
    images = None
    if (parts_dir / 'images.npy').exists():
        images = np.load(parts_dir / 'images.npy')
    
    with open(clusters_dir / 'cluster_metadata.json', 'r') as f:
        cluster_metadata = json.load(f)
    
    return {
        'masks': masks,
        'images': images,
        'image_labels': labels,
        'parts_metadata': parts_metadata,
        'cluster_labels': cluster_labels,
        'part_to_image': part_to_image,
        'part_to_slot': part_to_slot,
        'part_to_class': part_to_class,
        'cluster_metadata': cluster_metadata
    }

@st.cache_data
def load_existing_labels(labels_file):
    """Load existing cluster labels if they exist"""
    labels_file = Path(labels_file)
    if labels_file.exists():
        with open(labels_file, 'r') as f:
            return json.load(f)
    return {}

def save_cluster_labels(labels_dict, labels_file):
    """Save cluster labels to JSON file"""
    labels_file = Path(labels_file)
    labels_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(labels_file, 'w') as f:
        json.dump(labels_dict, f, indent=2)
    
    st.success(f"✓ Labels saved to {labels_file}")

def get_cluster_samples(cluster_id, data, max_samples=20):
    """Get sample parts from a specific cluster"""
    # Find all parts in this cluster
    cluster_mask = data['cluster_labels'] == cluster_id
    part_indices = np.where(cluster_mask)[0]
    
    # Sample randomly if too many
    if len(part_indices) > max_samples:
        sampled_indices = np.random.choice(part_indices, max_samples, replace=False)
    else:
        sampled_indices = part_indices
    
    samples = []
    for part_idx in sampled_indices:
        image_idx = data['part_to_image'][part_idx]
        slot_idx = data['part_to_slot'][part_idx]
        class_label = data['part_to_class'][part_idx]
        
        # Get the mask for this part
        mask = data['masks'][image_idx, slot_idx]
        
        # Get image if available
        image = None
        if data['images'] is not None:
            image = data['images'][image_idx]
        
        samples.append({
            'part_idx': part_idx,
            'image_idx': image_idx,
            'slot_idx': slot_idx,
            'class_label': class_label,
            'mask': mask,
            'image': image
        })
    
    return samples

def display_cluster_samples(samples, class_names, cols=5):
    """Display cluster samples in a grid with high-res overlay"""
    import cv2
    
    num_samples = len(samples)
    num_rows = (num_samples + cols - 1) // cols
    
    for row in range(num_rows):
        columns = st.columns(cols)
        for col_idx in range(cols):
            sample_idx = row * cols + col_idx
            if sample_idx >= num_samples:
                break
            
            sample = samples[sample_idx]
            
            with columns[col_idx]:
                if sample['image'] is not None:
                    # Create overlay
                    img = sample['image'].copy() # [H, W, 3] uint8
                    mask = sample['mask'] # [H, W] float
                    
                    # Resize mask to match image if needed (though they should match)
                    if mask.shape != img.shape[:2]:
                        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                    
                    # Normalize mask for visualization
                    mask_norm = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
                    
                    # Create heatmap (red)
                    heatmap = np.zeros_like(img)
                    heatmap[:, :, 0] = 255 # Red channel
                    
                    # Apply mask opacity
                    # Stronger opacity for higher attention
                    opacity = mask_norm[:, :, np.newaxis] * 0.7
                    
                    # Blend
                    overlay = img * (1 - opacity) + heatmap * opacity
                    overlay = overlay.astype(np.uint8)
                    
                    # Upsample for display (e.g., 128x128)
                    display_size = (128, 128)
                    overlay_large = cv2.resize(overlay, display_size, interpolation=cv2.INTER_NEAREST)
                    
                    st.image(overlay_large, caption=f"{class_names[sample['class_label']]}\n(Img {sample['image_idx']})", clamp=True)
                else:
                    # Fallback to mask only
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.imshow(sample['mask'], cmap='hot', interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title(
                        f"Img {sample['image_idx']}, Slot {sample['slot_idx']}\n"
                        f"{class_names[sample['class_label']]}",
                        fontsize=8
                    )
                    st.pyplot(fig)
                    plt.close(fig)

@st.cache_resource
def get_inference_pipeline():
    """Load inference pipeline (cached)"""
    return InferencePipeline()

def render_labeling_page():
    st.title("🏷️ Part Cluster Labeling")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    parts_dir = st.sidebar.text_input(
        "Parts Directory",
        value="./parts/extracted"
    )
    
    clusters_dir = st.sidebar.text_input(
        "Clusters Directory",
        value="./parts/clusters"
    )
    
    labels_file = st.sidebar.text_input(
        "Labels Output File",
        value="./parts/labels/cluster_labels.json"
    )
    
    # Load data
    try:
        data = load_clustering_data(parts_dir, clusters_dir)
        st.sidebar.success("✓ Data loaded successfully")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Make sure you have run:\n1. `python experiments/extract_parts.py`\n2. `python experiments/cluster_parts.py`")
        return
    
    # Load existing labels
    cluster_labels_dict = load_existing_labels(labels_file)
    
    # Display statistics
    n_clusters = data['cluster_metadata']['n_clusters']
    n_parts = data['cluster_metadata']['n_parts']
    class_names = data['cluster_metadata']['classes']
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Statistics")
    st.sidebar.metric("Total Clusters", n_clusters)
    st.sidebar.metric("Total Parts", n_parts)
    st.sidebar.metric("Labeled Clusters", len(cluster_labels_dict))
    
    progress = len(cluster_labels_dict) / n_clusters * 100
    st.sidebar.progress(progress / 100)
    st.sidebar.caption(f"Progress: {progress:.1f}%")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Cluster Browser")
    
    with col2:
        # Filter options
        filter_option = st.selectbox(
            "Filter",
            ["All Clusters", "Unlabeled Only", "Labeled Only"]
        )
    
    # Get cluster IDs based on filter
    all_cluster_ids = list(range(n_clusters))
    labeled_ids = set(int(k) for k in cluster_labels_dict.keys())
    
    if filter_option == "Unlabeled Only":
        display_cluster_ids = [cid for cid in all_cluster_ids if cid not in labeled_ids]
    elif filter_option == "Labeled Only":
        display_cluster_ids = list(labeled_ids)
    else:
        display_cluster_ids = all_cluster_ids
    
    # Cluster selection
    if not display_cluster_ids:
        st.info("No clusters match the current filter.")
        return
    
    selected_cluster = st.selectbox(
        "Select Cluster to Label",
        options=display_cluster_ids,
        format_func=lambda x: f"Cluster {x}" + (f" - {cluster_labels_dict[str(x)]['label']}" if str(x) in cluster_labels_dict else "")
    )
    
    st.markdown("---")
    
    # Get cluster statistics
    cluster_mask = data['cluster_labels'] == selected_cluster
    cluster_size = cluster_mask.sum()
    cluster_classes = data['part_to_class'][cluster_mask]
    
    # Display cluster info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cluster ID", selected_cluster)
    with col2:
        st.metric("Number of Parts", cluster_size)
    with col3:
        # Most common class
        unique, counts = np.unique(cluster_classes, return_counts=True)
        most_common_idx = unique[np.argmax(counts)]
        most_common_pct = (counts.max() / len(cluster_classes)) * 100
        st.metric(
            "Dominant Class",
            f"{class_names[most_common_idx]} ({most_common_pct:.0f}%)"
        )
    
    # Class distribution
    st.subheader("Class Distribution in Cluster")
    class_dist = pd.DataFrame({
        'Class': [class_names[i] for i in unique],
        'Count': counts,
        'Percentage': (counts / len(cluster_classes)) * 100
    })
    st.dataframe(class_dist, hide_index=True)
    
    st.markdown("---")
    
    # Display samples
    st.subheader("Sample Parts from Cluster")
    num_samples = st.select_slider("Number of samples to display", options=[10, 25, 50], value=10)
    
    samples = get_cluster_samples(selected_cluster, data, max_samples=num_samples)
    display_cluster_samples(samples, class_names, cols=5)
    
    st.markdown("---")
    
    # Labeling interface
    st.header("🏷️ Label This Cluster")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Get current label if exists
        current_label = cluster_labels_dict.get(str(selected_cluster), {})
        
        semantic_label = st.text_input(
            "Semantic Label",
            value=current_label.get('label', ''),
            placeholder="e.g., 'wing', 'wheel', 'ear', 'background'"
        )
    
    with col2:
        confidence = st.select_slider(
            "Confidence",
            options=["Low", "Medium", "High"],
            value=current_label.get('confidence', 'Medium')
        )
    
    notes = st.text_area(
        "Notes (optional)",
        value=current_label.get('notes', ''),
        placeholder="Any observations about this cluster..."
    )
    
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        if st.button("💾 Save Label", type="primary", use_container_width=True):
            if semantic_label:
                cluster_labels_dict[str(selected_cluster)] = {
                    'label': semantic_label,
                    'confidence': confidence,
                    'notes': notes,
                    'cluster_size': int(cluster_size)
                }
                save_cluster_labels(cluster_labels_dict, labels_file)
                # Force reload to ensure UI updates
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("Please enter a semantic label")
    
    with col2:
        if st.button("⏭️ Skip", use_container_width=True):
            # Move to next unlabeled cluster
            unlabeled = [cid for cid in all_cluster_ids if cid not in labeled_ids]
            if unlabeled:
                st.session_state.selected_cluster = unlabeled[0]
                st.rerun()
    
    # Export labeled data
    st.markdown("---")
    st.header("📤 Export")
    
    if st.button("📥 Download Labels JSON"):
        st.download_button(
            label="Download cluster_labels.json",
            data=json.dumps(cluster_labels_dict, indent=2),
            file_name="cluster_labels.json",
            mime="application/json"
        )
    
    # Reset Labels Button
    st.markdown("---")
    st.header("⚠️ Reset All Labels")
    if st.button("Reset All Labels", type="secondary"):
        labels_path = Path("parts/labels/cluster_labels.json")
        if labels_path.exists():
            labels_path.unlink()
            st.cache_data.clear()
            st.success("All labels have been reset!")
            st.rerun()
        else:
            st.info("No labels found to reset.")
    
    # Show labeled clusters summary
    if cluster_labels_dict:
        st.subheader("Labeled Clusters Summary")
        labeled_df = pd.DataFrame([
            {
                'Cluster ID': int(k),
                'Label': v['label'],
                'Confidence': v['confidence'],
                'Size': v['cluster_size']
            }
            for k, v in cluster_labels_dict.items()
        ]).sort_values('Cluster ID')
        
        st.dataframe(labeled_df, hide_index=True)

def render_inference_page():
    st.title("🔮 Inference & Validation")
    st.markdown("Upload an image to discover its parts and validate semantic labels.")
    st.markdown("---")
    
    # Initialize pipeline
    with st.spinner("Loading model and centroids..."):
        try:
            pipeline = get_inference_pipeline()
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading pipeline: {e}")
            st.info("Ensure you have trained the model and run clustering first.")
            return

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Run inference
        with st.spinner("Processing image..."):
            results = pipeline.process_image(uploaded_file)
            
        st.markdown("### Results")
        
        # Display Original and Reconstruction
        col1, col2 = st.columns(2)
        with col1:
            st.image(results['original'], caption="Original Image", use_column_width=True)
        with col2:
            st.image(results['reconstruction'], caption="Model Reconstruction", use_column_width=True, clamp=True)
        
        # Display Prediction
        st.markdown("---")
        st.subheader("Prediction")
        pred_class = results.get('predicted_class', 'Unknown')
        probs = results.get('class_probs', {})
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Predicted Class", pred_class)
        
        with col2:
            if probs:
                st.write("Class Confidence:")
                for cls, prob in probs.items():
                    st.progress(prob, text=f"{cls}: {prob:.1%}")
            
        st.markdown("---")
        st.subheader("Discovered Parts")
        
        parts = results['parts']
        cols = 4
        num_rows = (len(parts) + cols - 1) // cols
        
        for row in range(num_rows):
            columns = st.columns(cols)
            for col_idx in range(cols):
                idx = row * cols + col_idx
                if idx >= len(parts):
                    break
                
                part = parts[idx]
                
                # Create overlay
                # Resize mask to match original image size for display
                import cv2
                orig_w, orig_h = results['original'].size
                mask_resized = cv2.resize(part['mask'], (orig_w, orig_h))
                
                # Create heatmap
                heatmap = plt.cm.hot(mask_resized)[:, :, :3]
                
                # Overlay on original
                orig_np = np.array(results['original']) / 255.0
                overlay = orig_np * 0.6 + heatmap * 0.4
                
                with columns[col_idx]:
                    st.image(overlay, caption=f"{part['label']}\n(Cluster {part['cluster_id']})", clamp=True)
                    # Clamp score to [0, 1] for progress bar
                    score = max(0.0, min(1.0, part['score']))
                    st.progress(score)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Labeling", "Inference & Validation"])
    
    if page == "Labeling":
        render_labeling_page()
    else:
        render_inference_page()

if __name__ == '__main__':
    main()
