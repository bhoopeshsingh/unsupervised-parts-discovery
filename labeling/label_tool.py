# labeling/label_tool.py
"""
Enhanced cluster labeling tool with foreground/background hints.
Uses DINO feature analysis to suggest which clusters are likely cat parts.

Run with: streamlit run labeling/label_tool.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, numpy as np, json, yaml, pickle
from pathlib import Path
from PIL import Image
import streamlit as st
from tqdm import tqdm
import torch.nn.functional as F


def load_data(config_path='configs/unified_config.yaml'):
    """Load cached data."""
    cfg  = yaml.safe_load(open(config_path))
    data = torch.load(cfg['dino']['features_cache'])
    labels_arr = torch.load('cache/cluster_labels.pt').numpy()
    n_clusters = cfg['clustering']['n_clusters']
    
    # Load cluster centers
    with open('cache/kmeans.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    
    return data, labels_arr, n_clusters, cfg, cluster_centers


def analyze_cluster_foreground_likelihood(cluster_centers, labels_arr):
    """
    Analyze which clusters are likely foreground (cat) vs background.
    
    Strategy:
    1. Clusters with high inter-cluster distance = distinct parts (likely cat)
    2. Clusters with patches concentrated in image center = likely cat
    3. Clusters with high feature norm = likely foreground
    
    Returns: dict of {cluster_id: foreground_score} (0-1, higher = more likely cat)
    """
    n_clusters = len(cluster_centers)
    scores = {}
    
    # Normalize cluster centers
    centers_norm = F.normalize(cluster_centers, dim=1)
    
    for i in range(n_clusters):
        # 1. Distinctiveness: how far is this cluster from others?
        similarities = (centers_norm[i:i+1] @ centers_norm.T).squeeze()
        similarities[i] = 0  # ignore self-similarity
        distinctiveness = 1.0 - similarities.mean().item()  # 0-1
        
        # 2. Feature magnitude: foreground typically has higher norms
        feat_norm = cluster_centers[i].norm().item()
        norm_score = min(feat_norm / 10.0, 1.0)  # normalize to 0-1
        
        # Combined score
        foreground_score = distinctiveness * 0.6 + norm_score * 0.4
        scores[i] = foreground_score
    
    return scores


def compute_patch_quality_cache(data, labels_arr, cache_path='cache/patch_quality.pt'):
    """Pre-compute quality scores for ALL patches once, save to disk."""
    if Path(cache_path).exists():
        return torch.load(cache_path)
    
    print('Computing patch quality scores (one-time, ~2-3 min)...')
    n_patches = len(data['image_ids'])
    brightness_scores = np.zeros(n_patches, dtype=np.float32)
    variance_scores   = np.zeros(n_patches, dtype=np.float32)
    spatial_centrality = np.zeros(n_patches, dtype=np.float32)
    
    n_images = len(data['image_paths'])
    
    for img_idx in tqdm(range(n_images), desc='Processing images'):
        try:
            img_path = data['image_paths'][img_idx]
            img = Image.open(img_path).convert('RGB').resize((224, 224))
            img_arr = np.array(img, dtype=np.float32)
            
            mask = (data['image_ids'] == img_idx).numpy()
            patch_indices = np.where(mask)[0]
            
            for local_idx, global_idx in enumerate(patch_indices):
                patch_idx = data['patch_ids'][global_idx].item()
                row_p = patch_idx // 28
                col_p = patch_idx % 28
                r0, c0 = row_p * 8, col_p * 8
                
                patch = img_arr[r0:r0+8, c0:c0+8]
                if patch.size > 0:
                    brightness_scores[global_idx] = patch.mean() / 255.0
                    variance_scores[global_idx]   = patch.std()
                    
                    # Spatial: distance from center (14, 14)
                    center_dist = np.sqrt((row_p - 14)**2 + (col_p - 14)**2)
                    spatial_centrality[global_idx] = 1.0 / (1.0 + center_dist / 10.0)
        except Exception:
            continue
    
    quality_cache = {
        'brightness': torch.from_numpy(brightness_scores),
        'variance': torch.from_numpy(variance_scores),
        'spatial_centrality': torch.from_numpy(spatial_centrality),
        'n_patches': n_patches
    }
    
    torch.save(quality_cache, cache_path)
    print(f'Saved to {cache_path}')
    return quality_cache


def compute_cluster_spatial_stats(cluster_id, data, labels_arr):
    """
    Compute where patches from this cluster are located spatially.
    Returns mean distance from image center.
    """
    cluster_mask = (labels_arr == cluster_id)
    cluster_indices = np.where(cluster_mask)[0]
    
    if len(cluster_indices) == 0:
        return 1.0  # max distance if empty
    
    # Sample to avoid slowness on huge clusters
    if len(cluster_indices) > 10000:
        cluster_indices = np.random.choice(cluster_indices, 10000, replace=False)
    
    patch_positions = data['patch_ids'][cluster_indices].numpy()
    rows = patch_positions // 28
    cols = patch_positions % 28
    
    center_distances = np.sqrt((rows - 14)**2 + (cols - 14)**2)
    return float(center_distances.mean())


def get_cluster_patches_fast(cluster_id, data, labels_arr, quality_cache,
                              n_samples=9,
                              brightness_threshold=0.15,
                              variance_threshold=10.0):
    """Fast patch selection using pre-computed quality scores."""
    cluster_mask = (labels_arr == cluster_id)
    cluster_indices = np.where(cluster_mask)[0]
    
    if len(cluster_indices) == 0:
        return []
    
    brightness = quality_cache['brightness'][cluster_indices].numpy()
    variance   = quality_cache['variance'][cluster_indices].numpy()
    spatial    = quality_cache['spatial_centrality'][cluster_indices].numpy()
    
    # Filter by quality
    quality_mask = (brightness >= brightness_threshold) & (variance >= variance_threshold)
    good_indices = cluster_indices[quality_mask]
    
    if len(good_indices) == 0:
        return []
    
    # Get scores for filtered patches
    brightness_subset = brightness[quality_mask]
    variance_subset   = variance[quality_mask]
    spatial_subset    = spatial[quality_mask]
    
    # Combined quality score (emphasize spatial centrality for cat detection)
    quality_scores = (brightness_subset * 0.2 + 
                     (variance_subset / 50.0) * 0.3 + 
                     spatial_subset * 0.5)  # spatial is most important
    
    # Sort by quality, take top N
    top_n_local = np.argsort(quality_scores)[::-1][:n_samples]
    top_n_global = good_indices[top_n_local]
    
    # Load patches
    patches = []
    for idx in top_n_global:
        img_idx   = int(data['image_ids'][idx].item())
        patch_idx = int(data['patch_ids'][idx].item())
        row_p = patch_idx // 28
        col_p = patch_idx % 28
        
        try:
            img = Image.open(data['image_paths'][img_idx]).convert('RGB').resize((224, 224))
            r0, c0 = row_p * 8, col_p * 8
            patch = img.crop((c0, r0, c0+8, r0+8)).resize((80, 80), Image.NEAREST)
            
            local_idx = np.where(cluster_indices == idx)[0][0]
            brightness_val = float(brightness[local_idx])
            variance_val   = float(variance[local_idx])
            spatial_val    = float(spatial[local_idx])
            
            patches.append((patch, str(data['image_paths'][img_idx]), 
                          int(patch_idx), brightness_val, variance_val, spatial_val))
        except Exception:
            continue
    
    return patches


# ── Streamlit UI ──────────────────────────────────────────────
def run_streamlit():
    st.set_page_config(page_title='Cluster Labeler', layout='wide')
    st.title('🏷️  Human Cluster Labeling Tool')
    st.caption('Label each cluster with the semantic part it represents.')
    
    # Load data
    data, labels_arr, n_clusters, cfg, cluster_centers = load_data()
    labels_path = cfg['concepts']['labels_path']
    
    # Compute cluster foreground scores (cached in session)
    if 'fg_scores' not in st.session_state:
        st.session_state.fg_scores = analyze_cluster_foreground_likelihood(
            cluster_centers, labels_arr
        )
    fg_scores = st.session_state.fg_scores
    
    # Initialize patch quality cache
    if 'quality_cache' not in st.session_state:
        with st.spinner('Computing patch quality (one-time, ~2-3 min)...'):
            st.session_state.quality_cache = compute_patch_quality_cache(data, labels_arr)
    quality_cache = st.session_state.quality_cache
    
    # Load existing labels
    existing = {}
    if Path(labels_path).exists():
        existing = json.load(open(labels_path))
    
    # Sidebar: Progress
    st.sidebar.header('Progress')
    labeled_count = sum(1 for v in existing.values() if v.get('label',''))
    st.sidebar.metric('Labeled', f'{labeled_count} / {n_clusters}')
    st.sidebar.progress(labeled_count / n_clusters)
    
    # Sidebar: Filters
    st.sidebar.header('Patch Quality Filters')
    brightness_thresh = st.sidebar.slider('Min brightness', 0.0, 0.5, 0.15, 0.05)
    variance_thresh = st.sidebar.slider('Min variance', 0.0, 30.0, 10.0, 5.0)
    n_samples = st.sidebar.slider('Samples per cluster', 6, 15, 9, 3)
    
    # Sidebar: Cluster overview with foreground hints
    st.sidebar.header('Cluster Overview')
    for cid in range(n_clusters):
        size = int((labels_arr == cid).sum())
        fg_score = fg_scores[cid]
        label = existing.get(str(cid), {}).get('label', '—')
        
        # Color code by foreground likelihood
        if fg_score > 0.6:
            emoji = "🐱"  # likely cat
        elif fg_score > 0.4:
            emoji = "❓"  # uncertain
        else:
            emoji = "🌫️"  # likely background
        
        st.sidebar.caption(f'{emoji} C{cid}: {label[:15]} ({size//1000}k patches, fg={fg_score:.2f})')
    
    # Main: Cluster selection
    cluster_id = st.selectbox(
        'Select cluster to label:',
        list(range(n_clusters)),
        format_func=lambda x: f'Cluster {x}' + (
            f' — {existing.get(str(x),{}).get("label","")}' if str(x) in existing else '')
    )
    
    # Show cluster analysis
    cluster_size = int((labels_arr == cluster_id).sum())
    fg_score = fg_scores[cluster_id]
    spatial_mean = compute_cluster_spatial_stats(cluster_id, data, labels_arr)
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.metric('Patches', f'{cluster_size:,}')
    col_stat2.metric('Foreground Score', f'{fg_score:.2f}', 
                     help='0.0=likely background, 1.0=likely cat part')
    col_stat3.metric('Spatial Centrality', f'{spatial_mean:.1f}',
                     help='Lower=more centered (cat), higher=more edge (background)')
    
    # Recommendation
    if fg_score > 0.6 and spatial_mean < 8:
        st.success('✅ This cluster is likely a **cat part** (centered, distinct features)')
    elif fg_score < 0.4 or spatial_mean > 12:
        st.warning('⚠️ This cluster is likely **background/noise** (edge patches, less distinct)')
    else:
        st.info('❓ This cluster is **uncertain** — inspect patches carefully')
    
    # Load patches
    with st.spinner('Loading patches...'):
        patches = get_cluster_patches_fast(
            cluster_id, data, labels_arr, quality_cache,
            n_samples=n_samples,
            brightness_threshold=brightness_thresh,
            variance_threshold=variance_thresh
        )
    
    # Display patches
    if not patches:
        st.warning(f'⚠️ No quality patches found. This cluster is likely pure background.')
        st.info('💡 Label as "background" and uncheck "Include in classification"')
    else:
        st.success(f'✅ Showing {len(patches)} highest-quality patches (center-focused)')
        cols = st.columns(3)
        for i, (patch, path, pidx, brightness, variance, spatial) in enumerate(patches):
            cols[i % 3].image(patch, 
                            caption=f'p{pidx} | center={spatial:.2f}', 
                            width=120)
    
    st.divider()
    
    # Labeling form
    st.subheader(f'Label Cluster {cluster_id}')
    current = existing.get(str(cluster_id), {})
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        label = st.text_input(
            'Semantic label',
            value=current.get('label', ''),
            placeholder='e.g. eyes, ears, body, background',
            key=f'label_{cluster_id}'
        )
        notes = st.text_area(
            'Notes', 
            value=current.get('notes', ''), 
            height=60,
            key=f'notes_{cluster_id}'
        )
    
    with col2:
        # Auto-suggest include based on foreground score
        default_include = fg_score > 0.4 if str(cluster_id) not in existing else current.get('include', True)
        
        include = st.checkbox(
            'Include in classification',
            value=default_include,
            key=f'include_{cluster_id}',
            help='Auto-suggested based on foreground score'
        )
        confidence = st.select_slider(
            'Confidence',
            options=[1, 2, 3],
            value=current.get('confidence', 2),
            format_func=lambda x: {1:'😐 Unsure', 2:'🙂 Likely', 3:'😃 Certain'}[x],
            key=f'conf_{cluster_id}'
        )
    
    # Buttons
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button('💾 Save', type='primary', use_container_width=True):
            existing[str(cluster_id)] = {
                'label': label.strip(),
                'include': include,
                'confidence': confidence,
                'notes': notes.strip()
            }
            Path(labels_path).parent.mkdir(parents=True, exist_ok=True)
            json.dump(existing, open(labels_path, 'w'), indent=2)
            st.success(f'✅ Saved: Cluster {cluster_id} → "{label}"')
            st.rerun()
    
    with col_btn2:
        if st.button('⏩ Next', use_container_width=True):
            for cid in range(cluster_id + 1, n_clusters):
                if str(cid) not in existing or not existing[str(cid)].get('label'):
                    st.rerun()
                    break
    
    # Summary
    if labeled_count == n_clusters:
        st.balloons()
        st.success('🎉 All clusters labeled! Run: python experiments/run_dino_pipeline.py --stage concepts')


if __name__ == '__main__':
    run_streamlit()