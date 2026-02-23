# labeling/label_tool.py
"""
Two-tab Streamlit tool:
  Tab 1 — Cluster Labeling : label DINO patch clusters with semantic part names.
  Tab 2 — Classify & Explain : upload any image, get prediction + semantic part overlay.

Run with: streamlit run labeling/label_tool.py
"""
import sys, os
# Always run from project root so relative paths (cache/, configs/) resolve correctly
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

import io, tempfile
import torch, numpy as np, json, yaml, pickle
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from tqdm import tqdm
import torch.nn.functional as F
import cv2


def load_data(config_path='configs/config.yaml'):
    """Load cached data."""
    cfg  = yaml.safe_load(open(config_path))
    data = torch.load(cfg['dino']['features_cache'], weights_only=False)
    cluster_labels_path = cfg['dino'].get('cluster_labels_path', 'cache/cluster_labels.pt')
    labels_arr = torch.load(cluster_labels_path, weights_only=True).numpy()
    n_clusters = cfg['clustering']['n_clusters']
    
    # Load cluster centers (works for both KMeans and GaussianMixture)
    clusterer_path = cfg['dino'].get('clusterer_path', 'cache/kmeans.pkl')
    with open(clusterer_path, 'rb') as f:
        kmeans = pickle.load(f)
    centers_np = getattr(kmeans, 'cluster_centers_', None) or kmeans.means_
    cluster_centers = torch.tensor(centers_np, dtype=torch.float32)
    
    return data, labels_arr, n_clusters, cfg, cluster_centers


def analyze_cluster_foreground_likelihood(cluster_centers, labels_arr):
    """Analyze which clusters are likely foreground (cat) vs background."""
    n_clusters = len(cluster_centers)
    scores = {}
    
    # Normalize cluster centers
    centers_norm = F.normalize(cluster_centers, dim=1)
    
    for i in range(n_clusters):
        # Distinctiveness: how far is this cluster from others?
        similarities = (centers_norm[i:i+1] @ centers_norm.T).squeeze()
        similarities[i] = 0
        distinctiveness = 1.0 - similarities.mean().item()
        
        # Feature magnitude
        feat_norm = cluster_centers[i].norm().item()
        norm_score = min(feat_norm / 10.0, 1.0)
        
        # Combined score
        foreground_score = distinctiveness * 0.6 + norm_score * 0.4
        scores[i] = foreground_score
    
    return scores


def compute_patch_quality(data, labels_arr):
    """Compute quality scores for all patches."""
    n_patches = len(data['image_ids'])
    brightness_scores = np.zeros(n_patches, dtype=np.float32)
    variance_scores   = np.zeros(n_patches, dtype=np.float32)
    spatial_centrality = np.zeros(n_patches, dtype=np.float32)

    n_images = len(data['image_paths'])

    for img_idx in tqdm(range(n_images), desc='Computing patch quality'):
        try:
            img_path = data['image_paths'][img_idx]
            img = Image.open(img_path).convert('RGB').resize((224, 224))
            img_arr = np.array(img, dtype=np.float32)

            mask = (data['image_ids'] == img_idx).numpy()
            patch_indices = np.where(mask)[0]

            for global_idx in patch_indices:
                patch_idx = data['patch_ids'][global_idx].item()
                row_p = patch_idx // 28
                col_p = patch_idx % 28
                r0, c0 = row_p * 8, col_p * 8

                patch = img_arr[r0:r0+8, c0:c0+8]
                if patch.size > 0:
                    brightness_scores[global_idx] = patch.mean() / 255.0
                    variance_scores[global_idx]   = patch.std()
                    center_dist = np.sqrt((row_p - 14)**2 + (col_p - 14)**2)
                    spatial_centrality[global_idx] = 1.0 / (1.0 + center_dist / 10.0)
        except Exception:
            continue

    return {
        'brightness': torch.from_numpy(brightness_scores),
        'variance': torch.from_numpy(variance_scores),
        'spatial_centrality': torch.from_numpy(spatial_centrality),
        'n_patches': n_patches
    }


def compute_cluster_spatial_stats(cluster_id, data, labels_arr):
    """Compute where patches from this cluster are located spatially."""
    cluster_mask = (labels_arr == cluster_id)
    cluster_indices = np.where(cluster_mask)[0]
    
    if len(cluster_indices) == 0:
        return 1.0
    
    # Sample to avoid slowness
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
                              variance_threshold=10.0,
                              context_size=72,
                              part_box_size=24):
    """
    Fast patch selection WITH CONTEXT WINDOWS.
    Shows patches with surrounding image region and red box highlighting the part region.
    
    Args:
        context_size: Size of context window (default 72 = more face context)
        part_box_size: Size of red highlighted "part" box in pixels (default 24 = 3×3 patches,
                       so the part is visually closer to eye/nose scale instead of tiny 8×8)
    """
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
    
    # Combined quality score
    quality_scores = (brightness_subset * 0.2 + 
                     (variance_subset / 50.0) * 0.3 + 
                     spatial_subset * 0.5)
    
    # Sort by quality, take top N
    top_n_local = np.argsort(quality_scores)[::-1][:n_samples]
    top_n_global = good_indices[top_n_local]
    
    # Load patches WITH CONTEXT
    patches = []
    for idx in top_n_global:
        img_idx   = int(data['image_ids'][idx].item())
        patch_idx = int(data['patch_ids'][idx].item())
        row_p = patch_idx // 28
        col_p = patch_idx % 28
        
        try:
            img = Image.open(data['image_paths'][img_idx]).convert('RGB').resize((224, 224))
            img_arr = np.array(img)
            
            # Patch center in pixel coordinates
            patch_center_r = row_p * 8 + 4
            patch_center_c = col_p * 8 + 4
            
            # Context window around patch
            half_ctx = context_size // 2
            r_start = max(0, patch_center_r - half_ctx)
            r_end   = min(224, patch_center_r + half_ctx)
            c_start = max(0, patch_center_c - half_ctx)
            c_end   = min(224, patch_center_c + half_ctx)
            
            # Crop context region
            context_crop = img_arr[r_start:r_end, c_start:c_end].copy()
            
            # Red box: center on the 8×8 patch, use part_box_size so part is visible (e.g. 24 = 3× patch)
            patch_r_in_ctx = row_p * 8 - r_start
            patch_c_in_ctx = col_p * 8 - c_start
            patch_center_r = patch_r_in_ctx + 4
            patch_center_c = patch_c_in_ctx + 4
            half_box = part_box_size // 2
            box_r0 = max(0, patch_center_r - half_box)
            box_c0 = max(0, patch_center_c - half_box)
            box_r1 = min(context_crop.shape[0], box_r0 + part_box_size)
            box_c1 = min(context_crop.shape[1], box_c0 + part_box_size)
            thickness = max(2, part_box_size // 12)
            
            context_with_box = context_crop.copy()
            cv2.rectangle(
                context_with_box,
                (box_c0, box_r0),
                (box_c1, box_r1),
                (255, 0, 0),
                thickness
            )
            
            # Convert to PIL
            context_pil = Image.fromarray(context_with_box)
            
            local_idx = np.where(cluster_indices == idx)[0][0]
            brightness_val = float(brightness[local_idx])
            variance_val   = float(variance[local_idx])
            spatial_val    = float(spatial[local_idx])
            
            patches.append((context_pil, str(data['image_paths'][img_idx]), 
                          int(patch_idx), brightness_val, variance_val, spatial_val))
        except Exception:
            continue
    
    return patches


# ── Classification helpers ─────────────────────────────────────

@st.cache_resource(show_spinner="Loading DINO model…")
def load_dino_extractor(model_name: str, device: str, image_size: int):
    from src.models.dino_extractor import DinoExtractor
    return DinoExtractor(model_name=model_name, device=device, image_size=image_size)


@st.cache_resource(show_spinner="Loading classifier…")
def load_classifier_and_vectors(classifier_path: str, vectors_path: str):
    from src.pipeline.one_class_classifier import CatConceptOneClassClassifier
    clf = CatConceptOneClassClassifier.load(classifier_path)
    saved = torch.load(vectors_path, weights_only=False)
    return clf, saved["vectors"]


def run_classify_tab(cfg):
    """Classify & Explain tab: upload image → prediction + semantic part map."""
    from src.pipeline.concept_classifier import (
        get_spatial_concept_map,
        render_dissertation_explanation,
    )

    st.header("Upload an image to classify")
    st.caption(
        "The system will identify which semantic cat parts are present "
        "and explain its prediction."
    )

    # ── check prerequisites ────────────────────────────────────
    clf_path = cfg["classification"].get("classifier_path", "cache/concept_classifier.pkl")
    vec_path = cfg["concepts"].get("vectors_cache", "cache/concept_vectors.pt")
    lbl_path = cfg["concepts"].get("labels_path", "cache/labels.json")

    missing = [p for p in [clf_path, vec_path, lbl_path] if not Path(p).exists()]
    if missing:
        st.error(
            "Pipeline cache files not found. Run the full pipeline first:\n"
            "```\npython experiments/run_pipeline.py --stage all\n```\n"
            f"Missing: {missing}"
        )
        return

    # ── load models (cached across reruns) ────────────────────
    extractor = load_dino_extractor(
        cfg["dino"]["model"],
        cfg["dino"]["device"],
        cfg["dino"]["image_size"],
    )
    clf, vectors = load_classifier_and_vectors(clf_path, vec_path)

    # ── file uploader ──────────────────────────────────────────
    uploaded = st.file_uploader(
        "Choose an image (jpg / png)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.info("👆 Upload an image to get started.")
        return

    # Show upload preview
    pil_img = Image.open(uploaded).convert("RGB")
    col_prev, col_info = st.columns([1, 2])
    col_prev.image(pil_img, caption="Uploaded image", use_container_width=True)
    col_info.markdown(f"**Filename:** `{uploaded.name}`")
    col_info.markdown(f"**Size:** {pil_img.width} × {pil_img.height} px")
    col_info.markdown(f"**Concepts available:** {', '.join(list(vectors.keys()))}")

    classify_btn = col_info.button("🔍 Classify", type="primary", use_container_width=True)

    if not classify_btn:
        return

    # ── run inference ──────────────────────────────────────────
    with st.spinner("Extracting features and classifying…"):
        # Save upload to temp file (DINO extractor needs a file path)
        suffix = Path(uploaded.name).suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        fg_threshold = cfg["dino"].get("fg_threshold", 0.5)
        all_feats, fg_mask = extractor.extract_all_patches_with_fg_mask(
            tmp_path, fg_threshold=fg_threshold
        )
        fg_feats = all_feats[fg_mask]
        concept_names = list(vectors.keys())

        # Per-concept activation scores (max cosine similarity across fg patches)
        img_norm = F.normalize(fg_feats, dim=-1)
        concept_scores_dict = {}
        for c_name, vec in vectors.items():
            v_norm = F.normalize(vec.unsqueeze(0), dim=1).squeeze(0)
            concept_scores_dict[c_name] = (img_norm @ v_norm).max().item()

        score_vec = np.array([[concept_scores_dict[c] for c in concept_names]])
        preds, _, confs = clf.predict(score_vec)
        explanation = clf.explain(score_vec, concept_names)

        result = {
            "prediction": preds[0],
            "confidence": float(confs[0]),
            "concept_scores": concept_scores_dict,
            # deviation_from_cat: positive = above typical cat = pushes toward cat
            "contributions": {
                item["concept"]: item["deviation_from_cat"] for item in explanation
            },
        }
        concept_map, patch_sims = get_spatial_concept_map(
            concept_names, vectors, all_feats
        )
        os.unlink(tmp_path)

    # ── prediction banner ──────────────────────────────────────
    pred = result["prediction"].upper()   # "CAT" or "NOT_CAT"
    conf = result["confidence"]
    is_cat = pred == "CAT"
    banner_color = "#1a237e" if is_cat else "#b71c1c"
    emoji = "🐱" if is_cat else "❓"

    st.markdown(
        f"""
        <div style="background:{banner_color};padding:14px 20px;border-radius:10px;
                    text-align:center;margin:12px 0">
            <span style="color:white;font-size:1.6rem;font-weight:700">
                {emoji} Prediction: {pred}
            </span>
            <span style="color:#cfd8dc;font-size:1.1rem;margin-left:16px">
                Confidence: {conf:.1%}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── semantic explanation figure ────────────────────────────
    st.subheader("Semantic Part Map & Concept Contributions")

    with st.spinner("Rendering explanation…"):
        # Re-open the uploaded image via BytesIO for the figure
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as _t:
            tmp_path = _t.name
        pil_img.save(tmp_path)
        fig = render_dissertation_explanation(
            image_path=tmp_path,
            concept_map=concept_map,
            concept_names=concept_names,
            result=result,
            fg_mask=fg_mask,
            save_path=None,          # we'll render inline
        )
        # Render to buffer and display
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        os.unlink(tmp_path)

    st.image(buf, use_container_width=True)

    # ── concept activation table ───────────────────────────────
    st.subheader("Concept Activations")
    scores = result["concept_scores"]
    contribs = result["contributions"]
    rows = sorted(
        [
            {
                "Concept": n.replace("_", " "),
                "Activation": round(scores[n], 3),
                "vs avg cat": round(contribs[n], 4),
                "Signal": "↑ above avg" if contribs[n] >= 0 else "↓ below avg",
            }
            for n in concept_names
        ],
        key=lambda r: r["Activation"],
        reverse=True,
    )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # ── plain-language explanation ─────────────────────────────
    st.subheader("Plain-language explanation")
    # Top activated concepts by raw activation score
    top_activated = [r["Concept"] for r in rows if r["Activation"] > 0.5][:3]
    top_weak = [r["Concept"] for r in rows if r["Activation"] < 0.3][:2]

    if is_cat:
        parts = ", ".join(f"**{c}**" for c in top_activated) if top_activated else "cat-like patterns"
        st.success(
            f"**CAT** ({conf:.0%} confidence)\n\n"
            f"The concept activation profile — {parts} — falls **inside** the "
            f"learned cat distribution. The one-class classifier judges the overall "
            f"combination of all concept activations, not each concept individually."
        )
    else:
        weak = ", ".join(f"**{c}**" for c in top_weak) if top_weak else "most cat concepts"
        st.warning(
            f"**NOT A CAT** ({conf:.0%} confidence)\n\n"
            f"The overall concept activation profile falls **outside** the learned "
            f"cat distribution. Weak activations in {weak} pushed the profile away "
            f"from the cat boundary."
        )

    with st.expander("How to read these results"):
        st.markdown(
            "**Activation** — how strongly each semantic part (labelled by a human) "
            "was detected in this image. Higher = stronger match to that concept.\n\n"
            "**vs avg cat** — how this image's activation compares to the average cat "
            "in training. ↑ above avg = stronger than typical cat; ↓ below avg = weaker.\n\n"
            "**Important:** individual concept signals do not vote independently. "
            "The one-class SVM evaluates the *combined* 10-concept profile and asks: "
            "*does this pattern fall within the distribution of cat images?* "
            "A concept can be below average and the image still be classified as cat "
            "if the overall profile remains within the learned boundary.\n\n"
            "**Semantic Part Map** — each 8×8 patch coloured by its closest concept. "
            "Shows *where* in the image each part was detected."
        )


# ── Streamlit UI ──────────────────────────────────────────────
def run_streamlit():
    st.set_page_config(page_title='Parts Discovery', layout='wide')
    st.title('Unsupervised Parts Discovery')

    # Load config + data once (shared by both tabs)
    data, labels_arr, n_clusters, cfg, cluster_centers = load_data()

    tab_label, tab_classify = st.tabs(["🏷️ Label Clusters", "🔍 Classify & Explain"])

    # ── TAB 2: Classify & Explain ──────────────────────────────
    with tab_classify:
        run_classify_tab(cfg)

    # ── TAB 1: Cluster Labeling ────────────────────────────────
    with tab_label:
        st.caption('Label each cluster with the semantic part it represents.')
        st.caption('🔴 Red box = part region (adjust size in sidebar) | Surrounding area = context for easier identification')

        labels_path = cfg['concepts']['labels_path']

        # Compute cluster foreground scores
        if 'fg_scores' not in st.session_state:
            st.session_state.fg_scores = analyze_cluster_foreground_likelihood(
                cluster_centers, labels_arr
            )
        fg_scores = st.session_state.fg_scores

        # Compute patch quality scores (kept in session memory)
        if 'quality_cache' not in st.session_state:
            with st.spinner('Computing patch quality (~2-3 min)...'):
                st.session_state.quality_cache = compute_patch_quality(data, labels_arr)
        quality_cache = st.session_state.quality_cache

        # Load existing labels
        existing = {}
        if Path(labels_path).exists():
            existing = json.load(open(labels_path))

        # Sidebar: Progress
        st.sidebar.header('Progress')
        labeled_count = sum(1 for v in existing.values() if v.get('label', ''))
        st.sidebar.metric('Labeled', f'{labeled_count} / {n_clusters}')
        st.sidebar.progress(labeled_count / n_clusters)

        # Sidebar: Filters
        st.sidebar.header('Display Settings')
        brightness_thresh = st.sidebar.slider('Min brightness', 0.0, 0.5, 0.15, 0.05)
        variance_thresh = st.sidebar.slider('Min variance', 0.0, 30.0, 10.0, 5.0)
        n_samples = st.sidebar.slider('Samples per cluster', 6, 50, 9, 3)
        context_size = st.sidebar.slider('Context window size', 48, 128, 72, 8,
                                         help='Larger = more surrounding face/context visible')
        part_box_size = st.sidebar.slider('Part box size (red square)', 8, 48, 24, 8,
                                          help='8 = single patch (tiny), 24 = ~3× patches (e.g. eye scale), 32 = larger part')

        # Sidebar: Cluster overview
        st.sidebar.header('Cluster Overview')
        for cid in range(n_clusters):
            size = int((labels_arr == cid).sum())
            fg_score = fg_scores[cid]
            label = existing.get(str(cid), {}).get('label', '—')
            if fg_score > 0.6:
                emoji = "🐱"
            elif fg_score > 0.4:
                emoji = "❓"
            else:
                emoji = "🌫️"
            st.sidebar.caption(f'{emoji} C{cid}: {label[:15]} ({size//1000}k, fg={fg_score:.2f})')

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
        with st.spinner('Loading patches with context...'):
            patches = get_cluster_patches_fast(
                cluster_id, data, labels_arr, quality_cache,
                n_samples=n_samples,
                brightness_threshold=brightness_thresh,
                variance_threshold=variance_thresh,
                context_size=context_size,
                part_box_size=part_box_size
            )

        # Display patches
        if not patches:
            st.warning('⚠️ No quality patches found. This cluster is likely pure background.')
            st.info('💡 Label as "background" and uncheck "Include in classification"')
        else:
            st.success(f'✅ Showing {len(patches)} highest-quality patches')
            cols = st.columns(3)
            for i, (patch, path, pidx, brightness, variance, spatial) in enumerate(patches):
                cols[i % 3].image(patch,
                                  caption=f'patch {pidx} (center={spatial:.2f})',
                                  width=150)

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
                format_func=lambda x: {1: '😐 Unsure', 2: '🙂 Likely', 3: '😃 Certain'}[x],
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
        labeled_count = sum(1 for v in existing.values() if v.get('label', ''))
        if labeled_count == n_clusters:
           # st.balloons()
            st.success('🎉 All clusters labeled! Run: python experiments/run_pipeline.py --stage concepts')


if __name__ == '__main__':
    run_streamlit()