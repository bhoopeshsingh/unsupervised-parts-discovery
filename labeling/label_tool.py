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
import torch.nn.functional as F
import cv2


def compute_cluster_class_distribution(data, labels_arr, n_clusters):
    """
    For each cluster, compute fraction of patches from each class (bird/car/cat).
    Returns dict: {cluster_id: {'bird': 0.3, 'car': 0.5, 'cat': 0.2, 'dominant': 'car'}}
    """
    from collections import Counter
    class_names  = data['class_names']          # e.g. ['bird', 'car', 'cat']
    image_labels = data['image_labels']          # per-image class index, list[N_images]
    image_ids    = data['image_ids'].numpy()     # per-patch image index [N_patches]

    # Build per-patch class string
    patch_classes = [class_names[image_labels[iid]] for iid in image_ids]

    dist = {}
    for cid in range(n_clusters):
        mask    = (labels_arr == cid)
        indices = np.where(mask)[0]
        counts  = Counter(patch_classes[i] for i in indices)
        total   = sum(counts.values()) or 1
        fracs   = {cls: counts.get(cls, 0) / total for cls in class_names}
        dominant = max(fracs, key=fracs.get)
        fracs['dominant'] = dominant
        fracs['is_mixed'] = fracs[dominant] < 0.70   # mixed if no class > 70%
        dist[cid] = fracs
    return dist


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

    class_dist = compute_cluster_class_distribution(data, labels_arr, n_clusters)
    return data, labels_arr, n_clusters, cfg, cluster_centers, class_dist


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
                              spatial_centrality_threshold=0.3,
                              context_size=72,
                              part_box_size=24):
    """
    Patch selection WITH CONTEXT WINDOWS, sorted by representativeness.

    Patches are sorted by cosine similarity to the cluster mean feature vector:
      - First n_core patches  = most representative (closest to cluster centre)
      - Last  n_boundary      = boundary cases  (farthest from cluster centre)

    This gives the labeler a much clearer signal about what the cluster represents
    and where its edges are, compared to random or quality-only sampling.

    Args:
        context_size: Size of context window (default 72 = more face context)
        part_box_size: Size of red highlighted "part" box in pixels
    """
    cluster_mask = (labels_arr == cluster_id)
    cluster_indices = np.where(cluster_mask)[0]

    if len(cluster_indices) == 0:
        return []

    brightness = quality_cache['brightness'][cluster_indices].numpy()
    variance   = quality_cache['variance'][cluster_indices].numpy()
    spatial    = quality_cache['spatial_centrality'][cluster_indices].numpy()

    # ── Representativeness: cosine similarity to cluster mean ──────────────
    # Uses raw DINO features (works for both 384-dim and 1152-dim multilayer)
    cluster_feats = data['features'][cluster_indices]          # [M, D]
    cluster_mean  = F.normalize(cluster_feats.mean(0, keepdim=True), dim=1)  # [1, D]
    cluster_feats_norm = F.normalize(cluster_feats, dim=1)     # [M, D]
    cosine_sims   = (cluster_feats_norm @ cluster_mean.T).squeeze(1).numpy()  # [M]
    # cosine_sims[i] = 1.0 → perfect match to cluster centre (most representative)
    # cosine_sims[i] = low → boundary/outlier patch

    # ── Filter by quality + spatial centrality (remove edge/background patches) ──
    # spatial_centrality = 1/(1+dist) → 1.0=image centre, ~0.0=image edge
    # Keep patches with value >= threshold (i.e. sufficiently central)
    quality_ok = (
        (brightness >= brightness_threshold) &
        (variance >= variance_threshold) &
        (spatial >= spatial_centrality_threshold)
    )
    good_local = np.where(quality_ok)[0]

    if len(good_local) == 0:
        return []

    good_global  = cluster_indices[good_local]
    good_sims    = cosine_sims[good_local]

    # ── Select: n_core most representative + n_boundary boundary cases ─────
    sorted_by_sim = np.argsort(good_sims)[::-1]   # descending: best first
    n_core     = max(6, n_samples - 3)
    n_boundary = n_samples - n_core
    core_local     = sorted_by_sim[:n_core]
    boundary_local = sorted_by_sim[-n_boundary:] if len(sorted_by_sim) > n_core else []

    selected_local  = np.concatenate([core_local, boundary_local]).astype(int)
    selected_global = good_global[selected_local]
    selected_sims   = good_sims[selected_local]
    is_boundary     = np.array([False] * len(core_local) + [True] * len(boundary_local))
    
    # Load patches WITH CONTEXT
    patches = []
    for _, (idx, sim, is_bdry) in enumerate(
        zip(selected_global, selected_sims, is_boundary)
    ):
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

            context_crop = img_arr[r_start:r_end, c_start:c_end].copy()

            # Red box for core, orange box for boundary cases
            patch_r_in_ctx = row_p * 8 - r_start
            patch_c_in_ctx = col_p * 8 - c_start
            pr_center = patch_r_in_ctx + 4
            pc_center = patch_c_in_ctx + 4
            half_box  = part_box_size // 2
            box_r0 = max(0, pr_center - half_box)
            box_c0 = max(0, pc_center - half_box)
            box_r1 = min(context_crop.shape[0], box_r0 + part_box_size)
            box_c1 = min(context_crop.shape[1], box_c0 + part_box_size)
            thickness  = max(2, part_box_size // 12)
            box_colour = (255, 140, 0) if is_bdry else (255, 0, 0)  # orange=boundary, red=core

            context_with_box = context_crop.copy()
            cv2.rectangle(context_with_box, (box_c0, box_r0), (box_c1, box_r1),
                          box_colour, thickness)

            context_pil = Image.fromarray(context_with_box)

            local_idx = np.where(cluster_indices == idx)[0][0]
            patches.append((
                context_pil,
                str(data['image_paths'][img_idx]),
                int(patch_idx),
                float(brightness[local_idx]),
                float(variance[local_idx]),
                float(spatial[local_idx]),
                float(sim),        # representativeness score
                bool(is_bdry),     # True = boundary case
            ))
        except Exception:
            continue

    return patches


# ── Classification helpers ─────────────────────────────────────

@st.cache_resource(show_spinner="Loading DINO model…")
def load_dino_extractor(model_name: str, device: str, image_size: int,
                        use_multilayer: bool = False):
    from src.models.dino_extractor import DinoExtractor
    return DinoExtractor(model_name=model_name, device=device, image_size=image_size,
                         use_multilayer=use_multilayer)


@st.cache_resource(show_spinner="Loading classifier…")
def load_classifier_and_vectors(classifier_path: str, vectors_path: str):
    import pickle
    with open(classifier_path, "rb") as f:
        payload = pickle.load(f)
    # New format: dict with keys clf, class_names, concept_names, …
    if isinstance(payload, dict):
        clf        = payload["clf"]
        class_names = payload["class_names"]
    else:
        # Legacy one-class wrapper — still works for backwards compat
        clf        = payload
        class_names = ["cat"]
    saved = torch.load(vectors_path, weights_only=False)
    return clf, saved["vectors"], class_names


def run_classify_tab(cfg):
    """Classify & Explain tab: upload image → prediction + semantic part map."""
    from src.pipeline.concept_classifier import (
        compute_image_concept_scores,
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

    # ── file uploader ──────────────────────────────────────────
    uploaded = st.file_uploader(
        "Choose an image (jpg / png)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.info("👆 Upload an image to get started.")
        return

    # ── load models only once an image is present ──────────────
    extractor = load_dino_extractor(
        cfg["dino"]["model"],
        cfg["dino"]["device"],
        cfg["dino"]["image_size"],
        cfg["dino"].get("use_multilayer", False),
    )
    clf, vectors, class_names = load_classifier_and_vectors(clf_path, vec_path)

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

        # Load clusterer + label mapping for cluster-proportion scoring
        clusterer_path = cfg["dino"].get("clusterer_path", "cache/kmeans.pkl")
        with open(clusterer_path, "rb") as _f:
            _model = pickle.load(_f)
        from src.pipeline.patch_clusterer import PatchClusterer
        _clusterer = PatchClusterer.load(clusterer_path)
        _human_labels = json.load(open(cfg["concepts"]["labels_path"]))
        concept_to_cluster = {
            meta["label"]: int(cid)
            for cid, meta in _human_labels.items()
            if meta.get("label", "").strip() and meta.get("include", True)
        }
        fg_patch_ids = torch.where(fg_mask)[0]
        concept_scores_dict = compute_image_concept_scores(
            fg_feats, vectors,
            clusterer=_clusterer, concept_to_cluster=concept_to_cluster,
            patch_ids=fg_patch_ids,
        )

        # Load the concept names the classifier's StandardScaler was fitted on.
        # This is the ground truth for column order and count — must not drift.
        _scores_cache = cfg["concepts"].get("scores_cache", "cache/concept_scores.pt")
        clf_concept_names = torch.load(_scores_cache, weights_only=True)["concept_names"]

        # Build score_vec in the exact order/shape the scaler expects.
        # Concepts missing from current scoring default to 0.0.
        score_vec = np.array([[concept_scores_dict.get(c, 0.0) for c in clf_concept_names]])

        # 3-class logistic regression
        pred_idx   = clf.predict(score_vec)[0]
        pred_proba = clf.predict_proba(score_vec)[0]       # [n_classes]
        pred_label = class_names[pred_idx]
        confidence = float(pred_proba[pred_idx])

        # Contributions = logistic regression coefficients × score (per concept)
        coef_row = clf.coef_[pred_idx]                     # [n_concepts]
        mean_score = score_vec[0].mean()
        contributions = {
            c: float(coef_row[i] * (score_vec[0][i] - mean_score))
            for i, c in enumerate(clf_concept_names)
        }

        result = {
            "prediction": pred_label,
            "confidence": confidence,
            "class_proba": {cls: float(p) for cls, p in zip(class_names, pred_proba)},
            "concept_scores": {c: concept_scores_dict.get(c, 0.0) for c in clf_concept_names},
            "contributions": contributions,
        }

        # For spatial map: only concepts that exist in vectors can be visualised
        concept_names = [c for c in clf_concept_names if c in vectors]
        concept_map, patch_sims = get_spatial_concept_map(
            concept_names, vectors, all_feats
        )
        os.unlink(tmp_path)

    # ── prediction banner ──────────────────────────────────────
    pred = result["prediction"]
    conf = result["confidence"]
    cls_colors  = {"cat": "#1a237e", "car": "#b71c1c", "bird": "#1b5e20"}
    cls_emojis  = {"cat": "🐱", "car": "🚗", "bird": "🐦"}
    banner_color = cls_colors.get(pred, "#37474f")
    emoji        = cls_emojis.get(pred, "❓")

    st.markdown(
        f"""
        <div style="background:{banner_color};padding:14px 20px;border-radius:10px;
                    text-align:center;margin:12px 0">
            <span style="color:white;font-size:1.6rem;font-weight:700">
                {emoji} Prediction: {pred.upper()}
            </span>
            <span style="color:#cfd8dc;font-size:1.1rem;margin-left:16px">
                Confidence: {conf:.1%}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Per-class probability breakdown
    if "class_proba" in result:
        prob_cols = st.columns(len(result["class_proba"]))
        for i, (cls, prob) in enumerate(result["class_proba"].items()):
            prob_cols[i].metric(
                f"{cls_emojis.get(cls,'')} {cls.upper()}",
                f"{prob:.1%}",
                delta="predicted" if cls == pred else None,
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
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0.1)
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
                "Contribution": round(contribs[n], 4),
                "Signal": "↑ supports" if contribs[n] >= 0 else "↓ opposes",
            }
            for n in concept_names
        ],
        key=lambda r: r["Activation"],
        reverse=True,
    )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # ── plain-language explanation ─────────────────────────────
    st.subheader("Plain-language explanation")
    top_activated = [r["Concept"] for r in rows if r["Activation"] > 0.5][:3]
    parts = ", ".join(f"**{c}**" for c in top_activated) if top_activated else "dominant concept patterns"
    st.success(
        f"**{pred.upper()}** ({conf:.0%} confidence)\n\n"
        f"The concept activation profile — {parts} — best matches the "
        f"**{pred}** class. The classifier scores each image on {len(clf_concept_names)} "
        f"named semantic concepts and picks the class whose concept profile fits best."
    )
    with st.expander("How to read these results"):
        st.markdown(
            "**Activation** — how strongly each semantic concept (labelled by a human) "
            "was detected in this image. Higher = stronger match to that concept.\n\n"
            "**Contribution** — logistic regression coefficient × activation deviation from mean. "
            "Positive = pushed prediction toward this class; negative = pushed away.\n\n"
            "**Important:** the 3-class classifier (cat | car | bird) scores each image "
            "against all named concepts and picks the class whose concept profile fits best. "
            "A car image should activate car-related concepts strongly and cat/bird concepts weakly.\n\n"
            "**Semantic Part Map** — each 8×8 patch coloured by its closest concept. "
            "Shows *where* in the image each part was detected."
        )


# ── Streamlit UI ──────────────────────────────────────────────
def run_streamlit():
    st.set_page_config(page_title='Parts Discovery', layout='wide')
    st.title('Unsupervised Parts Discovery')

    # Load config + data once (shared by both tabs)
    data, labels_arr, n_clusters, cfg, cluster_centers, class_dist = load_data()

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

        # Load pre-computed patch quality (generated by --stage cluster)
        quality_cache_path = cfg['dino'].get('patch_quality_cache', 'cache/patch_quality.pt')
        if not Path(quality_cache_path).exists():
            st.error(
                f"Patch quality cache not found: `{quality_cache_path}`\n\n"
                "Re-run the cluster stage to generate it:\n"
                "```\npython experiments/run_pipeline.py --stage cluster\n```"
            )
            st.stop()
        if 'quality_cache' not in st.session_state:
            st.session_state.quality_cache = torch.load(quality_cache_path, weights_only=False)
        quality_cache = st.session_state.quality_cache

        # Load existing labels
        existing = {}
        if Path(labels_path).exists():
            existing = json.load(open(labels_path))

        # Initialize selected cluster in session state
        if 'selected_cluster' not in st.session_state:
            st.session_state.selected_cluster = 0
        # Clamp in case n_clusters changed
        if st.session_state.selected_cluster >= n_clusters:
            st.session_state.selected_cluster = 0

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
        spatial_centrality_thresh = st.sidebar.slider(
            'Min spatial centrality', 0.0, 1.0, 0.3, 0.05,
            help='1.0 = image centre, 0.0 = image edge. Higher = only show central object patches. 0.0 = no filter.'
        )

        # Sidebar: Clickable cluster list
        st.sidebar.header('Clusters')
        cls_emoji = {'bird': '🟦', 'car': '🟥', 'cat': '🟩'}
        for cid in range(n_clusters):
            size = int((labels_arr == cid).sum())
            clabel = existing.get(str(cid), {}).get('label', '—')
            dom = class_dist[cid]['dominant']
            mixed_flag = '⚠️' if class_dist[cid]['is_mixed'] else ''
            emoji = cls_emoji.get(dom, '⬜')
            btn_label = f'{emoji}{mixed_flag} C{cid}: {clabel[:12]} ({size//1000}k)'
            is_selected = st.session_state.selected_cluster == cid
            if st.sidebar.button(
                btn_label,
                key=f'cluster_btn_{cid}',
                type='primary' if is_selected else 'secondary',
                use_container_width=True,
            ):
                st.session_state.selected_cluster = cid
                st.rerun()

        cluster_id = st.session_state.selected_cluster

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
        if fg_score > 0.5 and spatial_mean < 8:
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
                spatial_centrality_threshold=spatial_centrality_thresh,
                context_size=context_size,
                part_box_size=part_box_size
            )

        # Display patches
        if not patches:
            st.warning('⚠️ No quality patches found. This cluster is likely pure background.')
            st.info('💡 Label as "background" and uncheck "Include in classification"')
        else:
            core_patches     = [p for p in patches if not p[7]]
            boundary_patches = [p for p in patches if p[7]]

            st.success(
                f'✅ {len(core_patches)} core patches (🔴 red box = most representative) '
                f'+ {len(boundary_patches)} boundary patches (🟠 orange box = cluster edge)'
            )
            st.caption('Core patches = closest to cluster centre. '
                       'Boundary patches = farthest from centre (helps spot where the cluster bleeds into other concepts).')

            st.markdown('**Core patches** — what this cluster mostly looks like')
            cols = st.columns(3)
            for i, (patch, path, pidx, brightness, variance, spatial, sim, _) in enumerate(core_patches):
                cols[i % 3].image(patch,
                                  caption=f'sim={sim:.2f} | patch {pidx}',
                                  width=150)

            if boundary_patches:
                st.markdown('**Boundary patches** — edge of cluster (may bleed into adjacent concept)')
                cols2 = st.columns(3)
                for i, (patch, path, pidx, brightness, variance, spatial, sim, _) in enumerate(boundary_patches):
                    cols2[i % 3].image(patch,
                                       caption=f'sim={sim:.2f} | patch {pidx}',
                                       width=150)

        st.divider()

        # ── Class distribution for this cluster ───────────────────────────────
        dist = class_dist[cluster_id]
        dominant_cls = dist['dominant']
        class_names_list = [c for c in data['class_names']]

        st.subheader(f'Label Cluster {cluster_id}')

        # Class breakdown bar
        bar_cols = st.columns(len(class_names_list))
        cls_colors = {'bird': '🟦', 'car': '🟥', 'cat': '🟩'}
        for i, cls in enumerate(class_names_list):
            pct = dist.get(cls, 0)
            bar_cols[i].metric(
                f"{cls_colors.get(cls,'⬜')} {cls.upper()}",
                f"{pct:.0%}",
                delta="dominant" if cls == dominant_cls else None,
            )

        if dist['is_mixed']:
            st.warning(f'⚠️ Mixed cluster — no single class dominates (>{70:.0f}%). Label with caution or mark as background.')
        else:
            st.info(f'✅ Dominant class: **{dominant_cls.upper()}** ({dist[dominant_cls]:.0%} of patches) — label as `{dominant_cls}: <part name>`')

        current = existing.get(str(cluster_id), {})

        # Auto-suggest label prefix from dominant class if no existing label
        default_label = current.get('label', '')
        if not default_label and not dist['is_mixed']:
            default_label = f'{dominant_cls}: '

        col1, col2 = st.columns([2, 1])

        with col1:
            label = st.text_input(
                'Semantic label  (format: class: part — e.g. "cat: nose", "car: windshield")',
                value=default_label,
                placeholder='e.g. cat: nose, car: windshield, bird: wing',
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
                        st.session_state.selected_cluster = cid
                        st.rerun()
                        break

        # Summary
        labeled_count = sum(1 for v in existing.values() if v.get('label', ''))
        if labeled_count == n_clusters:
           # st.balloons()
            st.success('🎉 All clusters labeled! Run: python experiments/run_pipeline.py --stage concepts')


if __name__ == '__main__':
    run_streamlit()