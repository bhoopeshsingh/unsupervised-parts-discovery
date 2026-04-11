# labeling/lab_cluster_labeler.py
"""
Lab Report Cluster Labeling Tool — Streamlit GUI (Panel-Structured)

Two-tab interface:
  Tab 1 — Label Clusters : SME inspects per-panel clusters and assigns clinical labels
                           with optional LLM suggestion (via lab_concept_advisor.py)
  Tab 2 — Classify       : upload a CSV / manual entry → concept activations + explanation

Panel-structured cluster keys: "cbc:0", "biochem:3", "lipid:1"
  Each panel (CBC / BMP / Lipid) has its own independent GMM clustering.
  Labels are saved in cache/lab/labels.json with "panel:cluster_id" keys.

Run with:  streamlit run labeling/lab_cluster_labeler.py
"""

import json
import os
import pickle
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

CONFIG = "configs/config_lab.yaml"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading pipeline cache…")
def load_data():
    """
    Load panel-structured cluster data from the new pipeline.
    Returns everything needed by both tabs.
    """
    cfg       = yaml.safe_load(open(CONFIG))
    cache_dir = Path(cfg["lab_data"]["cache_dir"])

    # Features (flat 34-dim deviation, all records)
    feat_data     = torch.load(cache_dir / "features.pt", weights_only=False)
    features      = feat_data["features"]        # [N, 34]
    feature_cols  = feat_data["feature_cols"]    # ordered list
    record_ids    = feat_data["record_ids"]

    # Per-panel cluster labels and fitted GMM models
    cluster_data      = torch.load(cache_dir / "panel_clusters.pt", weights_only=False)
    per_panel_labels  = cluster_data["per_panel_labels"]   # {panel_name: np.array [N]}
    panel_names       = cluster_data["panel_names"]        # ['cbc', 'biochem', 'lipid']
    n_clusters        = cluster_data["n_clusters"]         # clusters per panel

    with open(cache_dir / "panel_clusterers.pkl", "rb") as f:
        clusterers = pickle.load(f)                        # {panel_name: PatchClusterer}

    records_df = pd.read_csv(cache_dir / "records.csv")

    return (
        features, feature_cols, record_ids,
        per_panel_labels, panel_names, n_clusters,
        clusterers, records_df, cfg,
    )


def get_feature_labels(cfg) -> dict:
    labels = {}
    for panel in cfg["lab_data"]["panels"].values():
        labels.update(panel["features"])
    return labels


def get_clinical_ranges(cfg) -> dict:
    return cfg["embedding"]["clinical_ranges"]


def get_panel_feature_cols(cfg, panel_name: str) -> list:
    """Return feature columns belonging to a specific panel."""
    return list(cfg["lab_data"]["panels"][panel_name]["features"].keys())


# ---------------------------------------------------------------------------
# Deviation profile chart
# ---------------------------------------------------------------------------

def plot_cluster_profile(
    panel_name: str,
    cluster_id: int,
    features: torch.Tensor,
    panel_labels: np.ndarray,
    feature_cols: list,
    feat_labels: dict,
    ranges: dict,
    panel_feature_cols: list,
):
    """
    Horizontal bar chart of mean clinical deviation for this panel's cluster.
    Panel-owned features are shown prominently; other features shown as context.
    """
    mask          = panel_labels == cluster_id
    cluster_feats = features[mask].numpy()
    pop_feats     = features.numpy()

    cluster_means = cluster_feats.mean(axis=0)
    pop_means     = pop_feats.mean(axis=0)

    labels_short = [feat_labels.get(c, c).split(" (")[0][:28] for c in feature_cols]

    colors = []
    alphas = []
    for c, v in zip(feature_cols, cluster_means):
        in_panel = c in panel_feature_cols
        if abs(v) < 1.0:
            colors.append("#2ecc71" if in_panel else "#a9dfbf")
        elif abs(v) < 2.0:
            colors.append("#f39c12" if in_panel else "#fdebd0")
        else:
            colors.append("#e74c3c" if in_panel else "#f5b7b1")
        alphas.append(1.0 if in_panel else 0.45)

    fig, ax = plt.subplots(figsize=(8, max(5, len(feature_cols) * 0.34)))
    y = np.arange(len(feature_cols))

    for i, (yi, val, col, alpha) in enumerate(zip(y, cluster_means, colors, alphas)):
        ax.barh(yi, val, color=col, alpha=alpha)

    ax.plot(pop_means, y, "k|", markersize=8, label="Population mean", zorder=5)
    ax.axvline(0,  color="gray",     linewidth=0.8, linestyle="--")
    ax.axvline(-1, color="#27ae60",  linewidth=0.5, linestyle=":")
    ax.axvline(1,  color="#27ae60",  linewidth=0.5, linestyle=":")

    ax.set_yticks(y)
    ax.set_yticklabels(labels_short, fontsize=8)
    ax.set_xlabel("Deviation from normal range  (0=normal, ±1=boundary, >±2=abnormal)")
    ax.set_title(
        f"Panel '{panel_name.upper()}' — Cluster {cluster_id}  (n={int(mask.sum()):,} records)",
        fontweight="bold",
    )

    legend_patches = [
        mpatches.Patch(color="#2ecc71", label=f"Panel '{panel_name}' — Normal"),
        mpatches.Patch(color="#f39c12", label=f"Panel '{panel_name}' — Mild"),
        mpatches.Patch(color="#e74c3c", label=f"Panel '{panel_name}' — Abnormal"),
        mpatches.Patch(color="#a9dfbf", alpha=0.5, label="Other panels (context)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=7)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Representative records
# ---------------------------------------------------------------------------

def get_representative_records(
    panel_name: str,
    cluster_id: int,
    features: torch.Tensor,
    panel_labels: np.ndarray,
    record_ids: list,
    records_df: pd.DataFrame,
    feature_cols: list,
    topk: int = 10,
) -> pd.DataFrame:
    """Top-K records closest to the cluster centre (most representative)."""
    mask        = panel_labels == cluster_id
    cluster_idx = np.where(mask)[0]
    if len(cluster_idx) == 0:
        return pd.DataFrame()

    cluster_feats = features[cluster_idx]
    cluster_mean  = F.normalize(cluster_feats.mean(0, keepdim=True), dim=1)
    sims          = (F.normalize(cluster_feats, dim=1) @ cluster_mean.T).squeeze(1).numpy()
    top_local     = np.argsort(sims)[::-1][:topk]
    top_global    = cluster_idx[top_local]

    ids  = [record_ids[i] for i in top_global]
    rows = records_df[records_df["record_id"].isin(ids)][
        ["record_id"] + feature_cols
    ].copy()
    rows = rows.set_index("record_id").reindex(ids).reset_index()
    rows.insert(0, "similarity", np.round(sims[top_local], 3))
    return rows


def style_record_table(df, feature_cols, feat_labels, ranges):
    """Colour cells: green=normal, yellow=mild, red=abnormal."""
    display = df.rename(
        columns={c: feat_labels.get(c, c).split(" (")[0][:20] for c in feature_cols}
    )

    def colour_cell(val, col):
        if col not in ranges or (isinstance(val, float) and np.isnan(val)):
            return ""
        lo, hi = ranges[col]
        mid    = (lo + hi) / 2.0
        half   = max((hi - lo) / 2.0, 1e-6)
        dev    = abs((val - mid) / half)
        if dev < 1.0:
            return "background-color:#d5f5e3;color:#145a32"
        if dev < 2.0:
            return "background-color:#fef9e7;color:#7d6608"
        return "background-color:#fadbd8;color:#7b241c"

    styler = display.style
    for col in feature_cols:
        short = feat_labels.get(col, col).split(" (")[0][:20]
        if short in display.columns:
            styler = styler.applymap(lambda v, c=col: colour_cell(v, c), subset=[short])
    return styler.format(precision=2)


# ---------------------------------------------------------------------------
# Tab 1 — Label Clusters
# ---------------------------------------------------------------------------

def run_label_tab(
    features, feature_cols, record_ids,
    per_panel_labels, panel_names, n_clusters,
    clusterers, records_df, cfg,
):
    labels_path  = cfg["concepts"]["labels_path"]
    feat_labels  = get_feature_labels(cfg)
    ranges       = get_clinical_ranges(cfg)
    topk         = cfg["concepts"].get("topk_records", 10)
    total_keys   = len(panel_names) * n_clusters

    existing = json.load(open(labels_path)) if Path(labels_path).exists() else {}

    # ── Session state: selected = "panel:cluster_id" ──────────────
    if "lab_selected" not in st.session_state:
        st.session_state.lab_selected = f"{panel_names[0]}:0"

    # ── Sidebar: progress ──────────────────────────────────────────
    labeled_count = sum(1 for v in existing.values() if v.get("label", ""))
    st.sidebar.header("Progress")
    st.sidebar.metric("Labeled", f"{labeled_count} / {total_keys}")
    st.sidebar.progress(labeled_count / total_keys)

    # ── Sidebar: panel-grouped cluster list ────────────────────────
    st.sidebar.header("Clusters by Panel")
    for pname in panel_names:
        panel_labels_arr = per_panel_labels[pname]
        panel_label_cfg  = cfg["lab_data"]["panels"][pname]["label"]
        with st.sidebar.expander(f"📋 {panel_label_cfg}", expanded=True):
            for cid in range(n_clusters):
                key       = f"{pname}:{cid}"
                size      = int((panel_labels_arr == cid).sum())
                clabel    = existing.get(key, {}).get("label", "—")
                is_labeled = bool(existing.get(key, {}).get("label", ""))
                emoji      = "✅" if is_labeled else "🔲"
                btn_label  = f"{emoji} C{cid}: {clabel[:14]}  ({size:,})"
                is_selected = st.session_state.lab_selected == key
                if st.button(
                    btn_label,
                    key=f"lab_btn_{key}",
                    type="primary" if is_selected else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.lab_selected = key
                    st.rerun()

    # ── Parse selected ─────────────────────────────────────────────
    sel_panel, sel_cid_str = st.session_state.lab_selected.rsplit(":", 1)
    sel_cid    = int(sel_cid_str)
    panel_cfg  = cfg["lab_data"]["panels"][sel_panel]
    panel_feature_cols = list(panel_cfg["features"].keys())
    panel_labels_arr   = per_panel_labels[sel_panel]
    cluster_size       = int((panel_labels_arr == sel_cid).sum())

    st.caption(
        "Each cluster = a pattern of lab values discovered **without** labels. "
        "Assign a clinical name based on the deviation profile and representative records below."
    )
    st.caption("🟢 Normal  🟡 Mildly abnormal  🔴 Significantly abnormal  "
               "*(faded bars = other panels shown for context)*")

    # ── Cluster stats ──────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Panel",   panel_cfg["label"])
    c2.metric("Cluster", f"#{sel_cid}")
    c3.metric("Records", f"{cluster_size:,}")
    c4.metric("% of total", f"{100*cluster_size/len(panel_labels_arr):.1f}%")

    # ── Deviation profile ──────────────────────────────────────────
    st.subheader("Clinical Deviation Profile")
    fig = plot_cluster_profile(
        sel_panel, sel_cid, features, panel_labels_arr,
        feature_cols, feat_labels, ranges, panel_feature_cols,
    )
    st.pyplot(fig)
    plt.close(fig)

    # ── Representative records ─────────────────────────────────────
    st.subheader(f"Top {topk} Most Representative Records")
    rep_df = get_representative_records(
        sel_panel, sel_cid, features, panel_labels_arr,
        record_ids, records_df, feature_cols, topk=topk,
    )
    if rep_df.empty:
        st.warning("No records found.")
    else:
        styled = style_record_table(rep_df, feature_cols, feat_labels, ranges)
        st.dataframe(styled, use_container_width=True)

    st.divider()

    # ── LLM Advisor ────────────────────────────────────────────────
    st.subheader(f"Label  —  {panel_cfg['label']}  Cluster {sel_cid}")
    current = existing.get(f"{sel_panel}:{sel_cid}", {})

    with st.expander("🤖 Ask AI for a label suggestion", expanded=False):
        ai_backend = st.radio(
            "LLM backend",
            ["ollama", "anthropic"],
            horizontal=True,
            key=f"ai_backend_{sel_panel}_{sel_cid}",
        )
        ollama_model = "llama3"
        if ai_backend == "ollama":
            ollama_model = st.text_input(
                "Ollama model name",
                value="llama3",
                key=f"ollama_model_{sel_panel}_{sel_cid}",
                help="Run: ollama pull llama3  (or meditron, biomistral, mistral)",
            )

        if st.button("🔬 Generate clinical label suggestion",
                     key=f"ai_btn_{sel_panel}_{sel_cid}"):
            with st.spinner("Querying medical LLM…"):
                from src.pipeline.lab_concept_advisor import (
                    build_deviation_profile,
                    format_profile_for_prompt,
                    query_anthropic,
                    query_ollama,
                )

                # Build a synthetic flat labels array for this panel's cluster
                # so the advisor's profile builder works without modification
                profile = build_deviation_profile(
                    cluster_id=sel_cid,
                    features=features,
                    cluster_labels=panel_labels_arr,
                    feature_cols=feature_cols,
                    feature_labels=feat_labels,
                    clinical_ranges=ranges,
                )
                profile_text = format_profile_for_prompt(profile, feat_labels)

                if ai_backend == "anthropic":
                    result = query_anthropic(profile_text)
                else:
                    result = query_ollama(
                        profile_text,
                        model=ollama_model,
                        host="http://localhost:11434",
                    )
                result["profile_text"] = profile_text
                result["n_records"]    = profile["n_records"]

            if result.get("error"):
                st.error(f"LLM error: {result['error']}")
                if ai_backend == "ollama":
                    st.info("Is Ollama running?  `ollama serve`  then  `ollama pull llama3`")
            else:
                st.session_state[f"ai_suggestion_{sel_panel}_{sel_cid}"] = result

        suggestion_key = f"ai_suggestion_{sel_panel}_{sel_cid}"
        if suggestion_key in st.session_state:
            r = st.session_state[suggestion_key]
            conf_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(
                str(r.get("CONFIDENCE", "")).lower(), "⚪"
            )
            st.markdown(
                f"**Suggested label:** `{r.get('PRIMARY_LABEL', '—')}`  "
                f"{conf_color} Confidence: {r.get('CONFIDENCE', '—')}"
            )
            st.markdown(f"**Key findings:** {r.get('KEY_FINDINGS', '—')}")
            st.markdown(f"**Differentials:** {r.get('DIFFERENTIALS', '—')}")
            st.caption(f"SME note: {r.get('SME_NOTE', '')}")

            if st.button("✅ Accept AI suggestion",
                         key=f"ai_accept_{sel_panel}_{sel_cid}"):
                st.session_state[f"prefill_{sel_panel}_{sel_cid}"] = \
                    r.get("PRIMARY_LABEL", "")
                st.rerun()

    # ── Label form ─────────────────────────────────────────────────
    prefill_key   = f"prefill_{sel_panel}_{sel_cid}"
    default_label = st.session_state.pop(prefill_key, None)
    if default_label is None:
        default_label = current.get("label", "")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        label_input = st.text_input(
            "Clinical label",
            value=default_label,
            placeholder="e.g. iron_deficiency_anaemia, CKD_pattern, dyslipidaemia",
            key=f"lbl_{sel_panel}_{sel_cid}",
        )
        notes_input = st.text_area(
            "Notes",
            value=current.get("notes", ""),
            height=60,
            key=f"notes_{sel_panel}_{sel_cid}",
        )
    with col_b:
        include_input = st.checkbox(
            "Include in classification",
            value=current.get("include", True),
            key=f"inc_{sel_panel}_{sel_cid}",
        )
        confidence_input = st.select_slider(
            "Confidence",
            options=[1, 2, 3],
            value=current.get("confidence", 2),
            format_func=lambda x: {1: "😐 Unsure", 2: "🙂 Likely", 3: "😃 Certain"}[x],
            key=f"conf_{sel_panel}_{sel_cid}",
        )

    col_save, col_next = st.columns(2)
    with col_save:
        if st.button("💾 Save", type="primary", use_container_width=True):
            existing[f"{sel_panel}:{sel_cid}"] = {
                "label":      label_input.strip(),
                "include":    include_input,
                "confidence": confidence_input,
                "notes":      notes_input.strip(),
            }
            Path(labels_path).parent.mkdir(parents=True, exist_ok=True)
            json.dump(existing, open(labels_path, "w"), indent=2)
            st.success(f'✅ Saved: {sel_panel}:{sel_cid} → "{label_input}"')
            st.rerun()

    with col_next:
        if st.button("⏩ Next unlabeled", use_container_width=True):
            for pname in panel_names:
                for cid in range(n_clusters):
                    k = f"{pname}:{cid}"
                    if not existing.get(k, {}).get("label", ""):
                        st.session_state.lab_selected = k
                        st.rerun()

    labeled_count = sum(1 for v in existing.values() if v.get("label", ""))
    if labeled_count == total_keys:
        st.success(
            "🎉 All clusters labeled! Run:\n"
            "```\npython experiments/run_lab_pipeline.py --stage concepts\n"
            "python experiments/run_lab_pipeline.py --stage classify\n```"
        )


# ---------------------------------------------------------------------------
# Tab 2 — Classify a new record
# ---------------------------------------------------------------------------

SAMPLE_DIR = "data/lab_sample_records"

SAMPLE_DESCRIPTIONS = {
    "healthy_normal":          "All values within normal range — expected healthy adult.",
    "iron_deficiency_anaemia": "Low Hgb/MCV/MCH, high RDW + elevated platelets — microcytic anaemia.",
    "metabolic_syndrome":      "High glucose, high TG, low HDL, high LDL, high total cholesterol.",
    "chronic_kidney_disease":  "Elevated creatinine/BUN, high K⁺, low bicarb, normocytic renal anaemia.",
    "liver_stress":            "Markedly elevated ALT/AST/ALP and bilirubin — hepatocellular injury.",
}


def _extract_features_for_record(record_df, feature_cols, cfg, cache_dir):
    """Impute missing values with training medians and return feature tensor [1, 34]."""
    from src.models.lab_extractor import LabExtractor
    train_df = pd.read_csv(cache_dir / "records.csv")
    for col in feature_cols:
        if col not in record_df.columns or pd.isna(record_df[col].iloc[0]):
            record_df[col] = train_df[col].median()
    extractor = LabExtractor.load(str(cache_dir / "extractor.pkl"), CONFIG)
    return extractor.transform(record_df[feature_cols], feature_cols)   # [1, 34]


def _compute_concept_scores_for_record(feat_tensor, cfg, cache_dir):
    """
    Run the full panel-structured concept scoring pipeline on a single record.

    Steps:
      1. Load PanelFTTransformer → extract panel-patches [1, n_panels, patch_dim]
      2. Apply per-panel GMM (from panel_clusterers.pkl) → soft proba [1, k] per panel
      3. Concatenate → [1, n_panels * k] concept score vector
    Returns (score_vec [1, C], concept_names list[str])
    """
    from sklearn.preprocessing import normalize as sk_normalize
    from src.models.panel_ft_transformer import PanelFTTransformer, build_panel_ids

    weights_path    = cfg["transformer"]["weights_path"]
    patches_meta_pt = cfg["transformer"]["panel_patches_path"]
    scores_cache    = cfg["concepts"].get("scores_cache", str(cache_dir / "concept_scores.pt"))
    labels_path     = cfg["concepts"]["labels_path"]
    device          = cfg["transformer"]["pretrain"]["device"]

    if not Path(weights_path).exists():
        return None, None

    # Load model + metadata
    model = PanelFTTransformer.load(weights_path, device=device)
    model.eval()

    patches_meta = torch.load(patches_meta_pt, weights_only=False)
    panel_names  = patches_meta["panel_names"]
    k            = cfg["transformer"]["n_clusters_per_panel"]

    # Extract panel-patches for this record
    x = feat_tensor.to(device)
    with torch.no_grad():
        panel_patches = model.extract_panel_patches(x)   # [1, n_panels, patch_dim]

    with open(cache_dir / "panel_clusterers.pkl", "rb") as f:
        clusterers = pickle.load(f)

    human_labels = json.load(open(labels_path)) if Path(labels_path).exists() else {}

    score_blocks  = []
    concept_names = []
    for p, pname in enumerate(panel_names):
        patch = panel_patches[0, p, :].cpu().numpy().reshape(1, -1)   # [1, D]
        clusterer = clusterers[pname]
        X = sk_normalize(patch, norm="l2")
        if clusterer.pca is not None:
            X = sk_normalize(clusterer.pca.transform(X), norm="l2")
        proba = clusterer.gmm.predict_proba(X).astype(np.float32)     # [1, k]
        score_blocks.append(proba)
        for cid in range(k):
            key   = f"{pname}:{cid}"
            label = human_labels.get(key, {}).get("label", "").strip()
            concept_names.append(f"{pname}: {label}" if label else f"{pname}: cluster_{cid}")

    score_vec = np.concatenate(score_blocks, axis=1)   # [1, n_panels*k]
    return score_vec, concept_names


def _show_classification_results(score_vec, concept_names, cfg, cache_dir):
    """Render concept activation scores + classifier prediction."""
    import pickle

    clf_path = cfg["classification"]["classifier_path"]
    if not Path(clf_path).exists():
        st.warning("Classifier not trained yet. Run `--stage classify` first.")
        return

    # Lab pipeline saves a plain dict (no scaler — GMM scores are already 0-1)
    with open(clf_path, "rb") as f:
        payload = pickle.load(f)

    raw_clf      = payload["clf"]
    class_names  = payload["class_names"]
    # Lab classifier was trained without scaling
    pred_class_id  = raw_clf.predict(score_vec)[0]
    pred_proba_all = raw_clf.predict_proba(score_vec)[0]
    pred           = class_names[pred_class_id]
    conf           = float(pred_proba_all.max())

    palette = ["#1a237e","#1b5e20","#b71c1c","#4a148c","#e65100","#006064","#37474f","#3e2723"]
    banner_color = palette[pred_class_id % len(palette)]
    st.markdown(
        f"""<div style="background:{banner_color};padding:14px 20px;border-radius:10px;
                text-align:center;margin:12px 0">
            <span style="color:white;font-size:1.5rem;font-weight:700">
                Prediction: {pred.replace('_',' ').title()}
            </span>
            <span style="color:#cfd8dc;font-size:1.1rem;margin-left:16px">
                Confidence: {conf:.1%}
            </span>
        </div>""",
        unsafe_allow_html=True,
    )

    with st.expander("Class probability breakdown"):
        rows = sorted(
            [{"Condition": class_names[i].replace("_"," ").title(),
              "Probability": round(float(p), 4)}
             for i, p in enumerate(pred_proba_all)],
            key=lambda r: r["Probability"], reverse=True,
        )
        st.dataframe(rows, use_container_width=True, hide_index=True)

    # Concept activation chart
    st.subheader("Concept Activation Scores")
    score_vals = score_vec[0]
    sorted_idx = np.argsort(score_vals)[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, len(concept_names) * 0.45)))
    bar_colors = ["#2980b9" if score_vals[i] > 0.3 else "#95a5a6" for i in sorted_idx]
    ax.barh(
        [concept_names[i].replace("_", " ") for i in sorted_idx],
        [score_vals[i] for i in sorted_idx],
        color=bar_colors,
    )
    ax.axvline(0.3, color="red", linestyle="--", linewidth=0.8, label="Threshold (0.3)")
    ax.set_xlabel("GMM cluster membership probability")
    ax.set_title("How strongly does this record match each clinical concept?")
    ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    signal_label = lambda s: ("🔴 Strong" if s > 0.3 else ("🟡 Partial" if s > 0.1 else "⚪ Weak"))
    table_rows = [
        {
            "Clinical Concept": concept_names[i].replace("_"," ").title(),
            "Activation": round(float(score_vals[i]), 4),
            "Signal": signal_label(score_vals[i]),
        }
        for i in sorted_idx
    ]
    st.dataframe(table_rows, use_container_width=True, hide_index=True)


def run_report_tab(cfg):
    """Show the full classification report from the last --stage classify run."""
    import pickle
    import pandas as pd

    st.header("Classification Report — Lab Experiment")
    st.caption(
        "Logistic regression trained on unsupervised concept scores. "
        "Labels (DIQ010, BPQ020) were never used during feature learning or clustering."
    )

    clf_path = cfg["classification"]["classifier_path"]
    if not Path(clf_path).exists():
        st.warning("Classifier not found. Run `--stage classify` first.")
        return

    with open(clf_path, "rb") as f:
        payload = pickle.load(f)

    report = payload.get("classification_report")
    if report is None:
        st.warning("No classification report saved. Re-run `--stage classify` to regenerate.")
        return

    # ── Accuracy summary ──────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Test Accuracy",  f"{payload['test_accuracy']:.1%}")
    col2.metric("Train Accuracy", f"{payload['train_accuracy']:.1%}")
    chance = 1 / len(payload["class_names"])
    col3.metric("Chance Baseline", f"{chance:.1%}")

    st.divider()

    # ── Per-class precision / recall / F1 ────────────────────────────────
    st.subheader("Per-class Performance")
    rows = []
    for cls in payload["class_names"]:
        m = report[cls]
        rows.append({
            "Class":     cls.replace("_", " ").title(),
            "Precision": round(m["precision"], 3),
            "Recall":    round(m["recall"],    3),
            "F1-Score":  round(m["f1-score"],  3),
            "Support":   int(m["support"]),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # ── Macro / weighted averages ─────────────────────────────────────────
    st.subheader("Averages")
    avg_rows = []
    for key in ("macro avg", "weighted avg"):
        if key in report:
            m = report[key]
            avg_rows.append({
                "Average":   key.title(),
                "Precision": round(m["precision"], 3),
                "Recall":    round(m["recall"],    3),
                "F1-Score":  round(m["f1-score"],  3),
                "Support":   int(m["support"]),
            })
    st.dataframe(avg_rows, use_container_width=True, hide_index=True)

    # ── F1 bar chart ──────────────────────────────────────────────────────
    st.subheader("F1-Score by Class")
    fig, ax = plt.subplots(figsize=(6, 3))
    classes = [r["Class"] for r in rows]
    f1s     = [r["F1-Score"] for r in rows]
    colors  = ["#1a237e", "#b71c1c", "#1b5e20"][: len(classes)]
    ax.barh(classes, f1s, color=colors)
    ax.axvline(chance, color="grey", linestyle="--", linewidth=0.8, label=f"Chance ({chance:.0%})")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1-Score")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(f"Label source: {payload.get('label_source', 'NHANES')}")


def run_classify_tab(cfg):
    st.header("Classify a new lab record")
    st.caption(
        "Enter a patient's lab values to see which clinical concept patterns "
        "it activates and get a classification with explanation."
    )

    cache_dir    = Path(cfg["lab_data"]["cache_dir"])
    feat_labels  = get_feature_labels(cfg)
    feature_cols = list(feat_labels.keys())
    ranges       = get_clinical_ranges(cfg)

    # Prerequisites
    weights_path = cfg["transformer"]["weights_path"]
    clf_path     = cfg["classification"]["classifier_path"]
    missing      = [p for p in [weights_path, clf_path] if not Path(p).exists()]
    if missing:
        st.error(
            "Pipeline not fully run. Complete these stages first:\n"
            "```\npython experiments/run_lab_pipeline.py --stage pretrain\n"
            "python experiments/run_lab_pipeline.py --stage encode\n"
            "python experiments/run_lab_pipeline.py --stage cluster\n"
            "# label clusters in this GUI, then:\n"
            "python experiments/run_lab_pipeline.py --stage concepts\n"
            "python experiments/run_lab_pipeline.py --stage classify\n```\n"
            f"Missing: {missing}"
        )
        return

    tab_manual, tab_upload, tab_sample = st.tabs(
        ["📝 Manual Entry", "📤 Upload CSV", "📁 Load Sample"]
    )

    def _classify_and_show(record_df):
        with st.spinner("Extracting features & computing concept scores…"):
            feat_tensor = _extract_features_for_record(record_df, feature_cols, cfg, cache_dir)
            score_vec, concept_names = _compute_concept_scores_for_record(
                feat_tensor, cfg, cache_dir
            )
        if score_vec is None:
            st.error("Transformer weights not found.")
            return
        _show_classification_results(score_vec, concept_names, cfg, cache_dir)

    with tab_manual:
        st.info("Enter lab values below. Leave blank → imputed with population median.")
        input_vals = {}
        for pname, panel_cfg_inner in cfg["lab_data"]["panels"].items():
            st.subheader(panel_cfg_inner["label"])
            cols = st.columns(3)
            for i, (col_code, col_label) in enumerate(panel_cfg_inner["features"].items()):
                v = cols[i % 3].text_input(col_label, value="", key=f"inp_{col_code}")
                input_vals[col_code] = v
        if st.button("🔍 Classify", type="primary", key="classify_manual"):
            record = {col: (float(v.strip()) if v.strip() else float("nan"))
                      for col, v in input_vals.items()}
            _classify_and_show(pd.DataFrame([record]))

    with tab_upload:
        st.info("Upload a CSV with NHANES column names. Missing columns imputed.")
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_upload")
        if uploaded is not None:
            upload_df = pd.read_csv(uploaded)
            upload_df.columns = [c.upper() for c in upload_df.columns]
            st.success(f"Loaded {len(upload_df)} record(s)")
            row_idx = 0
            if len(upload_df) > 1:
                row_idx = st.selectbox("Select row", range(len(upload_df)),
                                       format_func=lambda i: f"Row {i+1}",
                                       key="upload_row")
            st.dataframe(upload_df.iloc[[row_idx]], use_container_width=True)
            if st.button("🔍 Classify", type="primary", key="classify_upload"):
                _classify_and_show(upload_df.iloc[[row_idx]].reset_index(drop=True))

    with tab_sample:
        sample_files = sorted(Path(SAMPLE_DIR).glob("*.csv")) if Path(SAMPLE_DIR).exists() else []
        if not sample_files:
            st.warning(f"No sample files in `{SAMPLE_DIR}/`")
        else:
            sample_map  = {f.stem: f for f in sample_files}
            chosen_key  = st.selectbox(
                "Select a sample record",
                list(sample_map.keys()),
                format_func=lambda k: k.replace("_", " ").title(),
                key="sample_select",
            )
            desc = SAMPLE_DESCRIPTIONS.get(chosen_key, "")
            if desc:
                st.caption(f"**Pattern:** {desc}")
            sample_df = pd.read_csv(sample_map[chosen_key])
            sample_df.columns = [c.upper() for c in sample_df.columns]
            display_cols = [c for c in feature_cols if c in sample_df.columns]
            styled = style_record_table(sample_df[display_cols].head(1), display_cols,
                                        feat_labels, ranges)
            st.dataframe(styled, use_container_width=True)
            if st.button("🔍 Classify", type="primary", key="classify_sample"):
                _classify_and_show(sample_df.iloc[[0]].reset_index(drop=True))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_streamlit():
    st.set_page_config(page_title="Lab Concept Discovery", layout="wide")
    st.title("Lab Report — Unsupervised Concept Discovery")

    cfg       = yaml.safe_load(open(CONFIG))
    cache_dir = Path(cfg["lab_data"]["cache_dir"])

    # Prerequisites check
    required = {
        "features.pt":        cache_dir / "features.pt",
        "panel_clusters.pt":  cache_dir / "panel_clusters.pt",
        "panel_clusterers.pkl": cache_dir / "panel_clusterers.pkl",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        st.error(
            "Pipeline cache not ready. Run:\n"
            "```\npython experiments/run_lab_pipeline.py --stage pretrain\n"
            "python experiments/run_lab_pipeline.py --stage encode\n"
            "python experiments/run_lab_pipeline.py --stage cluster\n```\n"
            f"Missing files: {missing}"
        )
        st.stop()

    (features, feature_cols, record_ids,
     per_panel_labels, panel_names, n_clusters,
     clusterers, records_df, cfg) = load_data()

    tab_label, tab_classify, tab_report = st.tabs(
        ["🏷️ Label Clusters", "🔍 Classify Record", "📊 Classification Report"]
    )

    with tab_label:
        run_label_tab(
            features, feature_cols, record_ids,
            per_panel_labels, panel_names, n_clusters,
            clusterers, records_df, cfg,
        )

    with tab_classify:
        run_classify_tab(cfg)

    with tab_report:
        run_report_tab(cfg)


if __name__ == "__main__":
    run_streamlit()
