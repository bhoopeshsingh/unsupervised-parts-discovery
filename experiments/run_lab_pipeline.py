# experiments/run_lab_pipeline.py
"""
Phase 2: NHANES lab reports → unsupervised concepts → LogReg on diagnoses
the model never trained on (same headline claim as the image pipeline).

Rough map vs images: DINO patches / GMM / concept scores / LogReg here become
panel patches, PanelFT + SSL, GMM per panel, then clinical outcome prediction.

Stages: load (merge .xpt) → extract → pretrain → encode → cluster → (label in UI)
→ concepts → classify → validate.

Examples:
  python experiments/run_lab_pipeline.py --stage all
  python experiments/run_lab_pipeline.py --stage classify
  python experiments/run_lab_pipeline.py --stage validate
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml


CONFIG = "configs/config_lab.yaml"


# ---------------------------------------------------------------------------
# Cache cleanup helper
# ---------------------------------------------------------------------------

def _clear_files(paths, reason=""):
    cleared = [p for p in paths if Path(p).exists()]
    if cleared:
        tag = f" ({reason})" if reason else ""
        print(f"\n  🗑  Clearing stale cache{tag}:")
        for p in cleared:
            Path(p).unlink()
            print(f"       ✗  {p}")


# ---------------------------------------------------------------------------
# Stage: download
# ---------------------------------------------------------------------------

def stage_download(config_path: str = CONFIG):
    """Print curl lines for each .xpt listed in config_lab.yaml (does not run curl for you)."""
    cfg = yaml.safe_load(open(config_path))
    dcfg = cfg["lab_data"]
    file_bases = dcfg["file_bases"]
    val_bases = dcfg.get("validation_files") or {}
    # Lab panels + demographics + validation-only (HbA1c, diabetes/BP questionnaires)
    bases = list(file_bases.values()) + list(val_bases.values())
    cycles = dcfg["cycles"]

    print("\nFiles below match config_lab.yaml — save them under data/lab_samples (mkdir first):\n")
    print("mkdir -p data/lab_samples && cd data/lab_samples")
    print()
    for c in cycles:
        year = c["year"]
        suffix = c["suffix"]
        url_base = c["url_base"].rstrip("/")
        print(f"# --- {year} ({suffix}) ---")
        for base in bases:
            fname = f"{base}{suffix}.xpt"
            print(f"curl -fLO {url_base}/{fname}")
        print()
    print("Then from the project root:")
    print("  python experiments/run_lab_pipeline.py --stage load")
    print("  python experiments/run_lab_pipeline.py --stage all   # or run L2–L8 individually")
    print("\nIf curl 404s, check that cycle’s DataFiles page — CDC names move occasionally.")


# ---------------------------------------------------------------------------
# Stage: load
# ---------------------------------------------------------------------------

def stage_load(cfg, config_path: str = CONFIG):
    print("\n" + "=" * 60)
    print("STAGE L1: Load & Merge NHANES Panels")
    print("=" * 60)
    from src.data.lab_loader import load_nhanes

    features_df, full_df, feature_cols = load_nhanes(config_path)

    cache_dir = Path(cfg["lab_data"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    features_df.to_csv(cache_dir / "records.csv", index=False)
    full_df.to_csv(cache_dir / "records_with_demo.csv", index=False)

    # Cycle distribution summary
    for yr, grp in features_df.groupby("cycle_year"):
        print(f"     {yr}: {len(grp):,} records")

    meta = {
        "feature_cols": feature_cols,
        "n_records": len(features_df),
        "n_features": len(feature_cols),
    }
    torch.save(meta, cache_dir / "meta.pt")

    print(f"\n  ✓  Saved {len(features_df):,} records → {cache_dir}/records.csv")
    return features_df, feature_cols


# ---------------------------------------------------------------------------
# Stage: extract
# ---------------------------------------------------------------------------

def stage_extract(cfg):
    print("\n" + "=" * 60)
    print("STAGE L2: Feature Extraction — clinical deviation encoding")
    print("=" * 60)

    cache_dir = Path(cfg["lab_data"]["cache_dir"])
    records_path = cache_dir / "records.csv"
    if not records_path.exists():
        print("ERROR: records.csv not found. Run --stage load first.")
        sys.exit(1)

    import pandas as pd
    from src.models.lab_extractor import LabExtractor

    features_df = pd.read_csv(records_path)
    meta = torch.load(cache_dir / "meta.pt", weights_only=False)
    feature_cols = meta["feature_cols"]

    # Clear downstream stale files
    _clear_files(
        [
            str(cache_dir / "features.pt"),
            str(cache_dir / "clusters.pt"),
            str(cache_dir / "kmeans.pkl"),
            str(cache_dir / "kmeans_meta.pt"),
            str(cache_dir / "kmeans_centers.pt"),
            str(cache_dir / "kmeans_pca.pkl"),
            cfg["concepts"]["labels_path"],
            cfg["concepts"].get("vectors_cache", str(cache_dir / "concept_vectors.pt")),
            cfg["concepts"].get("scores_cache",  str(cache_dir / "concept_scores.pt")),
            cfg["classification"]["classifier_path"],
        ],
        reason="re-extract invalidates all downstream cache",
    )

    extractor = LabExtractor(CONFIG)
    features_tensor = extractor.fit_transform(features_df, feature_cols)

    print(f"  Embedding method : {extractor.method}")
    print(f"  Feature tensor   : {list(features_tensor.shape)}  (records × features)")

    extractor.save(str(cache_dir / "extractor.pkl"))

    torch.save(
        {
            "features":    features_tensor,
            "record_ids":  features_df["record_id"].tolist(),
            "cycle_years": features_df["cycle_year"].tolist(),
            "feature_cols": feature_cols,
            "n_records":   len(features_df),
            "method":      extractor.method,
        },
        cache_dir / "features.pt",
    )
    print(f"  ✓  Saved features → {cache_dir}/features.pt")
    return features_tensor, feature_cols


# ---------------------------------------------------------------------------
# Stage: pretrain  — two-scale SSL pre-training on clinical deviation features
# ---------------------------------------------------------------------------

def stage_pretrain(cfg):
    print("\n" + "=" * 60)
    print("STAGE L3: SSL Pre-training  (PanelFTTransformer — two-scale)")
    print("=" * 60)

    cache_dir = Path(cfg["lab_data"]["cache_dir"])
    features_path = cache_dir / "features.pt"
    if not features_path.exists():
        print("ERROR: features.pt not found. Run --stage extract first.")
        sys.exit(1)

    from src.models.panel_ft_transformer import PanelFTTransformer, build_panel_ids

    data = torch.load(features_path, weights_only=False)
    features = data["features"]                      # [N, 34] clinical deviation tensor
    saved_feature_cols = data["feature_cols"]        # original fixed order from stage_extract

    tcfg = cfg["transformer"]
    panel_ids, feature_cols, panel_names = build_panel_ids(
        CONFIG,
        shuffle_panels=tcfg.get("shuffle_panels", False),
        random_seed=tcfg.get("shuffle_panels_seed", None),
    )

    # Reorder tensor columns to match the (possibly shuffled) feature_cols so
    # panel_ids[i] correctly describes the feature at position i in the tensor.
    if feature_cols != saved_feature_cols:
        col_idx = [saved_feature_cols.index(c) for c in feature_cols]
        features = features[:, col_idx]
        print(f"  Panel order shuffled — tensor columns reindexed to match.")

    pcfg = tcfg["pretrain"]

    print(f"  Panels : {panel_names}")
    print(f"  Counts : { {n: panel_ids.count(i) for i, n in enumerate(panel_names)} }")

    model = PanelFTTransformer(
        n_features=len(feature_cols),
        panel_ids=panel_ids,
        n_panels=len(panel_names),
        d_model=tcfg["d_model"],
        n_heads=tcfg["n_heads"],
        n_layers=tcfg["n_layers"],
        dropout=tcfg["dropout"],
        extract_layers=tuple(tcfg["extract_layers"]),
    )

    model.pretrain(
        features=features,
        n_epochs=pcfg["n_epochs"],
        batch_size=pcfg["batch_size"],
        lr=pcfg["lr"],
        device=pcfg["device"],
        save_path=tcfg["weights_path"],
    )

    # Clear downstream stale files
    _clear_files(
        [
            tcfg["panel_patches_path"],
            str(cache_dir / "panel_clusters.pt"),
            str(cache_dir / "panel_clusterers.pkl"),
            cfg["concepts"]["labels_path"],
            cfg["concepts"].get("vectors_cache", str(cache_dir / "concept_vectors.pt")),
            cfg["concepts"].get("scores_cache",  str(cache_dir / "concept_scores.pt")),
            cfg["classification"]["classifier_path"],
        ],
        reason="new transformer weights invalidate downstream cache",
    )
    return model


# ---------------------------------------------------------------------------
# Stage: encode  — extract per-panel multi-granularity patches
# ---------------------------------------------------------------------------

def stage_encode(cfg):
    """
    Uses the pre-trained PanelFTTransformer to extract multi-granularity
    panel-patch vectors for every record.

    Each record → n_panels panel-patches, each of size len(extract_layers)*d_model.
    Stored as [N*n_panels, patch_dim] with panel_source metadata.

    This is the analog of DINO's extract_patches — patches are panel-patches,
    and the multi-layer concatenation mirrors Phase 1's multilayer feature extraction.
    """
    print("\n" + "=" * 60)
    print("STAGE L4: Panel-Patch Encoding  (intermediate layer extraction)")
    print("=" * 60)

    cache_dir = Path(cfg["lab_data"]["cache_dir"])
    features_path = cache_dir / "features.pt"
    weights_path = cfg["transformer"]["weights_path"]

    if not features_path.exists():
        print("ERROR: features.pt not found. Run --stage extract first.")
        sys.exit(1)
    if not Path(weights_path).exists():
        print("ERROR: transformer.pt not found. Run --stage pretrain first.")
        sys.exit(1)

    from src.models.panel_ft_transformer import PanelFTTransformer, build_panel_ids

    data = torch.load(features_path, weights_only=False)
    features = data["features"]                      # [N, 34]
    record_ids = data["record_ids"]
    saved_feature_cols = data["feature_cols"]

    panel_ids, feature_cols, panel_names = build_panel_ids(
        CONFIG,
        shuffle_panels=cfg["transformer"].get("shuffle_panels", False),
        random_seed=cfg["transformer"].get("shuffle_panels_seed", None),
    )

    if feature_cols != saved_feature_cols:
        col_idx = [saved_feature_cols.index(c) for c in feature_cols]
        features = features[:, col_idx]

    n_panels = len(panel_names)
    device = cfg["transformer"]["pretrain"]["device"]

    model = PanelFTTransformer.load(weights_path, device=device)
    model.eval()

    print(f"  Loaded transformer: {model.n_layers}L d={model.d_model}")
    print(f"  Extract layers: {model.extract_layers}  →  patch_dim={len(model.extract_layers)*model.d_model}")

    # Process in batches to avoid OOM
    batch_size = 512
    all_patches = []
    for start in range(0, len(features), batch_size):
        batch = features[start:start + batch_size].to(device)
        patches = model.extract_panel_patches(batch)    # [B, n_panels, patch_dim]
        all_patches.append(patches.cpu())
    patches_all = torch.cat(all_patches, dim=0)         # [N, n_panels, patch_dim]

    N, P, D = patches_all.shape
    print(f"  Panel patches : {N} records × {P} panels × {D} dims")

    # Flatten to [N*n_panels, D] with metadata for clustering
    patches_flat = patches_all.reshape(N * P, D)        # [N*P, D]
    record_idx = torch.arange(N).repeat_interleave(P)   # [N*P] — which record
    panel_idx = torch.arange(P).repeat(N)               # [N*P] — which panel

    patches_path = cfg["transformer"]["panel_patches_path"]
    torch.save(
        {
            "patches":      patches_flat,         # [N*P, D]  — features for clustering
            "record_idx":   record_idx,           # [N*P] int — record source
            "panel_idx":    panel_idx,            # [N*P] int — panel source (0=cbc,1=bmp,2=lipid)
            "panel_names":  panel_names,
            "n_records":    N,
            "n_panels":     P,
            "patch_dim":    D,
            "record_ids":   record_ids,
        },
        patches_path,
    )
    print(f"  ✓  Panel patches saved → {patches_path}")

    _clear_files(
        [
            str(cache_dir / "panel_clusters.pt"),
            str(cache_dir / "panel_clusterers.pkl"),
            cfg["concepts"]["labels_path"],
            cfg["concepts"].get("vectors_cache", str(cache_dir / "concept_vectors.pt")),
            cfg["concepts"].get("scores_cache",  str(cache_dir / "concept_scores.pt")),
            cfg["classification"]["classifier_path"],
        ],
        reason="new panel patches invalidate cluster/concept cache",
    )
    return patches_flat, panel_names


# ---------------------------------------------------------------------------
# Stage: cluster
# ---------------------------------------------------------------------------

def stage_cluster(cfg):
    """
    Per-panel GMM clustering on multi-granularity panel-patches.

    Runs one GMM independently per panel (CBC / BMP / Lipid).
    Each panel gets n_clusters_per_panel clusters → panel-level clinical concepts.

    This mirrors Phase 1's PatchClusterer but on panel-patches instead of
    spatial image patches. Concept vector per record = concat of per-panel
    soft GMM memberships = [n_panels * n_clusters_per_panel] dimensional.
    """
    print("\n" + "=" * 60)
    print("STAGE L5: Per-Panel GMM Clustering")
    print("=" * 60)

    cache_dir = Path(cfg["lab_data"]["cache_dir"])
    patches_path = cfg["transformer"]["panel_patches_path"]

    if not Path(patches_path).exists():
        print("ERROR: panel_patches.pt not found. Run --stage encode first.")
        sys.exit(1)

    from src.pipeline.patch_clusterer import PatchClusterer
    import pickle

    data = torch.load(patches_path, weights_only=False)
    patches   = data["patches"]          # [N*P, D]
    panel_idx = data["panel_idx"]        # [N*P]
    panel_names = data["panel_names"]
    n_panels  = data["n_panels"]

    k = cfg["transformer"]["n_clusters_per_panel"]
    ccfg = cfg["clustering"]

    per_panel_labels = {}    # panel_name → [N] cluster labels
    clusterers = {}          # panel_name → PatchClusterer (fitted)

    for p, name in enumerate(panel_names):
        mask = (panel_idx == p)
        panel_patches = patches[mask]    # [N, D] — all records' patches for this panel

        print(f"\n  Panel '{name}' ({panel_patches.shape[0]:,} patches × {panel_patches.shape[1]} dims)")

        clusterer = PatchClusterer(
            n_clusters=k,
            random_seed=ccfg.get("random_seed", 42),
            use_spatial_features=False,
            use_pca=True,
            pca_dims=min(50, panel_patches.shape[1] - 1),
            method="gmm",
            gmm_max_fit_samples=ccfg.get("gmm_max_fit_samples", 50000),
        )

        labels = clusterer.fit(panel_patches)
        sizes = np.bincount(labels, minlength=k).tolist()
        print(f"    Cluster sizes: {sizes}")

        per_panel_labels[name] = labels
        clusterers[name] = clusterer

    # Save per-panel cluster labels and fitted clusterers
    torch.save(
        {
            "per_panel_labels": per_panel_labels,    # {panel_name: [N] int array}
            "panel_names":      panel_names,
            "n_clusters":       k,
        },
        cache_dir / "panel_clusters.pt",
    )
    with open(cache_dir / "panel_clusterers.pkl", "wb") as f:
        pickle.dump(clusterers, f)

    _clear_files(
        [
            cfg["concepts"]["labels_path"],
            cfg["concepts"].get("vectors_cache", str(cache_dir / "concept_vectors.pt")),
            cfg["concepts"].get("scores_cache",  str(cache_dir / "concept_scores.pt")),
            cfg["classification"]["classifier_path"],
        ],
        reason="re-cluster invalidates labels, concept vectors, scores, classifier",
    )
    print("\n  ⚠  Label clusters in the GUI before running --stage concepts:")
    print("       streamlit run labeling/lab_cluster_labeler.py")

    return clusterers, per_panel_labels


# ---------------------------------------------------------------------------
# Stage: concepts
# ---------------------------------------------------------------------------

def stage_concepts(cfg):
    """
    Build concept vectors and activation scores from per-panel labeled clusters.

    Concept activation score per record = concatenation of per-panel GMM soft
    memberships → [n_panels * n_clusters_per_panel] dimensional vector.

    This directly mirrors Phase 1's concept scoring:
      Phase 1: record score = fraction of image patches in each concept cluster
      Phase 2: record score = per-panel soft memberships concatenated

    Human labels use key format "panel:cluster_id" e.g. "cbc:3", "biochem:7".
    """
    print("\n" + "=" * 60)
    print("STAGE L6: Build Concept Vectors from Labeled Per-Panel Clusters")
    print("=" * 60)

    import pickle
    from sklearn.preprocessing import normalize as sk_normalize

    cache_dir = Path(cfg["lab_data"]["cache_dir"])
    labels_path = cfg["concepts"]["labels_path"]

    if not Path(labels_path).exists():
        print(f"ERROR: {labels_path} not found.")
        print("Label clusters first:  streamlit run labeling/lab_cluster_labeler.py")
        sys.exit(1)

    human_labels = json.load(open(labels_path))   # {"cbc:3": {"label": "Iron def.", ...}, ...}

    patches_data = torch.load(cfg["transformer"]["panel_patches_path"], weights_only=False)
    panel_names  = patches_data["panel_names"]
    n_records    = patches_data["n_records"]
    patches_flat = patches_data["patches"]         # [N*P, D]
    panel_idx_all = patches_data["panel_idx"]      # [N*P]

    cluster_data = torch.load(cache_dir / "panel_clusters.pt", weights_only=False)
    per_panel_labels = cluster_data["per_panel_labels"]   # {panel: [N] int}
    k = cluster_data["n_clusters"]

    with open(cache_dir / "panel_clusterers.pkl", "rb") as f:
        clusterers = pickle.load(f)

    # ── Build per-panel GMM soft membership scores ──────────────────────────
    # For each panel: [N, k] soft probabilities from the fitted GMM.
    # Requires re-applying the same PCA + normalize preprocessing that PatchClusterer used.

    concept_names = []       # flat list: "cbc:Iron Def.", "biochem:CKD", ...
    panel_score_blocks = []  # list of [N, k] arrays, one per panel

    for p, pname in enumerate(panel_names):
        mask = (panel_idx_all == p)
        panel_patches = patches_flat[mask]           # [N, D]
        clusterer = clusterers[pname]

        # Apply the same preprocessing pipeline as during fitting
        X = sk_normalize(panel_patches.numpy(), norm="l2")
        if clusterer.pca is not None:
            X = sk_normalize(clusterer.pca.transform(X), norm="l2")

        # GMM soft probabilities [N, k]
        proba = clusterer.gmm.predict_proba(X).astype(np.float32)
        panel_score_blocks.append(proba)

        # Build concept names for this panel's labeled clusters
        for cid in range(k):
            key = f"{pname}:{cid}"
            meta = human_labels.get(key, {})
            label = meta.get("label", "").strip()
            included = meta.get("include", True)
            if label and included:
                concept_names.append(f"{pname}: {label}")
            else:
                concept_names.append(f"{pname}: cluster_{cid}")

    # Concatenate per-panel scores → [N, n_panels * k] concept vector per record
    all_scores = np.concatenate(panel_score_blocks, axis=1)   # [N, P*k]

    print(f"\n  Concept vector shape: {all_scores.shape}  ({n_records} records × {all_scores.shape[1]} concept dims)")
    print(f"  = {len(panel_names)} panels × {k} clusters/panel")

    # Build concept vectors (cluster centroids in original feature space)
    # These are the per-panel patch centroid vectors — used for LLM labeling
    features_data = torch.load(cache_dir / "features.pt", weights_only=False)
    features = features_data["features"]   # [N, 34] flat deviation (human-interpretable)

    concept_vectors = {}
    for p, pname in enumerate(panel_names):
        panel_cluster_labels = per_panel_labels[pname]    # [N] int
        for cid in range(k):
            mask = (panel_cluster_labels == cid)
            if mask.sum() == 0:
                continue
            centroid = features[mask].mean(dim=0)   # [34] — human-readable deviation profile
            key = f"{pname}: cluster_{cid}"
            human_label = human_labels.get(f"{pname}:{cid}", {}).get("label", "")
            if human_label:
                key = f"{pname}: {human_label}"
            concept_vectors[key] = centroid
            print(f"  {key:40s}  n={mask.sum():,}")

    vectors_cache = cfg["concepts"].get("vectors_cache", str(cache_dir / "concept_vectors.pt"))
    torch.save({"vectors": concept_vectors, "concept_names": concept_names}, vectors_cache)
    print(f"\n  ✓  {len(concept_vectors)} concept vectors → {vectors_cache}")

    scores_cache = cfg["concepts"].get("scores_cache", str(cache_dir / "concept_scores.pt"))
    torch.save(
        {
            "scores":        torch.tensor(all_scores, dtype=torch.float32),
            "concept_names": concept_names,
            "panel_names":   panel_names,
            "n_clusters":    k,
        },
        scores_cache,
    )
    print(f"  ✓  Concept scores [{all_scores.shape[0]} × {all_scores.shape[1]}] → {scores_cache}")
    return concept_vectors, all_scores, concept_names


# ---------------------------------------------------------------------------
# Stage: classify
# ---------------------------------------------------------------------------

def stage_classify(cfg):
    """
    ┌─────────────────────────────────────────────────────────────────────┐
    │  DISSERTATION CORE CLAIM PROOF                                      │
    │                                                                     │
    │  Input : concept scores [N, 30]  — learned without any labels      │
    │  Output: predict diabetes / hypertension / normal                   │
    │                                                                     │
    │  The classifier never saw DIQ010 or BPQ020 during any prior stage. │
    │  If it predicts accurately, unsupervised concepts are clinically    │
    │  meaningful — directly differentiating us from TCAV and CBM.       │
    └─────────────────────────────────────────────────────────────────────┘

    Label construction (from NHANES questionnaire data, zero training overlap):
      diabetes     — DIQ010 == 1  (physician confirmed)
      hypertension — BPQ020 == 1  (physician confirmed), no diabetes
      normal       — neither condition flagged

    Mirrors image classify stage exactly:
      Image: concept scores [N, 30] → LogReg → cat / car / bird
      Lab:   concept scores [N, 30] → LogReg → diabetes / hypertension / normal
    """
    print("\n" + "=" * 60)
    print("STAGE L7: Concept Classifier — predict diabetes/hypertension from concept scores")
    print()
    print("  Core claim: concepts discovered without supervision predict")
    print("  physician-confirmed diagnoses they were never trained on.")
    print("=" * 60)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    import pandas as pd

    cache_dir    = Path(cfg["lab_data"]["cache_dir"])
    scores_cache = cfg["concepts"].get("scores_cache", str(cache_dir / "concept_scores.pt"))

    if not Path(scores_cache).exists():
        print("ERROR: concept scores not found. Run --stage concepts first.")
        sys.exit(1)

    records_with_demo = cache_dir / "records_with_demo.csv"
    if not records_with_demo.exists():
        print("ERROR: records_with_demo.csv not found. Run --stage load first.")
        sys.exit(1)

    # ── Load concept scores ────────────────────────────────────────────────
    scores_data   = torch.load(scores_cache, weights_only=False)
    scores        = scores_data["scores"].numpy()    # [N, 30]
    concept_names = scores_data["concept_names"]

    # ── Load clinical labels (never used in any prior stage) ──────────────
    full_df  = pd.read_csv(records_with_demo)
    diq      = full_df.get("DIQ010")   # 1=Yes, 2=No, 3=Borderline, 7/9=unknown
    bpq      = full_df.get("BPQ020")   # 1=Yes, 2=No

    has_diq = diq is not None and diq.notna().any()
    has_bpq = bpq is not None and bpq.notna().any()

    if not has_diq:
        print("  ⚠  DIQ010 (diabetes) not found in records_with_demo.csv.")
        print("     Re-run --stage load to regenerate with validation columns.")
        sys.exit(1)

    # ── Build 3-class labels ──────────────────────────────────────────────
    #   diabetes     (class 0) — DIQ010 == 1
    #   hypertension (class 1) — BPQ020 == 1, no diabetes
    #   normal       (class 2) — neither
    #
    # Records with unknown/refused answers are excluded from classifier
    # training but clustering was still performed on them — no leakage.
    labels     = np.full(len(full_df), -1, dtype=int)  # -1 = unknown, excluded
    class_names = ["diabetes", "hypertension", "normal"]

    diabetic_mask = (diq == 1)
    hyper_mask    = (bpq == 1) & ~diabetic_mask if has_bpq else np.zeros(len(full_df), bool)
    normal_mask   = (diq == 2) & ((bpq == 2) if has_bpq else True)

    labels[diabetic_mask] = 0
    labels[hyper_mask]    = 1
    labels[normal_mask]   = 2

    # Keep only records with a known label
    known_mask = labels >= 0
    X = scores[known_mask]
    y = labels[known_mask]

    class_counts = {c: (y == i).sum() for i, c in enumerate(class_names)}
    print(f"\n  Records with known labels : {known_mask.sum():,} / {len(full_df):,}")
    for cls, n in class_counts.items():
        base_rate = n / known_mask.sum()
        print(f"    {cls:14s}: {n:,}  (base rate {base_rate:.1%})")

    # Balance: cap majority class at 3× minority to avoid trivial accuracy
    min_n   = min(class_counts.values())
    cap     = min(min_n * 3, max(class_counts.values()))
    indices = []
    rng     = np.random.default_rng(42)
    for cls_idx in range(len(class_names)):
        idx = np.where(y == cls_idx)[0]
        idx = rng.choice(idx, size=min(len(idx), cap), replace=False)
        indices.append(idx)
    indices = np.concatenate(indices)
    X_bal, y_bal = X[indices], y[indices]
    print(f"\n  After balancing: {len(X_bal):,} records (cap={cap:,} per class)")

    # ── Train / test split ────────────────────────────────────────────────
    test_size = cfg["classification"].get("test_size", 0.2)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_bal, y_bal,
        test_size=test_size,
        stratify=y_bal,
        random_state=cfg["classification"].get("random_seed", 42),
    )

    # ── Logistic Regression on concept scores — same as image pipeline ────
    clf = LogisticRegression(
        C=cfg["classification"].get("C", 1.0),
        max_iter=cfg["classification"].get("max_iter", 1000),
        solver="lbfgs",
        random_state=cfg["classification"].get("random_seed", 42),
    )
    clf.fit(X_tr, y_tr)

    train_acc = accuracy_score(y_tr, clf.predict(X_tr))
    test_acc  = accuracy_score(y_te, clf.predict(X_te))

    print(f"\n  Train accuracy : {train_acc:.1%}")
    print(f"  Test  accuracy : {test_acc:.1%}  (chance = {1/len(class_names):.1%})")
    print()
    print(classification_report(y_te, clf.predict(X_te), target_names=class_names))

    # ── Save — same dict format as image pipeline ─────────────────────────
    report_dict = classification_report(
        y_te, clf.predict(X_te), target_names=class_names, output_dict=True
    )
    payload = {
        "clf":              clf,
        "class_names":      class_names,
        "concept_names":    concept_names,
        "test_accuracy":    test_acc,
        "train_accuracy":   train_acc,
        "classification_report": report_dict,
        # Metadata about what these labels represent — for dissertation write-up
        "label_source":     "NHANES questionnaire (DIQ010, BPQ020) — never used in training",
        "core_claim":       "Unsupervised concept scores predict physician-confirmed diagnoses",
    }

    classifier_path = cfg["classification"]["classifier_path"]
    Path(classifier_path).parent.mkdir(parents=True, exist_ok=True)
    with open(classifier_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"  ✓  Classifier saved → {classifier_path}")
    print(f"\n  ── Core claim result ──────────────────────────────────────────")
    print(f"  Concept scores learned without any diagnosis labels predict")
    print(f"  {class_names} at {test_acc:.1%} accuracy")
    print(f"  (baseline chance = {1/len(class_names):.1%})")
    return clf


# ---------------------------------------------------------------------------
# Stage: validate — per-cluster enrichment analysis
# ---------------------------------------------------------------------------

def stage_validate(cfg):
    """
    ┌─────────────────────────────────────────────────────────────────────┐
    │  CLUSTER ENRICHMENT ANALYSIS — quantified core claim proof          │
    │                                                                     │
    │  For each discovered concept cluster, compute:                      │
    │    - % of cluster members with diabetes vs population base rate     │
    │    - % of cluster members with hypertension vs population base rate │
    │    - mean HbA1c vs population mean                                  │
    │    - enrichment ratio = cluster rate / base rate                    │
    │                                                                     │
    │  This is the image-pipeline equivalent of "patch purity" —         │
    │  instead of "96% of patches in cluster 4 are bird patches",        │
    │  we show "cluster 4 has 5.6× the diabetes rate of the population"  │
    │                                                                     │
    │  Answers the examiner question: "How do you know these clusters     │
    │  are clinically meaningful and not just noise?"                     │
    └─────────────────────────────────────────────────────────────────────┘
    """
    print("\n" + "=" * 60)
    print("STAGE L8: Cluster Enrichment Analysis — core claim evidence")
    print()
    print("  For each concept cluster: enrichment vs clinical ground truth")
    print("  (diagnoses never used in any prior stage)")
    print("=" * 60)

    import pandas as pd

    cache_dir = Path(cfg["lab_data"]["cache_dir"])

    # ── Load cluster assignments ──────────────────────────────────────────
    cluster_data     = torch.load(cache_dir / "panel_clusters.pt", weights_only=False)
    per_panel_labels = cluster_data["per_panel_labels"]   # {panel: np.array [N]}
    panel_names      = cluster_data["panel_names"]
    k                = cluster_data["n_clusters"]

    # ── Load human labels ─────────────────────────────────────────────────
    labels_path  = cfg["concepts"]["labels_path"]
    human_labels = json.load(open(labels_path)) if Path(labels_path).exists() else {}

    # ── Load validation columns ───────────────────────────────────────────
    full_df = pd.read_csv(cache_dir / "records_with_demo.csv")

    diq   = full_df.get("DIQ010")    # 1=diabetes
    bpq   = full_df.get("BPQ020")    # 1=hypertension
    hba1c = full_df.get("LBXGH")     # HbA1c %

    has_diq   = diq   is not None and diq.notna().any()
    has_bpq   = bpq   is not None and bpq.notna().any()
    has_hba1c = hba1c is not None and hba1c.notna().any()

    # Population base rates
    base_diabetes = (diq   == 1).mean() if has_diq   else None
    base_hyper    = (bpq   == 1).mean() if has_bpq   else None
    base_hba1c    = hba1c.mean()        if has_hba1c else None

    if has_diq:   print(f"\n  Population base rates:")
    if has_diq:   print(f"    Diabetes     : {base_diabetes:.1%}  ({(diq==1).sum():,} records)")
    if has_bpq:   print(f"    Hypertension : {base_hyper:.1%}  ({(bpq==1).sum():,} records)")
    if has_hba1c: print(f"    Mean HbA1c   : {base_hba1c:.2f}%")

    # ── Per-cluster enrichment ────────────────────────────────────────────
    results = []  # for saving to CSV

    for pname in panel_names:
        cluster_labels = per_panel_labels[pname]   # [N] int
        print(f"\n  {'─'*60}")
        print(f"  Panel: {pname.upper()}")
        print(f"  {'─'*60}")

        for cid in range(k):
            mask = (cluster_labels == cid)
            n    = mask.sum()
            key  = f"{pname}:{cid}"
            meta = human_labels.get(key, {})
            label = meta.get("label", f"cluster_{cid}") or f"cluster_{cid}"
            concept_key = f"{pname}: {label}"

            row = {"panel": pname, "cluster_id": cid, "label": label, "n": n}

            enrichments = []
            if has_diq:
                diab_rate   = (diq[mask]  == 1).mean()
                diab_enrich = diab_rate / base_diabetes if base_diabetes > 0 else 0
                row["diabetes_rate"]        = diab_rate
                row["diabetes_enrichment"]  = diab_enrich
                enrichments.append(f"diabetes={diab_rate:.0%} ({diab_enrich:.1f}×)")

            if has_bpq:
                hyper_rate   = (bpq[mask]  == 1).mean()
                hyper_enrich = hyper_rate / base_hyper if base_hyper > 0 else 0
                row["hypertension_rate"]        = hyper_rate
                row["hypertension_enrichment"]  = hyper_enrich
                enrichments.append(f"hypertension={hyper_rate:.0%} ({hyper_enrich:.1f}×)")

            if has_hba1c:
                mean_hba1c = hba1c[mask].mean()
                row["mean_hba1c"] = mean_hba1c
                enrichments.append(f"HbA1c={mean_hba1c:.2f}%")

            enrich_str = "  |  ".join(enrichments)
            flag = "  ◀ HIGH" if any(
                row.get(k, 0) >= 2.0
                for k in ["diabetes_enrichment", "hypertension_enrichment"]
            ) else ""
            print(f"  [{cid:2d}] {concept_key:40s}  n={n:,}  {enrich_str}{flag}")
            results.append(row)

    # ── Save enrichment table ─────────────────────────────────────────────
    enrich_df  = pd.DataFrame(results)
    enrich_path = cache_dir / "cluster_enrichment.csv"
    enrich_df.to_csv(enrich_path, index=False)
    print(f"\n  ✓  Enrichment table saved → {enrich_path}")
    print(f"     (clusters with ≥2× enrichment flagged with ◀ HIGH)")

    # ── Top-line summary for dissertation ────────────────────────────────
    if "diabetes_enrichment" in enrich_df.columns:
        top = enrich_df.sort_values("diabetes_enrichment", ascending=False).iloc[0]
        print(f"\n  ── Top result for dissertation ─────────────────────────────")
        print(f"  Best cluster for diabetes:")
        print(f"    Panel      : {top['panel']}")
        print(f"    Label      : {top['label']}")
        print(f"    Enrichment : {top['diabetes_enrichment']:.1f}× population base rate")
        if "mean_hba1c" in top:
            print(f"    Mean HbA1c : {top['mean_hba1c']:.2f}%  (diagnostic threshold = 6.5%)")
        print(f"\n  Write-up sentence:")
        print(f"  \"Concept '{top['label']}' (discovered without any diagnosis labels)")
        print(f"   has {top['diabetes_enrichment']:.1f}× the diabetes prevalence of the general")
        print(f"   population, validating its clinical interpretability.\"")

    return enrich_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Lab Report Concept Discovery Pipeline")
    parser.add_argument(
        "--stage",
        default="all",
        choices=[
            "all", "download", "load", "extract",
            "pretrain", "encode", "cluster",
            "concepts", "classify", "validate",
        ],
    )
    parser.add_argument("--config", default=CONFIG)
    args = parser.parse_args()

    if args.stage == "download":
        stage_download(config_path=args.config)
        return

    cfg = yaml.safe_load(open(args.config))

    # Ensure cache dir exists
    Path(cfg["lab_data"]["cache_dir"]).mkdir(parents=True, exist_ok=True)

    if args.stage in ("all", "load"):
        stage_load(cfg, config_path=args.config)
    if args.stage in ("all", "extract"):
        stage_extract(cfg)
    if args.stage in ("all", "pretrain"):
        stage_pretrain(cfg)
    if args.stage in ("all", "encode"):
        stage_encode(cfg)
    if args.stage in ("all", "cluster"):
        stage_cluster(cfg)
    if args.stage == "concepts":
        stage_concepts(cfg)
    if args.stage == "classify":
        stage_classify(cfg)
    if args.stage == "validate":
        stage_validate(cfg)

    print("\n✅ Pipeline stage complete!")


if __name__ == "__main__":
    main()
