# experiments/run_pipeline.py
"""
End-to-end DINO parts discovery pipeline.
Runs all stages in sequence, logs to W&B.

Usage:
  python experiments/run_pipeline.py --stage all
  python experiments/run_pipeline.py --stage finetune   # fine-tune DINO (run before extract)
  python experiments/run_pipeline.py --stage extract
  python experiments/run_pipeline.py --stage cluster
  python experiments/run_pipeline.py --stage concepts
  python experiments/run_pipeline.py --stage classify
  python experiments/run_pipeline.py --stage explain --image path/to/image.jpg
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def _clear_files(paths, reason=""):
    """Delete cache files that exist; print a summary of what was removed."""
    cleared = [p for p in paths if Path(p).exists()]
    if cleared:
        tag = f" ({reason})" if reason else ""
        print(f"\n  🗑  Clearing stale cache{tag}:")
        for p in cleared:
            Path(p).unlink()
            print(f"       ✗  {p}")
    return cleared


def _clusterer_siblings(clusterer_path: str):
    """Return all auto-generated files that accompany a saved clusterer."""
    base = clusterer_path.replace(".pkl", "")
    return [
        clusterer_path,
        base + "_centers.pt",
        base + "_meta.pt",
        base + "_pca.pkl",
    ]


def stage_extract(cfg):
    print("\n" + "=" * 60)
    print("STAGE 1+2: Feature Extraction + Caching")
    print("=" * 60)

    # Wipe every downstream file — features are changing so nothing below is valid
    clusterer_path = cfg["dino"].get("clusterer_path", "cache/kmeans.pkl")
    _clear_files(
        [
            cfg["dino"]["features_cache"],
            cfg["dino"].get("patch_quality_cache", "cache/patch_quality.pt"),
            cfg["dino"].get("cluster_labels_path", "cache/cluster_labels.pt"),
            *_clusterer_siblings(clusterer_path),
            cfg["concepts"]["labels_path"],
            cfg["concepts"].get("vectors_cache", "cache/concept_vectors.pt"),
            cfg["concepts"].get("scores_cache", "cache/concept_scores.pt"),
            cfg["classification"].get("classifier_path", "cache/concept_classifier.pkl"),
        ],
        reason="re-extract invalidates all downstream cache",
    )

    from experiments.extract_features import extract_all

    # If fine-tuned weights exist, load them before extracting
    ft_path = cfg.get("finetune", {}).get("save_path", "cache/dino_finetuned.pt")
    if Path(ft_path).exists():
        print(f"Fine-tuned weights found at {ft_path} — will be loaded during extraction.")
    use_ml = cfg["dino"].get("use_multilayer", False)
    print(f"Feature mode: {'multi-layer (8,10,12) → 1152-dim' if use_ml else 'single-layer → 384-dim'}")
    result = extract_all(finetune_weights=ft_path if Path(ft_path).exists() else None)
    stage_patch_quality(cfg)
    return result


def stage_cluster(cfg):
    print("\n" + "=" * 60)
    print("STAGE 3: Patch Clustering")
    print("=" * 60)
    from src.pipeline.patch_clusterer import PatchClusterer

    data = torch.load(cfg["dino"]["features_cache"], weights_only=False)
    ccfg = cfg["clustering"]
    use_spatial = ccfg.get("use_spatial_features", False)
    use_pca = ccfg.get("use_pca", False)
    pca_dims = ccfg.get("pca_dims", 64)
    method = ccfg.get("method", "kmeans")

    print(f"  n_clusters={ccfg['n_clusters']}, method={method}, use_pca={use_pca}"
          + (f" (→{pca_dims}d)" if use_pca else "")
          + f", use_spatial={use_spatial}")

    clusterer = PatchClusterer(
        n_clusters=ccfg["n_clusters"],
        random_seed=ccfg.get("random_seed", 42),
        use_spatial_features=use_spatial,
        spatial_weight=ccfg.get("spatial_weight", 0.15),
        use_pca=use_pca,
        pca_dims=pca_dims,
        method=method,
        gmm_max_fit_samples=ccfg.get("gmm_max_fit_samples", 150_000),
    )
    labels = clusterer.fit(
        data["features"],
        patch_ids=data["patch_ids"] if use_spatial else None,
    )

    # Optional: spatial coherence smoothing — clean up isolated stray patches
    use_smoothing = ccfg.get("use_spatial_smoothing", False)
    if use_smoothing:
        print("  Applying spatial coherence smoothing...")
        labels = clusterer.smooth_labels(
            labels,
            image_ids=data["image_ids"].numpy(),
            patch_ids=data["patch_ids"].numpy(),
            n_iter=1,
        )

    cluster_labels_path = cfg["dino"].get("cluster_labels_path", "cache/cluster_labels.pt")
    clusterer_path = cfg["dino"].get("clusterer_path", "cache/kmeans.pkl")
    torch.save(torch.tensor(labels), cluster_labels_path)
    clusterer.save(clusterer_path)

    # Cluster IDs are reassigned on every run — everything built on top of them is stale.
    _clear_files(
        [
            cfg["concepts"]["labels_path"],
            cfg["concepts"].get("vectors_cache", "cache/concept_vectors.pt"),
            cfg["concepts"].get("scores_cache", "cache/concept_scores.pt"),
            cfg["classification"].get("classifier_path", "cache/concept_classifier.pkl"),
        ],
        reason="re-cluster invalidates labels, concept vectors, scores, and classifier",
    )
    print("\n  ⚠  Re-label clusters in the GUI before running --stage concepts:")
    print("       streamlit run labeling/label_tool.py")

    return clusterer, labels


def stage_patch_quality(cfg):
    """
    Pre-compute per-patch brightness, variance, and spatial centrality for every
    foreground patch in the features cache.  Saves to cache/patch_quality.pt so
    the labeling GUI can load it instantly without touching any images at startup.
    """
    print("\n" + "=" * 60)
    print("STAGE 3b: Patch Quality Pre-computation")
    print("=" * 60)

    data = torch.load(cfg["dino"]["features_cache"], weights_only=False)
    n_patches = len(data["image_ids"])
    n_images  = len(data["image_paths"])

    brightness        = np.zeros(n_patches, dtype=np.float32)
    variance_scores   = np.zeros(n_patches, dtype=np.float32)
    spatial_centrality = np.zeros(n_patches, dtype=np.float32)

    for img_idx in tqdm(range(n_images), desc="  Patch quality"):
        try:
            img_arr = np.array(
                Image.open(data["image_paths"][img_idx]).convert("RGB").resize((224, 224)),
                dtype=np.float32,
            )
            patch_indices = np.where((data["image_ids"] == img_idx).numpy())[0]
            for global_idx in patch_indices:
                patch_idx = data["patch_ids"][global_idx].item()
                row_p, col_p = divmod(int(patch_idx), 28)
                r0, c0 = row_p * 8, col_p * 8
                patch = img_arr[r0:r0 + 8, c0:c0 + 8]
                if patch.size > 0:
                    brightness[global_idx]         = patch.mean() / 255.0
                    variance_scores[global_idx]    = patch.std()
                    center_dist = np.sqrt((row_p - 14) ** 2 + (col_p - 14) ** 2)
                    spatial_centrality[global_idx] = 1.0 / (1.0 + center_dist / 10.0)
        except Exception:
            continue

    out = {
        "brightness":         torch.from_numpy(brightness),
        "variance":           torch.from_numpy(variance_scores),
        "spatial_centrality": torch.from_numpy(spatial_centrality),
        "n_patches":          n_patches,
    }
    cache_path = cfg["dino"].get("patch_quality_cache", "cache/patch_quality.pt")
    torch.save(out, cache_path)
    print(f"  Saved patch quality cache → {cache_path}  ({n_patches:,} patches)")
    return out


def stage_concepts(cfg):
    print("\n" + "=" * 60)
    print("STAGE 4+5: Build Concept Vectors + Compute Scores")
    print("=" * 60)
    from src.pipeline.concept_builder import (
        build_concept_vectors,
        compute_concept_scores_all,
    )

    vectors = build_concept_vectors()
    scores, concept_names = compute_concept_scores_all()
    return vectors, scores, concept_names


def stage_classify(cfg):
    print("\n" + "=" * 60)
    print("STAGE 6: One-Class Concept Classifier  (cat vs not-cat)")
    print("=" * 60)
    from src.pipeline.one_class_classifier import CatConceptOneClassClassifier

    scores_data = torch.load(
        cfg["concepts"]["scores_cache"], weights_only=False
    )
    scores = scores_data["scores"].numpy()
    concept_names = scores_data["concept_names"]

    print(f"Training on {len(scores)} cat images × {len(concept_names)} concepts")
    clf = CatConceptOneClassClassifier(method="ocsvm", nu=0.1)
    clf.fit(scores)

    preds, dec_scores, confs = clf.predict(scores)
    cat_rate = sum(p == "cat" for p in preds) / len(preds)
    print(f"Cat recall (train): {cat_rate:.1%}  |  avg confidence: {confs.mean():.3f}")

    classifier_path = cfg["classification"].get(
        "classifier_path", "cache/concept_classifier.pkl"
    )
    clf.save(classifier_path)
    print(f"Saved one-class classifier → {classifier_path}")
    return clf


def stage_finetune(cfg):
    """
    Fine-tune last 2 DINO transformer blocks using cluster pseudo-labels.
    Run BEFORE --stage extract so the improved weights are used for feature extraction.

    Sequence:
      1. python experiments/run_pipeline.py --stage extract    # initial features
      2. python experiments/run_pipeline.py --stage cluster    # initial clusters
      3. python experiments/run_pipeline.py --stage finetune   # improve DINO
      4. python experiments/run_pipeline.py --stage extract    # re-extract with fine-tuned DINO
      5. python experiments/run_pipeline.py --stage cluster    # re-cluster improved features
    """
    print("\n" + "=" * 60)
    print("STAGE FT: DINO Semantic Fine-tuning")
    print("=" * 60)
    from src.models.dino_extractor import DinoExtractor
    from src.models.dino_finetuner import DinoSemanticFinetuner, ImagePathDataset
    from src.pipeline.patch_clusterer import PatchClusterer
    from torch.utils.data import DataLoader

    ftcfg = cfg.get("finetune", {})
    n_epochs        = ftcfg.get("n_epochs", 3)
    lr              = float(ftcfg.get("lr", 1e-5))
    n_pairs         = ftcfg.get("n_pairs", 512)
    batch_size      = ftcfg.get("batch_size", 16)
    save_path       = ftcfg.get("save_path", "cache/dino_finetuned.pt")

    # Check pre-requisites
    features_cache = cfg["dino"]["features_cache"]
    clusterer_path = cfg["dino"].get("clusterer_path", "cache/kmeans.pkl")
    if not Path(features_cache).exists() or not Path(clusterer_path).exists():
        print("ERROR: Run --stage extract and --stage cluster first.")
        sys.exit(1)

    # Load extractor (frozen DINO)
    extractor = DinoExtractor(
        model_name=cfg["dino"]["model"],
        device=cfg["dino"]["device"],
        image_size=cfg["dino"]["image_size"],
    )

    # Re-enable gradients for fine-tuning (DinoExtractor freezes them by default)
    for p in extractor.model.parameters():
        p.requires_grad = False   # reset; finetuner will selectively unfreeze

    # Load fitted clusterer (for pseudo-labels)
    clusterer = PatchClusterer.load(clusterer_path)
    print(f"Clusterer loaded: {clusterer.n_clusters} clusters, method={clusterer.method}")

    # Image dataloader — cat images only
    data = torch.load(features_cache, weights_only=False)
    image_paths = data["image_paths"]
    dataset = ImagePathDataset(image_paths, image_size=cfg["dino"]["image_size"])
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    print(f"Dataset: {len(dataset)} images  |  batch_size={batch_size}")

    # Create finetuner — pass use_multilayer so it unfreezes the right blocks
    # and uses the same feature extraction as the extract stage
    use_multilayer = cfg["dino"].get("use_multilayer", False)
    finetuner = DinoSemanticFinetuner(
        extractor.model,
        device=extractor.device,
        lr=lr,
        use_multilayer=use_multilayer,
    )

    print(f"\nFine-tuning for {n_epochs} epoch(s)  (lr={lr}, n_pairs={n_pairs})")
    for epoch in range(1, n_epochs + 1):
        loss = finetuner.train_epoch(dataloader, clusterer, n_pairs=n_pairs)
        print(f"  Epoch {epoch}/{n_epochs}  loss={loss:.4f}")

    finetuner.save(save_path)
    print(f"\nDone. Now re-run extract + cluster to use the improved features:")
    print(f"  python experiments/run_pipeline.py --stage extract")
    print(f"  python experiments/run_pipeline.py --stage cluster")
    return save_path


def stage_explain(cfg, image_path: str):
    print("\n" + "=" * 60)
    print(f"STAGE 7: Explain prediction for {image_path}")
    print("=" * 60)
    import json, pickle
    from src.models.dino_extractor import DinoExtractor
    from src.pipeline.one_class_classifier import CatConceptOneClassClassifier
    from src.pipeline.concept_classifier import (
        compute_image_concept_scores,
        get_spatial_concept_map,
        render_dissertation_explanation,
        render_explanation,
    )
    from src.pipeline.patch_clusterer import PatchClusterer

    extractor = DinoExtractor(
        model_name=cfg["dino"]["model"],
        device=cfg["dino"]["device"],
        image_size=cfg["dino"]["image_size"],
        use_multilayer=cfg["dino"].get("use_multilayer", False),
    )

    # Load fine-tuned weights if available
    ft_path = cfg.get("finetune", {}).get("save_path", "cache/dino_finetuned.pt")
    if Path(ft_path).exists():
        from src.models.dino_finetuner import DinoSemanticFinetuner
        DinoSemanticFinetuner.load_weights_into_extractor(extractor, ft_path)

    classifier_path = cfg["classification"].get(
        "classifier_path", "cache/concept_classifier.pkl"
    )
    clf = CatConceptOneClassClassifier.load(classifier_path)
    saved = torch.load(cfg["concepts"]["vectors_cache"], weights_only=False)
    vectors = saved["vectors"]
    concept_names = list(vectors.keys())
    fg_threshold = cfg["dino"].get("fg_threshold", 0.5)

    all_feats, fg_mask = extractor.extract_all_patches_with_fg_mask(
        image_path, fg_threshold=fg_threshold
    )
    fg_feats = all_feats[fg_mask]
    fg_patch_ids = torch.where(fg_mask)[0]   # actual patch indices (0-783)

    # Load clusterer + label mapping for cluster-proportion scoring
    clusterer_path = cfg["dino"].get("clusterer_path", "cache/kmeans.pkl")
    clusterer = PatchClusterer.load(clusterer_path)
    human_labels = json.load(open(cfg["concepts"]["labels_path"]))
    concept_to_cluster = {
        meta["label"]: int(cid)
        for cid, meta in human_labels.items()
        if meta.get("label", "").strip() and meta.get("include", True)
    }

    concept_scores = compute_image_concept_scores(
        fg_feats, vectors,
        clusterer=clusterer, concept_to_cluster=concept_to_cluster,
        patch_ids=fg_patch_ids,
    )

    score_vec = np.array([[concept_scores[c] for c in concept_names]])
    preds, _, confs = clf.predict(score_vec)
    explanation = clf.explain(score_vec, concept_names)

    result = {
        "prediction": preds[0],
        "confidence": float(confs[0]),
        "concept_scores": concept_scores,
        "contributions": {
            item["concept"]: item["deviation_from_cat"] for item in explanation
        },
    }

    print(f"\nPrediction : {result['prediction']}")
    print(f"Confidence : {result['confidence']:.2%}")
    print("\nConcept Activations (sorted by strength):")
    for c, score in sorted(concept_scores.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(score * 20)
        print(f"  {c:20s} {score:.3f} {bar}")

    concept_map, _ = get_spatial_concept_map(concept_names, vectors, all_feats)

    explanation_dir = cfg["classification"].get("explanation_dir", "cache/")
    stem = Path(image_path).stem

    dis_path = str(Path(explanation_dir) / f"explain_semantic_{stem}.png")
    render_dissertation_explanation(
        image_path=image_path,
        concept_map=concept_map,
        concept_names=concept_names,
        result=result,
        fg_mask=fg_mask,
        save_path=dis_path,
    )

    bar_path = str(Path(explanation_dir) / f"explanation_{stem}.png")
    render_explanation(result, save_path=bar_path)

    return result


def log_to_wandb(cfg, acc, concept_names):
    try:
        import wandb
        wandb.init(
            project=cfg.get("wandb", {}).get("project", "dino-parts-discovery"),
            name="dino-concept-pipeline",
            config={
                "model": cfg["dino"]["model"],
                "n_clusters": cfg["clustering"]["n_clusters"],
                "n_concepts": len(concept_names),
                "clustering_method": cfg["clustering"].get("method", "kmeans"),
            },
        )
        wandb.log({"n_concepts": len(concept_names)})
        for img_path in Path("cache").glob("part_map_img*.png"):
            wandb.log({f"part_maps/{img_path.name}": wandb.Image(str(img_path))})
        for img_path in Path("cache").glob("explanation_*.png"):
            wandb.log({f"explanations/{img_path.name}": wandb.Image(str(img_path))})
        wandb.finish()
        print("W&B logging complete.")
    except Exception as e:
        print(f"W&B logging skipped: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "extract", "cluster", "concepts", "classify",
                 "explain", "finetune"],
    )
    parser.add_argument("--image", default=None, help="Image path for explain stage")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.stage == "finetune":
        stage_finetune(cfg)
        return

    if args.stage in ("all", "extract"):
        stage_extract(cfg)
    if args.stage in ("all", "cluster"):
        stage_cluster(cfg)
    if args.stage in ("all", "concepts"):
        if not Path(cfg["concepts"]["labels_path"]).exists():
            print("ERROR: labels.json not found. Run labeling tool first:")
            print("  streamlit run labeling/label_tool.py")
            sys.exit(1)
        stage_concepts(cfg)
    if args.stage in ("all", "classify"):
        if not Path(cfg["concepts"]["scores_cache"]).exists():
            print("ERROR: concept scores not found. Run concepts stage first.")
            sys.exit(1)
        clf = stage_classify(cfg)
        scores_data = torch.load(cfg["concepts"]["scores_cache"], weights_only=False)
        log_to_wandb(cfg, 0.0, scores_data["concept_names"])
    if args.stage == "explain":
        if not args.image:
            print("ERROR: --image required for explain stage")
            sys.exit(1)
        stage_explain(cfg, args.image)

    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()
