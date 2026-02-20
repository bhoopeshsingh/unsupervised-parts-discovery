# experiments/run_dino_pipeline.py
"""
End-to-end DINO parts discovery pipeline.
Runs all stages in sequence, logs to W&B.

Usage:
  python experiments/run_dino_pipeline.py --stage all
  python experiments/run_dino_pipeline.py --stage cluster
  python experiments/run_dino_pipeline.py --stage classify
  python experiments/run_dino_pipeline.py --stage explain --image path/to/image.jpg
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from pathlib import Path


def stage_extract(cfg):
    print("\n" + "=" * 60)
    print("STAGE 1+2: Feature Extraction + Caching")
    print("=" * 60)
    from experiments.extract_dino_features import extract_all
    return extract_all()


def stage_cluster(cfg):
    print("\n" + "=" * 60)
    print("STAGE 3: Patch Clustering")
    print("=" * 60)
    from src.parts.patch_clusterer import PatchClusterer

    data = torch.load(cfg["dino"]["features_cache"], weights_only=False)
    ccfg = cfg["clustering"]
    use_spatial = ccfg.get("use_spatial_features", False)
    use_pca = ccfg.get("use_pca", False)
    pca_dims = ccfg.get("pca_dims", 64)

    print(f"  n_clusters={ccfg['n_clusters']}, use_pca={use_pca}"
          + (f" (→{pca_dims}d)" if use_pca else "")
          + f", use_spatial={use_spatial}")

    clusterer = PatchClusterer(
        n_clusters=ccfg["n_clusters"],
        random_seed=ccfg.get("random_seed", 42),
        use_spatial_features=use_spatial,
        spatial_weight=ccfg.get("spatial_weight", 0.15),
        use_pca=use_pca,
        pca_dims=pca_dims,
    )
    labels = clusterer.fit(
        data["features"],
        patch_ids=data["patch_ids"] if use_spatial else None,
    )
    cluster_labels_path = cfg["dino"].get("cluster_labels_path", "cache/cluster_labels.pt")
    clusterer_path = cfg["dino"].get("clusterer_path", "cache/kmeans.pkl")
    torch.save(torch.tensor(labels), cluster_labels_path)
    clusterer.save(clusterer_path)
    return clusterer, labels


def stage_concepts(cfg):
    print("\n" + "=" * 60)
    print("STAGE 4+5: Build Concept Vectors + Compute Scores")
    print("=" * 60)
    from src.concepts.concept_builder import (
        build_concept_vectors,
        compute_concept_scores_all,
    )

    vectors = build_concept_vectors()
    scores, concept_names = compute_concept_scores_all()
    return vectors, scores, concept_names


def stage_classify(cfg):
    print("\n" + "=" * 60)
    print("STAGE 6: Concept Classification")
    print("=" * 60)
    from src.classification.concept_classifier import ConceptClassifier

    scores_data = torch.load(
        cfg["concepts"]["scores_cache"], weights_only=False
    )
    scores = scores_data["scores"]
    concept_names = scores_data["concept_names"]
    image_labels = scores_data["image_labels"]
    class_names = scores_data["class_names"]

    clf = ConceptClassifier(
        C=cfg["classification"]["C"],
        max_iter=cfg["classification"]["max_iter"],
        random_state=cfg["classification"].get("random_seed", 42),
    )
    acc = clf.fit(
        scores,
        image_labels,
        concept_names,
        class_names,
        test_size=cfg["classification"].get("test_size", 0.2),
    )
    classifier_path = cfg["classification"].get("classifier_path", "cache/concept_classifier.pkl")
    clf.save(classifier_path)
    return clf, acc


def stage_explain(cfg, image_path: str):
    print("\n" + "=" * 60)
    print(f"STAGE 7: Explain prediction for {image_path}")
    print("=" * 60)
    from src.models.dino_extractor import DinoExtractor
    from src.classification.concept_classifier import (
        ConceptClassifier,
        render_explanation,
    )

    extractor = DinoExtractor(
        model_name=cfg["dino"]["model"],
        device=cfg["dino"]["device"],
        image_size=cfg["dino"]["image_size"],
    )
    classifier_path = cfg["classification"].get("classifier_path", "cache/concept_classifier.pkl")
    clf = ConceptClassifier.load(classifier_path)
    saved = torch.load(
        cfg["concepts"]["vectors_cache"], weights_only=False
    )
    vectors = saved["vectors"]
    fg_threshold = cfg["dino"].get("fg_threshold", 0.5)
    img_feats = extractor.extract_foreground_patches(image_path, fg_threshold=fg_threshold)
    result = clf.predict_with_explanation(img_feats, vectors)

    print(f"\nPrediction : {result['prediction']}")
    print(f"Confidence : {result['confidence']:.2%}")
    print("\nConcept Activations:")
    for c, score in sorted(
        result["concept_scores"].items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        bar = "█" * int(score * 20)
        print(f"  {c:20s} {score:.3f} {bar}")

    explanation_dir = cfg["classification"].get("explanation_dir", "cache/")
    save_path = str(Path(explanation_dir) / f"explanation_{Path(image_path).stem}.png")
    render_explanation(result, save_path=save_path)
    return result


def log_to_wandb(cfg, acc, concept_names):
    """Log results to W&B — reuses your existing W&B setup."""
    try:
        import wandb
        wandb.init(
            project=cfg.get("wandb", {}).get(
                "project", "dino-parts-discovery"
            ),
            name="dino-concept-pipeline",
            config={
                "model": cfg["dino"]["model"],
                "n_clusters": cfg["clustering"]["n_clusters"],
                "n_concepts": len(concept_names),
                "clf_C": cfg["classification"]["C"],
            },
        )
        wandb.log({
            "test_accuracy": acc,
            "n_concepts": len(concept_names),
        })
        for img_path in Path("cache").glob("part_map_img*.png"):
            wandb.log({
                f"part_maps/{img_path.name}": wandb.Image(str(img_path)),
            })
        for img_path in Path("cache").glob("explanation_*.png"):
            wandb.log({
                f"explanations/{img_path.name}": wandb.Image(str(img_path)),
            })
        wandb.finish()
        print("W&B logging complete.")
    except Exception as e:
        print(f"W&B logging skipped: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "extract", "cluster", "concepts", "classify", "explain"],
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Image path for explain stage",
    )
    parser.add_argument(
        "--config",
        default="configs/unified_config.yaml",
    )
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

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
            print(
                "ERROR: concept scores not found. Run concepts stage first: "
                "python experiments/run_dino_pipeline.py --stage concepts"
            )
            sys.exit(1)
        clf, acc = stage_classify(cfg)
        scores_data = torch.load(
            cfg["concepts"]["scores_cache"], weights_only=False
        )
        log_to_wandb(cfg, acc, scores_data["concept_names"])
    if args.stage == "explain":
        if not args.image:
            print("ERROR: --image required for explain stage")
            sys.exit(1)
        stage_explain(cfg, args.image)

    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()
