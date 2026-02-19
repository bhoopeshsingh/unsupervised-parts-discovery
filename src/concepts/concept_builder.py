# src/concepts/concept_builder.py
"""
Build concept vectors from human-labeled clusters and compute concept scores.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml


def build_concept_vectors(
    config_path: str = "configs/unified_config.yaml",
) -> dict:
    """
    For each labeled, included cluster: compute the mean DINO feature vector.
    This mean vector is the concept vector — it represents what that semantic
    part looks like in DINO feature space.
    Returns: dict of {label: tensor[384]}
    """
    cfg = yaml.safe_load(open(config_path))
    data = torch.load(cfg["dino"]["features_cache"], weights_only=False)
    cluster_labels_path = cfg["dino"].get("cluster_labels_path", "cache/cluster_labels.pt")
    labels_arr = torch.load(cluster_labels_path, weights_only=True).numpy()
    human_labels = json.load(
        open(cfg["concepts"]["labels_path"])
    )
    features = data["features"]  # [N, 384]

    concept_vectors = {}
    concept_meta = {}
    print("Building concept vectors:")
    for cluster_id_str, meta in human_labels.items():
        cluster_id = int(cluster_id_str)
        label = meta.get("label", "").strip()
        include = meta.get("include", True)
        if not label or not include:
            print(f'  Cluster {cluster_id}: "{label}" — EXCLUDED')
            continue
        mask = labels_arr == cluster_id
        if mask.sum() == 0:
            print(f"  Cluster {cluster_id}: empty — skipped")
            continue
        cluster_feats = features[mask]  # [M, 384]
        mean_vec = cluster_feats.mean(dim=0)  # [384]
        mean_vec = F.normalize(mean_vec, dim=0)
        concept_vectors[label] = mean_vec
        concept_meta[label] = {
            "cluster_id": cluster_id,
            "n_patches": int(mask.sum()),
            "confidence": meta.get("confidence", 2),
            "notes": meta.get("notes", ""),
        }
        print(f'  Cluster {cluster_id}: "{label}" — {mask.sum():,} patches')

    vectors_path = cfg["concepts"]["vectors_cache"]
    Path(vectors_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"vectors": concept_vectors, "meta": concept_meta},
        vectors_path,
    )
    print(f"Saved {len(concept_vectors)} concept vectors → {vectors_path}")
    return concept_vectors


def compute_concept_scores_all(
    config_path: str = "configs/unified_config.yaml",
):
    """
    Pre-compute concept activation scores for every image.
    Score = max cosine similarity across all 784 patches.
    Saves scores matrix [N_images, N_concepts] to cache.
    """
    cfg = yaml.safe_load(open(config_path))
    data = torch.load(cfg["dino"]["features_cache"], weights_only=False)
    saved = torch.load(
        cfg["concepts"]["vectors_cache"], weights_only=False
    )
    concept_vectors = saved["vectors"]
    image_paths = data["image_paths"]
    image_ids = data["image_ids"]
    features = data["features"]
    image_labels = data["image_labels"]
    concept_names = list(concept_vectors.keys())
    n_images = len(image_paths)
    n_concepts = len(concept_names)
    scores_matrix = torch.zeros(n_images, n_concepts)

    print(
        f"Computing concept scores for {n_images} images × {n_concepts} concepts..."
    )
    for img_idx in range(n_images):
        mask = image_ids == img_idx
        img_feats = features[mask]  # [784, 384]
        img_norm = F.normalize(img_feats, dim=-1)
        for c_idx, c_name in enumerate(concept_names):
            vec = F.normalize(
                concept_vectors[c_name].unsqueeze(0), dim=1
            ).squeeze(0)
            sims = img_norm @ vec  # [784]
            scores_matrix[img_idx, c_idx] = sims.max()

    scores_cache = cfg["concepts"]["scores_cache"]
    torch.save(
        {
            "scores": scores_matrix,
            "concept_names": concept_names,
            "image_labels": image_labels,
            "class_names": data["class_names"],
        },
        scores_cache,
    )
    print(f"Saved concept scores {scores_matrix.shape} → {scores_cache}")
    return scores_matrix, concept_names


if __name__ == "__main__":
    build_concept_vectors()
    compute_concept_scores_all()
