# src/pipeline/concept_builder.py
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
    config_path: str = "configs/config.yaml",
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
    config_path: str = "configs/config.yaml",
):
    """
    Pre-compute concept activation scores for every image using
    CLUSTER ASSIGNMENT PROPORTION.

    Score = fraction of the image's patches that were assigned to each
    labeled concept cluster.

    Why this works:
      - A cat image genuinely has 10-15% patches in cat_ear cluster,
        5-8% in eye cluster, etc. — because the clusterer was trained on cats
      - A non-cat image's patches scatter across clusters differently,
        giving low proportions in ALL cat-specific clusters
      - This gives a much wider score range (0.0–0.20) vs cosine similarity
        (0.40–0.80 for everything), making the OneClassSVM boundary meaningful

    Saves scores matrix [N_images, N_concepts] to cache.
    """
    cfg = yaml.safe_load(open(config_path))
    import pickle
    data = torch.load(cfg["dino"]["features_cache"], weights_only=False)
    cluster_labels_path = cfg["dino"].get("cluster_labels_path", "cache/cluster_labels.pt")
    cluster_labels = torch.load(cluster_labels_path, weights_only=True).numpy()

    human_labels = json.load(open(cfg["concepts"]["labels_path"]))
    # Map: concept_name → cluster_id
    concept_to_cluster = {
        meta["label"]: int(cid)
        for cid, meta in human_labels.items()
        if meta.get("label", "").strip() and meta.get("include", True)
    }

    image_ids = data["image_ids"].numpy()
    image_paths = data["image_paths"]
    image_labels = data["image_labels"]
    concept_names = list(concept_to_cluster.keys())
    n_images = len(image_paths)
    n_concepts = len(concept_names)
    scores_matrix = np.zeros((n_images, n_concepts), dtype=np.float32)

    print(
        f"Computing concept scores for {n_images} images × {n_concepts} concepts "
        f"[cluster proportion scoring]..."
    )
    for img_idx in range(n_images):
        mask = image_ids == img_idx
        img_cluster_labels = cluster_labels[mask]   # cluster id per patch
        n_patches = len(img_cluster_labels)
        if n_patches == 0:
            continue
        for c_idx, c_name in enumerate(concept_names):
            cluster_id = concept_to_cluster[c_name]
            scores_matrix[img_idx, c_idx] = (
                img_cluster_labels == cluster_id
            ).sum() / n_patches

    scores_cache = cfg["concepts"]["scores_cache"]
    torch.save(
        {
            "scores": torch.tensor(scores_matrix),
            "concept_names": concept_names,
            "image_labels": image_labels,
            "class_names": data["class_names"],
        },
        scores_cache,
    )
    print(f"Saved concept scores {scores_matrix.shape} → {scores_cache}")
    print(f"Score ranges: " + ", ".join(
        f"{concept_names[i]}=[{scores_matrix[:,i].min():.3f},{scores_matrix[:,i].max():.3f}]"
        for i in range(n_concepts)
    ))
    return torch.tensor(scores_matrix), concept_names


if __name__ == "__main__":
    build_concept_vectors()
    compute_concept_scores_all()
