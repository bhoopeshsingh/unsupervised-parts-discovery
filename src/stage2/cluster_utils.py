# cluster_utils.py
"""
cluster_utils.py
----------------
Utilities for building a global part dictionary from saved part embeddings.

Primary functions:
 - load_part_embeddings: loads embeddings and metadata saved by trainer
 - flatten_part_embeddings: converts (N_images, K, D) -> (N_images*K, D)
 - cluster_embeddings: clusters flattened embeddings (HDBSCAN if available, else Agglomerative/KMeans)
 - get_cluster_representatives: picks nearest-to-centroid exemplars per cluster
 - save_clusters: writes clusters.json & representative samples manifest
 - save_representative_patches (best-effort): if metadata contains image/patch references, save example patch images

Notes:
 - This module is intentionally conservative: if HDBSCAN is not installed it falls back to sklearn.
 - Input assumptions (trainer artifact format):
     * embeddings_path -> numpy .npy file with shape (N_images, K, D) or (N_parts, D)
     * meta_path -> JSON list/dict with per-row metadata. Expected formats:
         - If embeddings is (N_images, K, D), meta should be a list with length N_images, each entry contains { "image_path": "...", "label": int }
         - If embeddings is flattened (N_parts, D), meta can be a list of length N_parts with {"image_path":..., "part_index":..}
   The function is defensive and documents what it expects in saved clusters.json.

Reference (methodology inspiration):
 - /mnt/data/PDiscoFormer-Relaxing Part Discovery.pdf
"""

import os
import json
import math
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

# Try to import HDBSCAN (preferred). If not available, fall back to sklearn.
try:
    import hdbscan
    _HDBSCAN_AVAILABLE = True
except Exception:
    _HDBSCAN_AVAILABLE = False

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import pairwise_distances_argmin_min


# Path to uploaded reference (provenance)
PAPER_PATH = "/mnt/data/PDiscoFormer-Relaxing Part Discovery.pdf"


def load_part_embeddings(embeddings_path: str, meta_path: Optional[str] = None) -> Tuple[np.ndarray, Optional[Any]]:
    """
    Loads embeddings and optional meta.

    Returns:
      embeddings: np.ndarray
        - If loaded array has shape (N_images, K, D), returned as-is.
        - If (N_parts, D) returned as-is.
      meta: parsed JSON or None
    """
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    embeddings = np.load(embeddings_path, allow_pickle=True)
    meta = None
    if meta_path is not None and os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    return embeddings, meta


def flatten_part_embeddings(embeddings: np.ndarray, meta: Optional[Any] = None) -> Tuple[np.ndarray, List[Dict]]:
    """
    Converts embeddings into a 2D array of shape (N_parts, D) and builds per-row metadata.

    Cases handled:
     - embeddings shape == (N_images, K, D) -> flatten to (N_images*K, D)
       metadata per row will include {"image_index": i, "part_index": k, ...} and copy any top-level meta
     - embeddings shape == (N_parts, D) -> returns as-is and tries to align meta to rows

    Returns:
      flat_embeddings: np.ndarray (N_parts, D)
      flat_meta: list of dicts (length N_parts)
    """
    arr = np.asarray(embeddings)
    if arr.ndim == 3:
        N, K, D = arr.shape
        flat = arr.reshape((N * K, D))
        flat_meta = []
        # create row-level meta
        for i in range(N):
            row_meta = meta[i] if (meta is not None and i < len(meta)) else {}
            for k in range(K):
                entry = {"image_index": i, "part_index": k}
                # copy possible image_path, label if present
                if isinstance(row_meta, dict):
                    if "image_path" in row_meta:
                        entry["image_path"] = row_meta["image_path"]
                    if "label" in row_meta:
                        entry["image_label"] = row_meta["label"]
                flat_meta.append(entry)
        return flat, flat_meta
    elif arr.ndim == 2:
        flat = arr
        flat_meta = []
        if meta is not None:
            # if meta is a list matching rows, return it
            if isinstance(meta, list) and len(meta) == flat.shape[0]:
                flat_meta = meta
            else:
                # otherwise create empty placeholders
                for i in range(flat.shape[0]):
                    flat_meta.append({})
        else:
            for i in range(flat.shape[0]):
                flat_meta.append({})
        return flat, flat_meta
    else:
        raise ValueError(f"Unexpected embeddings array shape: {arr.shape}")


def cluster_embeddings(
    embeddings: np.ndarray,
    method: str = "hdbscan",
    min_cluster_size: int = 20,
    n_clusters: Optional[int] = None,
    random_state: int = 0
) -> Tuple[np.ndarray, Dict]:
    """
    Cluster flattened embeddings.

    Args:
      embeddings: (N_parts, D)
      method: "hdbscan" | "agglomerative" | "kmeans"
      min_cluster_size: used for HDBSCAN
      n_clusters: required for agglomerative/kmeans
      random_state: for deterministic fallback kmeans

    Returns:
      labels: np.ndarray (N_parts,) cluster labels (-1 for noise)
      info: dict with clustering metadata
    """
    N, D = embeddings.shape

    if method == "hdbscan" and _HDBSCAN_AVAILABLE:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, min_cluster_size), metric="euclidean")
        labels = clusterer.fit_predict(embeddings)
        info = {
            "method": "hdbscan",
            "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
            "probabilities": getattr(clusterer, "probabilities_", None)
        }
        return labels, info

    # Fallbacks
    if method == "hdbscan" and not _HDBSCAN_AVAILABLE:
        method = "agglomerative" if n_clusters is None else "kmeans"

    if method == "agglomerative":
        if n_clusters is None:
            # heuristic: sqrt(N/2)
            n_clusters = max(2, int(math.sqrt(N / 2)))
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clusterer.fit_predict(embeddings)
        info = {"method": "agglomerative", "n_clusters": int(n_clusters)}
        return labels, info

    if method == "kmeans":
        if n_clusters is None:
            n_clusters = max(2, int(math.sqrt(N / 2)))
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = km.fit_predict(embeddings)
        info = {"method": "kmeans", "n_clusters": int(n_clusters)}
        return labels, info

    raise ValueError(f"Unsupported clustering method: {method}")


def get_cluster_representatives(
    embeddings: np.ndarray,
    labels: np.ndarray,
    topk: int = 10
) -> Dict[int, Dict]:
    """
    For each cluster id (excluding -1 noise), compute centroid and return top-k
    nearest indices as representatives.

    Returns:
      mapping: cluster_id -> { "centroid": [...], "representatives": [idx1, idx2, ...] }
    """
    result = {}
    unique_labels = sorted(set(int(x) for x in labels))
    for lab in unique_labels:
        if lab == -1:
            continue
        idxs = np.where(labels == lab)[0]
        cluster_emb = embeddings[idxs]
        centroid = cluster_emb.mean(axis=0)
        # find nearest samples to centroid
        rep_idxs, _ = pairwise_distances_argmin_min(centroid.reshape(1, -1), cluster_emb)
        # pairwise_distances_argmin_min returns indices relative to cluster_emb
        # compute distances for full ranking
        dists = np.linalg.norm(cluster_emb - centroid.reshape(1, -1), axis=1)
        order = np.argsort(dists)[:topk]
        representatives = [int(idxs[o]) for o in order.tolist()]
        result[int(lab)] = {
            "size": int(len(idxs)),
            "centroid_norm": float(np.linalg.norm(centroid)),
            "representatives": representatives
        }
    return result


def save_clusters(
    out_dir: str,
    labels: np.ndarray,
    flat_meta: List[Dict],
    representatives: Dict[int, Dict],
    info: Dict
) -> str:
    """
    Save clusters.json + manifest. Returns path to clusters.json.

    clusters.json schema:
      {
        "clusters": {
            "<cluster_id>": {
                "size": int,
                "representatives": [row_index, ...],
                "members": [row_index, ...]
            },
            ...
        },
        "info": {...}
      }
    """
    os.makedirs(out_dir, exist_ok=True)
    clusters = {}
    unique_labels = sorted(set(int(x) for x in labels))
    for lab in unique_labels:
        members = [int(i) for i in np.where(labels == lab)[0].tolist()]
        if lab == -1:
            cluster_name = "noise"
        else:
            cluster_name = str(lab)
        clusters[cluster_name] = {
            "size": len(members),
            "members": members,
            "representatives": representatives.get(lab, {}).get("representatives", []),
        }

    out = {
        "clusters": clusters,
        "info": info,
        "meta_example": flat_meta[0] if len(flat_meta) > 0 else {}
    }

    out_path = os.path.join(out_dir, "clusters.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out_path


def save_representative_samples(
    embeddings: np.ndarray,
    flat_meta: List[Dict],
    representatives: Dict[int, Dict],
    out_dir: str,
    images_root: Optional[str] = None,
    top_k: int = 5
) -> Dict[int, List[str]]:
    """
    Best-effort: If flat_meta entries include 'image_path' and optionally patch coordinates,
    attempt to copy or crop the representative images to out_dir/cluster_<id>/.

    Returns:
      mapping cluster_id -> list of saved filepaths
    If no image_path info is available, returns mapping cluster_id -> list of "row_index" strings.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = {}

    for lab, info in representatives.items():
        reps = info.get("representatives", [])[:top_k]
        cluster_dir = os.path.join(out_dir, f"cluster_{lab}")
        os.makedirs(cluster_dir, exist_ok=True)
        saved_paths = []
        for rid in reps:
            meta = flat_meta[rid] if rid < len(flat_meta) else {}
            img_path = meta.get("image_path", None)
            # If image path exists and file exists, copy to cluster_dir (optionally crop if patch coords present)
            if img_path is not None and os.path.exists(img_path):
                # safe filename
                base = os.path.basename(img_path)
                dest = os.path.join(cluster_dir, f"{rid}__{base}")
                try:
                    # simple copy (no cropping unless coords provided)
                    from shutil import copyfile
                    copyfile(img_path, dest)
                    saved_paths.append(dest)
                except Exception:
                    saved_paths.append(f"{rid}")
            else:
                # no image path -> just store the row index
                saved_paths.append(str(rid))
        saved[lab] = saved_paths
    return saved


# ------------------------------
# High-level pipeline function
# ------------------------------
def build_clusters(
    embeddings_path: str,
    meta_path: Optional[str] = None,
    out_dir: str = "./clusters_out",
    method: str = "hdbscan",
    min_cluster_size: int = 20,
    n_clusters: Optional[int] = None,
    topk: int = 10,
    images_root: Optional[str] = None
) -> str:
    """
    High-level convenience function.

    Steps:
     1. load embeddings + meta
     2. flatten embeddings to (N_parts, D)
     3. cluster (HDBSCAN preferred)
     4. compute representatives
     5. save clusters.json and sample images (best-effort)

    Returns:
      path to clusters.json
    """
    embeddings, meta = load_part_embeddings(embeddings_path, meta_path)
    flat_emb, flat_meta = flatten_part_embeddings(embeddings, meta)

    labels, info = cluster_embeddings(
        flat_emb, method=method, min_cluster_size=min_cluster_size, n_clusters=n_clusters
    )

    reps = get_cluster_representatives(flat_emb, labels, topk=topk)

    clusters_json_path = save_clusters(out_dir, labels, flat_meta, reps, info)

    samples_dir = os.path.join(out_dir, "cluster_samples")
    saved_samples = save_representative_samples(flat_emb, flat_meta, reps, samples_dir, top_k=min(5, topk))

    # augment clusters.json with sample filepaths if available
    with open(clusters_json_path, "r") as f:
        clusters_blob = json.load(f)
    clusters_blob["sample_files"] = saved_samples
    with open(clusters_json_path, "w") as f:
        json.dump(clusters_blob, f, indent=2)

    return clusters_json_path


# ------------------------------
# CLI usage
# ------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build global clusters from part embeddings")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to part_embeddings.npy")
    parser.add_argument("--meta", type=str, default=None, help="Optional metadata JSON path")
    parser.add_argument("--out", type=str, default="./clusters_out", help="Output directory")
    parser.add_argument("--method", type=str, default="hdbscan", choices=["hdbscan", "agglomerative", "kmeans"])
    parser.add_argument("--min_cluster_size", type=int, default=20)
    parser.add_argument("--n_clusters", type=int, default=None)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    path = build_clusters(
        embeddings_path=args.embeddings,
        meta_path=args.meta,
        out_dir=args.out,
        method=args.method,
        min_cluster_size=args.min_cluster_size,
        n_clusters=args.n_clusters,
        topk=args.topk
    )
    print(f"Clusters saved to: {path}")