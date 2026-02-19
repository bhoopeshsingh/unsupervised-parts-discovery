# src/parts/patch_clusterer.py
"""
Patch clusterer using MiniBatchKMeans on DINO patch features.
Produces part maps (28x28 cluster labels per image) and visualisation helpers.
"""
import pickle
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize


class PatchClusterer:
    def __init__(self, n_clusters: int = 8, random_seed: int = 42):
        self.n_clusters = n_clusters
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            n_init=10,
            max_iter=300,
            batch_size=4096,
            random_state=random_seed,
        )
        self.labels_ = None
        self.centers_ = None

    def fit(self, features: torch.Tensor) -> np.ndarray:
        """
        Args:
            features [N, 384] — all patch features
        Returns:
            cluster labels [N]
        """
        print(f"Clustering {features.shape[0]:,} patches into {self.n_clusters} clusters...")
        X = normalize(features.numpy(), norm="l2")
        self.kmeans.fit(X)
        self.labels_ = self.kmeans.labels_
        self.centers_ = torch.tensor(
            self.kmeans.cluster_centers_, dtype=torch.float32
        )
        print(f"Done. Cluster sizes: {np.bincount(self.labels_)}")
        return self.labels_

    def predict(self, features: torch.Tensor) -> np.ndarray:
        X = normalize(features.numpy(), norm="l2")
        return self.kmeans.predict(X)

    def get_part_map(
        self, image_idx: int, image_ids: torch.Tensor
    ) -> np.ndarray:
        """Returns [28, 28] grid of cluster labels for one image."""
        mask = (image_ids == image_idx).numpy()
        patch_labels = self.labels_[mask]
        return patch_labels.reshape(28, 28)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.kmeans, f)
        centers_path = path.replace(".pkl", "_centers.pt")
        torch.save(self.centers_, centers_path)
        print(f"Clusterer saved to {path}")

    @classmethod
    def load(cls, path: str, n_clusters: int = 8):
        obj = cls(n_clusters=n_clusters)
        with open(path, "rb") as f:
            obj.kmeans = pickle.load(f)
        obj.centers_ = torch.tensor(
            obj.kmeans.cluster_centers_, dtype=torch.float32
        )
        centers_path = path.replace(".pkl", "_centers.pt")
        if Path(centers_path).exists():
            obj.centers_ = torch.load(centers_path, weights_only=True)
        return obj


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def visualise_part_map(
    image_path: str,
    part_map: np.ndarray,
    n_clusters: int = 8,
    alpha: float = 0.55,
    save_path: str = None,
):
    """
    Overlay a [28,28] part map onto the original image.
    Each cluster gets a distinct colour.
    """
    img = np.array(
        Image.open(image_path).convert("RGB").resize((224, 224))
    )
    part_map_full = np.array(
        Image.fromarray(part_map.astype(np.uint8)).resize(
            (224, 224), Image.NEAREST
        )
    )
    cmap = cm.get_cmap("tab10", n_clusters)
    overlay = cmap(part_map_full / max(n_clusters - 1, 1))[:, :, :3]
    blended = (1 - alpha) * (img / 255.0) + alpha * overlay
    blended = np.clip(blended, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(
        part_map_full, cmap="tab10", vmin=0, vmax=n_clusters - 1
    )
    axes[1].set_title("Part Map")
    axes[1].axis("off")
    axes[2].imshow(blended)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def visualise_cluster_samples(
    cluster_id: int,
    features: torch.Tensor,
    labels: np.ndarray,
    image_ids: torch.Tensor,
    patch_ids: torch.Tensor,
    image_paths: list,
    n_samples: int = 9,
    save_path: str = None,
):
    """
    Show n_samples image patches from a given cluster.
    This is the input to your human labeling step.
    """
    cluster_mask = labels == cluster_id
    idxs = np.where(cluster_mask)[0]
    if len(idxs) == 0:
        print(f"Cluster {cluster_id} is empty")
        return
    sample_idxs = np.random.choice(
        idxs, min(n_samples, len(idxs)), replace=False
    )
    cols = 3
    rows = (len(sample_idxs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()
    for ax_i, idx in enumerate(sample_idxs):
        img_idx = image_ids[idx].item()
        patch_idx = patch_ids[idx].item()
        row_p = patch_idx // 28
        col_p = patch_idx % 28
        img = (
            Image.open(image_paths[img_idx])
            .convert("RGB")
            .resize((224, 224))
        )
        img_arr = np.array(img)
        r0, c0 = row_p * 8, col_p * 8
        patch = img_arr[r0 : r0 + 8, c0 : c0 + 8]
        patch_large = np.array(
            Image.fromarray(patch).resize((64, 64), Image.NEAREST)
        )
        axes[ax_i].imshow(patch_large)
        axes[ax_i].set_title(f"img{img_idx} p{patch_idx}", fontsize=8)
        axes[ax_i].axis("off")
    for ax in axes[len(sample_idxs) :]:
        ax.axis("off")
    fig.suptitle(
        f"Cluster {cluster_id} — Sample Patches",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()
