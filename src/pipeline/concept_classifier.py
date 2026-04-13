# src/pipeline/concept_classifier.py
"""
Linear probe classifier on concept activation scores with explainability.
"""
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ConceptClassifier:
    """
    Linear probe classifier on top of concept activation scores.
    Intentionally simple — interpretability requires linearity.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        self.clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            class_weight="balanced",  # handles class imbalance (e.g. 3738 cats vs 400 birds)
        )
        self.scaler = StandardScaler()
        self.concept_names = None
        self.class_names = None

    def fit(
        self,
        scores: torch.Tensor,
        labels: list,
        concept_names: list,
        class_names: list,
        test_size: float = 0.2,
    ):
        """
        scores: [N_images, N_concepts]
        labels: [N_images] integer class ids
        """
        self.concept_names = concept_names
        self.class_names = class_names
        X = scores.numpy() if isinstance(scores, torch.Tensor) else scores
        y = np.array(labels)
        stratify = y if len(np.unique(y)) > 1 else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
        X_tr = self.scaler.fit_transform(X_tr)
        X_te = self.scaler.transform(X_te)
        self.clf.fit(X_tr, y_tr)
        y_pred = self.clf.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        print(f"\nClassification Accuracy: {acc:.3f}")
        print(
            classification_report(
                y_te,
                y_pred,
                target_names=[
                    class_names[i] for i in sorted(np.unique(y_te))
                ],
            )
        )
        self._print_concept_weights()
        return acc

    def _print_concept_weights(self):
        print("\nConcept weights per class:")
        for i, cname in enumerate(self.class_names):
            if i >= len(self.clf.coef_):
                continue
            weights = (
                self.clf.coef_[i]
                if len(self.clf.coef_) > 1
                else self.clf.coef_[0]
            )
            sorted_idx = np.argsort(np.abs(weights))[::-1]
            print(f'  Class "{cname}":')
            for j in sorted_idx[:5]:
                print(
                    f"    {self.concept_names[j]:20s} weight={weights[j]:+.3f}"
                )

    def predict_with_explanation(
        self,
        image_features: torch.Tensor,
        concept_vectors: dict,
    ) -> dict:
        """
        Given DINO features for ONE image [784, 384],
        returns prediction + full concept breakdown.
        """
        scores = {}
        img_norm = F.normalize(image_features, dim=-1)  # [784, 384]
        for c_name, vec in concept_vectors.items():
            v_norm = F.normalize(vec.unsqueeze(0), dim=1).squeeze(0)
            sims = img_norm @ v_norm  # [784]
            scores[c_name] = sims.max().item()

        score_vec = np.array(
            [[scores[c] for c in self.concept_names]]
        )
        score_vec_scaled = self.scaler.transform(score_vec)
        pred_class_id = self.clf.predict(score_vec_scaled)[0]
        pred_proba = self.clf.predict_proba(score_vec_scaled)[0]
        pred_class_name = self.class_names[pred_class_id]
        confidence = pred_proba.max()

        coef_idx = (
            pred_class_id
            if len(self.clf.coef_) > 1
            else 0
        )
        weights = self.clf.coef_[coef_idx]
        contributions = {
            c: scores[c] * weights[i]
            for i, c in enumerate(self.concept_names)
        }
        return {
            "prediction": pred_class_name,
            "confidence": float(confidence),
            "concept_scores": scores,
            "contributions": contributions,
            "class_proba": {
                self.class_names[i]: float(p)
                for i, p in enumerate(pred_proba)
            },
        }

    def save(self, path: str):
        payload = {
            "clf": self.clf,
            "scaler": self.scaler,
            "concept_names": self.concept_names,
            "class_names": self.class_names,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"Classifier saved → {path}")

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = cls()
        obj.clf = payload["clf"]
        obj.scaler = payload["scaler"]
        obj.concept_names = payload["concept_names"]
        obj.class_names = payload["class_names"]
        return obj


def compute_image_concept_scores(
    image_features: torch.Tensor,
    concept_vectors: dict,
    clusterer=None,
    concept_to_cluster: dict = None,
    patch_ids: torch.Tensor = None,
) -> dict:
    """
    Compute per-concept activation score for a single image.

    If clusterer is provided: uses CLUSTER ASSIGNMENT PROPORTION —
      fraction of patches assigned to each concept's cluster.
      This matches the training-time scoring and is discriminating:
      cats have meaningful proportions in specific clusters; non-cats don't.

    Fallback (no clusterer): top-10 mean cosine similarity to concept vector.
    """
    if clusterer is not None and concept_to_cluster is not None:
        patch_labels = clusterer.predict(image_features, patch_ids=patch_ids)
        n_patches = len(patch_labels)
        scores = {}
        for c_name, cluster_id in concept_to_cluster.items():
            scores[c_name] = float((patch_labels == cluster_id).sum()) / n_patches
        return scores

    # Fallback: top-10 mean cosine similarity
    img_norm = F.normalize(image_features, dim=-1)
    scores = {}
    for c_name, vec in concept_vectors.items():
        v_norm = F.normalize(vec.unsqueeze(0), dim=1).squeeze(0)
        sims = img_norm @ v_norm
        k = min(10, sims.shape[0])
        scores[c_name] = sims.topk(k).values.mean().item()
    return scores


def get_spatial_concept_map(
    concept_names: list,
    concept_vectors: dict,
    image_features: torch.Tensor,
) -> tuple:
    """
    Assign each of the 784 image patches to its nearest concept by cosine similarity.

    Args:
        concept_names:   ordered list of concept name strings
        concept_vectors: dict {name: tensor [384]}
        image_features:  [784, 384] — ALL patch features for the image

    Returns:
        concept_map [28, 28] int  — concept index per patch
        patch_sims  [784, N]  float — similarity of each patch to each concept
    """
    img_norm = F.normalize(image_features, dim=-1)  # [784, 384]
    vecs = torch.stack([
        F.normalize(concept_vectors[n].float(), dim=0) for n in concept_names
    ])  # [N, 384]
    patch_sims = (img_norm @ vecs.T).numpy()        # [784, N]
    concept_idx = patch_sims.argmax(axis=1)         # [784]
    return concept_idx.reshape(28, 28), patch_sims


def render_dissertation_explanation(
    image_path: str,
    concept_map: np.ndarray,
    concept_names: list,
    result: dict,
    fg_mask: np.ndarray = None,
    save_path: str = None,
):
    """
    Dissertation-quality 3-panel figure:
      [Original image] | [Semantic part overlay] | [Concept activation bar chart]

    Each patch in the overlay is coloured by its dominant concept (e.g. cat_ears,
    cat_eyes, cat_fur).  Background patches (low attention) are dimmed.
    """
    from PIL import Image as PILImage
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec

    # Assign grey to concepts with zero/negligible activation; give active
    # concepts distinct colours from a 20-colour qualitative palette so no
    # two active concepts share the same hue.
    INACTIVE_GREY = (0.72, 0.72, 0.72)          # light grey
    ACTIVE_PALETTE = [                           # 20 perceptually distinct colours
        "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
        "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
        "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
        "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#000000",
    ]

    scores = result["concept_scores"]
    ACTIVE_THRESHOLD = 0.05                      # scores below this → grey

    active_concepts = [n for n in concept_names if scores.get(n, 0) > ACTIVE_THRESHOLD]
    colour_iter = iter(ACTIVE_PALETTE)
    concept_colors: dict = {}
    for name in concept_names:
        if name in active_concepts:
            hex_col = next(colour_iter, "#555555")   # fallback if >20 active
            r, g, b = int(hex_col[1:3], 16), int(hex_col[3:5], 16), int(hex_col[5:7], 16)
            concept_colors[name] = (r / 255, g / 255, b / 255)
        else:
            concept_colors[name] = INACTIVE_GREY

    # ---- load & resize image --------------------------------------------------
    img = np.array(PILImage.open(image_path).convert("RGB").resize((224, 224)))

    # ---- build semantic colour overlay ----------------------------------------
    # concept_map: [28, 28] int → upsample to [224, 224]
    map_upscaled = np.array(
        PILImage.fromarray(concept_map.astype(np.uint8)).resize(
            (224, 224), PILImage.NEAREST
        )
    )
    # colour grid: each pixel gets concept colour
    overlay = np.zeros((224, 224, 3), dtype=np.float32)
    for idx, name in enumerate(concept_names):
        mask = map_upscaled == idx
        overlay[mask] = concept_colors[name]

    # dim background patches
    alpha = np.ones((224, 224), dtype=np.float32) * 0.6
    if fg_mask is not None:
        fg_28 = fg_mask.reshape(28, 28).numpy().astype(np.uint8)
        fg_224 = np.array(
            PILImage.fromarray(fg_28 * 255).resize((224, 224), PILImage.NEAREST)
        ) / 255.0
        alpha = 0.80 * fg_224   # background=0 (no overlay), foreground=0.80

    blended = (img / 255.0) * (1 - alpha[:, :, None]) + overlay * alpha[:, :, None]
    blended = np.clip(blended, 0, 1)

    # ---- bar chart data -------------------------------------------------------
    contribs = result["contributions"]
    sorted_names = sorted(concept_names, key=lambda c: abs(contribs[c]), reverse=True)
    bar_values = [contribs[n] for n in sorted_names]
    bar_scores = [scores[n] for n in sorted_names]
    bar_colors = [
        (*concept_colors[n], 0.85) for n in sorted_names
    ]

    # ---- layout ---------------------------------------------------------------
    # Single-row GridSpec: 3 panels side by side.
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(
        1, 3,
        width_ratios=[1, 1, 1.4],
        wspace=0.35,
    )

    # Panel 1 — original
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img)
    ax0.set_title("Input Image", fontsize=12, fontweight="bold")
    ax0.axis("off")

    # Panel 2 — semantic overlay
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(blended)
    ax1.set_title("Semantic Part Map", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # Panel 3 — bar chart
    ax2 = fig.add_subplot(gs[0, 2])
    y_pos = np.arange(len(sorted_names))
    bars = ax2.barh(y_pos, bar_values, color=bar_colors, edgecolor="white", height=0.65)
    for i, (bar, score) in enumerate(zip(bars, bar_scores)):
        x_text = bar.get_width() + (max(bar_values) - min(bar_values)) * 0.02
        ax2.text(
            x_text, bar.get_y() + bar.get_height() / 2,
            f"act={score:.2f}",
            va="center", ha="left", fontsize=8, color="#444",
        )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(
        [n.replace("_", " ") for n in sorted_names], fontsize=9
    )
    ax2.set_xlabel("Concept contribution\n(activation × weight)", fontsize=9)
    ax2.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax2.spines[["top", "right"]].set_visible(False)
    pred = result["prediction"].upper()
    conf = result["confidence"]
    ax2.set_title(
        f"Prediction: {pred}  ({conf:.0%})",
        fontsize=12,
        fontweight="bold",
        color="#1a237e" if pred == "CAT" else "#b71c1c",
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Dissertation figure saved → {save_path}")
    plt.close(fig)
    return fig


def render_explanation(result: dict, save_path: str = None):
    """
    Render a horizontal bar chart showing concept contributions.
    This is the explainability visualisation for your dissertation.
    """
    contribs = result["contributions"]
    concepts = list(contribs.keys())
    values = [contribs[c] for c in concepts]
    scores = [result["concept_scores"][c] for c in concepts]
    sorted_idx = sorted(
        range(len(values)),
        key=lambda i: abs(values[i]),
        reverse=True,
    )
    concepts = [concepts[i] for i in sorted_idx]
    values = [values[i] for i in sorted_idx]
    scores = [scores[i] for i in sorted_idx]
    colors = ["#2196F3" if v >= 0 else "#F44336" for v in values]

    fig, ax = plt.subplots(
        figsize=(9, max(3, len(concepts) * 0.6 + 1))
    )
    bars = ax.barh(
        concepts,
        values,
        color=colors,
        edgecolor="white",
        height=0.6,
    )
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"act={score:.2f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#555",
        )
    ax.set_xlabel(
        "Concept Contribution (activation × weight)",
        fontsize=10,
    )
    ax.set_title(
        f"Prediction: {result['prediction'].upper()}, "
        f"confidence={result['confidence']:.0%}",
        fontsize=12,
        fontweight="bold",
    )
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Explanation saved → {save_path}")
    plt.show()
    return fig
