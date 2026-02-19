# src/classification/concept_classifier.py
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
            multi_class="auto",
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
