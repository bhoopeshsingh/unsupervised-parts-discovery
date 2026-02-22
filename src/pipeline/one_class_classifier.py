import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
import pickle, torch


class CatConceptOneClassClassifier:
    """
    Trained ONLY on cat concept vectors.
    Returns: cat_score ∈ [-1, 1], threshold at 0.
    Negative = not cat (with how negative = how confident).
    """

    def __init__(self, method='ocsvm', nu=0.1):
        self.method = method
        self.scaler = StandardScaler()

        if method == 'ocsvm':
            # nu = expected fraction of outliers
            # 0.1 means "assume 10% of training data might be noisy"
            self.clf = OneClassSVM(kernel='rbf', nu=nu, gamma='scale')
        elif method == 'elliptic':
            # Fits a Gaussian to cat concept space
            # More interpretable: Mahalanobis distance from cat centroid
            self.clf = EllipticEnvelope(contamination=nu)

    def fit(self, cat_concept_scores):
        """
        cat_concept_scores: (N_cat, n_concepts) — only cat images
        """
        X = self.scaler.fit_transform(cat_concept_scores)
        self.clf.fit(X)

        # Store cat centroid for explainability
        self.cat_centroid = X.mean(axis=0)
        self.concept_std = X.std(axis=0)
        print(f"Fitted on {len(X)} cat images, {X.shape[1]} concepts")
        return self

    def predict(self, concept_scores, return_scores=True):
        """
        Returns predictions and confidence scores.
        score > 0  → cat    (higher = more confident)
        score < 0  → not cat (lower = more confident not-cat)
        """
        X = self.scaler.transform(concept_scores)

        if self.method == 'ocsvm':
            # decision_function gives signed distance from boundary
            raw_scores = self.clf.decision_function(X)
        else:
            raw_scores = self.clf.decision_function(X)

        # Normalise to [-1, 1] range roughly
        scores = np.tanh(raw_scores / raw_scores.std())
        predictions = ['cat' if s > 0 else 'not_cat' for s in scores]

        # Confidence = how far from decision boundary
        confidence = np.abs(scores)
        confidence = np.clip(confidence, 0, 1)

        return predictions, scores, confidence

    def explain(self, concept_scores, concept_names):
        """Which concepts pushed toward / away from cat?"""
        X = self.scaler.transform(concept_scores)
        deviations = X[0] - self.cat_centroid  # how far from typical cat

        explanation = sorted([
            {
                'concept': concept_names[i],
                'activation': float(concept_scores[0, i]),
                'deviation_from_cat': float(deviations[i]),
                'pushes': 'not_cat' if deviations[i] < -0.5 else
                'cat' if deviations[i] > 0.5 else 'neutral'
            }
            for i in range(len(concept_names))
        ], key=lambda x: abs(x['deviation_from_cat']), reverse=True)

        return explanation

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)


# ── Usage in your pipeline ─────────────────────────────────────

def train_one_class_classifier(concept_scores_dict, concept_names, cfg):
    """
    concept_scores_dict: {'cat': (N, K) array, 'bird': (M, K) array}
    Trains on cat only, validates separation on bird.
    """
    cat_scores = concept_scores_dict['cat']
    bird_scores = concept_scores_dict['bird']

    # Try multiple nu values, pick best separation
    best_nu, best_gap = 0.1, -np.inf

    for nu in [0.05, 0.1, 0.15, 0.2]:
        clf = CatConceptOneClassClassifier(method='ocsvm', nu=nu)
        clf.fit(cat_scores)

        _, cat_sc, _ = clf.predict(cat_scores)
        _, bird_sc, _ = clf.predict(bird_scores)

        # Gap between cat scores and bird scores
        gap = cat_sc.mean() - bird_sc.mean()
        print(f"nu={nu}: cat_mean={cat_sc.mean():.3f}, "
              f"bird_mean={bird_sc.mean():.3f}, gap={gap:.3f}")

        if gap > best_gap:
            best_gap = gap
            best_nu = nu

    print(f"\nBest nu={best_nu} (gap={best_gap:.3f})")
    final_clf = CatConceptOneClassClassifier(method='ocsvm', nu=best_nu)
    final_clf.fit(cat_scores)

    # Evaluation
    cat_preds, cat_sc, cat_conf = final_clf.predict(cat_scores)
    bird_preds, bird_sc, bird_conf = final_clf.predict(bird_scores)

    cat_acc = sum(p == 'cat' for p in cat_preds) / len(cat_preds)
    bird_acc = sum(p == 'not_cat' for p in bird_preds) / len(bird_preds)

    print(f"Cat accuracy:  {cat_acc:.1%}")
    print(f"Bird accuracy: {bird_acc:.1%}")
    print(f"Avg cat confidence:  {cat_conf.mean():.3f}")
    print(f"Avg bird confidence: {bird_conf.mean():.3f}")

    return final_clf