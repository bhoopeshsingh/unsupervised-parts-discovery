# Technical Design: Learning Interpretable Feature Spaces for Medical Image Classification

> **Dissertation Topic Alignment**: "Learning Interpretable Feature Spaces via Domain-Informed Encoders"
> **Application Domain**: Medical image classification — positive (disease) vs negative (healthy) cases
> **Proof-of-Concept Domain**: Cat (positive) vs non-cat (negative) using masked image dataset

---

## Table of Contents

1. [Big Picture](#1-big-picture)
2. [How the Work Aligns With the Dissertation Topic](#2-how-the-work-aligns-with-the-dissertation-topic)
3. [Pipeline Overview](#3-pipeline-overview)
4. [Stage 1 — Domain-Informed Encoder (DINO Feature Extraction)](#4-stage-1--domain-informed-encoder-dino-feature-extraction)
5. [Stage 2 — Foreground Masking](#5-stage-2--foreground-masking)
6. [Stage 3 — Unsupervised Part Discovery (GMM Clustering)](#6-stage-3--unsupervised-part-discovery-gmm-clustering)
7. [Stage 4 — DINO Semantic Fine-Tuning (Domain Knowledge via Loss)](#7-stage-4--dino-semantic-fine-tuning-domain-knowledge-via-loss)
8. [Stage 5 — Human Concept Labeling (Domain Knowledge Injection)](#8-stage-5--human-concept-labeling-domain-knowledge-injection)
9. [Stage 6 — Concept Vector Building](#9-stage-6--concept-vector-building)
10. [Stage 7 — Concept Activation Scoring](#10-stage-7--concept-activation-scoring)
11. [Stage 8 — One-Class Classifier (OneClassSVM)](#11-stage-8--one-class-classifier-oneclasssvm)
12. [Stage 9 — Explanation Generation](#12-stage-9--explanation-generation)
13. [Three Channels of Domain Knowledge](#13-three-channels-of-domain-knowledge)
14. [Connection to Cited Literature](#14-connection-to-cited-literature)
15. [Evaluation Strategy](#15-evaluation-strategy)
16. [Medical Mapping Table](#16-medical-mapping-table)
17. [Pipeline Execution Commands](#17-pipeline-execution-commands)

---

## 1. Big Picture

**The core problem with black-box classifiers in medicine**: A radiologist cannot trust a model that says "tumour present" without knowing *which visual features* caused that decision, *where* they are in the scan, and *how strongly* each one contributed.

**Your system solves this** by forcing the encoder to organise its internal feature space around human-interpretable semantic parts — so that both the prediction and its explanation emerge from the same structured space.

```
INPUT IMAGE
    │
    ▼
[DINO Encoder] — pretrained on 1.2M images, understands visual structure
    │  784 patch embeddings × 384 dimensions
    ▼
[Foreground Mask] — attention-guided: keep only clinically relevant patches
    │  ~300 foreground patches per image
    ▼
[GMM Clustering] — unsupervised: find recurring visual patterns across all positive images
    │  12 clusters (each = candidate semantic part)
    ▼
[Human Labeling] — domain expert names each cluster: cat_ear, cat_eye, cat_fur...
    │                 in medicine: lesion_margin, irregular_texture, healthy_tissue...
    ▼
[Concept Scoring] — per image: what fraction of its patches match each named concept?
    │  [cat_ear=0.16, cat_eye=0.04, light_fur=0.01, cat_face=0.003]
    ▼
[OneClassSVM] — trained ONLY on positive (cat/disease) concept profiles
    │  "does this image's concept fingerprint look like a positive case?"
    ▼
[Explanation] — 3-panel figure: original | coloured concept map | contribution bar chart
```

---

## 2. How the Work Aligns With the Dissertation Topic

| Supervisor Objective | Your Implementation | File |
|---|---|---|
| Design encoder with domain knowledge | DINO ViT fine-tuned with semantic consistency loss | `src/models/dino_finetuner.py` |
| Domain knowledge via **loss function** | Same-cluster patches pulled together; different pushed apart | `dino_finetuner.py:semantic_consistency_loss()` |
| Domain knowledge via **input representation** | Foreground masking removes background noise | `extract_features.py:fg_threshold=0.6` |
| Domain knowledge via **architecture modification** | Spatial (row, col) features appended to patch embeddings | `patch_clusterer.py:_augment_with_spatial()` |
| Interpretable latent features | Each cluster dimension named by domain expert | `labeling/label_tool.py` |
| Evaluate interpretability vs performance | Cluster purity, spatial coherence, classification accuracy | `scripts/validate_clusters.py` |
| Demonstrate in healthcare domain | Cat(+)/Non-cat(−) as proxy for disease/healthy | One-class setup mirrors clinical anomaly detection |

---

## 3. Pipeline Overview

```
experiments/run_pipeline.py --stage extract    → Stage 1+2: DINO extraction + fg masking
experiments/run_pipeline.py --stage cluster    → Stage 3:   GMM clustering
experiments/run_pipeline.py --stage finetune   → Stage 4:   DINO semantic fine-tuning (optional)
streamlit run labeling/label_tool.py           → Stage 5:   Human concept labeling
experiments/run_pipeline.py --stage concepts   → Stage 6+7: Concept vectors + scoring
experiments/run_pipeline.py --stage classify   → Stage 8:   Train OneClassSVM
experiments/run_pipeline.py --stage explain \
  --image data/masked/train/cat/001.png        → Stage 9:   Generate explanation
```

**Recommended full sequence** (first-time run):

```bash
# 1. Initial extraction (no fine-tuning yet)
python experiments/run_pipeline.py --stage extract

# 2. Initial clustering
python experiments/run_pipeline.py --stage cluster

# 3. Fine-tune DINO to improve cluster separation
python experiments/run_pipeline.py --stage finetune

# 4. Re-extract with improved encoder
python experiments/run_pipeline.py --stage extract

# 5. Re-cluster improved features
python experiments/run_pipeline.py --stage cluster

# 6. Label clusters (Streamlit UI)
streamlit run labeling/label_tool.py

# 7. Build concept vectors and scores
python experiments/run_pipeline.py --stage concepts

# 8. Train one-class classifier
python experiments/run_pipeline.py --stage classify

# 9. Explain a prediction
python experiments/run_pipeline.py --stage explain --image data/masked/train/cat/001.png
```

---

## 4. Stage 1 — Domain-Informed Encoder (DINO Feature Extraction)

**File**: `experiments/extract_features.py` → calls `src/models/dino_extractor.py`
**Config**: `dino.model: dino_vits8`, `dino.image_size: 224`
**Output**: `cache/dino_features.pt`

### What DINO Is

DINO (Self-**Di**stillation with **No** labels) is a Vision Transformer (ViT) trained via self-supervised learning on 1.2 million images. It learns rich visual features without any class labels — purely by predicting its own outputs under different augmentations.

**Why DINO over a standard CNN?**
- ViT splits the image into fixed-size patches, giving you a **spatial map of features** — exactly what you need for part discovery
- DINO's CLS token attention naturally focuses on the object, not the background — provides free foreground masking
- Features are semantically smooth: visually similar patches (e.g. two cat ears) land close in embedding space

### Internal Mechanics

```
Input image: [1, 3, 224, 224]  (RGB, normalised with ImageNet mean/std)
     │
     ▼ ViT-S/8: divide into non-overlapping 8×8 patches
     │
     Shape: 224/8 = 28 patches per row/col → 28×28 = 784 patches total
     │
     ▼ Add learnable CLS token → sequence of 785 tokens
     │
     ▼ 12 transformer blocks (self-attention + MLP), embedding dim = 384
     │
     ▼ model.get_intermediate_layers(image_tensor, n=1)[0]
     │  → [B, 785, 384]  (last block output)
     │
     ▼ Drop CLS token (index 0) → [B, 784, 384]
     │
     ▼ .squeeze(0).cpu()  → [784, 384] per image
```

**Code** (`dino_extractor.py:extract_patches`):
```python
feats = self.model.get_intermediate_layers(image_tensor, n=1)[0]
# feats: [B, 785, 384] — index 0 is CLS token
return feats[:, 1:, :]  # drop CLS → [B, 784, 384]
```

**What each 384-dim vector means**: Each of the 784 vectors encodes the semantic content of one 8×8 pixel patch — its texture, colour, shape context, and position relative to other patches. Patches from cat ears across different images cluster together because their semantic content is similar.

### Feature Cache Structure

```python
cache = {
    "features":      tensor[N_total_patches, 384],  # all fg patch embeddings
    "image_ids":     tensor[N_total_patches],        # which image each patch belongs to
    "patch_ids":     tensor[N_total_patches],        # which of 784 positions (0-783)
    "image_paths":   list[str],                      # file paths
    "image_labels":  list[int],                      # class id per image
    "class_names":   list[str],                      # ["cat"]
    "grid_size":     28,
    "feat_dim":      384,
    "fg_threshold":  0.6,
}
```

N_total_patches = N_images × ~300 (after foreground masking removes ~40% background).

---

## 5. Stage 2 — Foreground Masking

**File**: `experiments/extract_features.py` (lines 106–123)
**Config**: `dino.fg_threshold: 0.6`

### Why Mask the Background?

If you include background patches (sky, wall, table) in the clustering, clusters form around background textures rather than object parts. In medical imaging, this is equivalent to the model learning "scanner background" as a feature — not useful.

### How DINO Attention Provides Free Foreground Segmentation

DINO's CLS token attends to the most semantically relevant patches in the image (the object). By thresholding this attention, you get a segmentation mask with **no extra annotation**.

### Internal Mechanics

```
Extract attention from last ViT block:
  model.get_last_selfattention(img_tensor)
  → [1, num_heads, 785, 785]   (6 attention heads for ViT-S)

CLS token row = index 0 of the attention matrix:
  attn[0, :, 0, 1:]            → [6, 784]  (CLS → patch attention, all heads)
  .mean(dim=0)                  → [784]     (average across heads)

Threshold at the 60th percentile:
  threshold = cls_attn.quantile(0.6)
  fg_mask = cls_attn > threshold   → [784] bool
  # keeps top 40% most-attended patches = foreground

Safety fallback:
  if len(fg_indices) < 10:    # if masking is too aggressive
      keep top 25% anyway
```

**Code** (`extract_features.py`):
```python
attn = extractor.extract_attention(img_tensor)   # [1, heads, 785, 785]
cls_attn = attn[0, :, 0, 1:].mean(dim=0).cpu()  # [784] mean across heads
threshold = cls_attn.quantile(fg_threshold)
fg_mask = cls_attn > threshold                   # [784] boolean
```

**Effect**: For a typical cat image with fg_threshold=0.6, ~300 of 784 patches survive (the cat body). Background patches (floor, wall) are discarded.

---

## 6. Stage 3 — Unsupervised Part Discovery (GMM Clustering)

**File**: `src/pipeline/patch_clusterer.py`
**Config**: `clustering.method: gmm`, `n_clusters: 10`, `use_pca: true`, `use_spatial_features: true`
**Output**: `cache/kmeans.pkl` (the GMM model), `cache/cluster_labels.pt`

### Why Gaussian Mixture Model (not KMeans)?

| Property | KMeans | GMM |
|---|---|---|
| Cluster shape | Spherical only | Ellipsoidal — handles elongated clusters |
| Part sizes | Assumes equal size | Handles unequal part sizes (ear is small, fur is large) |
| Assignment | Hard (one cluster per patch) | Soft probabilities (we use argmax = hard) |
| Scalability | Scales easily | Slow on full dataset → subsample 150k for fitting |

Cat ears are small (few patches per image) while cat fur covers most of the body. KMeans would split fur into many small clusters to balance sizes. GMM correctly allocates one large cluster for fur and several small ones for fine-grained features.

### Feature Preparation Pipeline

Before clustering, features go through three transformations:

**Step 1: L2 Normalisation**
```python
X = normalize(features.numpy(), norm="l2")
# [N, 384] — each vector has unit length
# Makes cosine distance equivalent to Euclidean distance
```

**Step 2: PCA Dimensionality Reduction (384 → 64)**
```python
self.pca = PCA(n_components=64, random_state=42)
X = self.pca.fit_transform(X)    # [N, 64]
X = normalize(X, norm="l2")      # re-normalise after projection
# Retains ~85-90% of variance, removes noise dimensions
# Speeds up GMM fitting significantly
```

**Step 3: Spatial Feature Augmentation**
```python
rows = (patch_ids // 28) / 28.0  # normalised row position [0, 1]
cols = (patch_ids % 28)  / 28.0  # normalised col position [0, 1]
spatial = np.stack([rows, cols], axis=1) * spatial_weight  # scale by 0.4
X = np.hstack([features, spatial])  # [N, 66]  (64 semantic + 2 spatial)
X = normalize(X, norm="l2")
```

The spatial_weight=0.4 means position contributes 40% relative to the semantic content. This encodes the domain prior that **anatomical parts occupy consistent spatial positions** — cat eyes are always near the top-centre, not scattered randomly. In medical imaging: a tumour at the lung apex has different clinical significance than one at the base.

### GMM Fitting (Scalability)

```python
# 1.4M patches total → subsample 150k for fitting
if N > 150_000:
    idx = rng.choice(N, 150_000, replace=False)
    self.gmm.fit(X[idx])   # fit on subsample
else:
    self.gmm.fit(X)

# Predict ALL patches (fast — just evaluate Gaussian likelihoods)
self.labels_ = self.gmm.predict(X)
```

GMM internal parameters:
- `covariance_type='diag'` — each cluster has its own diagonal covariance (no cross-term correlations). This scales as O(K×D) not O(K×D²), making it feasible in 64 dims.
- `n_init=3` — run EM algorithm 3 times with different initialisations, keep best
- Each cluster is characterised by: mean vector `μₖ` [64+2], diagonal covariance `Σₖ` [64+2], mixing weight `πₖ`

### What Comes Out

```
cache/cluster_labels.pt  — tensor[N_total_patches] of cluster IDs (0 to 11)
cache/kmeans.pkl          — the fitted GaussianMixture object
cache/kmeans_meta.pt      — hyperparameters (n_clusters, method, spatial_weight, etc.)
cache/kmeans_pca.pkl      — the fitted PCA object (needed for inference)
```

Example cluster distribution: `[82401, 65234, 110293, 44821, ...]` — unequal sizes reflecting that fur is more common than eyes.

---

## 7. Stage 4 — DINO Semantic Fine-Tuning (Domain Knowledge via Loss)

**File**: `src/models/dino_finetuner.py`
**Config**: `finetune.n_epochs: 3`, `finetune.lr: 1e-5`, `finetune.n_pairs: 512`
**Output**: `cache/dino_finetuned.pt`
**Run**: `python experiments/run_pipeline.py --stage finetune`

### The Problem This Solves

After the first round of clustering on a frozen DINO, the cluster boundaries are approximate — the encoder wasn't trained to produce semantically distinct part features. Fine-tuning the encoder to *respect* the cluster structure makes the feature space more interpretable.

### Which Parts of the Model Are Updated

```python
# Freeze ALL parameters first
for param in self.model.parameters():
    param.requires_grad = False

# Unfreeze ONLY the last 2 transformer blocks (blocks.10 and blocks.11)
for name, param in self.model.named_parameters():
    if "blocks.10" in name or "blocks.11" in name:
        param.requires_grad = True
```

ViT-S has 12 transformer blocks (0–11). Freezing the first 10 preserves the general visual features learned from ImageNet. Only the final attention heads and MLPs are updated — they do the final semantic organisation of features.

**Trainable parameters**: ~4.7M out of 21M total (last 2 blocks only).

### The Semantic Consistency Loss

This is where **domain knowledge enters via the loss function** — the key contribution the supervisor is looking for.

```python
def semantic_consistency_loss(self, embeddings, pseudo_labels, margin=0.5, n_pairs=512):
    embeddings = F.normalize(embeddings, dim=1)   # L2 normalise

    # Sample random pairs of patches from the current batch
    idx_a = torch.randint(0, N, (n_pairs,))
    idx_b = torch.randint(0, N, (n_pairs,))

    # Cosine similarity between each pair
    sim = (embeddings[idx_a] * embeddings[idx_b]).sum(dim=1)   # [-1, 1]

    # Binary: 1 if same cluster (same semantic part), 0 if different
    same = (pseudo_labels[idx_a] == pseudo_labels[idx_b]).float()

    # Loss:
    #   same cluster  → minimise (1 - sim)  i.e. pull together toward sim=1
    #   diff cluster  → minimise relu(sim - margin)  i.e. push apart below margin
    loss = same * (1 - sim) + (1 - same) * F.relu(sim - margin)
    return loss.mean()
```

**Intuition**:
- If patches A and B are from the same cluster (both cat ears), their embeddings should be similar (cos sim → 1). Loss penalises when they are far apart.
- If patches A and B are from different clusters (one cat ear, one cat eye), their embeddings should be dissimilar (cos sim < 0.5). Loss penalises when they are too close.

After fine-tuning, the feature space has **wider separation between semantic parts** — making subsequent clustering and scoring more discriminating.

### The Bootstrap Loop

Fine-tuning uses the cluster pseudo-labels as supervision. After fine-tuning, you re-extract features (with the improved encoder) and re-cluster. This is a self-improving loop:

```
Initial DINO → Cluster → Pseudo-labels
                              ↓
                     Fine-tune DINO with pseudo-labels
                              ↓
                     Better DINO → Re-cluster → Better pseudo-labels
```

---

## 8. Stage 5 — Human Concept Labeling (Domain Knowledge Injection)

**File**: `labeling/label_tool.py`
**Run**: `streamlit run labeling/label_tool.py`
**Output**: `cache/labels.json`

### Why This Is the Critical Step

This is the **neurosymbolic bridge** — the point where a human domain expert turns unlabeled cluster IDs into named, meaningful concepts. Without this step, the system has discovered structure but cannot name it.

In medical imaging, this is equivalent to a radiologist reviewing clusters of image patches and saying:
- "Cluster 3 — all these patches look like irregular lesion boundaries"
- "Cluster 7 — these are all healthy parenchymal tissue"
- "Cluster 11 — this is just background/noise, exclude it"

### What the Labeling UI Shows

For each cluster, the expert sees:
1. **9 representative patch thumbnails** (8×8 pixels each, enlarged to 64×64) — randomly sampled from the cluster
2. **Context windows** — the full 224×224 image with a red box highlighting where the patch came from
3. **Part map overlay** — the full image with all 784 patches coloured by cluster assignment
4. **Cluster statistics** — how many patches, which images, confidence level

### The labels.json Format

```json
{
  "0": {"label": "cat_ear",   "include": true,  "confidence": 3, "notes": "top triangular structures"},
  "1": {"label": "cat_eye",   "include": true,  "confidence": 3, "notes": "round bright patches"},
  "2": {"label": "light_fur", "include": true,  "confidence": 2, "notes": "light-coloured body fur"},
  "3": {"label": "cat_face",  "include": true,  "confidence": 3, "notes": "facial feature patches"},
  "4": {"label": "",          "include": false,  "notes": "background/floor — excluded"},
  ...
}
```

**include: false** clusters are permanently excluded from concept scoring — they contribute zero information to the classifier. This is the equivalent of a radiologist saying "this cluster is just scanner artefact, ignore it."

---

## 9. Stage 6 — Concept Vector Building

**File**: `src/pipeline/concept_builder.py:build_concept_vectors()`
**Output**: `cache/concept_vectors.pt`

### What a Concept Vector Is

For each labeled, included cluster, compute the **mean of all patch embeddings** assigned to that cluster across all positive (cat) images.

```python
mask = labels_arr == cluster_id          # which of N_total_patches belong to this cluster
cluster_feats = features[mask]           # [M, 384]  — all patches of this concept
mean_vec = cluster_feats.mean(dim=0)     # [384]     — centroid in feature space
mean_vec = F.normalize(mean_vec, dim=0) # unit vector
concept_vectors[label] = mean_vec
```

**Interpretation**: The concept vector for "cat_ear" is the average feature representation of every ear patch across all cat images in your dataset. It is the **prototype** of what an ear looks like in DINO's feature space — a 384-dimensional fingerprint.

These concept vectors are the **interpretable axes of your feature space**. Instead of 384 anonymous dimensions, you have a named direction in space for each semantic part:

```
Latent space axes:
  dim_0 to dim_383: (abstract, uninterpretable)

  vs.

  "cat_ear" direction:  unit vector pointing toward ear-like features
  "cat_eye" direction:  unit vector pointing toward eye-like features
  "light_fur" direction: unit vector pointing toward fur-like features
```

---

## 10. Stage 7 — Concept Activation Scoring

**File**: `src/pipeline/concept_builder.py:compute_concept_scores_all()`
**Config**: `concepts.scores_cache: cache/concept_scores.pt`

### The Key Design Decision: Cluster Proportion, Not Cosine Similarity

The scoring method is the most critical design choice in the pipeline. Two options were considered:

**Option A — Cosine Similarity (REJECTED)**
```python
# For each concept, find the maximum patch similarity
sims = img_patches @ concept_vector.T   # [784]
score = sims.topk(10).values.mean()    # top-10 mean
```
Problem: Cosine similarity gave scores of **0.4–0.8 for ALL images** — cats, birds, airplanes, random noise. The SVM boundary was meaningless because the score space had no discriminating power.

**Option B — Cluster Proportion (USED)**
```python
# For each image, run clusterer to assign each patch to a cluster
patch_labels = clusterer.predict(image_fg_patches)   # [K] cluster ids

# Score = fraction of patches landing in this concept's cluster
for concept_name, cluster_id in concept_to_cluster.items():
    score = (patch_labels == cluster_id).sum() / len(patch_labels)
```

**Why this works**:
- The GMM was fitted exclusively on cat images. Cat patches naturally land in cat-specific clusters.
- A bird image's patches scatter into clusters differently — they land in clusters that represent feathers, beaks, sky — not cat ears or cat eyes. They get **score ≈ 0.0** for all cat-specific concepts.
- Score range: **0.0–0.38** (discriminating) vs cosine similarity's **0.4–0.8** (not discriminating).

### Concrete Score Examples

| Image | cat_ear | cat_eye | light_fur | cat_face |
|---|---|---|---|---|
| Cat | 0.161 | 0.045 | 0.006 | 0.169 |
| Bird | 0.000 | 0.000 | 0.080 | 0.000 |
| Airplane | 0.000 | 0.000 | 0.002 | 0.001 |
| Human face | 0.005 | 0.021 | 0.013 | 0.018 |

Cat images have meaningful non-zero scores in cat-specific clusters. Non-cat images are near zero.

### Output: Concept Score Matrix

```
cache/concept_scores.pt:
    "scores":        tensor[N_images, N_concepts]  — the score matrix
    "concept_names": ["cat_ear", "cat_eye", "light_fur", "cat_face", ...]
    "image_labels":  [0, 0, 0, 0, ...]  (all cats = class 0)
    "class_names":   ["cat"]
```

---

## 11. Stage 8 — One-Class Classifier (OneClassSVM)

**File**: `src/pipeline/one_class_classifier.py`
**Config**: `classifier_path: cache/concept_classifier.pkl`
**Output**: The fitted SVM model

### Why One-Class, Not Binary?

Binary classifiers (cat vs bird) require labeled examples of both classes. In medicine:
- You need confirmed disease examples **and** confirmed healthy examples
- Healthy examples are easy to collect, but **labeling is expensive and time-consuming**
- You may encounter diseases at test time that weren't in your training set

One-class classification (anomaly detection) trains only on positive examples:
- **Training**: fit only on disease images
- **Inference**: "is this new image inside or outside the disease distribution?"
- **Advantage**: naturally handles novel negatives — anything that doesn't look like the disease is flagged

### How OneClassSVM Works Internally

```python
clf = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
```

**nu = 0.05**: 5% of training samples are expected to be outliers (noise/mislabeled). Smaller nu → tighter boundary around the positive class.

**RBF kernel**: Maps the concept score vectors into a high-dimensional space where a hyperplane (the boundary) can separate the positive cluster from the origin.

**Training** (cat concept scores only):
```python
X = self.scaler.fit_transform(cat_concept_scores)  # standardise each concept
self.clf.fit(X)
self.cat_centroid = X.mean(axis=0)   # store for explanation
self.concept_std = X.std(axis=0)
```

**Inference**:
```python
X = self.scaler.transform(concept_scores)  # same standardisation
raw_scores = self.clf.decision_function(X)
# positive → inside boundary → CAT
# negative → outside boundary → NOT CAT

# Normalise to [-1, 1]
scores = np.tanh(raw_scores / raw_scores.std())
predictions = ['cat' if s > 0 else 'not_cat' for s in scores]
confidence = np.abs(scores)  # distance from boundary = confidence
```

**Explanation**:
```python
deviations = X[0] - self.cat_centroid   # how far from typical cat profile
# deviation < -0.5 → this concept is BELOW typical cat level
# deviation > +0.5 → this concept is ABOVE typical cat level
```

### Performance

- **Cat recall (training set)**: 93.4% — 93.4% of cat images correctly classified as CAT
- **Average confidence**: 0.754 — well away from the decision boundary
- **Non-cat**: correctly gives NOT CAT with high confidence for birds, airplanes, random images

---

## 12. Stage 9 — Explanation Generation

**File**: `src/pipeline/concept_classifier.py:render_dissertation_explanation()`
**Output**: `cache/explain_semantic_{image_name}.png`

### The 3-Panel Explanation Figure

**Panel 1 — Original Image**: The input image as-is. No modification.

**Panel 2 — Semantic Part Map**: A coloured overlay where each of the 784 patches is coloured according to its dominant concept.

```python
# Assign each patch to nearest concept vector by cosine similarity
img_norm = F.normalize(image_features, dim=-1)   # [784, 384]
vecs = torch.stack([
    F.normalize(concept_vectors[n].float(), dim=0) for n in concept_names
])                                                 # [N_concepts, 384]
patch_sims = (img_norm @ vecs.T).numpy()          # [784, N_concepts]
concept_idx = patch_sims.argmax(axis=1)           # [784] — dominant concept per patch
concept_map = concept_idx.reshape(28, 28)         # [28, 28]
```

The 28×28 map is then upscaled to 224×224 (nearest neighbour) and blended with the original image. Background patches (fg_mask=False) are dimmed with alpha=0.25 to visually de-emphasise them.

**Panel 3 — Concept Contribution Bar Chart**: Horizontal bars showing each concept's contribution.

```python
# contribution = activation score × deviation from typical cat
# Positive bar → this concept pushed TOWARD positive prediction
# Negative bar → this concept pushed AWAY FROM positive prediction
contributions = {
    concept: activation × deviation_from_cat_centroid
    for concept, activation in concept_scores.items()
}
```

### What the Clinician Reads

```
Prediction:  CAT  (82%)

Concept Activations (sorted by strength):
  cat_face     0.169  ████████████████████
  cat_ear      0.161  ███████████████████
  cat_eye      0.045  █████
  light_fur    0.006  █

Evidence: cat_face, cat_ear, cat_eye
```

In medical terms: "Positive (disease detected, 82% confidence). Evidence: lesion_margin (16.9%), irregular_core (16.1%), disrupted_boundary (4.5%)."

---

## 13. Three Channels of Domain Knowledge

The supervisor explicitly asks for **multiple strategies** for embedding domain knowledge. Your system implements all three simultaneously:

### Channel 1: Input Representation
- **What**: Foreground masking removes background patches before any learning occurs
- **How**: DINO CLS attention thresholded at 60th percentile (`fg_threshold=0.6`)
- **Domain prior encoded**: "Relevant features are in the foreground, not the background"
- **Medical analogy**: Focus on the organ/region of interest, not the scanner table or surrounding tissue

### Channel 2: Architecture Modification
- **What**: Spatial (row, col) coordinates appended to patch embeddings before clustering
- **How**: `spatial_weight=0.4` scales position relative to semantic content
- **Domain prior encoded**: "Anatomical parts occupy consistent spatial positions"
- **Medical analogy**: A tumour at the lung apex vs the lung base — position matters for diagnosis

### Channel 3: Loss Function
- **What**: Semantic consistency loss during DINO fine-tuning
- **How**: Same-cluster patches pulled together; different-cluster patches pushed apart
- **Domain prior encoded**: "Semantically similar patches (same part) should have similar representations"
- **Medical analogy**: All patches showing the same tissue type should cluster together in feature space

### Channel 4: Human Symbolic Labeling (Bonus)
- **What**: Expert names clusters, excludes irrelevant ones
- **How**: Streamlit UI with context windows and part map overlays
- **Domain prior encoded**: Explicit semantic vocabulary from the domain expert
- **Medical analogy**: Radiologist labels cluster 3 as "ground glass opacity" — a specific, clinically meaningful finding

---

## 14. Connection to Cited Literature

| Paper | Relevance to Your Work |
|---|---|
| **Melchior (2022)** — Symmetry transformations in encoders | Your spatial augmentation encodes the invariance prior that anatomical parts appear at consistent locations. Position is explicitly represented, not discarded. |
| **Dash et al. (2022)** — Survey of domain knowledge strategies | You implement all three strategies they survey: input modification (fg masking), loss function (contrastive loss), architecture modification (spatial features). |
| **Zhan et al. (2021)** — Neurosymbolic encoders | Your human labeling step is precisely the "symbolic program" layer. The neural encoder discovers structure; the human provides symbolic names. This is neural + symbolic = neurosymbolic. |
| **Bouadi et al. (2024)** — Semantic-guided feature engineering | Your cluster-proportion features are semantically guided: each feature dimension is named, verified by a human, and corresponds to a specific visual concept. |
| **Worrall et al. (2017)** — Interpretable encoder-decoder networks | Your concept vectors disentangle the feature space into named, non-overlapping semantic directions — the same goal as disentangled encoder-decoder representations. |

---

## 15. Evaluation Strategy

### Interpretability Metrics

| Metric | How Measured | Target |
|---|---|---|
| Cluster visual coherence | Human labeling confidence score (1-3 per cluster) | Avg confidence > 2 |
| Spatial coherence | Part map visual inspection — are concept regions contiguous? | Connected blobs, not scattered pixels |
| Concept coverage | Fraction of fg patches covered by included concepts | > 70% |
| Label consensus | Would two independent labelers agree on the cluster name? | Qualitative |

### Predictive Performance

| Metric | Value | Notes |
|---|---|---|
| Cat recall (training) | 93.4% | 93.4% of cat images correctly classified as CAT |
| Avg confidence (cat) | 0.754 | Well above the 0 boundary |
| Non-cat true negative | Bird → NOT CAT (100%) | cat_face=0.000, cat_ear=0.000 |
| Confidence calibration | High confidence aligns with visual plausibility | Qualitative check |

### Interpretability vs Performance Trade-off

| Hyperparameter | More Interpretable ↑ | Better Accuracy ↑ | Trade-off |
|---|---|---|---|
| n_clusters ↑ | Finer-grained concepts | Harder to label, smaller scores | Sweet spot at K=10-12 |
| fg_threshold ↑ | Cleaner foreground | Fewer patches → lower recall | 0.6 balances both |
| spatial_weight ↑ | More coherent spatial parts | Less flexibility for scattered parts | 0.4 gives good maps |
| GMM vs KMeans | Handles unequal part sizes | Slower fitting | GMM preferred |
| Fine-tuning | Wider cluster separation | Requires two rounds of extract+cluster | Recommended |

---

## 16. Medical Mapping Table

| Your Proof-of-Concept | Medical Translation |
|---|---|
| Cat images | Histopathology slides / MRI scans with confirmed diagnosis |
| Non-cat images (birds, planes) | Healthy tissue scans / negative control cases |
| DINO patch features [784, 384] | Local texture/structure features from scan patches |
| cat_ear cluster | Lesion margin cluster |
| cat_eye cluster | Irregular core cluster |
| light_fur cluster | Parenchymal texture cluster |
| Foreground masking | ROI masking — focus on organ of interest, not scanner table |
| OneClassSVM trained on cats | Disease detector trained only on confirmed disease cases |
| nu = 0.05 | 5% of training cases assumed to be mislabeled or atypical |
| CAT prediction | "Disease present" |
| NOT CAT prediction | "Healthy / No disease detected" |
| Concept score [cat_ear=0.16] | Feature profile [lesion_margin=0.16] |
| Explanation bar chart | Feature-level audit trail for the radiologist |

### Why One-Class Mirrors Clinical Reality

1. **Annotated disease cases are scarce**: Labeling a scan "disease positive" requires expert radiologist time. Labeling "healthy negative" is easier but less discriminating.
2. **Novel disease variants**: At test time, a scanner might encounter a variant of the disease not seen in training. A binary classifier fails; a one-class detector flags any image that doesn't fit the disease profile.
3. **Class imbalance**: Disease prevalence in the real world may be 1 in 1000. Training a binary classifier on balanced data creates unrealistic performance expectations.

---

## 17. Pipeline Execution Commands

```bash
# Setup
cd /Users/dev/ml-projects/unsupervised-parts-discovery
pip install -r requirements.txt

# Step 1: Extract DINO features (15-20 min for ~1000 images on M4 Mac)
python experiments/run_pipeline.py --stage extract

# Step 2: Find optimal number of clusters (optional — generates elbow plot)
python experiments/find_optimal_k.py

# Step 3: Cluster patches into semantic parts
python experiments/run_pipeline.py --stage cluster

# Step 4: Fine-tune DINO (optional — improves separation, ~30 min)
python experiments/run_pipeline.py --stage finetune
# Then re-run extract and cluster with improved encoder:
python experiments/run_pipeline.py --stage extract
python experiments/run_pipeline.py --stage cluster

# Step 5: Human labeling — open Streamlit UI
streamlit run labeling/label_tool.py
# → Label each cluster in the "Label Clusters" tab
# → Press "Save Labels" when done

# Step 6: Build concept vectors and compute scores
python experiments/run_pipeline.py --stage concepts

# Step 7: Train one-class classifier
python experiments/run_pipeline.py --stage classify

# Step 8: Explain a prediction
python experiments/run_pipeline.py --stage explain --image data/masked/train/cat/001.png
# → Outputs: cache/explain_semantic_001.png
# → Outputs: cache/explanation_001.png

# Validate cluster quality
python scripts/validate_clusters.py
```

---

## Key File Reference

| File | Purpose |
|---|---|
| `configs/config.yaml` | All hyperparameters — single source of truth |
| `src/models/dino_extractor.py` | DINO ViT-S/8 wrapper, patch extraction, attention maps |
| `src/models/dino_finetuner.py` | Semantic consistency fine-tuning, ImagePathDataset |
| `src/pipeline/patch_clusterer.py` | GMM/KMeans clustering, spatial feature augmentation, PCA |
| `src/pipeline/concept_builder.py` | Concept vector building, cluster proportion scoring |
| `src/pipeline/concept_classifier.py` | compute_image_concept_scores, spatial concept map, explanation rendering |
| `src/pipeline/one_class_classifier.py` | CatConceptOneClassClassifier (OneClassSVM), explain() |
| `experiments/extract_features.py` | Batch DINO extraction with foreground masking |
| `experiments/run_pipeline.py` | Orchestrates all stages end-to-end |
| `experiments/find_optimal_k.py` | Elbow + silhouette analysis for K selection |
| `labeling/label_tool.py` | Streamlit UI — Label Clusters + Classify & Explain tabs |
| `scripts/validate_clusters.py` | Cluster quality metrics and visualisation |
| `cache/labels.json` | Human-assigned concept names per cluster |
| `cache/dino_features.pt` | Extracted patch features [N_patches, 384] |
| `cache/cluster_labels.pt` | Cluster assignment per patch [N_patches] |
| `cache/concept_vectors.pt` | Mean concept vectors {name: tensor[384]} |
| `cache/concept_scores.pt` | Concept activation scores [N_images, N_concepts] |
| `cache/concept_classifier.pkl` | Fitted OneClassSVM |
| `cache/kmeans.pkl` | Fitted GMM model |
| `cache/dino_finetuned.pt` | Fine-tuned DINO weights (after --stage finetune) |
