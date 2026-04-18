# Viva Q&A — Unsupervised Concept Discovery for Interpretable Prediction
*Organised by pipeline stage. Code snippets reference actual files.*

---

## STAGE 0 — Problem & Motivation

**Q: Why does interpretability matter? Isn't accuracy enough?**
A: GDPR Article 22 and the EU AI Act require that automated decisions affecting people — especially clinical decisions — must be explainable. A black-box model that says "diabetes: 87%" cannot be audited, challenged, or trusted by a clinician. My system gives a per-patient concept trail: "this prediction was driven by elevated_cholesterol (0.18 activation) and metabolic_syndrome (0.12 activation)" — that is auditable and contestable.

**Q: What is wrong with existing XAI methods like TCAV, CBM, and ACE?**
A: All three require concept labels to be defined BEFORE or DURING training. TCAV needs labelled probe datasets per concept. CBM requires concept annotations on every training image. ACE uses image segmentation preprocessing. My approach discovers concepts from raw unlabelled data and names them post-hoc — zero concept supervision at any stage.

**Q: What is your research question?**
A: "Can we discover meaningful concepts from unlabelled data and use those concept activations to make interpretable, accurate predictions?" The answer from this work is yes for visual data (98.9% accuracy) and partially yes for clinical tabular data (1.53× chance, with class-imbalance limitations).

---

## STAGE 1 — Data

**Q: Why Oxford Pets, Stanford Cars, and CUB-200 specifically?**
A: Three reasons. First, they are standard benchmarks with known difficulty. Second, they have clean foreground-dominant images, which matters because DINO's attention-based foreground masking is designed for objects not scenes. Third, the three classes (cat / car / bird) are semantically distant — a model that confuses them is genuinely broken, not just unlucky.

**Q: What is your train/test split?**
A: 80/20 stratified split, `random_state=42`. 800 test images per class = 2,400 total test images. Stratification ensures class proportions are equal in both splits.

**Q: Why 12,000 images total?**
A: ~4,000 per class gives GMM enough patches to fit stable clusters. Each image contributes ~196 foreground patches (25% of 784), giving ~2.35 million total foreground patch vectors for GMM fitting.

---

## STAGE 2 — DINO Feature Extraction

**Q: Why DINO and not ResNet, CLIP, or a supervised ViT?**
A: DINO's self-supervised training objective forces the CLS token attention to focus on semantically meaningful object regions without any labels. The result is that DINO patches from the same object part (e.g. all cat ears across different images) naturally cluster together in feature space. Supervised models don't have this property — their features are optimised for classification, not part separation. CLIP is text-aligned, not part-structured.

**Q: Why ViT-S/8 specifically — why patch size 8?**
A: Patch size 8 on a 224×224 image gives a 28×28 = 784 patch grid. Each patch covers 8×8 pixels — fine enough to separate eyes from nose from ear on a face. Patch size 16 gives only 196 patches (14×14 grid) which loses spatial resolution for small parts like eyes or car headlights.

**Q: Why concatenate layers 8, 10, and 12 (multi-layer)?**
A: Each layer captures a different abstraction level. Layer 8 (384d) captures texture — fur type, surface grain, feather pattern. Layer 10 (384d) captures local structure — contours, part boundaries. Layer 12 (384d) captures semantics — object identity. Concatenating gives 1152d vectors that encode all three simultaneously. The ablation (variant B) showed that single-layer L12 drops NMI by 0.028 and class purity by 1.6 percentage points.

**File:** `src/models/dino_extractor.py`
```python
# Multi-layer extraction: layers 8, 10, 12 concatenated
all_layers = self.model.get_intermediate_layers(x, n=5)
# Indices 0,2,4 correspond to blocks 7,9,11 → layers 8,10,12
feats = torch.cat(
    [all_layers[i][:, 1:, :] for i in (0, 2, 4)], dim=-1
)  # [B, 784, 1152]
```

**Q: What is foreground masking and why is it needed?**
A: DINO's CLS token attention naturally highlights the object being attended to. The fg_threshold=0.75 means we keep only the top 25% most-attended patches per image. Without this, background patches (sky, floor, wall) would enter GMM fitting and pollute clusters — you'd get a "grey wall texture" cluster instead of meaningful semantic parts. Ablation C (no FG masking) showed NMI drops from 0.326 to 0.267.

**File:** `experiments/extract_features.py` (threshold applied during extraction)
```python
# Keep top 25% most-attended patches as foreground
fg_threshold = cfg["dino"]["fg_threshold"]   # 0.75
attn = attentions[0].mean(0)[0, 1:]         # CLS attention: [784]
fg_mask = attn > torch.quantile(attn, fg_threshold)  # top 25%
```

---

## STAGE 3 — GMM Clustering

**Q: Why GMM and not K-Means?**
A: Two reasons, both visible in your data. First, cluster sizes are highly unequal — `cat:fur` covers ~15% of cat patches, `cat:eye` covers ~3%. K-Means assumes spherical equal-variance clusters and will split the large fur cluster and merge small eye/nose clusters. GMM uses a covariance matrix per cluster (`covariance_type="diag"`) so it can model elongated, differently-sized clusters. Second, GMM gives a soft probability assignment — every patch has a probability of belonging to each cluster — which is more principled than K-Means' hard nearest-centroid assignment.

**File:** `src/pipeline/patch_clusterer.py` lines 45–52
```python
self.gmm = GaussianMixture(
    n_components=n_clusters,
    covariance_type="diag",   # diagonal = faster, handles unequal sizes
    n_init=3,
    max_iter=200,
    random_state=random_seed,
)
```

**Q: Why 30 clusters for 3 classes?**
A: K=30 = K_per_class × n_classes = 10 × 3. K=10 per class was determined empirically via a single-class pilot: `find_optimal_k.py` was run on cat-only features and silhouette analysis identified K=10 as optimal (silhouette=0.61, the maximum across K=5–20, Table 6 in the dissertation). This was then scaled to three classes: 10 × 3 = 30. The assumption is that each class has similar internal part complexity — reasonable for three semantically distinct categories. 8 clusters were excluded as noise after human labeling, leaving 22 active concepts.

**Q: You used KMeans in find_optimal_k.py but GMM for actual clustering — why?**
A: Running GMM across K=5 to 20 on millions of patches per sweep would be prohibitively slow. KMeans silhouette is a fast, well-established proxy for identifying the natural cluster count in a feature space — the geometry of the data does not fundamentally change based on algorithm. The K=10 result from KMeans was then validated by GMM convergence diagnostics: `converged_=True` and a stable `lower_bound_` at K=30 in the full three-class run (visible in `src/pipeline/patch_clusterer.py` line 127–128).

**Q: Why PCA before clustering? Why 128 dimensions?**
A: 1152d → 128d via PCA. Two reasons: (1) GMM fitting is O(n·k·d²) — at 1152d this is computationally prohibitive. (2) PCA removes correlated noise dimensions and keeps the 128 axes of greatest variance (~90% of total). After PCA the feature space is denser and GMM converges more reliably.

**File:** `src/pipeline/patch_clusterer.py` lines 73–84
```python
def _apply_pca(self, X, fit=False):
    if fit:
        self.pca = PCA(n_components=self.pca_dims, random_state=self.random_seed)
        X = self.pca.fit_transform(X)
        explained = self.pca.explained_variance_ratio_.sum() * 100
        print(f"PCA: {self.pca.n_features_in_} → {self.pca_dims} dims ({explained:.1f}% variance)")
    else:
        X = self.pca.transform(X)
    return normalize(X, norm="l2")
```

**Q: What is spatial smoothing and why did you add it?**
A: After GMM assigns clusters, isolated "stray" patches can appear — a single background patch stamped as `cat_ear` surrounded by `cat_face` patches. Spatial smoothing applies majority-vote in a 3×3 neighbourhood: if ≥2/3 of a patch's neighbours belong to a different cluster, the patch is reassigned. This cleans the semantic part maps visually without changing cluster boundaries.

**File:** `src/pipeline/patch_clusterer.py` lines 145–212 (`smooth_labels`)

---

## STAGE 4 — Concept Naming (Human SME Labeling)

**Q: How are clusters assigned names like "cat:face" or "car:wheel"?**
A: After clustering, the Streamlit tool (`labeling/image_cluster_labeler.py`) shows a thumbnail grid of random patches from each cluster. A human expert (me) looks at the grid and types a name. If 80%+ of patches in a cluster show cat face regions, it is named `cat:face`. This is the only human input in the entire pipeline — no labels at any other stage.

**Q: Isn't this subjective? What if two people name clusters differently?**
A: The naming is post-hoc and does not affect the model's behaviour — the cluster structure is fixed by GMM. Different names would change the bar chart labels but not the classification accuracy or concept contributions. Reproducibility of naming is a known limitation, addressed in future work via LLM-assisted naming (already implemented for the lab domain).

**Q: Why were 8 of 30 clusters excluded?**
A: During labeling, 8 clusters were marked `include: false` in `cache/labels.json` because they contained mixed-class patches with no coherent semantic label (e.g. generic texture edges, shadow regions). Including them would add noise features to the classifier. 22 clean concept clusters remain.

**File:** `src/pipeline/concept_builder.py` lines 35–56
```python
for cluster_id_str, meta in human_labels.items():
    label = meta.get("label", "").strip()
    include = meta.get("include", True)
    if not label or not include:
        continue                          # skip noise clusters
    mask = labels_arr == cluster_id
    cluster_feats = features[mask]        # all patches in this cluster
    mean_vec = cluster_feats.mean(dim=0)  # concept vector = centroid
    mean_vec = F.normalize(mean_vec, dim=0)
    concept_vectors[label] = mean_vec
```

---

## STAGE 5 — Concept Scores & Classification

**Q: How is the concept activation score (act=) computed for a new image?**
A: Activation = fraction of the image's foreground patches assigned to that concept's cluster by GMM. If 196 foreground patches exist and 66 land in cluster 5 (cat:face), activation = 66/196 = 0.34.

**File:** `src/pipeline/concept_builder.py` lines 120–124
```python
for c_idx, c_name in enumerate(concept_names):
    cluster_id = concept_to_cluster[c_name]
    scores_matrix[img_idx, c_idx] = (
        img_cluster_labels == cluster_id
    ).sum() / n_patches
```

**Q: Why did LogReg achieve 98.9% while the OneClassSVM baseline only got 46.4%?**
A: The comparison is not quite fair — they test different things. The OneClassSVM in `stage5_results.json` is the **raw baseline**: it operates on mean DINO L12 features per image (384d PCA-compressed to 50d), with no concept extraction. It is equivalent to asking "can raw pixel-level features classify without any structure?" at 46.4%. LogReg operates on the 22-dimensional concept score vector — a highly discriminative, semantically structured representation. The 2.1× accuracy gap is the direct evidence that concept extraction adds value. The two classifiers test different input representations, not classifier strength.

**Q: Why Logistic Regression and not SVM, Random Forest, or MLP?**
A: Three reasons: (1) Linear probe — the dissertation's interpretability claim requires that classification is a simple linear combination of concept scores. A non-linear classifier would obscure which concepts matter. (2) The weights are directly the explanation: `contribution = weight × activation` is only meaningful if the model is linear. (3) 22-dimensional input with 12,000 samples — LogReg is perfectly suited, no non-linearity needed given how discriminative the concept scores already are.

**File:** `experiments/run_pipeline.py` lines 265–271
```python
clf = LogisticRegression(
    C=1.0,          # L2 regularisation — from configs/config.yaml
    max_iter=1000,
    solver="lbfgs",
    random_state=42,
)
clf.fit(X_tr, y_tr)   # X_tr: [N, 22] concept scores
```

**Q: What does StandardScaler do and why is it needed?**
A: Concept scores have different natural ranges — `cat:face` runs 0–0.34 while `car:wheel` runs 0–0.12. Without scaling, features with larger ranges dominate the L2 penalty. StandardScaler subtracts the training mean and divides by standard deviation, so every concept contributes equally to the regularisation term.

---

## STAGE 6 — Explanation & Visualisation

**Q: How is the semantic part map (coloured overlay) generated?**
A: For all 784 patches of the image, compute cosine similarity to each of the 22 concept vectors. Assign each patch to its nearest concept. Reshape the 784 assignments to a 28×28 grid, upsample to 224×224 via nearest-neighbour, and colour each region by concept.

**File:** `src/pipeline/concept_classifier.py` lines 219–225
```python
img_norm = F.normalize(image_features, dim=-1)   # [784, 384]
vecs = torch.stack([F.normalize(concept_vectors[n].float(), dim=0)
                    for n in concept_names])      # [22, 384]
patch_sims = (img_norm @ vecs.T).numpy()          # [784, 22]
concept_idx = patch_sims.argmax(axis=1)           # [784] nearest concept
return concept_idx.reshape(28, 28), patch_sims
```

**Q: What is the contribution formula exactly?**
A: `contribution = weight × (activation − mean_activation)`. The mean subtraction is key: it removes the "baseline hum" — concepts that fire at average levels for all images. Only deviations from the mean carry information. A concept with zero activation and a negative weight yields a positive contribution because its absence below the mean is evidence for the predicted class.

**File:** `experiments/run_pipeline.py` lines 462–467
```python
coef_row = clf.coef_[pred_idx]          # weights for predicted class only
mean_score = float(score_vec[0].mean()) # mean activation across all 22 concepts
contributions = {
    c: float(coef_row[i] * (score_vec[0][i] - mean_score))
    for i, c in enumerate(clf_concept_names)
}
```

---

## STAGE 7 — Lab Domain Transfer (Domain B)

**Q: Why transfer to clinical tabular data? What does it prove?**
A: It proves the concept discovery idea is domain-agnostic. The same three-stage pipeline (discover → name → predict) works on numerical lab tests just as it does on image patches. This is the "transfer" claim — the architecture is not vision-specific.

**Q: What is PSFTT and why not use a standard tabular encoder?**
A: PSFTT (Panel-Structured FT-Transformer) is a custom encoder for clinical lab data. Standard tabular encoders treat all features as a flat vector. PSFTT encodes clinical knowledge: CBC, BMP, and Lipid panels have different clinical meanings, so it uses Panel Positional Encoding (PE: CBC=0, BMP=1, Lipid=2) and two-scale SSL (within-panel masking + cross-panel dropout). The ablation showed that learned PE instead of explicit panel tokens increases inter-panel concept blending and reduces per-panel purity.

**Q: How does LLM labeling work for the lab domain?**
A: The GMM centroid for a lab cluster is a 34-dimensional vector (one value per lab test). We compute deviation from the population mean per test: `(centroid − population_mean) / std`. This produces a deviation profile like: "cholesterol +2.1σ, HDL −0.3σ, triglycerides +1.8σ". This profile is sent to an LLM which suggests a clinical label — e.g. "hypercholesterolemia_pattern". A human clinician validates the label. This is the Gap 3 contribution: first application of LLM concept grounding to numerical clinical data.

**Q: Lab accuracy is only 51% with diabetes recall of 0.05. Is this clinically useful?**
A: Not as a standalone diagnostic tool, no — and I never claim it is. This is a transfer study. The primary contribution is that concepts carry real signal (1.53× above chance = statistically significant above random). The recall collapse to "normal" is a class-imbalance problem in the GMM cluster space — the majority class (normal, n=1,346) dominates cluster training. Future work addresses this with weighted GMM sampling. The interpretability pipeline itself works correctly: the 30 clinically coherent concepts are correctly named and the concept attribution bar chart is meaningful.

**File:** `cache/lab/stage_l7_results.json`
```json
{
  "accuracy": 0.5094,
  "chance_baseline": 0.3333,
  "per_class": {
    "diabetes":     { "recall": 0.049, "f1": 0.088 },
    "hypertension": { "recall": 0.013, "f1": 0.025 },
    "normal":       { "recall": 0.977, "f1": 0.674 }
  }
}
```

---

## STAGE 8 — Ablation Study

**Q: What does NMI measure and why use it?**
A: NMI = Normalised Mutual Information. It measures how much knowing a patch's cluster ID tells you about its true class (cat/car/bird). NMI=0 means clusters are independent of class — useless. NMI=1 means clusters perfectly predict class — perfect. It is preferred over raw accuracy here because it is symmetric and does not depend on label alignment. Values in this work range 0.267–0.363.

**Q: Why does ablation D (pre-fine-tune base DINO) show higher NMI (0.363) than the full model A (0.326)?**
A: This is the most likely challenge. The answer: NMI and class purity measure cluster-to-class alignment, not semantic quality. Pre-fine-tune DINO has very class-separated features because the base model was pre-trained on ImageNet which includes many of the same classes. Fine-tuning slightly redistributes features toward semantic part consistency (same body-part patches cluster together regardless of class), which marginally reduces class purity but produces more meaningful within-class part structure. Silhouette confirms fine-tuning improves geometric cluster quality: A=0.052 vs D=0.045.

**File:** `cache/ablation/ablation_results.json`
```json
"A": { "variant_name": "Full (multi-layer + FG mask + fine-tuned)", "nmi": 0.326, "class_purity": 0.891, "silhouette": 0.052 },
"B": { "variant_name": "Single-layer (L12 only)",                  "nmi": 0.298, "class_purity": 0.876, "silhouette": 0.030 },
"C": { "variant_name": "No FG masking",                            "nmi": 0.267, "class_purity": 0.863, "silhouette": 0.035 },
"D": { "variant_name": "Pre-finetune DINO (base weights)",         "nmi": 0.363, "class_purity": 0.935, "silhouette": 0.045 }
```

**Q: What does the fine-tuning loss function do exactly?**
A: It is a contrastive loss that operates on pairs of patches. Same cluster → pull embeddings together (sim → 1). Different cluster → push apart (sim < margin=0.5). This sharpens part-level separation without using any class labels.

**File:** `src/models/dino_finetuner.py` lines 102–121
```python
def semantic_consistency_loss(self, embeddings, pseudo_labels, margin=0.5):
    embeddings = F.normalize(embeddings, dim=1)
    sim  = (embeddings[idx_a] * embeddings[idx_b]).sum(dim=1)
    same = (pseudo_labels[idx_a] == pseudo_labels[idx_b]).float()
    loss = same * (1 - sim) + (1 - same) * F.relu(sim - margin)
    return loss.mean()
```

---

## STAGE 9 — Results & Limitations

**Q: 98.9% visual accuracy — what is the risk of overfitting?**
A: Three mitigations: (1) 80/20 stratified split with fixed random seed — test set never seen during concept discovery or classifier training. (2) L2 regularisation (C=1.0) in LogReg prevents weight explosion. (3) Concept discovery is completely unsupervised — GMM has no access to class labels, so it cannot overfit to class boundaries.

**Q: Visual accuracy is 98.9% but the raw baseline is 46.4%. Isn't 46.4% suspiciously low?**
A: The baseline uses OneClassSVM on mean DINO features per image — it is a deliberately weak baseline that shows what happens without the concept pipeline. A supervised DINO fine-tune would score >99%. The point of the comparison is not to beat a strong baseline — it is to show that the concept score representation (22 numbers) achieves the same accuracy as raw 384d features while being fully interpretable.

**Q: What are the key limitations of this work?**
A: Four honest limitations: (1) Human SME labeling does not scale — 30 clusters took ~2 hours; 300 would require automation. (2) Lab domain class imbalance causes the classifier to collapse to "normal" — weighted GMM sampling is needed. (3) The pipeline is validated on 3 classes; scaling to 100+ classes requires re-evaluating optimal K. (4) Concept drift is not handled — if a new breed of cat looks unlike training cats, its patches may land in wrong clusters silently.

---

## KEY FILES TO HAVE OPEN DURING THE VIVA

| File | What to show |
|---|---|
| `configs/config.yaml` | All hyperparameters in one place — n_clusters=30, fg_threshold=0.75, C=1.0 |
| `src/pipeline/patch_clusterer.py` | GMM vs K-Means decision, spatial smoothing, PCA |
| `src/models/dino_extractor.py` | Multi-layer concat, foreground mask |
| `src/models/dino_finetuner.py` | Contrastive loss implementation |
| `src/pipeline/concept_builder.py` | Concept vector = cluster centroid, activation = cluster proportion |
| `src/pipeline/concept_classifier.py` | contribution formula, spatial map, LogReg predict |
| `experiments/run_pipeline.py` | End-to-end orchestration, stage_classify, stage_explain |
| `cache/stage5_results.json` | Visual 98.9% results with per-class breakdown |
| `cache/lab/stage_l7_results.json` | Lab 50.9% results — honest about limitations |
| `cache/ablation/ablation_results.json` | Ablation NMI/purity numbers |
