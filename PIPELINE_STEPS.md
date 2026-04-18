# Unsupervised Parts Discovery — Pipeline Steps

## Dataset Sources
- **cat** → Oxford Pets (cats only)
- **car** → Stanford Cars
- **bird** → CUB-200-2011

Raw downloads live in `data/v2/images_core/{train,val}/{cat,car,bird}/`
- train: cat=1920, car=3200, bird=3200
- val:   cat=480,  car=800,  bird=800

---

## Step 1 — Image Curation (run once, produces `data/v2/images/`)

### 1a. Subject-fraction filter (rembg)
Keeps images where the segmented subject occupies ≥70% of frame pixels.

```bash
python scripts/select_top_quality_images.py \
    --input  data/v2/images_core \
    --output data/v2/images \
    --method rembg \
    --min-subject-fraction 0.70 \
    --top-fraction 1.0 \
    --rembg-model u2net \
    --rembg-max-side 512
```

Result: ~1k–2k images per class per split (exact count depends on dataset quality).

### 1b. DINO-guided center crop → 128×128
Uses DINO CLS attention to find the tight bounding box of the subject,
expands by 8% margin, crops to a square, Lanczos-resizes to 128×128.

```bash
python scripts/center_subject_resize_128.py \
    --input  data/v2/images \
    --output data/v2/images_centered_128 \
    --config configs/config.yaml \
    --fg-quantile 0.75
```

### 1c. Bird pose filter (CLIP — sitting only)
Removes flying-bird images using CLIP zero-shot classification.
cat and car images are copied unchanged.

```bash
python scripts/filter_bird_class_sitting_clip.py \
    --input  data/v2/images_centered_128 \
    --output data/v2/images_sitting_birds_only \
    --min-p-sitting 0.55 \
    --device mps
```

### 1d. 3× augmentation
Adds 3 deterministic augmented copies per image (random crop, flip, colour jitter, rotation).
Final count: ~4k images per class.

```bash
python scripts/augment_dataset_x3.py \
    --input    data/v2/images_sitting_birds_only \
    --output   data/v2/images_aug \
    --variants 3 \
    --seed     42
```

### Final dataset path
Point `dino.data_root` in `configs/config.yaml` to:
```
data/v2/images_aug/train      # ~4k images per class
```

---

## Step 2 — Feature Extraction

Reads images from `dino.data_root`, extracts multi-layer DINO patch features,
applies foreground masking (keeps top `100 × (1 - fg_threshold)` % attended patches).

```bash
python experiments/run_pipeline.py --stage extract
```

Key config knobs (`configs/config.yaml`):
| Key | Value used | Effect |
|-----|-----------|--------|
| `dino.model` | `dino_vits8` | ViT-S/8 — 384-dim per layer |
| `dino.use_multilayer` | `true` | concat layers 8,10,12 → 1152-dim |
| `dino.fg_threshold` | `0.75` | keep top 25% attended patches |
| `dino.image_size` | `224` | DINO input resolution |

Output: `cache/dino_features.pt`

---

## Step 3 — Clustering

GMM on PCA-compressed features with spatial position weighting.

```bash
python experiments/run_pipeline.py --stage cluster
```

Key config knobs:
| Key | Value | Effect |
|-----|-------|--------|
| `clustering.n_clusters` | `30` | ~10 concepts per class |
| `clustering.method` | `gmm` | handles unequal cluster sizes |
| `clustering.use_pca` | `true` | 1152 → 128 dims |
| `clustering.use_spatial_features` | `true` | separates eyes vs nose etc. |
| `clustering.use_spatial_smoothing` | `true` | cleans up isolated stray patches |

Output: `cache/kmeans.pkl`, `cache/cluster_labels.pt`

---

## Step 4 — Manual Cluster Labeling

```bash
python -m streamlit run labeling/image_cluster_labeler.py
```

Assign a semantic label to each of the 30 clusters. Mark background/noise
clusters as **excluded** (`include: false`).

Known good labels (from original run):

| Class | Labels |
|-------|--------|
| cat | `cat: nose`, `cat: ear`, `cat: face`, `cat: leg`, `cat: fur`, `cat: eye`, `cat: moustach`, `cat: body_fur` |
| car | `car: side_fender`, `car: wheel`, `car: windshield`, `car: wheel_arch`, `car: bumpe_corner`, `car: headlight` |
| bird | `bird: neck`, `bird: tail`, `bird: feather`, `bird: wings`, `bird: beak`, `bird: eye`, `bird: leg` |
| noise | mark **exclude** |

Output: `cache/labels.json`

---

## Step 5 — Concept Vectors + Scores

```bash
python experiments/run_pipeline.py --stage concepts
```

Builds mean feature vector per labeled cluster (concept vector),
then scores each image by patch-assignment proportion per concept.

Output: `cache/concept_vectors.pt`, `cache/concept_scores.pt`

---

## Step 6 — Classification

```bash
python experiments/run_pipeline.py --stage classify
```

Logistic regression on concept activation scores → 3-class (cat / car / bird).

Output: `cache/concept_classifier.pkl`

---

## Full Re-run From Scratch (after data loss)

```bash
# 1. Recreate image dataset (Steps 1a–1d above)

# 2. Run pipeline
python experiments/run_pipeline.py --stage extract
python experiments/run_pipeline.py --stage cluster

# 3. Label clusters in Streamlit GUI
pkill -f streamlit
python -m streamlit run labeling/image_cluster_labeler.py

# 4. Finish pipeline
python experiments/run_pipeline.py --stage concepts
python experiments/run_pipeline.py --stage classify
```

## Python Environment
```bash
/Users/bhoopeshnandansingh/.venv/bin/python3
# or activate with: source /Users/bhoopeshnandansingh/.venv/bin/activate
```
