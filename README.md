# Unsupervised Parts Discovery — DINO Pipeline

> **Dissertation**: Learning Interpretable Feature Spaces via Domain-Informed Encoders
> **Application**: Explainable classification for medical image analysis (positive/negative cases)
> **Proof-of-concept domain**: Cat (positive) vs non-cat (negative) on masked image dataset

For full technical design, implementation internals, and dissertation alignment:
→ **[TECHNICAL_DESIGN.md](TECHNICAL_DESIGN.md)**

---

## Quick Start

```bash
cd /Users/dev/ml-projects/unsupervised-parts-discovery
pip install -r requirements.txt
```

## Pipeline (run in order)

```bash
# 1. Extract DINO patch features (~15-20 min for 1000 images on M4 Mac)
python experiments/run_pipeline.py --stage extract

# 2. Cluster patches into semantic parts
python experiments/run_pipeline.py --stage cluster

# 3. [Optional] Fine-tune DINO encoder — improves cluster separation
python experiments/run_pipeline.py --stage finetune
python experiments/run_pipeline.py --stage extract   # re-extract
python experiments/run_pipeline.py --stage cluster   # re-cluster

# 4. Label clusters via Streamlit UI
streamlit run labeling/label_tool.py

# 5. Build concept vectors + compute activation scores
python experiments/run_pipeline.py --stage concepts

# 6. Train one-class classifier (cat vs not-cat)
python experiments/run_pipeline.py --stage classify

# 7. Explain a prediction
python experiments/run_pipeline.py --stage explain --image data/masked/train/cat/001.png
```

## Project Structure

```
src/
  models/        dino_extractor.py, dino_finetuner.py
  pipeline/      patch_clusterer.py, concept_builder.py,
                 concept_classifier.py, one_class_classifier.py
  data/          data_classes.py, loaders.py, prepare_data.py
  utils/         __init__.py (load_config, set_seed, get_device)
experiments/
  extract_features.py   — batch DINO extraction with foreground masking
  find_optimal_k.py     — elbow + silhouette analysis for K selection
  run_pipeline.py       — orchestrates all stages
labeling/
  label_tool.py         — Streamlit UI (Label Clusters + Classify & Explain tabs)
scripts/
  validate_clusters.py  — cluster quality metrics
configs/
  config.yaml           — all hyperparameters
cache/                  — generated artefacts (gitignored)
```

## Key Hyperparameters (configs/config.yaml)

| Parameter | Value | Purpose |
|---|---|---|
| `dino.fg_threshold` | 0.6 | Keep top 40% attended patches (foreground masking) |
| `clustering.n_clusters` | 10 | Number of semantic part clusters |
| `clustering.method` | gmm | Gaussian Mixture Model (handles unequal part sizes) |
| `clustering.spatial_weight` | 0.4 | Weight of patch position in clustering |
| `clustering.use_pca` | true | 384 → 64 dims before clustering |
| `finetune.n_epochs` | 3 | Epochs for semantic consistency fine-tuning |
| `finetune.lr` | 1e-5 | Learning rate (only last 2 DINO blocks) |

## Hardware

Configured for **M4 Mac (MPS)**. Change `device: mps` → `cuda` for NVIDIA GPU.
- Feature extraction: ~15-20 min / 1000 images
- Clustering: ~2-5 min
- Fine-tuning: ~30 min (3 epochs)
- Explain single image: ~5 sec
