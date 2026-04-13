# Unsupervised Parts Discovery — Dissertation Pipeline

> **Dissertation**: *Learning Interpretable Feature Spaces via Domain-Informed Encoders*
> **Author**: Bhoopesh Singh
> **Branch**: `lab-uses-extension` (active development)

This repository implements a two-phase unsupervised concept discovery system that proves a single core claim across two domains:

> **"Concepts discovered without supervision predict clinical diagnoses they were never trained on."**

- **Phase 1 (Images)**: DINO ViT discovers visual parts (cat_ear, car_wheel, bird_beak) from unlabeled image patches → LogReg on concept scores achieves **98.9% accuracy** classifying cat/car/bird.
- **Phase 2 (Clinical Lab)**: PanelFTTransformer discovers clinical patterns (metabolic_syndrome, pre_diabetes, dyslipidemia) from NHANES lab panels → LogReg on concept scores achieves **50.9% accuracy** (vs 33.3% chance) predicting physician-confirmed diabetes/hypertension.

For full technical design, architecture, and dissertation alignment:
→ **[TECHNICAL_DESIGN.md](TECHNICAL_DESIGN.md)**

---

## Verified Results (from `cache/`)

| Claim | File | Value |
|---|---|---|
| Image concept classifier accuracy | `cache/stage5_results.json` | **98.9%** (vs 46.4% OneClassSVM baseline) |
| Image concept classifier AUC | `cache/stage5_results.json` | **0.9996** (vs 0.72 baseline) |
| Image concept classifier macro F1 | `cache/stage5_results.json` | **0.9888** |
| Lab classifier accuracy | `cache/lab/stage_l7_results.json` | **50.9%** (vs 33.3% chance) |
| Lab concept silhouette (full variant A) | `cache/ablation/ablation_results.json` | **0.0521** |
| Lab cluster enrichment | `cache/lab/cluster_enrichment.csv` | Up to **5.6× diabetes enrichment** |

Claims **NOT yet verified** (removed from dissertation pending evidence):
- ~~r=0.94 silhouette–semantic correlation~~ → actual r=0.56, not significant (p=0.44)
- ~~82% human semantic accuracy~~ → human evaluation not completed; run `--write-semantic-eval`

---

## Quick Start

```bash
pip install -r requirements.txt

# Generate all dissertation result JSON files (Priorities 1–4)
python experiments/generate_dissertation_results.py

# Skip the slow 10 GB DINO load (Priority 1) if already done
python experiments/generate_dissertation_results.py --skip-stage5

# After manually evaluating 50 random patches in the Streamlit labeller:
python experiments/generate_dissertation_results.py --write-semantic-eval 41
```

---

## Phase 1 — Image Pipeline (cat / car / bird)

Run stages in order (first time):

```bash
# S1+S2: Extract multi-layer DINO patch features + FG masking
# (~30 min for 12,000 images on M4 Mac, saves 10.8 GB to cache/dino_features.pt)
python experiments/run_pipeline.py --stage extract

# S3: Cluster patches into 30 semantic parts via GMM
python experiments/run_pipeline.py --stage cluster

# S4 (optional): Fine-tune last 2 DINO blocks to improve cluster separation
python experiments/run_pipeline.py --stage finetune
python experiments/run_pipeline.py --stage extract  # re-extract with improved weights
python experiments/run_pipeline.py --stage cluster  # re-cluster

# S5: Label clusters via Streamlit UI — assign names like cat_ear, car_wheel
streamlit run labeling/image_cluster_labeler.py

# S6: Build concept vectors + compute per-image concept activation scores
python experiments/run_pipeline.py --stage concepts

# S7: Train LogReg on concept scores → classify cat / car / bird
python experiments/run_pipeline.py --stage classify

# S8 (optional): Generate 3-panel explanation for a single image
python experiments/run_pipeline.py --stage explain --image data/cat/001.png
```

### Ablation Study (4 variants)

```bash
# Run all 4 ablation variants (A/B/C/D) — extracts features for each
python experiments/run_ablations.py

# Skip re-extraction if features already cached, recompute metrics only
python experiments/run_ablations.py --skip_extract

# Variant C only (memory-constrained: subsamples 2.4M of 9.4M patches)
python experiments/run_variant_C_subsampled.py
```

Ablation results are saved to `cache/ablation/ablation_results.json`.

---

## Phase 2 — Lab Pipeline (diabetes / hypertension / normal)

Data source: **NHANES 2017–2018** (CDC public dataset, 1,536 records, 34 lab tests across 3 panels).

```bash
# Download NHANES XPT files first:
python experiments/run_lab_pipeline.py --stage download

# Run entire pipeline end-to-end:
python experiments/run_lab_pipeline.py --stage all

# Or run individual stages:
python experiments/run_lab_pipeline.py --stage load      # L1: merge XPT panels
python experiments/run_lab_pipeline.py --stage extract   # L2: clinical deviation encoding
python experiments/run_lab_pipeline.py --stage pretrain  # L3: two-scale SSL (50 epochs)
python experiments/run_lab_pipeline.py --stage encode    # L4: panel-patch extraction
python experiments/run_lab_pipeline.py --stage cluster   # L5: per-panel GMM clustering

# Label clusters via Streamlit UI (LLM-assisted: Ollama or Claude API)
streamlit run labeling/lab_cluster_labeler.py

python experiments/run_lab_pipeline.py --stage concepts  # L6: concept vectors
python experiments/run_lab_pipeline.py --stage classify  # L7: LogReg → diabetes/hypertension
python experiments/run_lab_pipeline.py --stage validate  # L8: cluster enrichment analysis
```

### Lab Cluster Metrics

```bash
# Compare structured vs shuffled panel order (table in dissertation)
python experiments/compute_lab_cluster_metrics.py --tag shuffled --compare results.json
```

---

## Project Structure

```
src/
  models/
    dino_extractor.py         DINO ViT-S/8 feature extraction (single & multi-layer)
    dino_finetuner.py         Semantic consistency loss fine-tuning (last 2 blocks)
    lab_extractor.py          Clinical deviation encoding (LabExtractor)
    panel_ft_transformer.py   PanelFTTransformer — two-scale SSL pre-training
  pipeline/
    patch_clusterer.py        GMM / MiniBatchKMeans with PCA + spatial features
    concept_builder.py        Concept vectors + per-image activation scores
    concept_classifier.py     LogReg linear probe + 3-panel explanation render
    lab_concept_advisor.py    LLM-assisted cluster labeling (Ollama / Claude)
  data/
    lab_loader.py             NHANES XPT merge + validation column extraction
experiments/
  run_pipeline.py             Phase 1 orchestrator (stages: extract/cluster/finetune/…)
  run_lab_pipeline.py         Phase 2 orchestrator (stages: L1–L8)
  run_ablations.py            Ablation study: variants A, B, C, D
  run_variant_C_subsampled.py Memory-safe Variant C evaluation (mmap + 2.4M subsample)
  compute_lab_cluster_metrics.py  Silhouette/NMI for structured vs shuffled panels
  generate_dissertation_results.py  Generate all result JSON files (Priorities 1–4)
labeling/
  image_cluster_labeler.py    Streamlit UI for Phase 1 cluster labeling
  lab_cluster_labeler.py      Streamlit UI for Phase 2 (LLM-assisted)
configs/
  config.yaml                 Phase 1 hyperparameters (DINO, clustering, classification)
  config_lab.yaml             Phase 2 hyperparameters (PanelFT, panels, SSL)
cache/                        Generated artefacts (gitignored except JSON results)
  dino_features.pt            10.8 GB — 2.35M patches × 1152d (multi-layer L8+L10+L12)
  concept_scores.pt           [12000 images × 22 concepts]
  ablation/
    ablation_results.json     Silhouette, NMI, class purity for variants A–D
    {A,B,C,D}_features.pt     Per-variant feature caches
  lab/
    panel_patches.pt          [1536 records × 3 panels × 384d]
    concept_scores.pt         [1536 records × 30 concepts]
    cluster_enrichment.csv    Per-cluster diabetes/hypertension enrichment ratios
    classifier.pkl            Fitted LogReg (diabetes/hypertension/normal)
  stage5_results.json         Table 11: concept-based vs raw baseline (generated)
  semantic_correlation.json   Silhouette–semantic Pearson r (generated)
  semantic_eval.json          Human evaluation placeholder (manual step required)
  lab/stage_l7_results.json   Lab L7 accuracy + per-class metrics (generated)
```

---

## Key Hyperparameters

### Phase 1 — Image (`configs/config.yaml`)

| Parameter | Value | Purpose |
|---|---|---|
| `dino.model` | `dino_vits8` | ViT-S/8, 21M params, 384-dim patch features |
| `dino.use_multilayer` | `true` | Concat L8+L10+L12 → 1152d per patch |
| `dino.fg_threshold` | `0.5` | Keep top 50% CLS-attended patches |
| `clustering.n_clusters` | `30` | 10 per class (cat/car/bird) |
| `clustering.method` | `gmm` | GMM handles unequal part sizes |
| `clustering.use_pca` | `true` | 1152 → 128d before clustering |
| `finetune.n_epochs` | `3` | Last 2 DINO blocks only |
| `classification.test_size` | `0.2` | 80/20 stratified split |

### Phase 2 — Lab (`configs/config_lab.yaml`)

| Parameter | Value | Purpose |
|---|---|---|
| `transformer.d_model` | `128` | PanelFTTransformer hidden dim |
| `transformer.n_layers` | `6` | Transformer depth |
| `transformer.extract_layers` | `[1, 3, 5]` | Multi-granularity panel patches |
| `transformer.n_clusters_per_panel` | `10` | GMM clusters per CBC/BMP/Lipid panel |
| `clustering.pca_dims` | `20` | PCA before per-panel GMM |
| `classification.C` | `1.0` | LogReg regularisation |

---

## Hardware Notes

Tested on **Apple M4 Mac (MPS)** and **CPU**. Change `device: mps` → `cuda` for NVIDIA GPU.

| Stage | Approx Time (M4 Mac) |
|---|---|
| Phase 1 feature extraction (12k images) | ~30 min |
| Phase 1 clustering (2.35M patches) | ~10 min |
| Phase 1 DINO fine-tuning (3 epochs) | ~45 min |
| Phase 2 SSL pre-training (50 epochs) | ~20 min |
| Generate dissertation result JSONs | ~5 min (+10 min for Stage 5 with 10 GB load) |
