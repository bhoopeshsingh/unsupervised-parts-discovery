# MSc Dissertation — Master Plan
## "Learning Interpretable Feature Spaces via Domain-Informed Encoders"

> **NORTH STAR — Never deviate from this.**
> Every design decision, experiment, and result must trace back to the one problem statement below.

---

## The One Problem Being Solved

> **"Existing interpretable ML methods for clinical data require concepts to be defined before or during training. This work builds a pipeline that discovers concepts unsupervised, grounds them semantically through domain knowledge, and produces interpretable predictions — without requiring concept labels at training time."**

---

## Three Focused Gaps (from Literature)

| # | Paper | Their Stated Gap | Our Response |
|---|---|---|---|
| Gap 1 | TCAV (Kim, ICML 2018) & CBM (Koh, ICML 2020) | Require concept labels before/during training | Unsupervised-first discovery — concepts emerge from data, labeled post-hoc |
| Gap 2 | DINO (Caron, NeurIPS 2021) & FT-Transformer (Gorishniy, NeurIPS 2021) | Intermediate layer representations not analyzed | Multi-layer patch extraction (image) + panel-patch extraction (lab) |
| Gap 3 | ClusterLLM (EMNLP 2023) | LLM concept guidance restricted to textual data | LLM-mediated clinical concept naming from numerical deviation profiles |

---

## Three Design Pillars (directly answering the three gaps)

```
Pillar 1 — Unsupervised-First Concept Discovery        (answers Gap 1)
  GMM clustering on encoder features, no concept labels needed at any stage.

Pillar 2 — Domain-Informed Encoder                     (answers Gap 2)
  Three channels of domain knowledge:
    Channel A — Architecture:  multi-layer DINO / panel-structured PSFTT
    Channel B — Loss function: semantic consistency fine-tuning / two-scale SSL
    Channel C — Labeling:      human SME (image) / LLM-mediated (lab)

Pillar 3 — Post-Hoc Semantic Grounding                 (answers Gap 3)
  Image domain: human SME inspects patch clusters → names each concept
  Lab domain:   LLM reads deviation profile → suggests clinical label → human validates
```

---

## Two-Domain Architecture (same 3-pillar structure applied to both)

### Domain A — Visual (Image Pipeline)
| Stage | Component | File | Status |
|---|---|---|---|
| Extract | DINO ViT-S/8, multi-layer (L8+L10+L12), FG masking | `src/models/dino_extractor.py` | ✅ Done |
| Fine-tune | Semantic consistency loss (last 2 blocks) | `src/models/dino_finetuner.py` | ✅ Done |
| Cluster | GMM, 30 clusters, spatial features, PCA 1152→128 | `src/pipeline/patch_clusterer.py` | ✅ Done |
| **Label** | **Human SME labels patch clusters via Streamlit UI** | **`labeling/image_cluster_labeler.py`** | ⏳ After balanced data |
| Concepts | Concept vectors + per-image activation scores | `src/pipeline/concept_builder.py` | ❌ Pending labeling |
| Classify | Concept-score-based classifier | `src/pipeline/concept_classifier.py` | ❌ Pending concepts |
| Explain | Part map + contribution bar chart (3-panel figure) | `src/pipeline/concept_classifier.py` | ❌ Pending classify |

### Domain B — Tabular (Lab Pipeline)
| Stage | Component | File | Status |
|---|---|---|---|
| Load | NHANES 18,673 records, 34 lab features | `src/data/lab_loader.py` | ✅ Done |
| Extract | Clinical deviation encoding | `src/models/lab_extractor.py` | ✅ Done |
| **Pretrain** | **PanelFTTransformer, two-scale SSL** | **`src/models/panel_ft_transformer.py`** | ❌ Not run |
| Encode | Panel-patches [N × 3 × 384] from layers L2/L4/L6 | `experiments/run_lab_pipeline.py` | ❌ Pending pretrain |
| Cluster | Per-panel GMM (3 panels × 10 clusters = 30) | `experiments/run_lab_pipeline.py` | ❌ Pending encode |
| **Label** | **LLM suggests clinical label → human validates** | **`labeling/lab_cluster_labeler.py`** | ❌ Pending cluster |
| Concepts | Concept vectors + scores | `experiments/run_lab_pipeline.py` | ❌ Pending labeling |
| Classify | Concept-based classifier | `experiments/run_lab_pipeline.py` | ❌ Pending concepts |

---

## Execution Phases

| Phase | Planned | Built | Test | Claim |
|---|---|---|---|---|
| 1. Encoder | Domain-informed feature extraction | DINO multi-layer + PSFTT panel PE | Ablation variants B, C, D | Intermediate layers carry concept structure (Gap 2) |
| 2. Discovery | Unsupervised clustering | GMM on patch / panel-patch features | Cluster quality metrics | Concepts found without supervision (Gap 1) |
| 3. Grounding | Post-hoc semantic labeling | `image_cluster_labeler.py` / `lab_concept_advisor.py` | LLM confidence vs shuffled baseline | Domain knowledge injected post-hoc (Gap 3) |
| 4. Classification | Concept-based predictions | Concept vectors + SVM / LogReg | Accuracy: concept scores vs raw features | Interpretability without sacrificing accuracy |
| 5. Explanation | Auditable outputs | Part map + contribution bar chart | Human can trace prediction to named concept | Supports clinical audit requirement |

---

## Evaluation Strategy

### Ablation Table (one row per design decision)

| Variant | Change | Metric | Gap Evidence |
|---|---|---|---|
| A — Full | Baseline (all pillars active) | — | — |
| B — Single-layer | Remove intermediate layer concat | Silhouette, NMI, Class Purity ↓ | DINO Gap 2 |
| C — No FG masking | Remove attention-guided masking | Silhouette, NMI, Class Purity ↓ | ACE Gap 1 |
| D — Pre-finetune | Remove semantic fine-tuning | Silhouette, NMI, Class Purity ↓ | Pillar 2 Channel B |

Script: `experiments/run_ablations.py`

### Metrics Computed Per Variant
- **Silhouette Score** — cluster compactness & separation [-1, 1] ↑
- **Class Purity** — dominant class fraction per cluster [0, 1] ↑
- **NMI** — alignment of cluster assignments with true class labels [0, 1] ↑
- **Intra/Inter ratio** — distance within vs between clusters ↓
- **Labeling Confidence** — human/LLM confidence score per cluster (collected during labeling)

### LLM Labeling Validation (Gap 3 evidence)
- Run LLM advisor on real cluster centroids → record acceptance rate
- Run LLM advisor on **shuffled** centroids (random features) → expect low confidence
- Delta in acceptance rate = evidence that LLM concept naming is signal, not noise

### Classification Comparison
- Concept-vector SVM vs raw-feature SVM
- If accuracy within 5% AUC: interpretability is essentially free
- If accuracy gap > 5%: report honest trade-off table

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| GMM clusters not semantically coherent | Spatial smoothing + FG masking + multi-layer features pre-improve cluster quality before labeling |
| LLM labels clinically incorrect | Human clinician validation step; LLM provides DIFFERENTIALS and SME_NOTE for reviewer |
| Concept classifier underperforms raw features | Explicitly tested — inconclusive result is reported honestly as interpretability trade-off |
| Dataset imbalance biases concepts | Balanced download: 4000/4000/4000 (cat/car/bird) — in progress |
| Lab pipeline not executable in time | CPU-runnable config; MIMIC-IV external validation deferred to future work if needed |
| Ablations show no improvement | Inconclusive result is a valid finding — report and discuss in limitations chapter |

---

## Realistic Expected Outcomes

| Claim | Realistic Expectation | Fallback |
|---|---|---|
| Multi-layer > single-layer | Silhouette +5–15%, NMI improves | Report delta honestly; DINO paper confirms this was unexplored |
| FG masking improves purity | Class purity +5–10% | Qualitative attention map figures support the argument |
| LLM labels clinically valid | >70% accepted without modification by reviewer | Report acceptance rate as metric, include examples |
| Concept accuracy ≈ raw accuracy | Within 5% AUC | Explicit interpretability–accuracy trade-off table |
| Lab concepts align with known conditions | Named clusters match known clinical patterns | Qualitative alignment table (cluster name vs ICD code) |

---

## Dissertation Abstract (locked)

> "Concept-based explanation methods require domain experts to specify concepts before model training, limiting their use in clinical settings where concept definitions are costly to obtain. This dissertation presents an unsupervised concept discovery pipeline that learns interpretable clinical concepts from unlabeled data, grounds them semantically through human visual inspection (image domain) and LLM-mediated clinical labeling (tabular domain), and produces predictions traceable to named semantic concepts. Ablation experiments demonstrate that domain-informed encoder design choices — multi-layer feature extraction and panel-structured positional encoding — improve concept cluster quality, and that the resulting concept-based classifiers match the accuracy of black-box baselines while producing auditable explanations."

---

## Dissertation Chapter Map

| Chapter | Title | Content |
|---|---|---|
| 1 | Introduction | Problem statement, clinical motivation, 3 gaps, dissertation objectives |
| 2 | Literature Review | TCAV, CBM, ACE, DINO, FT-Transformer, Lab-MAE, ClusterLLM — gaps stated precisely |
| 3 | System Design | 3 pillars, 2 domains, architecture diagrams, design decisions traced to gaps |
| 4 | Implementation | Code walkthrough per stage, key design choices, domain-knowledge channels |
| 5 | Evaluation | Ablation table, LLM labeling validation, classification comparison, qualitative examples |
| 6 | Discussion | What results mean, what gaps are addressed, honest limitations |
| 7 | Conclusion | Summary, what was achieved, future work (MIMIC-IV, larger models) |

---

## Key Rule

> For every result section, use this structure:
> - *"Paper X identified gap Y as an open problem."*
> - *"We designed component Z to address it."*
> - *"Experiment E tested whether Z closes gap Y."*
> - *"Result R shows [metric], which [supports / partially supports / leaves open] the claim."*

Saying "partially supports" or "leaves open" is **more credible** than overclaiming.
