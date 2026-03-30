# Pipeline Architecture Analogy — Image ↔ Lab
## For dissertation report generation and co-worker reference

---

> **Dissertation Core Claim**
>
> *"Concepts discovered without supervision predict clinical diagnoses they were never trained on."*
>
> Both pipelines prove different facets of this claim using the **same 3-pillar architecture**
> applied to two entirely different data modalities. The image pipeline establishes the
> architecture's validity on a well-understood domain (visual parts). The lab pipeline
> then transfers the identical design to clinical tabular data — proving the architecture
> is domain-general, not domain-specific.

---

## The Analogy, Side by Side

```
IMAGE PIPELINE                              LAB PIPELINE
─────────────────────────────────────────   ──────────────────────────────────────────────
Data    : Oxford Pets / Stanford Cars /     Data    : NHANES 2013–2018
          CUB-200-2011                                18,673 patient records
          ~14,000 images, 3 classes                   34 lab test values per record
          Stored: 128×128 px                          3 clinical panels (CBC/BMP/Lipid)

Input   : 128×128 px image on disk          Input   : flat row of 34 numbers
          → DINO upscales to 224×224                   [WBC=6.2, Hgb=13.1, Glucose=105...]
          (Resize(256) → CenterCrop(224))              → encoded as clinical deviation
                                                        (deviation from normal range)

─────────────────────────────────────────   ──────────────────────────────────────────────
STEP 1: TOKENISE                            STEP 1: TOKENISE
─────────────────────────────────────────   ──────────────────────────────────────────────
Model   : DINO ViT-S/8                      Model   : PanelFTTransformer
File    : src/models/dino_extractor.py      File    : src/models/panel_ft_transformer.py
Library : torch.hub (facebookresearch/dino) Library : PyTorch nn.Module (custom)

What happens:                               What happens:
  ViT divides 224×224 image into              Each of the 34 lab tests becomes
  28×28 = 784 spatial patches.               its own vector token via:
  Each 8×8 pixel block = one patch.
                                               T_i = x_i * W_i + b_i
  Patch position encoded by                     ↑       ↑    ↑   ↑
  learned 2D positional embedding:            token  scalar weight bias
  patch(row=3, col=7) knows its               dim-d  value (d-dim)(d-dim)
  location in the image grid.
                                             Where x_i = deviation of test i
                                             from its clinical normal range,
                                             W_i ∈ R^d and b_i ∈ R^d are
                                             learned parameters unique to
                                             test i (no weight sharing).

                                             Panel membership encoded by:
                                               token_i += E_panel[panel(i)]
                                             where E_panel is a learned
                                             nn.Embedding(3, d_model).
                                             CBC tests get panel_id=0,
                                             biochem=1, lipid=2.

Key design:                                 Key design:
  Each patch knows WHERE it is              Each token knows WHICH PANEL
  (spatial position in image grid)          it belongs to (clinical system)

  → positional encoding = xy grid           → positional encoding = panel identity

Code reference:                             Code reference:
  self.model.get_intermediate_layers()        class FeatureTokenizer(nn.Module):
  — built into Facebook DINO                   def forward(self, x):
                                                 return (x.unsqueeze(-1)
                                                         * self.weight.unsqueeze(0)
                                                         + self.bias.unsqueeze(0))

─────────────────────────────────────────   ──────────────────────────────────────────────
STEP 2: TRANSFORMER LAYERS LEARN CONTEXT    STEP 2: TRANSFORMER LAYERS LEARN CONTEXT
─────────────────────────────────────────   ──────────────────────────────────────────────
Model   : ViT-S/8, 12 transformer blocks    Model   : 6 TransformerEncoderLayer blocks
          d_model=384, heads=6                        d_model=128, heads=4
          Pre-trained SSL (DINO objective)            Pre-trained SSL (two-scale, custom)

What each layer learns:                     What each layer learns:

  Layer  8 → local texture                   Layer 1 → within-test correlations
    patch sees its own pixel pattern            token sees its own deviation value
    fur texture, wheel edge, feather            WBC is aware of its own magnitude
    detail — very local information             co-deviations starting to form
                                                (neutrophils ↔ lymphocytes)

  Layer 10 → structural shapes               Layer 3 → panel-level patterns
    patch starts integrating context            tokens across same panel interact
    from neighbouring patches                   fully (14 CBC tests talk to each
    shape of ear, outline of tyre,              other); panel-level physiological
    silhouette of wing emerges                  pattern captured — e.g. iron
                                                deficiency anemia signature
                                                (low Hgb + low MCV + high RDW)

  Layer 12 → semantic identity               Layer 5 → cross-panel systemic patterns
    patch knows what object part               tokens span all 3 panels now
    it belongs to: "I am part of              glucose in biochem attends to
    a cat's face" vs "I am part               triglycerides in lipid panel
    of a car wheel"                            CLS token captures whole-body
                                               physiological state — metabolic
                                               syndrome, systemic inflammation

SSL pre-training:                           SSL pre-training:
  DINO contrastive self-distillation           Two-scale SSL (custom, novel):
  — student / teacher ViT pair                  Scale 1: within-panel masking
  — no labels used                               mask 40% of tests in one panel,
  — forces each patch to be                      reconstruct original deviations
    discriminative of its context                → early layers learn test
  Library: facebook DINO (pretrained)            co-deviation patterns

                                              Scale 2: cross-panel dropout
                                                drop entire panel, CLS token
                                                predicts dropped panel mean
                                                → late layers learn inter-panel
                                                physiological dependencies
                                              Library: PyTorch AdamW + CosineAnnealingLR

─────────────────────────────────────────   ──────────────────────────────────────────────
STEP 3: EXTRACT MULTI-GRANULARITY PATCHES   STEP 3: EXTRACT MULTI-GRANULARITY PANEL-PATCHES
─────────────────────────────────────────   ──────────────────────────────────────────────
File    : src/models/dino_extractor.py      File    : src/models/panel_ft_transformer.py
Method  : extract_patches()                 Method  : extract_panel_patches()
Stage   : run_pipeline.py stage_extract     Stage   : run_lab_pipeline.py stage_encode

What happens:                               What happens:
  For each image, take the hidden             For each patient record, take hidden
  states at layers 8, 10, 12.               states at layers 1, 3, 5.

  For each of 784 spatial patches:           For each of 3 panels (CBC/biochem/lipid):
    h_8   = hidden state at layer 8          h_1 = mean of all CBC tokens at layer 1
    h_10  = hidden state at layer 10         h_3 = mean of all CBC tokens at layer 3
    h_12  = hidden state at layer 12         h_5 = mean of all CBC tokens at layer 5

    patch_vec = concat(h_8, h_10, h_12)     panel_patch = concat(h_1, h_3, h_5)
              = 384 + 384 + 384                          = 128 + 128 + 128
              = 1152-dim per patch                       = 384-dim per panel-patch

  Per image: 784 patches × 1152-dim         Per record: 3 panel-patches × 384-dim
  Total dataset: ~1.7M patch vectors         Total dataset: 56,019 panel-patch vectors
                                                            (18,673 records × 3 panels)

Code reference:                             Code reference:
  all_layers = model.get_intermediate         _, intermediates = self.forward(x)
               _layers(image_tensor, n=5)     for p in range(self.n_panels):
  feats = torch.cat([                           feat_idx = (panel_ids == p)
    all_layers[i][:, 1:, :]                     panel_mean = h[:, feat_idx, :].mean(1)
    for i in (0, 2, 4)                        panel_patch = torch.cat(layer_vecs, -1)
  ], dim=-1)   # [B, 784, 1152]

Foreground masking:                         Clinical deviation encoding (input preprocessing):
  DINO CLS attention highlights               Raw lab values encoded as deviation
  foreground object; patches below            from clinical normal range before
  fg_threshold=0.75 are discarded.            entering tokenizer:
  This removes background sky,                  dev_i = 0               if in normal range
  road, plain wall patches.                    dev_i = (x - high) / σ   if above normal
  Library: torch + DINO attention              dev_i = (low - x) / σ    if below normal
  File: extract_features.py                  File: src/models/lab_extractor.py

─────────────────────────────────────────   ──────────────────────────────────────────────
STEP 4: CLUSTER PATCHES → CONCEPTS          STEP 4: CLUSTER PANEL-PATCHES → CONCEPTS
─────────────────────────────────────────   ──────────────────────────────────────────────
File    : src/pipeline/patch_clusterer.py   File    : src/pipeline/patch_clusterer.py
Stage   : run_pipeline.py stage_cluster     Stage   : run_lab_pipeline.py stage_cluster
Library : sklearn GaussianMixture           Library : sklearn GaussianMixture (same class)

What happens:                               What happens:
  1. PCA: 1152-dim → 128-dim                  1. PCA: 384-dim → 20-dim
     (retains ~90% variance)                     (retains ~90% variance)

  2. L2-normalise patch vectors               2. L2-normalise panel-patch vectors

  3. GMM with 30 components                   3. Per-panel GMM with 10 components each
     (global clustering across all               CBC:     GMM(10) on 18,673 CBC patches
     784 patches of all 14,000 images)           biochem: GMM(10) on 18,673 biochem patches
                                                 lipid:   GMM(10) on 18,673 lipid patches

  4. Each patch assigned to one of            4. Each panel-patch assigned to one of
     30 concept clusters                        10 concept clusters per panel

  Clusters discovered:                       Clusters discovered:
    cat: nose, cat: eye, cat: ear              cbc: 0..9  (e.g. anaemia pattern)
    cat: fur, bird: wing, bird: beak           biochem: 0..9 (e.g. diabetic pattern)
    car: wheel, car: headlight ...             lipid: 0..9 (e.g. dyslipidaemia)

  Spatial smoothing applied:                 No spatial smoothing (tabular data has
  majority-vote over 3×3 patch grid           no neighbour structure)
  to remove isolated stray patches.

─────────────────────────────────────────   ──────────────────────────────────────────────
STEP 5: LABEL CLUSTERS (CONCEPT NAMING)     STEP 5: LABEL CLUSTERS (CONCEPT NAMING)
─────────────────────────────────────────   ──────────────────────────────────────────────
File    : labeling/label_tool.py            File    : labeling/lab_label_tool.py
Library : Streamlit GUI                     Library : Streamlit GUI

Method  : HUMAN-PRIMARY                     Method  : LLM-MEDIATED
  SME inspects visual patch grid              LLM reads cluster centroid deviation
  for each cluster.                          profile (which tests are high/low).
  "These patches all show cat eyes"          "Pattern: Hgb↓ MCV↓ RDW↑ MCH↓ →
  → label: cat: eye                          iron deficiency anaemia"
                                             Human validates, not names.
  Labels saved:                              Labels saved:
    cache/labels.json                          cache/lab/labels.json
    {"15": {"label": "cat: eye", ...}}         {"cbc:3": {"label": "iron_def_anaemia"}}

  Key format: cluster_id (int)               Key format: panel:cluster_id (string)

─────────────────────────────────────────   ──────────────────────────────────────────────
STEP 6: CONCEPT ACTIVATION SCORES          STEP 6: CONCEPT ACTIVATION SCORES
─────────────────────────────────────────   ──────────────────────────────────────────────
File    : src/pipeline/concept_builder.py   File    : run_lab_pipeline.py stage_concepts

What happens:                               What happens:
  For each image:                             For each patient record:
    score_c = fraction of image's              score_c = soft GMM membership
              784 patches that belong                   probability for each
              to concept cluster c                      of the 30 concept clusters

  Result: [N_images, 30] matrix               Result: [N_records, 30] matrix
    each row = concept activation               each row = concept activation
    profile for one image                       profile for one patient

  Interpretation:                             Interpretation:
    Image with score(bird:wing)=0.42            Patient with score(biochem:CKD)=0.81
    has 42% of its patches in the              has 81% soft membership in the
    bird:wing cluster                          CKD pattern cluster

─────────────────────────────────────────   ──────────────────────────────────────────────
STEP 7: CLASSIFY ON CONCEPT SCORES          STEP 7: CLASSIFY ON CONCEPT SCORES
─────────────────────────────────────────   ──────────────────────────────────────────────
File    : run_pipeline.py stage_classify    File    : run_lab_pipeline.py stage_classify
Library : sklearn LogisticRegression        Library : sklearn LogisticRegression (same)

Input : concept scores [N, 30]              Input : concept scores [N, 30]
Labels: true class (cat/car/bird)           Labels: physician-confirmed diagnosis
         from folder structure                       DIQ010=1 → diabetes
         — available throughout                     BPQ020=1 → hypertension
                                                     neither  → normal
                                                     — NEVER used in any prior stage

Prediction target:                          Prediction target:
  What class is this image?                   What condition does this patient have?

  Cat/car/bird split is known from            Diabetes/hypertension/normal derived
  dataset folder structure — used             from NHANES questionnaire files
  as ground-truth class label                 (DIQ, BPQ) loaded at stage_load.
                                              These files were never touched by
                                              pretrain / encode / cluster stages.

This tests: can named visual concepts        This tests: can named clinical concepts
predict the object class?                    predict physician-confirmed diagnoses
                                             they were never trained on?
                                             ← This IS the dissertation core claim.

─────────────────────────────────────────   ──────────────────────────────────────────────
STEP 8: VALIDATE (lab only)                 STEP 8: VALIDATE
─────────────────────────────────────────   ──────────────────────────────────────────────
Image equivalent: patch purity metric       File    : run_lab_pipeline.py stage_validate
  "96% of patches in cluster 15 belong
   to bird images"                          Per-cluster enrichment analysis:
                                              For each cluster:
                                                diabetes_rate   = % diabetic members
                                                base_rate       = 14% (population)
                                                enrichment      = cluster / base
                                                mean_hba1c      = avg HbA1c in cluster

                                            Output:
                                              cache/lab/cluster_enrichment.csv

                                            Example dissertation result:
                                              "Concept 'diabetic_pattern' has
                                               5.6× the diabetes prevalence of
                                               the general population (71% vs 14%),
                                               mean HbA1c=8.3% — above the 6.5%
                                               diagnostic threshold — discovered
                                               without any label supervision."
```

---

## The FeatureTokenizer Explained

The line `T_i = x_i * W_i + b_i` is the core of the FT-Transformer tokenizer. Here's what every symbol means:

```
T_i  ∈ R^d        The output token for lab test i (a d=128 dimensional vector)
                   This is what enters the transformer — the "word embedding" equivalent

x_i  ∈ R          The input: a single scalar, the clinical deviation value of test i
                   e.g. x_glucose = +1.2  (glucose 1.2σ above upper normal range)
                        x_Hgb    = -2.1  (haemoglobin 2.1σ below lower normal range)
                        x_WBC    =  0.0  (WBC perfectly within normal range)

W_i  ∈ R^d        A learned weight vector, UNIQUE to test i
                   Not shared with any other test.
                   WBC has its own W_WBC. Glucose has its own W_glucose.
                   During SSL pre-training, W_WBC learns "what direction in
                   128-dim space encodes the concept of elevated WBC?"

b_i  ∈ R^d        A learned bias vector, also unique to test i
                   Encodes a "baseline" token shape even when x_i = 0

*   (multiply)     Element-wise scalar × vector: scales the direction W_i
                   by the magnitude of deviation x_i.
                   Large positive deviation → token points strongly in W_i direction.
                   Large negative deviation → token points strongly in -W_i direction.
                   Zero deviation → token = b_i (the resting position for test i)

+   (add)          Adds the bias: shifts the token to its baseline position
```

**Why not just use one shared linear layer?**

In standard tabular models, all features go through the same linear layer: `T = xW + b` where `x` is the full feature vector. This means every feature contributes equally regardless of its biological meaning. The FT-Transformer gives each feature its own weight vector so the model can learn that "a +2σ deviation in glucose" and "a +2σ deviation in creatinine" are completely different physiological signals — they should point in different directions in the 128-dim token space.

---

## Numbers at Each Stage

```
                    Image Pipeline              Lab Pipeline
                    ──────────────────────────  ──────────────────────────────
Input size          128×128 px (stored)         34 numbers per record
Model input         224×224 px (DINO upscales)  34 deviation values
Tokeniser output    784 patches × 384-dim       34 tokens × 128-dim
After panel PE      784 patches × 384-dim       34 tokens + 128-dim panel offset
Extract layers      8, 10, 12                   1, 3, 5  (0-indexed: layers 2,4,6)
Patch vector dim    384+384+384 = 1152-dim       128+128+128 = 384-dim per panel
Patches per input   784 spatial patches         3 panel-patches
Dataset patch total ~1.7M patches               56,019 panel-patches
PCA reduction       1152 → 128                  384 → 20
Clusters            30 (global GMM)             30 (10 per panel × 3 panels)
Concept score dim   30 per image                30 per patient record
Final classifier    LogReg [N, 30] → 3 classes  LogReg [N, 30] → 3 conditions
```

---

## Key Source Files

| Component | Image file | Lab file |
|-----------|-----------|---------|
| Encoder model | `src/models/dino_extractor.py` | `src/models/panel_ft_transformer.py` |
| Feature tokenizer | ViT patch embedding (built into DINO) | `FeatureTokenizer` class (line 34) |
| Data loader | `src/data/prepare_data.py` | `src/data/lab_loader.py` |
| Extractor | `LabExtractor` in `src/models/lab_extractor.py` | same |
| Clusterer | `src/pipeline/patch_clusterer.py` | same (reused) |
| Concept builder | `src/pipeline/concept_builder.py` | inline in `stage_concepts()` |
| Classifier | `src/pipeline/concept_classifier.py` | same (reused) |
| Pipeline runner | `experiments/run_pipeline.py` | `experiments/run_lab_pipeline.py` |
| Labeling GUI | `labeling/label_tool.py` | `labeling/lab_label_tool.py` |
| Config | `configs/config.yaml` | `configs/config_lab.yaml` |

---

## Grounding in Literature (Gap → Our Answer)

| Gap | Paper | Their limitation | Our answer |
|-----|-------|-----------------|------------|
| Gap 1 | TCAV (Kim, ICML 2018) / CBM (Koh, ICML 2020) | Concept labels required before/during training | Unsupervised GMM discovery; labels applied post-hoc |
| Gap 2 | DINO (Caron, NeurIPS 2021) / FT-Transformer (Gorishniy, NeurIPS 2021) | Intermediate layer representations never analysed for concept discovery | Multi-layer extraction from both models → concept clustering |
| Gap 3 | ClusterLLM (EMNLP 2023) | LLM concept guidance restricted to textual data | LLM reads numerical deviation profiles → clinical concept names |
