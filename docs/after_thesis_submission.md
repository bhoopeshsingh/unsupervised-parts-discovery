# After thesis submission: presentation, code, and viva

Your **submitted PDF is fixed**. You cannot change it unless your programme allows a formal correction. What follows is how to align **slides**, **repository**, and **what you say in the viva** with what you actually built.

## 1. Where the mismatch is

- **Submitted thesis (FR12 / §4.5.3):** may describe **OneClassSVM** on **concept scores** for the image domain.
- **Code and results scripts:** **multiclass logistic regression (LogReg)** on **30-dim concept scores** for **both** image and tabular pipelines; **OneClassSVM** appears in `generate_dissertation_results.py` only for the **raw L12 baseline**, not for the concept-based head.
- **Numbers (98.9%, 50.9%, etc.)** come from the LogReg-on-concepts setup; you are **not** changing results, only clarifying the **model class**.

## 2. Viva (verbal patch — no PDF edit)

Use one clear sentence if asked:

> “The implementation trains a **multiclass logistic regression** on concept activation scores for both domains; the raw-feature image baseline in the comparison script uses **OneClassSVM** in an OvR setup. The dissertation text in FR12 / §4.5.3 uses an older naming split; the **reported accuracies** correspond to the **LogReg-on-concepts** pipeline in the repository.”

If your examiners require a written addendum, ask your **module leader / registry** (not something to do in Git).

## 3. Presentation

- Regenerate the aligned deck:  
  `python scripts/build_aligned_viva_pptx.py`  
  (from repo root; copies `viva_presentation.pptx` → `viva_presentation_aligned.pptx` with dataset/title/LogReg label fixes).
- Or manually replace any **“OneClass SVM”** on the **concept** path with **“3-class LogReg”** / **“multiclass logistic regression”**, and keep **OneClassSVM** only if you show the **raw baseline** slide.

## 4. Code (this repo)

- **`experiments/run_pipeline.py` `stage_explain`:** updated to load the same **LogisticRegression** payload as `stage_classify` (with legacy fallback for `CatConceptOneClassClassifier`).
- **`src/pipeline/concept_classifier.py`:** dissertation figure title colours extended for **bird / car / cat**.

Optional later: a one-line footnote in **README** or **TECHNICAL_DESIGN.md** pointing to this file — only if you want the GitHub view to match the viva story.
