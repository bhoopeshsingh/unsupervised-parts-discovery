#!/usr/bin/env python3
"""
Generate all dissertation result JSON files.

Handles four priorities:
  Priority 1: cache/stage5_results.json        — Image classification comparison (Table 11)
  Priority 2: cache/semantic_correlation.json  — r between silhouette and semantic accuracy
  Priority 3: cache/semantic_eval.json         — Human evaluation flag (manual step)
  Priority 4: cache/lab/stage_l7_results.json  — Lab classifier accuracy (Stage L7)

Run:
  python experiments/generate_dissertation_results.py
"""

import json
import pickle
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.svm import OneClassSVM


# ─────────────────────────────────────────────────────────────────────────────
# Priority 1 — Stage 5: Image classification comparison (Table 11)
# ─────────────────────────────────────────────────────────────────────────────

def compute_stage5():
    print("\n" + "=" * 65)
    print("Priority 1 — Stage 5: Classification comparison (Table 11)")
    print("=" * 65)

    # ── Load concept scores ───────────────────────────────────────────────────
    scores_data   = torch.load("cache/concept_scores.pt", weights_only=False)
    X_concept     = scores_data["scores"].numpy()          # [N, C]
    y             = np.array(scores_data["image_labels"])  # [N]
    class_names   = scores_data["class_names"]             # ['bird','car','cat']
    concept_names = scores_data["concept_names"]

    print(f"\n  Concept scores : {X_concept.shape}  ({len(class_names)} classes)")

    # ── Shared train/test split (seed 42, 80/20) ──────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_concept, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── Concept-based classifier (LogReg) ────────────────────────────────────
    print("\n  [Concept-based] Logistic Regression on concept scores …")
    scaler_c = StandardScaler()
    X_tr_c = scaler_c.fit_transform(X_tr)
    X_te_c = scaler_c.transform(X_te)

    clf_c = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                               class_weight="balanced", random_state=42)
    clf_c.fit(X_tr_c, y_tr)

    y_pred_c = clf_c.predict(X_te_c)
    y_prob_c = clf_c.predict_proba(X_te_c)           # [N_te, 3]

    acc_c      = accuracy_score(y_te, y_pred_c)
    p_c, r_c, f1_c, _ = precision_recall_fscore_support(
        y_te, y_pred_c, average="macro", zero_division=0
    )
    y_te_bin   = label_binarize(y_te, classes=[0, 1, 2])
    auc_c      = roc_auc_score(y_te_bin, y_prob_c, multi_class="ovr", average="macro")

    report_c   = classification_report(y_te, y_pred_c, target_names=class_names,
                                        output_dict=True, zero_division=0)

    print(f"    Accuracy  : {acc_c:.4f}  ({acc_c:.1%})")
    print(f"    Macro F1  : {f1_c:.4f}")
    print(f"    Macro AUC : {auc_c:.4f}")

    # ── Raw baseline: OneClassSVM on mean L12 DINO features ──────────────────
    print("\n  [Raw baseline] Loading DINO features for per-image mean L12 …")
    print("  (This loads 10 GB — may take a minute)")
    dino_data   = torch.load("cache/dino_features.pt", weights_only=False)
    feats_all   = dino_data["features"]           # [N_patches, 1152]  float32
    image_ids   = dino_data["image_ids"]          # [N_patches]         int64
    img_labels  = np.array(dino_data["image_labels"])  # [N_images]

    # L12 = last 384 dimensions (features = L8|L10|L12 concatenated)
    feats_l12   = feats_all[:, -384:].numpy()     # [N_patches, 384]

    # Aggregate: mean per image
    n_images    = len(img_labels)
    X_raw       = np.zeros((n_images, 384), dtype=np.float32)
    counts      = np.zeros(n_images, dtype=np.int64)
    for patch_feat, img_id in zip(feats_l12, image_ids.numpy()):
        X_raw[img_id]  += patch_feat
        counts[img_id] += 1
    counts = np.maximum(counts, 1)
    X_raw /= counts[:, None]

    print(f"  Per-image raw L12 features: {X_raw.shape}")

    # Same train/test split (image indices match concept score order)
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X_raw, img_labels, test_size=0.2, stratify=img_labels, random_state=42
    )

    # OneClassSVM — one per class, OvR scoring
    print("  Training OneClassSVM (one per class) …")
    decision_scores = np.zeros((len(y_te_r), 3), dtype=np.float64)

    for cls_idx, cls_name in enumerate(class_names):
        mask_tr = (y_tr_r == cls_idx)
        X_cls   = X_tr_r[mask_tr]

        # PCA-reduce to 50d for speed (OneClassSVM doesn't scale to 384d easily)
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler as SS

        ss_r  = SS()
        X_cls_s = ss_r.fit_transform(X_cls)
        X_te_s  = ss_r.transform(X_te_r)

        pca = PCA(n_components=50, random_state=42)
        X_cls_p = pca.fit_transform(X_cls_s)
        X_te_p  = pca.transform(X_te_s)

        svm = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
        svm.fit(X_cls_p)
        decision_scores[:, cls_idx] = svm.decision_function(X_te_p)
        print(f"    {cls_name}: OneClassSVM trained on {mask_tr.sum()} images")

    # Predict = class with highest decision score
    y_pred_r = np.argmax(decision_scores, axis=1)
    acc_r    = accuracy_score(y_te_r, y_pred_r)
    p_r, r_r, f1_r, _ = precision_recall_fscore_support(
        y_te_r, y_pred_r, average="macro", zero_division=0
    )
    # AUC: shift scores to positive range, then treat as probabilities via softmax
    exp_scores = np.exp(decision_scores - decision_scores.max(axis=1, keepdims=True))
    proba_r    = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    y_te_r_bin = label_binarize(y_te_r, classes=[0, 1, 2])
    auc_r      = roc_auc_score(y_te_r_bin, proba_r, multi_class="ovr", average="macro")

    report_r   = classification_report(y_te_r, y_pred_r, target_names=class_names,
                                        output_dict=True, zero_division=0)

    print(f"\n  [Raw baseline] OneClassSVM on L12 features:")
    print(f"    Accuracy  : {acc_r:.4f}  ({acc_r:.1%})")
    print(f"    Macro F1  : {f1_r:.4f}")
    print(f"    Macro AUC : {auc_r:.4f}")

    # ── Build per-class metrics ───────────────────────────────────────────────
    def per_class_metrics(report, names):
        out = {}
        for n in names:
            row = report.get(n, {})
            out[n] = {
                "precision": round(row.get("precision", 0.0), 4),
                "recall":    round(row.get("recall",    0.0), 4),
                "f1":        round(row.get("f1-score",  0.0), 4),
                "support":   int(row.get("support",     0)),
            }
        return out

    result = {
        "concept_based": {
            "accuracy":      round(float(acc_c), 4),
            "macro_precision": round(float(p_c), 4),
            "macro_recall":  round(float(r_c), 4),
            "macro_F1":      round(float(f1_c), 4),
            "macro_AUC_ROC": round(float(auc_c), 4),
            "per_class":     per_class_metrics(report_c, class_names),
            "n_concepts":    X_concept.shape[1],
            "classifier":    "LogisticRegression (C=1.0, class_weight=balanced)",
        },
        "raw_baseline": {
            "accuracy":      round(float(acc_r), 4),
            "macro_precision": round(float(p_r), 4),
            "macro_recall":  round(float(r_r), 4),
            "macro_F1":      round(float(f1_r), 4),
            "macro_AUC_ROC": round(float(auc_r), 4),
            "per_class":     per_class_metrics(report_r, class_names),
            "features":      "mean DINO L12 (last 384d) per image, PCA→50d",
            "classifier":    "OneClassSVM (rbf, nu=0.1, OvR)",
        },
        "class_names":  class_names,
        "n_test":       int(len(y_te)),
        "split":        "80/20 stratified (random_state=42)",
        "note": (
            "Concept-based uses 22 labeled concept scores (8 noise clusters excluded). "
            "Raw baseline uses mean FG-masked DINO L12 patch features per image."
        ),
    }

    out_path = Path("cache/stage5_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n  ✓  Saved → {out_path}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Priority 2 — Silhouette–semantic correlation (§6.4, r=0.94 claim)
# ─────────────────────────────────────────────────────────────────────────────

def compute_semantic_correlation():
    print("\n" + "=" * 65)
    print("Priority 2 — Silhouette–semantic correlation (§6.4)")
    print("=" * 65)

    ablation = json.loads(Path("cache/ablation/ablation_results.json").read_text())

    # class_purity = fraction of patches in each cluster that belong to the
    # dominant class — this IS the semantic accuracy metric for the image pipeline
    variants = ["A", "B", "C", "D"]
    silhouettes      = [ablation[v]["silhouette"]    for v in variants]
    semantic_acc     = [ablation[v]["class_purity"]  for v in variants]

    print("\n  Variant | Silhouette | Semantic Accuracy (class purity)")
    print("  " + "-" * 50)
    for v, s, a in zip(variants, silhouettes, semantic_acc):
        print(f"  {v}       | {s:.4f}     | {a:.4f}")

    r, p_value = stats.pearsonr(silhouettes, semantic_acc)
    print(f"\n  Pearson r = {r:.4f}   (p = {p_value:.4f})")

    if abs(r) >= 0.90:
        print(f"  → Strong correlation (r={r:.2f}): silhouette predicts semantic quality")
    else:
        print(f"  ⚠  r = {r:.2f}, which is NOT r=0.94.")
        print("     The dissertation must report the actual value or remove this claim.")

    result = {
        "variants": {
            v: {
                "silhouette":        ablation[v]["silhouette"],
                "semantic_accuracy": ablation[v]["class_purity"],
                "semantic_metric":   "class_purity (fraction of patches in dominant-class cluster)",
                "nmi":               ablation[v]["nmi"],
                "variant_name":      ablation[v]["variant_name"],
            }
            for v in variants
        },
        "pearson_r":  round(float(r), 4),
        "p_value":    round(float(p_value), 4),
        "note": (
            "Semantic accuracy = class_purity from ablation study: "
            "fraction of patches in each cluster belonging to the same object class "
            "(bird/car/cat). Silhouette measures geometric cluster quality. "
            f"Actual Pearson r = {r:.4f} (not 0.94 — dissertation must be updated if different)."
        ),
    }

    out_path = Path("cache/semantic_correlation.json")
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n  ✓  Saved → {out_path}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Priority 3 — 82% semantic accuracy human evaluation flag
# ─────────────────────────────────────────────────────────────────────────────

def check_semantic_eval():
    print("\n" + "=" * 65)
    print("Priority 3 — Human evaluation semantic accuracy (§ claim: 82%)")
    print("=" * 65)

    out_path = Path("cache/semantic_eval.json")

    if out_path.exists():
        existing = json.loads(out_path.read_text())
        print(f"\n  Found existing: {existing}")
        return existing

    print("""
  ⚠  No cache/semantic_eval.json found.

  This file must be created MANUALLY from your Streamlit labelling session:
    1. Open: streamlit run labeling/image_cluster_labeler.py
    2. For a random sample of 50 patches (across all clusters), judge whether
       the patch correctly matches its assigned concept label.
    3. Count how many (out of 50) are correctly labelled.
    4. Then run this script again with --write-semantic-eval N_CORRECT
       (e.g. if 41/50 correct: python ... --write-semantic-eval 41)

  Until this is done, the 82% claim in the dissertation CANNOT be verified
  and should be removed or marked as [TODO: verify].
""")

    placeholder = {
        "n_patches":          50,
        "correct":            None,
        "semantic_accuracy":  None,
        "status":             "NOT_VERIFIED — human evaluation not yet completed",
        "action_required": (
            "Run Streamlit labeller, manually evaluate 50 random patches, "
            "then update this file with: correct=<n>, semantic_accuracy=<n/50>"
        ),
    }
    out_path.write_text(json.dumps(placeholder, indent=2))
    print(f"  Placeholder written → {out_path}")
    return placeholder


# ─────────────────────────────────────────────────────────────────────────────
# Priority 4 — Stage L7: Lab classifier accuracy
# ─────────────────────────────────────────────────────────────────────────────

def compute_stage_l7():
    print("\n" + "=" * 65)
    print("Priority 4 — Stage L7: Lab classifier accuracy")
    print("=" * 65)

    clf_path = Path("cache/lab/classifier.pkl")
    if not clf_path.exists():
        print("  ERROR: cache/lab/classifier.pkl not found.")
        print("  Run: python experiments/run_lab_pipeline.py --stage classify")
        return None

    with open(clf_path, "rb") as f:
        payload = pickle.load(f)

    test_acc   = payload["test_accuracy"]
    train_acc  = payload["train_accuracy"]
    report     = payload["classification_report"]
    class_names = payload["class_names"]  # ['diabetes','hypertension','normal']
    n_classes  = len(class_names)
    chance     = 1.0 / n_classes

    print(f"\n  Test accuracy  : {test_acc:.4f}  ({test_acc:.1%})")
    print(f"  Train accuracy : {train_acc:.4f}  ({train_acc:.1%})")
    print(f"  Chance baseline: {chance:.4f}  ({chance:.1%})")
    print(f"  Above chance by: {test_acc - chance:+.4f}  ({(test_acc - chance):.1%})")
    print()

    per_class = {}
    for cls in class_names:
        row = report.get(cls, {})
        per_class[cls] = {
            "precision": round(row.get("precision", 0.0), 4),
            "recall":    round(row.get("recall",    0.0), 4),
            "f1":        round(row.get("f1-score",  0.0), 4),
            "support":   int(row.get("support",     0)),
        }
        print(f"  {cls:14s}: P={per_class[cls]['precision']:.3f}  "
              f"R={per_class[cls]['recall']:.3f}  F1={per_class[cls]['f1']:.3f}  "
              f"n={per_class[cls]['support']}")

    macro = report.get("macro avg", {})

    result = {
        "accuracy":        round(float(test_acc),  4),
        "train_accuracy":  round(float(train_acc), 4),
        "chance_baseline": round(chance, 4),
        "n_classes":       n_classes,
        "class_names":     class_names,
        "macro_precision": round(float(macro.get("precision", 0.0)), 4),
        "macro_recall":    round(float(macro.get("recall",    0.0)), 4),
        "macro_F1":        round(float(macro.get("f1-score",  0.0)), 4),
        "per_class":       per_class,
        "label_source":    payload.get("label_source", "NHANES DIQ010, BPQ020"),
        "core_claim":      payload.get("core_claim", ""),
        "note": (
            f"Accuracy {test_acc:.1%} vs {chance:.1%} chance (3-class: "
            "diabetes/hypertension/normal). Labels from NHANES questionnaire, "
            "never used in any prior pipeline stage."
        ),
    }

    out_path = Path("cache/lab/stage_l7_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n  ✓  Saved → {out_path}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-semantic-eval", type=int, metavar="N_CORRECT",
                        help="Write semantic_eval.json with this many correct out of 50")
    parser.add_argument("--skip-stage5", action="store_true",
                        help="Skip Stage 5 (slow — loads 10 GB DINO features)")
    args = parser.parse_args()

    if args.write_semantic_eval is not None:
        n = args.write_semantic_eval
        result = {
            "n_patches":         50,
            "correct":           n,
            "semantic_accuracy": round(n / 50, 4),
            "status":            "VERIFIED — human evaluation completed",
        }
        Path("cache/semantic_eval.json").write_text(json.dumps(result, indent=2))
        print(f"Written cache/semantic_eval.json: {n}/50 = {n/50:.1%} semantic accuracy")
        return

    # Priority 2 — fast (no model loading)
    compute_semantic_correlation()

    # Priority 3 — flag check
    check_semantic_eval()

    # Priority 4 — fast (loads small pkl)
    compute_stage_l7()

    # Priority 1 — slow (loads 10 GB)
    if not args.skip_stage5:
        compute_stage5()
    else:
        print("\n[Skipped Stage 5 — pass without --skip-stage5 to run it]")

    print("\n" + "=" * 65)
    print("Done. Files written:")
    print("  cache/stage5_results.json        (Priority 1)")
    print("  cache/semantic_correlation.json  (Priority 2)")
    print("  cache/semantic_eval.json         (Priority 3 — manual step required)")
    print("  cache/lab/stage_l7_results.json  (Priority 4)")
    print("=" * 65)


if __name__ == "__main__":
    main()
