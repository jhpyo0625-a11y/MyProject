"""
T3.5 / T3.6 -- Classifier iteration after baseline missed the recall target.

Tries multiple pipelines on the existing ResNet18 embeddings. Each pipeline
uses StandardScaler (the likely missing piece in the baseline) followed by
a classifier variant. Reports CV Fail recall for each; promotes the winner.

Pipelines tried (in order of expected effectiveness):
  1. StandardScaler + LogReg (C grid)   -- baseline fix
  2. StandardScaler + SVM RBF (C grid)  -- typically best for frozen CNN features
  3. StandardScaler + GBT               -- nonlinear ensemble fallback

Threshold tuning and artifact serialization run only for the winning pipeline.
All runs logged to MLflow under the same experiment.
"""

import json
import pickle
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, classification_report,
    confusion_matrix, precision_recall_curve,
    precision_score, recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.config_loader import load_config
from src.training.dataset import LABEL_MAP, LABEL_NAMES, load_embeddings
from src.training.train import (
    COMP_FOLD, DELTA, TARGET_RECALL,
    binary_metrics, p_fail_from_proba, per_class_recall,
    plot_cm, plot_pr, tune_threshold,
)

cfg      = load_config()
REPORTS  = ROOT / "reports"
PROD_DIR = ROOT / cfg["paths"]["production_dir"]

# ---------------------------------------------------------------------------
# Pipeline catalogue
# ---------------------------------------------------------------------------

C_GRID = [0.01, 0.1, 1.0, 10.0]

def make_logreg(C):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            C=C, class_weight="balanced", max_iter=1000,
            random_state=cfg["training"]["random_seed"])),
    ])

def make_svm(C):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(
            C=C, kernel="rbf", class_weight="balanced",
            probability=True, random_state=cfg["training"]["random_seed"])),
    ])

def make_gbt():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=cfg["training"]["random_seed"])),
    ])


# ---------------------------------------------------------------------------
# CV runner (identical logic to train.py, parametrised on pipeline)
# ---------------------------------------------------------------------------

def cv_run(X, y3, y_bin, folds, is_aug, pipeline):
    oof_preds, oof_probs_l, oof_y3_l, oof_ybin_l = [], [], [], []
    fold_recalls = []

    for val_fold in range(COMP_FOLD):
        tr  = (folds != val_fold) & (folds != COMP_FOLD)
        val = (folds == val_fold) & ~is_aug

        pipeline.fit(X[tr], y3[tr])
        proba = pipeline.predict_proba(X[val])
        pred  = pipeline.predict(X[val])

        oof_preds.append(pred);     oof_probs_l.append(proba)
        oof_y3_l.append(y3[val]);   oof_ybin_l.append(y_bin[val])

        pfail    = p_fail_from_proba(proba, pipeline.classes_)
        pred_bin = (pfail > 0.5).astype(int)
        bm       = binary_metrics(y_bin[val], pred_bin)
        fold_recalls.append(bm["fail_recall"])

    oof_preds  = np.concatenate(oof_preds)
    oof_probs  = np.concatenate(oof_probs_l)
    oof_y3     = np.concatenate(oof_y3_l)
    oof_y_bin  = np.concatenate(oof_ybin_l)
    mean_recall = float(np.mean(fold_recalls))

    return mean_recall, oof_preds, oof_probs, oof_y3, oof_y_bin


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 62)
    print("T3.5/T3.6 -- Classifier iteration (baseline missed target)")
    print("=" * 62)

    X, y3, y_bin, folds, is_aug, labels_str = load_embeddings(cfg, ROOT)
    print(f"  Embeddings loaded: {len(X)} rows\n")

    db = ROOT / cfg["paths"]["experiments"] / "mlflow.db"
    db.parent.mkdir(exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + str(db).replace("\\", "/"))
    mlflow.set_experiment("coil-defect-detection")

    results = []   # (mean_fail_recall, name, pipeline, oof data)

    # ── Grid: LogReg + StandardScaler ────────────────────────
    print("LogReg + StandardScaler:")
    best_lr_recall, best_lr = -1.0, None
    for C in C_GRID:
        pl = make_logreg(C)
        recall, *oof = cv_run(X, y3, y_bin, folds, is_aug, pl)
        marker = " <--" if recall > best_lr_recall else ""
        print(f"  C={C:<6}  CV FailRecall={recall:.3f}{marker}")
        if recall > best_lr_recall:
            best_lr_recall = recall
            best_lr = (f"LogReg C={C}", pl, recall, oof)

    results.append(best_lr)

    # ── Grid: SVM RBF + StandardScaler ───────────────────────
    print("\nSVM (RBF) + StandardScaler:")
    best_svm_recall, best_svm = -1.0, None
    for C in C_GRID:
        pl = make_svm(C)
        recall, *oof = cv_run(X, y3, y_bin, folds, is_aug, pl)
        marker = " <--" if recall > best_svm_recall else ""
        print(f"  C={C:<6}  CV FailRecall={recall:.3f}{marker}")
        if recall > best_svm_recall:
            best_svm_recall = recall
            best_svm = (f"SVM RBF C={C}", pl, recall, oof)

    results.append(best_svm)

    # ── GBT + StandardScaler ──────────────────────────────────
    print("\nGradientBoosting + StandardScaler:")
    pl = make_gbt()
    recall, *oof = cv_run(X, y3, y_bin, folds, is_aug, pl)
    print(f"  CV FailRecall={recall:.3f}")
    results.append((f"GBT n=200 d=3", pl, recall, oof))

    # ── Pick winner ───────────────────────────────────────────
    print("\n" + "=" * 62)
    print("Summary:")
    results.sort(key=lambda r: r[2], reverse=True)
    for name, _, recall, _ in results:
        met = "MET" if recall >= TARGET_RECALL else "NOT MET"
        print(f"  {name:<25}  FailRecall={recall:.3f}  [{met}]")

    winner_name, winner_pl, winner_recall, winner_oof = results[0]
    oof_preds, oof_probs, oof_y3, oof_y_bin = winner_oof
    print(f"\nWinner: {winner_name}  (CV Fail recall = {winner_recall:.3f})")

    # ── Threshold tuning on winner ────────────────────────────
    threshold, p_fail_oof, thr_stats = tune_threshold(oof_probs, oof_y_bin, winner_pl.classes_)
    print(f"  Threshold    : {threshold:.4f}")
    print(f"  Fail recall  : {thr_stats['achieved_recall']:.3f}")
    print(f"  False-reject : {thr_stats['false_reject']:.3f}")

    # ── Confusion matrices ────────────────────────────────────
    pred_bin_thr = (p_fail_oof > threshold).astype(int)
    cm3 = confusion_matrix(oof_y3,    oof_preds,     labels=[0, 1, 2])
    cm2 = confusion_matrix(oof_y_bin, pred_bin_thr,  labels=[0, 1])

    plot_cm(cm3, LABEL_NAMES,
            f"3-class CM -- OOF -- {winner_name}",
            REPORTS / "confusion_matrix_3class.png")
    plot_cm(cm2, ["Pass", "Fail"],
            f"Binary CM -- OOF @ thr={threshold:.3f} -- {winner_name}",
            REPORTS / "confusion_matrix_binary.png")
    plot_pr(oof_y_bin, p_fail_oof, threshold,
            REPORTS / "precision_recall_curve.png")

    # ── Re-fit winner on folds 0-3, eval fold 4 ──────────────
    tr_final = (folds != COMP_FOLD)
    winner_pl.fit(X[tr_final], y3[tr_final])

    f4_mask  = (folds == COMP_FOLD) & ~is_aug
    f4_pred3 = winner_pl.predict(X[f4_mask])
    f4_proba = winner_pl.predict_proba(X[f4_mask])
    f4_pfail = p_fail_from_proba(f4_proba, winner_pl.classes_)
    f4_bin   = (f4_pfail > threshold).astype(int)
    f4_bm    = binary_metrics(y_bin[f4_mask], f4_bin)

    print(f"\nFold 4 eval: FailRecall={f4_bm['fail_recall']:.3f}  "
          f"FalseReject={f4_bm['false_reject']:.3f}")
    print(f"\n{classification_report(y3[f4_mask], f4_pred3, target_names=LABEL_NAMES)}")

    # ── MLflow log ────────────────────────────────────────────
    goal_met = winner_recall >= TARGET_RECALL
    with mlflow.start_run(run_name=f"iterate-{winner_name.replace(' ', '-')}") as run:
        mlflow.log_params({
            "classifier":    winner_name,
            "backbone":      cfg["model"]["backbone"],
            "aug_copies":    cfg["augmentation"]["minority_copies"],
            "target_recall": TARGET_RECALL,
            "threshold":     round(threshold, 6),
        })
        mlflow.log_metrics({
            "cv_fail_recall":        winner_recall,
            "thr_fail_recall":       thr_stats["achieved_recall"],
            "thr_fail_precision":    thr_stats["achieved_prec"],
            "thr_false_reject":      thr_stats["false_reject"],
            "fold4_fail_recall":     f4_bm["fail_recall"],
            "fold4_false_reject":    f4_bm["false_reject"],
        })
        for art in ["confusion_matrix_3class.png", "confusion_matrix_binary.png",
                    "precision_recall_curve.png"]:
            mlflow.log_artifact(str(REPORTS / art))

    # ── Save winner ───────────────────────────────────────────
    with open(PROD_DIR / "classifier.pkl", "wb") as f:
        pickle.dump(winner_pl, f)

    metadata = {
        "backbone":           cfg["model"]["backbone"],
        "embedding_dim":      int(X.shape[1]),
        "classifier":         winner_name,
        "classes":            LABEL_NAMES,
        "label_map":          LABEL_MAP,
        "pass_threshold":     round(threshold, 6),
        "review_band_delta":  DELTA,
        "cv_fail_recall":     round(winner_recall, 4),
        "thr_fail_recall":    round(thr_stats["achieved_recall"], 4),
        "thr_false_reject":   round(thr_stats["false_reject"], 4),
        "fold4_fail_recall":  round(f4_bm["fail_recall"], 4),
        "fold4_false_reject": round(f4_bm["false_reject"], 4),
        "trained_on_folds":   list(range(COMP_FOLD)),
        "comparison_fold":    COMP_FOLD,
        "n_train":            int(tr_final.sum()),
        "mlflow_run_id":      run.info.run_id,
        "date":               date.today().isoformat(),
    }
    with open(PROD_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    import yaml
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        live_cfg = yaml.safe_load(f)
    live_cfg["decision"]["pass_threshold"] = round(threshold, 6)
    with open(ROOT / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(live_cfg, f, default_flow_style=False, allow_unicode=True)

    verdict = "PASS" if goal_met else "FAIL -- needs backbone change (T3.5)"
    card = f"""# Model Card -- Coil Defect Inspection v1

**Date:** {date.today().isoformat()}
**Backbone:** {cfg['model']['backbone']} (frozen, ImageNet, global max pool)
**Classifier:** {winner_name} (with StandardScaler)
**MLflow run:** `{run.info.run_id}`

## CV Performance (OOF, folds 0-{COMP_FOLD-1})

| Metric | Value |
|--------|-------|
| Fail recall (target >={TARGET_RECALL:.0%}) | **{winner_recall:.1%}** |
| Fail recall @ tuned threshold | {thr_stats['achieved_recall']:.1%} |
| False-reject @ tuned threshold | {thr_stats['false_reject']:.1%} |
| Pass/Fail threshold | {threshold:.4f} |
| Review band | +/-{DELTA} |

## Fold {COMP_FOLD} Independent Check

| Metric | Value |
|--------|-------|
| Fail recall | {f4_bm['fail_recall']:.1%} |
| Fail precision | {f4_bm['fail_precision']:.1%} |
| False-reject | {f4_bm['false_reject']:.1%} |

## Verdict: {verdict}
"""
    (PROD_DIR / "model_card.md").write_text(card, encoding="utf-8")

    print(f"\n{'='*62}")
    print(f"VERDICT: {verdict}")
    print(f"  CV Fail recall  = {winner_recall:.1%}  (target {TARGET_RECALL:.0%})")
    print(f"  Fold 4 recall   = {f4_bm['fail_recall']:.1%}")
    print(f"  False-reject    = {thr_stats['false_reject']:.1%} (at tuned threshold)")
    print(f"  Threshold       = {threshold:.4f}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
