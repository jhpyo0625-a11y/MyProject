"""
T3.1-T3.7 -- Phase 3: training, CV evaluation, threshold tuning, serialization.

Single entry point for the full training pipeline:
  1. Load cached embeddings (src.training.dataset)
  2. 4-fold CV on folds 0-3  (fold 4 = comparison fold, never trained on)
  3. Collect out-of-fold (OOF) probabilities
  4. Tune the Pass/Fail threshold for >=95% Fail recall
  5. Train final classifier on all folds 0-3 combined
  6. Independent eval on fold 4
  7. Log params + metrics + artifacts to MLflow
  8. Serialize model, metadata, and model card

Run:
    python -m src.training.train
"""

import json
import pickle
import shutil
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.config_loader import load_config
from src.training.dataset import LABEL_MAP, LABEL_NAMES, load_embeddings

cfg      = load_config()
REPORTS  = ROOT / "reports"
PROD_DIR = ROOT / cfg["paths"]["production_dir"]
REPORTS.mkdir(exist_ok=True)
PROD_DIR.mkdir(parents=True, exist_ok=True)

COMP_FOLD     = cfg["training"]["comparison_fold"]   # 4
TARGET_RECALL = cfg["evaluation"]["target_fail_recall"]   # 0.95
DELTA         = cfg["decision"]["review_band_delta"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# predictor.py loads production artifacts in this precedence order; a
# higher-precedence file silently shadows a freshly-written lower one.
_ARTIFACT_PRECEDENCE = ["padim.pkl", "model.pt", "classifier.pkl"]


def warn_if_shadowed(prod_dir: Path, written: str) -> None:
    """Warn if a higher-precedence artifact will be loaded instead of `written`."""
    if written not in _ARTIFACT_PRECEDENCE:
        return
    rank = _ARTIFACT_PRECEDENCE.index(written)
    shadowers = [name for name in _ARTIFACT_PRECEDENCE[:rank]
                 if (prod_dir / name).exists()]
    if shadowers:
        print(f"  WARNING: wrote {written}, but {', '.join(shadowers)} has higher "
              f"precedence and will load instead. Remove it to deploy {written}.")


def make_clf():
    return LogisticRegression(
        C=cfg["training"]["logreg_C"],
        class_weight=cfg["training"]["class_weight"],
        max_iter=1000,
        random_state=cfg["training"]["random_seed"],
    )


def p_fail_from_proba(proba: np.ndarray, classes) -> np.ndarray:
    """Return P(Fail) = 1 - P(Pass) given predict_proba output."""
    pass_col = list(classes).index(0)   # Pass is encoded as 0
    return 1.0 - proba[:, pass_col]


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-class recall from 3-class arrays (index = LABEL_MAP value)."""
    result = {}
    for i, name in enumerate(LABEL_NAMES):
        mask = y_true == i
        result[name] = float((y_pred[mask] == i).mean()) if mask.any() else 0.0
    return result


def binary_metrics(y_bin: np.ndarray, pred_bin: np.ndarray) -> dict:
    return {
        "fail_recall":    float(recall_score(y_bin,    pred_bin, pos_label=1, zero_division=0)),
        "fail_precision": float(precision_score(y_bin, pred_bin, pos_label=1, zero_division=0)),
        "false_reject":   float((pred_bin[y_bin == 0] == 1).mean()) if (y_bin == 0).any() else 0.0,
    }


def plot_cm(cm: np.ndarray, classes: list, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(len(classes) * 2 + 1, len(classes) * 2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.colorbar(im, ax=ax)
    thresh = cm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def plot_pr(y_bin: np.ndarray, p_fail: np.ndarray, threshold: float, path: Path) -> None:
    prec, rec, thr = precision_recall_curve(y_bin, p_fail)
    ap = average_precision_score(y_bin, p_fail)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, lw=2, label=f"PR curve  AP={ap:.3f}")
    ax.axvline(TARGET_RECALL, color="gray", ls="--", lw=1,
               label=f"Target recall={TARGET_RECALL:.0%}")
    # operating point
    idx = int(np.searchsorted(thr, threshold))
    idx = min(idx, len(rec) - 1)
    ax.scatter([rec[idx]], [prec[idx]], s=120, color="red", zorder=5,
               label=f"Threshold={threshold:.3f}  "
                     f"recall={rec[idx]:.3f}  prec={prec[idx]:.3f}")
    ax.set_xlabel("Recall (Fail)"); ax.set_ylabel("Precision (Fail)")
    ax.set_title("Pass/Fail precision-recall curve (OOF, folds 0-3)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def tune_threshold(oof_probs: np.ndarray, oof_y_bin: np.ndarray, classes) -> tuple:
    """
    Find the largest P(Fail) threshold that achieves >= TARGET_RECALL on Fail.
    Returns (threshold, p_fail_array, stats_dict).
    """
    p_fail = p_fail_from_proba(oof_probs, classes)
    prec, rec, thr = precision_recall_curve(oof_y_bin, p_fail)

    # rec[:-1] aligns with thr; rec decreases as thr increases
    valid = rec[:-1] >= TARGET_RECALL
    if valid.any():
        threshold = float(thr[valid][-1])
    else:
        threshold = float(thr[0])
        print(f"  WARNING: cannot achieve {TARGET_RECALL:.0%} Fail recall at any threshold.")

    op_idx  = min(int(np.searchsorted(thr, threshold)), len(rec) - 1)
    pred_bin = (p_fail > threshold).astype(int)

    return threshold, p_fail, {
        "threshold":       threshold,
        "achieved_recall": float(rec[op_idx]),
        "achieved_prec":   float(prec[op_idx]),
        "false_reject":    float((p_fail[oof_y_bin == 0] > threshold).mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 62)
    print("Phase 3 -- Model Training & Evaluation")
    print("=" * 62)

    # ── data ──────────────────────────────────────────────────
    print("\nLoading embeddings...")
    X, y3, y_bin, folds, is_aug, labels_str = load_embeddings(cfg, ROOT)
    total    = len(X)
    cnt_orig = Counter(l for l, a in zip(labels_str, is_aug) if not a)
    cnt_aug  = Counter(l for l, a in zip(labels_str, is_aug) if a)
    print(f"  Total rows : {total}")
    for cls in LABEL_NAMES:
        print(f"  {cls:<6}: {cnt_orig[cls]} orig + {cnt_aug[cls]} aug")

    # ── MLflow setup ──────────────────────────────────────────
    db = ROOT / cfg["paths"]["experiments"] / "mlflow.db"
    db.parent.mkdir(exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + str(db).replace("\\", "/"))
    mlflow.set_experiment("coil-defect-detection")

    with mlflow.start_run(run_name="phase3-logreg-baseline") as run:

        mlflow.log_params({
            "backbone":      cfg["model"]["backbone"],
            "classifier":    "LogisticRegression",
            "C":             cfg["training"]["logreg_C"],
            "class_weight":  cfg["training"]["class_weight"],
            "n_cv_folds":    COMP_FOLD,        # 4 (not 5 -- fold 4 is held)
            "aug_copies":    cfg["augmentation"]["minority_copies"],
            "target_recall": TARGET_RECALL,
        })

        # ── 4-fold CV ─────────────────────────────────────────
        print(f"\n4-fold CV (folds 0-{COMP_FOLD-1}; fold {COMP_FOLD} held)...\n")
        oof_preds, oof_probs_l, oof_y3_l, oof_ybin_l = [], [], [], []
        fold_results = []

        for val_fold in range(COMP_FOLD):
            tr_mask  = (folds != val_fold) & (folds != COMP_FOLD)
            val_mask = (folds == val_fold) & ~is_aug

            X_tr, y_tr         = X[tr_mask],   y3[tr_mask]
            X_val, y_v3, y_vb  = X[val_mask],  y3[val_mask], y_bin[val_mask]

            clf = make_clf()
            clf.fit(X_tr, y_tr)

            pred  = clf.predict(X_val)
            proba = clf.predict_proba(X_val)
            pred_bin_05 = (p_fail_from_proba(proba, clf.classes_) > 0.5).astype(int)

            rc  = per_class_recall(y_v3, pred)
            bm  = binary_metrics(y_vb, pred_bin_05)

            fold_results.append({**rc, **bm})
            oof_preds.append(pred);     oof_probs_l.append(proba)
            oof_y3_l.append(y_v3);     oof_ybin_l.append(y_vb)

            print(f"  Fold {val_fold}  n_train={tr_mask.sum():4d}  n_val={val_mask.sum():3d}  |  "
                  f"FailRecall={bm['fail_recall']:.3f}  "
                  f"DentR={rc['Dent']:.3f}  LooseR={rc['Loose']:.3f}  "
                  f"FalseReject={bm['false_reject']:.3f}")

        oof_preds  = np.concatenate(oof_preds)
        oof_probs  = np.concatenate(oof_probs_l)
        oof_y3     = np.concatenate(oof_y3_l)
        oof_y_bin  = np.concatenate(oof_ybin_l)

        def mean_r(key): return float(np.mean([r[key] for r in fold_results if key in r]))
        cv = {
            "fail_recall":    mean_r("fail_recall"),
            "fail_precision": mean_r("fail_precision"),
            "false_reject":   mean_r("false_reject"),
            "recall_Dent":    mean_r("Dent"),
            "recall_Loose":   mean_r("Loose"),
            "recall_Pass":    mean_r("Pass"),
        }
        goal_met = cv["fail_recall"] >= TARGET_RECALL

        print(f"\n  Mean CV:")
        print(f"    Fail recall  = {cv['fail_recall']:.3f}  "
              f"({'MET' if goal_met else 'NOT MET'} -- target {TARGET_RECALL:.0%})")
        print(f"    Dent recall  = {cv['recall_Dent']:.3f}")
        print(f"    Loose recall = {cv['recall_Loose']:.3f}")
        print(f"    False-reject = {cv['false_reject']:.3f}")

        mlflow.log_metrics({f"cv_{k}": v for k, v in cv.items()})

        # ── threshold tuning ──────────────────────────────────
        print(f"\nTuning threshold for >={TARGET_RECALL:.0%} Fail recall...")
        threshold, p_fail_oof, thr_stats = tune_threshold(oof_probs, oof_y_bin, clf.classes_)
        print(f"  Threshold      : {threshold:.4f}")
        print(f"  Fail recall    : {thr_stats['achieved_recall']:.3f}")
        print(f"  Fail precision : {thr_stats['achieved_prec']:.3f}")
        print(f"  False-reject   : {thr_stats['false_reject']:.3f}  "
              f"({thr_stats['false_reject']*100:.1f}% of Pass coils rejected)")

        mlflow.log_metrics({
            "threshold":            threshold,
            "thr_fail_recall":      thr_stats["achieved_recall"],
            "thr_fail_precision":   thr_stats["achieved_prec"],
            "thr_false_reject":     thr_stats["false_reject"],
        })

        # ── confusion matrices & PR curve ─────────────────────
        pred_bin_thr = (p_fail_oof > threshold).astype(int)
        cm3 = confusion_matrix(oof_y3,    oof_preds,     labels=[0, 1, 2])
        cm2 = confusion_matrix(oof_y_bin, pred_bin_thr,  labels=[0, 1])

        plot_cm(cm3, LABEL_NAMES,
                f"3-class confusion matrix -- OOF (folds 0-{COMP_FOLD-1})",
                REPORTS / "confusion_matrix_3class.png")
        plot_cm(cm2, ["Pass", "Fail"],
                f"Binary confusion matrix -- OOF @ threshold {threshold:.3f}",
                REPORTS / "confusion_matrix_binary.png")
        plot_pr(oof_y_bin, p_fail_oof, threshold,
                REPORTS / "precision_recall_curve.png")

        # ── final model: train on folds 0-(COMP_FOLD-1) ───────
        print(f"\nTraining final model on folds 0-{COMP_FOLD-1}...")
        tr_final = (folds != COMP_FOLD)
        clf_final = make_clf()
        clf_final.fit(X[tr_final], y3[tr_final])
        print(f"  Samples used: {tr_final.sum()}")

        # ── fold 4 independent evaluation ─────────────────────
        print(f"\nEval on fold {COMP_FOLD} (comparison fold)...")
        f4_mask   = (folds == COMP_FOLD) & ~is_aug
        f4_pred3  = clf_final.predict(X[f4_mask])
        f4_proba  = clf_final.predict_proba(X[f4_mask])
        f4_pfail  = p_fail_from_proba(f4_proba, clf_final.classes_)
        f4_bin    = (f4_pfail > threshold).astype(int)
        f4_bm     = binary_metrics(y_bin[f4_mask], f4_bin)
        f4_rc     = per_class_recall(y3[f4_mask], f4_pred3)

        print(f"  n={f4_mask.sum()}  FailRecall={f4_bm['fail_recall']:.3f}  "
              f"FalseReject={f4_bm['false_reject']:.3f}")
        print(f"\n{classification_report(y3[f4_mask], f4_pred3, target_names=LABEL_NAMES)}")

        mlflow.log_metrics({f"fold4_{k}": v for k, v in {**f4_bm, **{f"recall_{k}": v for k,v in f4_rc.items()}}.items()})

        # ── serialize ─────────────────────────────────────────
        print("Saving artifacts...")
        with open(PROD_DIR / "classifier.pkl", "wb") as f:
            pickle.dump(clf_final, f)
        warn_if_shadowed(PROD_DIR, "classifier.pkl")

        metadata = {
            "backbone":           cfg["model"]["backbone"],
            "embedding_dim":      int(clf_final.n_features_in_),
            "classes":            LABEL_NAMES,
            "label_map":          LABEL_MAP,
            "pass_threshold":     round(threshold, 6),
            "review_band_delta":  DELTA,
            "target_fail_recall": TARGET_RECALL,
            "cv_fail_recall":     round(cv["fail_recall"], 4),
            "cv_dent_recall":     round(cv["recall_Dent"], 4),
            "cv_loose_recall":    round(cv["recall_Loose"], 4),
            "cv_false_reject":    round(cv["false_reject"], 4),
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

        # update live threshold in config.yaml
        import yaml
        with open(ROOT / "config.yaml", encoding="utf-8") as f:
            live_cfg = yaml.safe_load(f)
        live_cfg["decision"]["pass_threshold"] = round(threshold, 6)
        with open(ROOT / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(live_cfg, f, default_flow_style=False, allow_unicode=True)

        # model card
        verdict = "PASS" if goal_met else "FAIL -- iterate before deploying"
        card = f"""# Model Card -- Coil Defect Inspection v1

**Date:** {date.today().isoformat()}
**Backbone:** {cfg['model']['backbone']} (frozen, ImageNet, global max pool)
**Classifier:** LogisticRegression (C={cfg['training']['logreg_C']}, balanced weights)
**MLflow run:** `{run.info.run_id}`

## CV Performance (OOF, folds 0-{COMP_FOLD-1})

| Metric | Value |
|--------|-------|
| Fail recall (target ≥{TARGET_RECALL:.0%}) | **{cv['fail_recall']:.1%}** |
| Dent recall | {cv['recall_Dent']:.1%} |
| Loose recall | {cv['recall_Loose']:.1%} |
| False-reject rate | {cv['false_reject']:.1%} |
| Pass/Fail threshold | {threshold:.4f} |
| Review band | ±{DELTA} |

## Fold {COMP_FOLD} Independent Check

| Metric | Value |
|--------|-------|
| Fail recall | {f4_bm['fail_recall']:.1%} |
| Fail precision | {f4_bm['fail_precision']:.1%} |
| False-reject | {f4_bm['false_reject']:.1%} |

## Verdict: {verdict}
"""
        (PROD_DIR / "model_card.md").write_text(card, encoding="utf-8")

        for art in ["confusion_matrix_3class.png", "confusion_matrix_binary.png",
                    "precision_recall_curve.png"]:
            mlflow.log_artifact(str(REPORTS / art))
        mlflow.log_artifact(str(PROD_DIR / "classifier.pkl"))
        mlflow.log_artifact(str(PROD_DIR / "metadata.json"))

        # ── final verdict ─────────────────────────────────────
        print(f"\n{'='*62}")
        print(f"VERDICT: {verdict}")
        print(f"  CV Fail recall  = {cv['fail_recall']:.1%}  (target {TARGET_RECALL:.0%})")
        print(f"  Fold 4 recall   = {f4_bm['fail_recall']:.1%}")
        print(f"  False-reject    = {cv['false_reject']:.1%} CV / "
              f"{f4_bm['false_reject']:.1%} fold4")
        print(f"  Threshold       = {threshold:.4f}")
        print(f"  MLflow run      = {run.info.run_id}")
        print(f"{'='*62}")


if __name__ == "__main__":
    main()
