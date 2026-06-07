"""
T3.5 / Option 1 -- Fine-tune EfficientNet-B0 end-to-end.

Frozen ImageNet features failed (51% max CV Fail recall). This script
fine-tunes the backbone on the actual coil data so it learns task-specific
features (winding geometry, color shifts, deformation patterns).

Architecture
------------
- EfficientNet-B0 backbone; last 3 of 7 blocks unfrozen + conv_head unfrozen
- Global average pool (avg > max in backbone search)
- Dropout(0.4) + Linear(3) head

Training
--------
- Differential LR: backbone=5e-5, head=5e-4 (small backbone LR to preserve pretrained weights)
- Class-weighted CrossEntropyLoss (Pass 633 : Dent 74 : Loose 110)
- CosineAnnealingLR, early stopping on validation Fail recall
- Dynamic augmentation from saved crops (brightness, flip, affine)
- Input resized to 224x96 for CPU speed (4x faster than 448x192)

CV strategy
-----------
- 4-fold CV on folds 0-3; OOF probabilities -> threshold tuning
- Final model trained on folds 0-3 combined; evaluated on fold 4

Run:
    python -m src.training.finetune
"""

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import (
    classification_report, confusion_matrix, recall_score,
)
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.config_loader import load_config
from src.training.dataset import LABEL_MAP, LABEL_NAMES
from src.training.train import (
    COMP_FOLD, DELTA, TARGET_RECALL,
    binary_metrics, plot_cm, plot_pr, tune_threshold, p_fail_from_proba,
)

cfg = load_config()
CROPS_DIR = ROOT / cfg["paths"]["crops_dir"]
MANIFEST  = ROOT / cfg["paths"]["manifest"]
REPORTS   = ROOT / "reports"
PROD_DIR  = ROOT / cfg["paths"]["production_dir"]

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BACKBONE         = "efficientnet_b0"
N_UNFREEZE       = 3          # unfreeze last N of 7 EfficientNet blocks
DROPOUT          = 0.4
INPUT_W, INPUT_H = 224, 96    # crop resized for CPU speed (native=448x192)
BATCH_SIZE       = 8
BACKBONE_LR      = 5e-5
HEAD_LR          = 5e-4
WEIGHT_DECAY     = 1e-4
MAX_EPOCHS       = 40
PATIENCE         = 8
IMAGENET_MEAN    = [0.485, 0.456, 0.406]
IMAGENET_STD     = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CoilDataset(torch.utils.data.Dataset):
    def __init__(self, entries, transform):
        self.entries   = entries    # list of (crop_path, label_int)
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, label = self.entries[idx]
        arr = np.load(path)    # uint8 (192, 448, 3)
        img = Image.fromarray(arr).resize((INPUT_W, INPUT_H), Image.BILINEAR)
        return self.transform(img), label


TRAIN_TF = T.Compose([
    T.ColorJitter(brightness=0.3, contrast=0.3),
    T.RandomHorizontalFlip(0.5),
    T.RandomVerticalFlip(0.5),
    T.RandomAffine(degrees=5, translate=(0.05, 0.05)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

VAL_TF = T.Compose([
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CoilNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE, pretrained=True, num_classes=0, global_pool="avg",
        )
        # Freeze all backbone params, then selectively unfreeze
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Unfreeze last N_UNFREEZE blocks
        blocks = list(self.backbone.blocks.children())
        for block in blocks[-N_UNFREEZE:]:
            for p in block.parameters():
                p.requires_grad = True

        # Always unfreeze conv_head + bn2 + act2 (the output projection)
        for name, mod in self.backbone.named_modules():
            if any(k in name for k in ("conv_head", "bn2", "act2")):
                for p in mod.parameters():
                    p.requires_grad = True

        n_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.backbone.parameters())
        print(f"  Backbone trainable params: {n_trainable:,} / {n_total:,} "
              f"({100*n_trainable/n_total:.1f}%)")

        self.head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(self.backbone.num_features, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def make_optimizer(model):
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    return torch.optim.Adam([
        {"params": backbone_params,         "lr": BACKBONE_LR},
        {"params": model.head.parameters(), "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_loader(model, loader):
    """Return (all_probs np(N,3), all_true np(N,))."""
    model.eval()
    probs_l, true_l = [], []
    for imgs, labels in loader:
        probs_l.append(torch.softmax(model(imgs), dim=-1).numpy())
        true_l.append(labels.numpy())
    return np.concatenate(probs_l), np.concatenate(true_l)


def fail_recall_at_half(probs, y_true):
    """Binary Fail recall at threshold=0.5 (used as early-stopping criterion)."""
    p_fail   = 1.0 - probs[:, 0]
    pred_bin = (p_fail > 0.5).astype(int)
    y_bin    = (y_true != 0).astype(int)
    return float(recall_score(y_bin, pred_bin, pos_label=1, zero_division=0))


# ---------------------------------------------------------------------------
# Per-fold training
# ---------------------------------------------------------------------------

def train_fold(val_fold, entries_with_fold, class_weights_t):
    """
    Train one fold. Returns (best_state_dict, oof_probs, oof_true).
    entries_with_fold: list of (path, label_int, fold_int)
    """
    tr_raw = [(p, l) for p, l, f in entries_with_fold
              if f != val_fold and f != COMP_FOLD]
    vl_raw = [(p, l) for p, l, f in entries_with_fold if f == val_fold]

    tr_loader = torch.utils.data.DataLoader(
        CoilDataset(tr_raw, TRAIN_TF),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    vl_loader = torch.utils.data.DataLoader(
        CoilDataset(vl_raw, VAL_TF),
        batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0,
    )

    model     = CoilNet()
    optimizer = make_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-7,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    best_recall  = -1.0
    best_state   = None
    patience_cnt = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        loss = train_epoch(model, tr_loader, optimizer, criterion)
        scheduler.step()
        probs, true = eval_loader(model, vl_loader)
        recall = fail_recall_at_half(probs, true)

        if recall > best_recall:
            best_recall  = recall
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"    ep {epoch:3d}  loss={loss:.4f}  "
                  f"val_fail_recall={recall:.3f}  best={best_recall:.3f}")

        if patience_cnt >= PATIENCE:
            print(f"    Early stop at epoch {epoch}  best={best_recall:.3f}")
            break

    model.load_state_dict(best_state)
    probs, true = eval_loader(model, vl_loader)
    return model, probs, true


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 62)
    print("Fine-tuning EfficientNet-B0 end-to-end")
    print(f"  Backbone: {BACKBONE}  unfreeze last {N_UNFREEZE} blocks")
    print(f"  Input: {INPUT_W}x{INPUT_H}  batch={BATCH_SIZE}  "
          f"lr_backbone={BACKBONE_LR}  lr_head={HEAD_LR}")
    print("=" * 62)

    # Load manifest
    with open(MANIFEST, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    entries = []
    for r in rows:
        p = CROPS_DIR / r["label"] / f"{Path(r['filepath']).stem}.npy"
        if p.exists():
            entries.append((p, LABEL_MAP[r["label"]], int(r["fold"])))

    print(f"\nOriginal crops found: {len(entries)}")
    cnt = {cls: sum(1 for _, l, _ in entries if l == LABEL_MAP[cls]) for cls in LABEL_NAMES}
    for cls, n in cnt.items():
        print(f"  {cls}: {n}")

    # Class weights (balanced)
    all_labels = [l for _, l, _ in entries]
    cw = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=np.array(all_labels))
    cw_tensor = torch.FloatTensor(cw)
    print(f"\nClass weights: Pass={cw[0]:.2f}  Dent={cw[1]:.2f}  Loose={cw[2]:.2f}")

    # MLflow
    db = ROOT / cfg["paths"]["experiments"] / "mlflow.db"
    db.parent.mkdir(exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + str(db).replace("\\", "/"))
    mlflow.set_experiment("coil-defect-detection")

    with mlflow.start_run(run_name="finetune-efficientnet_b0") as run:
        mlflow.log_params({
            "backbone":       BACKBONE,
            "n_unfreeze":     N_UNFREEZE,
            "input_size":     f"{INPUT_W}x{INPUT_H}",
            "dropout":        DROPOUT,
            "backbone_lr":    BACKBONE_LR,
            "head_lr":        HEAD_LR,
            "max_epochs":     MAX_EPOCHS,
            "patience":       PATIENCE,
            "batch_size":     BATCH_SIZE,
            "approach":       "finetune_end_to_end",
        })

        # 4-fold CV
        print(f"\n4-fold CV (folds 0-{COMP_FOLD-1})...")
        oof_probs_l, oof_true_l = [], []
        fold_recalls = []

        for val_fold in range(COMP_FOLD):
            print(f"\n  --- Fold {val_fold} ---")
            t0 = time.perf_counter()
            model, oof_probs, oof_true = train_fold(val_fold, entries, cw_tensor)
            elapsed = time.perf_counter() - t0

            recall = fail_recall_at_half(oof_probs, oof_true)
            print(f"  Fold {val_fold} best val FailRecall={recall:.3f}  ({elapsed:.0f}s)")

            oof_probs_l.append(oof_probs)
            oof_true_l.append(oof_true)
            fold_recalls.append(recall)

        oof_probs_all = np.concatenate(oof_probs_l)
        oof_true_all  = np.concatenate(oof_true_l)
        oof_y_bin     = (oof_true_all != 0).astype(int)
        oof_preds_all = oof_probs_all.argmax(axis=1)

        cv_recall = float(np.mean(fold_recalls))
        goal_met  = cv_recall >= TARGET_RECALL
        print(f"\n  Mean CV Fail recall = {cv_recall:.3f}  "
              f"({'MET' if goal_met else 'NOT MET'} -- target {TARGET_RECALL:.0%})")

        # Threshold tuning
        print("\nTuning threshold...")

        # Build a dummy classes array matching the model's 3-class output
        class_arr = np.array([0, 1, 2])
        p_fail_oof  = 1.0 - oof_probs_all[:, 0]
        from sklearn.metrics import precision_recall_curve
        prec, rec, thr = precision_recall_curve(oof_y_bin, p_fail_oof)
        valid = rec[:-1] >= TARGET_RECALL
        threshold = float(thr[valid][-1]) if valid.any() else float(thr[0])
        op_idx    = min(int(np.searchsorted(thr, threshold)), len(rec) - 1)
        pred_bin  = (p_fail_oof > threshold).astype(int)
        thr_stats = {
            "threshold":       threshold,
            "achieved_recall": float(rec[op_idx]),
            "achieved_prec":   float(prec[op_idx]),
            "false_reject":    float((p_fail_oof[oof_y_bin == 0] > threshold).mean()),
        }
        print(f"  Threshold={threshold:.4f}  "
              f"recall={thr_stats['achieved_recall']:.3f}  "
              f"false_reject={thr_stats['false_reject']:.3f}")

        # Confusion matrices and PR curve
        pred_bin_thr = (p_fail_oof > threshold).astype(int)
        cm3 = confusion_matrix(oof_true_all, oof_preds_all, labels=[0, 1, 2])
        cm2 = confusion_matrix(oof_y_bin,    pred_bin_thr,  labels=[0, 1])

        plot_cm(cm3, LABEL_NAMES,
                f"3-class CM -- OOF -- finetune {BACKBONE}",
                REPORTS / "confusion_matrix_3class.png")
        plot_cm(cm2, ["Pass", "Fail"],
                f"Binary CM -- OOF @ thr={threshold:.3f}",
                REPORTS / "confusion_matrix_binary.png")
        plot_pr(oof_y_bin, p_fail_oof, threshold,
                REPORTS / "precision_recall_curve.png")

        # Final model: train on all folds 0-3
        print(f"\nTraining final model on folds 0-{COMP_FOLD-1}...")
        tr_final_raw = [(p, l) for p, l, f in entries if f != COMP_FOLD]
        tr_loader    = torch.utils.data.DataLoader(
            CoilDataset(tr_final_raw, TRAIN_TF),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        )

        final_model  = CoilNet()
        opt_final    = make_optimizer(final_model)
        sch_final    = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_final, T_max=MAX_EPOCHS, eta_min=1e-7,
        )
        criterion    = nn.CrossEntropyLoss(weight=cw_tensor)

        for epoch in range(1, MAX_EPOCHS + 1):
            train_epoch(final_model, tr_loader, opt_final, criterion)
            sch_final.step()

        # Fold 4 eval
        print(f"Evaluating on fold {COMP_FOLD}...")
        f4_raw     = [(p, l) for p, l, f in entries if f == COMP_FOLD]
        f4_loader  = torch.utils.data.DataLoader(
            CoilDataset(f4_raw, VAL_TF),
            batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0,
        )
        f4_probs, f4_true = eval_loader(final_model, f4_loader)
        f4_pfail   = 1.0 - f4_probs[:, 0]
        f4_bin     = (f4_pfail > threshold).astype(int)
        f4_bm      = binary_metrics((f4_true != 0).astype(int), f4_bin)
        f4_preds   = f4_probs.argmax(axis=1)

        print(f"  n={len(f4_true)}  FailRecall={f4_bm['fail_recall']:.3f}  "
              f"FalseReject={f4_bm['false_reject']:.3f}")
        print(f"\n{classification_report(f4_true, f4_preds, target_names=LABEL_NAMES)}")

        # Log to MLflow
        mlflow.log_metrics({
            "cv_fail_recall":     cv_recall,
            "thr_fail_recall":    thr_stats["achieved_recall"],
            "thr_fail_precision": thr_stats["achieved_prec"],
            "thr_false_reject":   thr_stats["false_reject"],
            "threshold":          threshold,
            "fold4_fail_recall":  f4_bm["fail_recall"],
            "fold4_false_reject": f4_bm["false_reject"],
        })
        for art in ["confusion_matrix_3class.png", "confusion_matrix_binary.png",
                    "precision_recall_curve.png"]:
            mlflow.log_artifact(str(REPORTS / art))

        # Save model
        PROD_DIR.mkdir(parents=True, exist_ok=True)
        model_path = PROD_DIR / "model.pt"
        torch.save({
            "state_dict":  final_model.state_dict(),
            "backbone":    BACKBONE,
            "n_unfreeze":  N_UNFREEZE,
            "input_w":     INPUT_W,
            "input_h":     INPUT_H,
            "dropout":     DROPOUT,
            "n_classes":   3,
            "label_map":   LABEL_MAP,
            "threshold":   threshold,
        }, model_path)

        metadata = {
            "approach":          "finetune_end_to_end",
            "backbone":          BACKBONE,
            "n_unfreeze_blocks": N_UNFREEZE,
            "input_size":        f"{INPUT_W}x{INPUT_H}",
            "classes":           LABEL_NAMES,
            "label_map":         LABEL_MAP,
            "pass_threshold":    round(threshold, 6),
            "review_band_delta": DELTA,
            "cv_fail_recall":    round(cv_recall, 4),
            "thr_fail_recall":   round(thr_stats["achieved_recall"], 4),
            "thr_false_reject":  round(thr_stats["false_reject"], 4),
            "fold4_fail_recall": round(f4_bm["fail_recall"], 4),
            "fold4_false_reject":round(f4_bm["false_reject"], 4),
            "n_train":           len(tr_final_raw),
            "mlflow_run_id":     run.info.run_id,
        }
        with open(PROD_DIR / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        import yaml
        with open(ROOT / "config.yaml", encoding="utf-8") as f:
            live_cfg = yaml.safe_load(f)
        live_cfg["decision"]["pass_threshold"] = round(threshold, 6)
        with open(ROOT / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(live_cfg, f, default_flow_style=False, allow_unicode=True)

        verdict = "PASS" if goal_met else "FAIL -- see next steps"
        card = f"""# Model Card -- Coil Defect Inspection v2 (fine-tuned)

**Approach:** End-to-end fine-tuning (EfficientNet-B0, last {N_UNFREEZE} blocks)
**Input:** {INPUT_W}x{INPUT_H} px (resized from 448x192 crops)
**MLflow run:** `{run.info.run_id}`

## CV Performance (OOF, folds 0-{COMP_FOLD-1})

| Metric | Value |
|--------|-------|
| Fail recall (target >={TARGET_RECALL:.0%}) | **{cv_recall:.1%}** |
| Fail recall @ tuned threshold | {thr_stats['achieved_recall']:.1%} |
| False-reject @ tuned threshold | {thr_stats['false_reject']:.1%} |
| Pass/Fail threshold | {threshold:.4f} |
| Review band | +/-{DELTA} |

## Fold {COMP_FOLD} Independent Check

| Metric | Value |
|--------|-------|
| Fail recall | {f4_bm['fail_recall']:.1%} |
| False-reject | {f4_bm['false_reject']:.1%} |

## Verdict: {verdict}
"""
        (PROD_DIR / "model_card.md").write_text(card, encoding="utf-8")
        mlflow.log_artifact(str(PROD_DIR / "metadata.json"))

        print(f"\n{'='*62}")
        print(f"VERDICT: {verdict}")
        print(f"  CV Fail recall  = {cv_recall:.1%}  (target {TARGET_RECALL:.0%})")
        print(f"  Fold 4 recall   = {f4_bm['fail_recall']:.1%}")
        print(f"  False-reject    = {thr_stats['false_reject']:.1%} (CV)")
        print(f"  Threshold       = {threshold:.4f}")
        print(f"  Model saved  -> {model_path}")
        print(f"{'='*62}")


if __name__ == "__main__":
    main()
