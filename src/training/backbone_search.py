"""
T3.5 -- Backbone search: try multiple (backbone, pool_mode) combinations.

Extracts embeddings on-the-fly from saved crops (no cache rebuild needed)
and runs a fast 4-fold CV with the best classifier from T3.6 (LogReg C=10
with StandardScaler).  Reports CV Fail recall for every combination.

Combinations tried:
  resnet18     x {max, avg}
  efficientnet_b0 x {max, avg}

The winning combination is printed at the end. To promote it:
  1. Update config.yaml -> model.backbone
  2. Run src/features/build_cache.py  (rebuilds cache for production)
  3. Run src/features/build_augmented_cache.py
  4. Re-run src/training/train.py to serialize final model

Run:
    python -m src.training.backbone_search
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import timm
import torchvision.transforms.functional as TF
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.config_loader import load_config
from src.training.dataset import LABEL_MAP, LABEL_NAMES
from src.training.train import COMP_FOLD, TARGET_RECALL, binary_metrics, p_fail_from_proba

cfg = load_config()

CROPS_DIR  = ROOT / cfg["paths"]["crops_dir"]
EMB_DIR    = ROOT / cfg["paths"]["embeddings_dir"]
MANIFEST   = ROOT / cfg["paths"]["manifest"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

SEARCH_GRID = [
    ("resnet18",        "max"),
    ("resnet18",        "avg"),
    ("efficientnet_b0", "max"),
    ("efficientnet_b0", "avg"),
]


# ---------------------------------------------------------------------------

def make_model(backbone: str):
    model = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="")
    model.eval()
    return model


@torch.no_grad()
def embed_batch(model, arrs: list, pool: str) -> np.ndarray:
    """arrs: list of uint8 (H,W,3) numpy arrays -> (N, D) float32"""
    tensors = []
    for arr in arrs:
        t = torch.from_numpy(arr.copy()).float() / 255.0
        t = t.permute(2, 0, 1)
        t = TF.normalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        tensors.append(t)
    batch  = torch.stack(tensors)           # (N, C, H, W)
    feats  = model(batch)                   # (N, D, H', W')
    if pool == "max":
        pooled = feats.amax(dim=(2, 3))     # (N, D)
    else:
        pooled = feats.mean(dim=(2, 3))     # (N, D)
    return pooled.numpy()


def load_all_crops_and_meta():
    """Load manifest metadata and return list of (crop_path, label, fold, is_aug)."""
    with open(MANIFEST, encoding="utf-8") as f:
        rows = {Path(r["filepath"]).stem: r for r in csv.DictReader(f)}

    # also load index.csv for augmented entries
    index_csv = EMB_DIR / "index.csv"
    entries = []
    with open(index_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stem   = row["stem"]
            label  = row["label"]
            fold   = int(row["fold"])
            is_aug = row["is_augmented"] == "True"

            # derive crop path
            if is_aug:
                # aug stem = {original_stem}_aug{i}
                orig_stem = "_aug".join(stem.split("_aug")[:-1])
                crop_path = CROPS_DIR / label / f"{orig_stem}.npy"
            else:
                crop_path = CROPS_DIR / label / f"{stem}.npy"

            if crop_path.exists():
                entries.append((crop_path, LABEL_MAP[label], fold, is_aug, label))

    return entries


def extract_embeddings(entries, model, pool, batch_size=32):
    """Run forward pass in batches. Returns X (N,D), y3, y_bin, folds, is_aug."""
    X_list, y3_list, yb_list, fold_list, aug_list = [], [], [], [], []

    for start in range(0, len(entries), batch_size):
        batch = entries[start:start + batch_size]
        arrs  = [np.load(e[0]) for e in batch]
        embs  = embed_batch(model, arrs, pool)
        for emb, (_, y3, fold, is_aug, lbl) in zip(embs, batch):
            X_list.append(emb)
            y3_list.append(y3)
            yb_list.append(0 if lbl == "Pass" else 1)
            fold_list.append(fold)
            aug_list.append(is_aug)

    return (
        np.array(X_list,    dtype=np.float32),
        np.array(y3_list,   dtype=np.int32),
        np.array(yb_list,   dtype=np.int32),
        np.array(fold_list, dtype=np.int32),
        np.array(aug_list,  dtype=bool),
    )


def cv_recall(X, y3, y_bin, folds, is_aug):
    """4-fold CV with LogReg C=10 + StandardScaler. Returns mean CV Fail recall."""
    recalls = []
    for val_fold in range(COMP_FOLD):
        tr  = (folds != val_fold) & (folds != COMP_FOLD)
        val = (folds == val_fold) & ~is_aug

        pl = Pipeline([
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(
                C=10.0, class_weight="balanced",
                max_iter=1000, random_state=cfg["training"]["random_seed"])),
        ])
        pl.fit(X[tr], y3[tr])

        proba    = pl.predict_proba(X[val])
        pfail    = p_fail_from_proba(proba, pl.classes_)
        pred_bin = (pfail > 0.5).astype(int)
        bm       = binary_metrics(y_bin[val], pred_bin)
        recalls.append(bm["fail_recall"])

    return float(np.mean(recalls))


# ---------------------------------------------------------------------------

def main():
    print("=" * 62)
    print("T3.5 -- Backbone search")
    print(f"Grid: {len(SEARCH_GRID)} combinations")
    print("=" * 62)

    entries = load_all_crops_and_meta()
    print(f"\nLoaded {len(entries)} crop entries from index.\n")

    results = []
    for backbone, pool in SEARCH_GRID:
        t0 = time.perf_counter()
        print(f"  {backbone:<20} pool={pool}  ", end="", flush=True)

        model = make_model(backbone)
        X, y3, y_bin, folds, is_aug = extract_embeddings(entries, model, pool)
        recall = cv_recall(X, y3, y_bin, folds, is_aug)
        elapsed = time.perf_counter() - t0

        met = "MET" if recall >= TARGET_RECALL else "NOT MET"
        print(f"FailRecall={recall:.3f}  [{met}]  ({elapsed:.0f}s)")
        results.append((recall, backbone, pool))
        del model   # free memory between runs

    print("\n" + "=" * 62)
    print("Results (sorted by Fail recall):")
    results.sort(reverse=True)
    for recall, backbone, pool in results:
        met = "MET" if recall >= TARGET_RECALL else "NOT MET"
        print(f"  {backbone:<20} pool={pool:<4}  FailRecall={recall:.3f}  [{met}]")

    best_recall, best_backbone, best_pool = results[0]
    print(f"\nBest: {best_backbone}  pool={best_pool}  "
          f"CV Fail recall={best_recall:.3f}")
    if best_recall >= TARGET_RECALL:
        print(f"\nTarget MET. To promote:")
        print(f"  1. Set config.yaml model.backbone = {best_backbone}")
        print(f"  2. Run: python -m src.features.build_cache")
        print(f"  3. Run: python -m src.features.build_augmented_cache")
        print(f"  4. Run: python -m src.training.train")
    else:
        print(f"\nTarget NOT MET by any combination.")
        print("Consider: fine-tuning the backbone end-to-end, or tightening")
        print("the crop to isolate a single coil from the frame.")


if __name__ == "__main__":
    main()
