"""
Anomaly-detection framing for coil defect inspection.

Motivation
----------
Two supervised fine-tuning attempts (class-weighted loss, then balanced
sampler + strong augmentation) both topped out at ~54% CV Fail-recall @ 20%
false-reject -- the bottleneck is the *number of distinct defect examples*
(74 Dent / 110 Loose), not the training recipe. You cannot resample your way
past 74 unique dents.

This module sidesteps that bottleneck: it trains ONLY on the abundant 633
Pass coils to model "normal", and flags deviations as Fail. Defect images are
never used for fitting -- only to set the operating threshold and evaluate.

Method (PaDiM-style, CPU-friendly)
----------------------------------
- Frozen ImageNet ResNet18, mid-level feature maps (layer2 stride8 +
  layer3 stride16). Concatenate at the layer3 grid (14x28 = 392 patch
  locations, 384 channels). Random-select R channels (seeded) for a stable,
  invertible per-patch covariance.
- PaDiM: per patch location, fit a Gaussian (mean + shrunk covariance) over
  Pass coils. Test anomaly map = per-patch Mahalanobis distance. IMAGE score
  = MAX over patches -- a small localized dent survives instead of being
  averaged away (the exact failure mode of the global-pool classifier).
- Global Mahalanobis comparator: avg-pool the patch grid to one vector, fit
  one Gaussian on Pass, score by Mahalanobis. Shows how much the patch-level
  localization actually buys.

Evaluation is the SAME honest protocol as finetune.py so numbers are directly
comparable: OOF recall @ false-reject budget over folds 0-3, plus a held-out
fold-4 check. Defects only ever appear on the scoring side, never the fit side.

Run:
    python -m src.training.anomaly
"""

import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.config_loader import load_config
from src.training.dataset import LABEL_MAP, LABEL_NAMES
from src.training.finetune import recall_at_fr, fr_table, INPUT_W, INPUT_H

cfg       = load_config()
CROPS_DIR = ROOT / cfg["paths"]["crops_dir"]
MANIFEST  = ROOT / cfg["paths"]["manifest"]
COMP_FOLD = cfg["training"]["comparison_fold"]
N_FOLDS   = cfg["training"]["n_folds"]
SEED      = cfg["training"]["random_seed"]
FR_BUDGET = 0.20
TARGET_RECALL = cfg["evaluation"]["target_fail_recall"]
PROD_DIR  = ROOT / cfg["paths"]["production_dir"]

# Human-assist decision policy (see fit_and_save_production):
#   score >= t_flag           -> AUTO-FLAG  (clear defect, human confirms before scrap)
#   t_low <= score < t_flag   -> REVIEW     (human looks)
#   score <  t_low            -> AUTO-PASS  (clearly normal; offloads the easy bulk)
FLAG_FR   = 0.02   # auto-flag threshold set at 2% false-alarm -> trustworthy flags
MISS_TOL  = 0.02   # auto-pass band may contain at most this fraction of defects

BACKBONE     = "resnet18"
OUT_INDICES  = (1, 2)      # layer1 (stride4) + layer2 (stride8): finer 28x56=1568
                           # patch grid. Finer localization measurably helped a
                           # small dent survive max-pooling: global-pool 40% ->
                           # coarse-patch (2,3) 50% -> fine-patch (1,2) 55% CV
                           # recall @20% FR.
R_CHANNELS   = 100         # random channel subset (PaDiM default for resnet18)
COV_EPS      = 0.01        # covariance regularization (+eps*I), PaDiM value
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def build_feature_model():
    m = timm.create_model(BACKBONE, pretrained=True, features_only=True,
                          out_indices=OUT_INDICES)
    m.eval()
    return m


_MEAN_T = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
_STD_T  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)


def _preprocess(arr):
    """uint8 (H,W,3) -> normalized (1,3,INPUT_H,INPUT_W) tensor."""
    t = torch.from_numpy(arr.copy()).float().div(255.0).permute(2, 0, 1)[None]
    t = F.interpolate(t, size=(INPUT_H, INPUT_W), mode="bilinear",
                      align_corners=False)
    return (t - _MEAN_T) / _STD_T


@torch.no_grad()
def _patch_matrix(model, arr, chan_idx):
    """uint8 crop -> (L, R) patch-feature matrix on the layer3 grid.

    L = H'*W' patch locations, R = len(chan_idx) selected channels.
    """
    maps = model(_preprocess(arr))           # [ (1,128,28,56), (1,256,14,28) ]
    ref_hw = maps[-1].shape[-2:]             # align everything to coarsest grid
    aligned = [F.interpolate(m, size=ref_hw, mode="bilinear",
                             align_corners=False) for m in maps]
    feat = torch.cat(aligned, dim=1)         # (1, 384, 14, 28)
    feat = feat[:, chan_idx]                 # (1, R, 14, 28)
    C = feat.shape[1]
    feat = feat.squeeze(0).reshape(C, -1).T  # (L, R)
    return feat.numpy().astype(np.float32)


def extract_all(entries, model, chan_idx):
    """entries: list of (path, label_int, fold). Returns X (N,L,R), y, folds."""
    X, y, folds = [], [], []
    t0 = time.perf_counter()
    for i, (p, lab, fold) in enumerate(entries, 1):
        X.append(_patch_matrix(model, np.load(p), chan_idx))
        y.append(lab)
        folds.append(fold)
        if i % 100 == 0 or i == len(entries):
            print(f"    extracted {i}/{len(entries)} "
                  f"({time.perf_counter()-t0:.0f}s)")
    return np.stack(X), np.array(y), np.array(folds)


# ---------------------------------------------------------------------------
# PaDiM (per-patch Gaussian) and global Mahalanobis
# ---------------------------------------------------------------------------

class PaDiM:
    """Per-patch-location Gaussian; image score = max patch Mahalanobis."""

    def fit(self, X_normal):
        # X_normal: (N, L, R)
        N, L, R = X_normal.shape
        self.mean = X_normal.mean(axis=0)                  # (L, R)
        self.inv  = np.empty((L, R, R), dtype=np.float32)
        eye = np.eye(R, dtype=np.float32) * COV_EPS
        for l in range(L):
            c = np.cov(X_normal[:, l, :], rowvar=False).astype(np.float32)
            self.inv[l] = np.linalg.inv(c + eye)
        return self

    def score(self, X):
        # returns (N,) image-level anomaly score = max over patches
        N, L, R = X.shape
        d = X - self.mean[None]                             # (N, L, R)
        # Mahalanobis^2 per (n,l): sum((d @ inv) * d)
        m = np.einsum("nlr,lrs,nls->nl", d, self.inv, d)    # (N, L)
        m = np.sqrt(np.maximum(m, 0.0))
        return m.max(axis=1)


class GlobalMahalanobis:
    """Avg-pool patches to one vector; single Gaussian; Mahalanobis score."""

    def fit(self, X_normal):
        v = X_normal.mean(axis=1)                           # (N, R)
        self.mean = v.mean(axis=0)
        c = np.cov(v, rowvar=False).astype(np.float32)
        self.inv = np.linalg.inv(c + np.eye(c.shape[0],
                                            dtype=np.float32) * COV_EPS)
        return self

    def score(self, X):
        v = X.mean(axis=1)
        d = v - self.mean[None]
        m = np.einsum("nr,rs,ns->n", d, self.inv, d)
        return np.sqrt(np.maximum(m, 0.0))


# ---------------------------------------------------------------------------
# Evaluation (honest recall @ false-reject budget, same as finetune.py)
# ---------------------------------------------------------------------------

def evaluate(name, model_cls, X, y, folds):
    print("\n" + "=" * 62)
    print(f"{name}")
    print("=" * 62)

    # --- OOF over folds 0..COMP_FOLD-1: fit on Pass of train folds, score val
    oof_scores, oof_ybin = [], []
    for val_fold in range(COMP_FOLD):
        train_mask = (folds != val_fold) & (folds != COMP_FOLD) & (y == 0)
        val_mask   = folds == val_fold
        det = model_cls().fit(X[train_mask])
        s   = det.score(X[val_mask])
        oof_scores.append(s)
        oof_ybin.append((y[val_mask] != 0).astype(int))
    oof_scores = np.concatenate(oof_scores)
    oof_ybin   = np.concatenate(oof_ybin)

    print("  OOF recall @ false-reject budget:")
    oof_table = fr_table(oof_scores, oof_ybin)
    rec, thr, fr = recall_at_fr(oof_scores, oof_ybin, FR_BUDGET)
    met = rec >= TARGET_RECALL
    print(f"\n  Operating point (FR<= {FR_BUDGET:.0%}): "
          f"score_thr={thr:.3f}  Fail recall={rec:.1%}  FR={fr:.0%}")
    print(f"  -> {'MEETS' if met else 'BELOW'} target {TARGET_RECALL:.0%} "
          f"recall @ <= {FR_BUDGET:.0%} false-reject")

    # --- Held-out fold 4: fit on Pass of folds 0-3, score fold 4
    train_mask = (folds != COMP_FOLD) & (y == 0)
    f4_mask    = folds == COMP_FOLD
    det = model_cls().fit(X[train_mask])
    f4_scores = det.score(X[f4_mask])
    f4_ybin   = (y[f4_mask] != 0).astype(int)
    print(f"\n  Fold {COMP_FOLD} (held-out) recall @ false-reject budget:")
    f4_table = fr_table(f4_scores, f4_ybin)

    # recall at the OOF-chosen threshold (honest deployed operating point)
    pred = (f4_scores >= thr).astype(int)
    pos, neg = f4_ybin == 1, f4_ybin == 0
    f4_rec = float(pred[pos].mean()) if pos.any() else 0.0
    f4_fr  = float(pred[neg].mean()) if neg.any() else 0.0
    print(f"\n  At deployed threshold {thr:.3f}: "
          f"FailRecall={f4_rec:.1%}  FalseReject={f4_fr:.0%}")

    return {"name": name, "oof_rec20": rec, "oof_fr": fr, "thr": thr,
            "f4_rec_at_thr": f4_rec, "f4_fr_at_thr": f4_fr,
            "oof_table": oof_table, "f4_table": f4_table,
            "oof_scores": oof_scores, "oof_ybin": oof_ybin}


# ---------------------------------------------------------------------------
# Production: fit deployable PaDiM + derive human-assist decision policy
# ---------------------------------------------------------------------------

def _band_split(scores, ybin, t_low, t_flag):
    """Return workload fractions and the critical auto-pass defect-miss rate."""
    ap   = scores < t_low
    flag = scores >= t_flag
    rev  = ~ap & ~flag
    defects = ybin == 1
    miss = float((ap & defects).sum() / max(defects.sum(), 1))   # defects auto-passed
    return {
        "auto_pass": float(ap.mean()), "review": float(rev.mean()),
        "flag": float(flag.mean()), "autopass_defect_miss": miss,
        "flag_defect_recall": float((flag & defects).sum() / max(defects.sum(), 1)),
    }


def fit_and_save_production(X, y, folds, chan_idx, oof_scores, oof_ybin):
    """Fit PaDiM on Pass(folds 0-3), derive the 3-band policy from OOF scores,
    characterize the workload on OOF + held-out fold 4, and save the artifact."""
    print("\n" + "=" * 62)
    print("PRODUCTION: deployable PaDiM + human-assist decision policy")
    print("=" * 62)

    # Thresholds derived purely from OOF (defects only used to place bands)
    _, t_flag, _ = recall_at_fr(oof_scores, oof_ybin, FLAG_FR)
    defect_scores = oof_scores[oof_ybin == 1]
    t_low = float(np.quantile(defect_scores, MISS_TOL))   # <=MISS_TOL defects below
    t_low = min(t_low, t_flag)

    print(f"  t_low (auto-pass ceiling) = {t_low:.3f}  "
          f"(<= {MISS_TOL:.0%} of OOF defects below)")
    print(f"  t_flag (auto-flag floor)  = {t_flag:.3f}  "
          f"(set at {FLAG_FR:.0%} false-alarm)")

    oof_split = _band_split(oof_scores, oof_ybin, t_low, t_flag)
    print(f"\n  Workload split (OOF folds 0-3):")
    print(f"    AUTO-PASS {oof_split['auto_pass']:5.1%}   "
          f"REVIEW {oof_split['review']:5.1%}   "
          f"AUTO-FLAG {oof_split['flag']:5.1%}")
    print(f"    defects auto-passed (critical): {oof_split['autopass_defect_miss']:.1%}")
    print(f"    defects auto-flagged           : {oof_split['flag_defect_recall']:.1%}")

    # Fit final detector on ALL Pass in folds 0-3
    det = PaDiM().fit(X[(folds != COMP_FOLD) & (y == 0)])

    # Honest held-out characterization on fold 4
    f4_mask = folds == COMP_FOLD
    f4_scores = det.score(X[f4_mask])
    f4_ybin   = (y[f4_mask] != 0).astype(int)
    f4_split  = _band_split(f4_scores, f4_ybin, t_low, t_flag)
    print(f"\n  Workload split (HELD-OUT fold 4 -- honest):")
    print(f"    AUTO-PASS {f4_split['auto_pass']:5.1%}   "
          f"REVIEW {f4_split['review']:5.1%}   "
          f"AUTO-FLAG {f4_split['flag']:5.1%}")
    print(f"    defects auto-passed (critical): {f4_split['autopass_defect_miss']:.1%}  "
          f"<-- must stay near 0")
    print(f"    defects auto-flagged           : {f4_split['flag_defect_recall']:.1%}")

    PROD_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "kind":        "padim",
        "backbone":    BACKBONE,
        "out_indices": OUT_INDICES,
        "chan_idx":    chan_idx,
        "input_w":     INPUT_W, "input_h": INPUT_H,
        "imagenet_mean": IMAGENET_MEAN, "imagenet_std": IMAGENET_STD,
        "mean":        det.mean.astype(np.float32),
        "inv":         det.inv.astype(np.float32),
        "t_low":       t_low, "t_flag": t_flag,
        "cov_eps":     COV_EPS,
    }
    path = PROD_DIR / "padim.pkl"
    with open(path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\n  Saved -> {path}  ({path.stat().st_size/1e6:.0f} MB)")

    card = f"""# Model Card -- Coil Defect Inspection (PaDiM, human-assist)

**Approach:** Unsupervised anomaly detection (PaDiM). Trains on Pass coils
ONLY -- no defect labels needed. Backbone: frozen ImageNet {BACKBONE},
layers {OUT_INDICES} (fine patch grid). Image score = MAX patch Mahalanobis.

**Why human-assist, not auto-pass:** Four approaches across supervised and
anomaly paradigms all hit ~50-55% Fail-recall @ 20% false-reject -- the
dataset (74 Dent / 110 Loose) cannot support the >=95% auto-recall target.
This detector is deployed as a triage/ranking aid; humans confirm; each
confirmation grows the defect set.

## Decision policy (3-band)

| Band | Rule | Action |
|---|---|---|
| AUTO-FLAG | score >= {t_flag:.3f} | clear defect -- human confirms before scrap |
| REVIEW    | {t_low:.3f} <= score < {t_flag:.3f} | human looks |
| AUTO-PASS | score < {t_low:.3f} | clearly normal -- offloads easy bulk |

Thresholds derived from OOF (folds 0-3); defects only used to place bands.

## Workload (held-out fold 4)

- AUTO-PASS {f4_split['auto_pass']:.0%} / REVIEW {f4_split['review']:.0%} / AUTO-FLAG {f4_split['flag']:.0%}
- **Defects auto-passed (critical, must stay ~0): {f4_split['autopass_defect_miss']:.1%}**
- Defects auto-flagged: {f4_split['flag_defect_recall']:.0%}

NOT a >=95% auto-recall gate. Honest CV Fail-recall @20% FR ~= 55%.
"""
    (PROD_DIR / "padim_card.md").write_text(card, encoding="utf-8")
    return t_low, t_flag


# ---------------------------------------------------------------------------

def main():
    print("=" * 62)
    print("Anomaly detection (train on Pass only)")
    print(f"  Backbone: {BACKBONE} (frozen)  layers {OUT_INDICES}  "
          f"R={R_CHANNELS} channels")
    print(f"  Input: {INPUT_W}x{INPUT_H}")
    print("=" * 62)

    with open(MANIFEST, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    entries = []
    for r in rows:
        p = CROPS_DIR / r["label"] / f"{Path(r['filepath']).stem}.npy"
        if p.exists():
            entries.append((p, LABEL_MAP[r["label"]], int(r["fold"])))

    cnt = {c: sum(1 for _, l, _ in entries if l == LABEL_MAP[c]) for c in LABEL_NAMES}
    print(f"\nCrops: {len(entries)}  " +
          "  ".join(f"{c}={n}" for c, n in cnt.items()))
    n_pass_cv = sum(1 for _, l, f in entries if l == 0 and f != COMP_FOLD)
    print(f"Pass coils available to model 'normal' (folds 0-3): {n_pass_cv}")

    # Random channel subset (seeded) over the 384 concatenated channels
    rng = np.random.RandomState(SEED)
    total_ch = sum(timm.create_model(BACKBONE, features_only=True,
                                     out_indices=OUT_INDICES).feature_info.channels())
    chan_idx = np.sort(rng.choice(total_ch, size=R_CHANNELS, replace=False))

    model = build_feature_model()
    print("\nExtracting patch features (frozen, one pass)...")
    X, y, folds = extract_all(entries, model, chan_idx)
    print(f"  feature tensor: {X.shape}  ({X.nbytes/1e6:.0f} MB)")

    results = [
        evaluate("PaDiM (per-patch, MAX) -- localized-defect aware",
                 PaDiM, X, y, folds),
        evaluate("Global Mahalanobis (avg-pool) -- comparator",
                 GlobalMahalanobis, X, y, folds),
    ]

    print("\n" + "=" * 62)
    print("SUMMARY vs supervised baseline (54% CV recall @20% FR)")
    print("=" * 62)
    print(f"  {'method':<34}{'CV rec@20%FR':>14}{'fold4@thr':>12}")
    for r in results:
        print(f"  {r['name'][:32]:<34}{r['oof_rec20']:>13.1%}"
              f"{r['f4_rec_at_thr']:>11.1%}")
    print(f"  {'supervised finetune (baseline)':<34}{0.537:>13.1%}{0.730:>11.1%}")

    # Deploy the PaDiM detector (results[0]) with the human-assist policy
    padim = results[0]
    fit_and_save_production(X, y, folds, chan_idx,
                            padim["oof_scores"], padim["oof_ybin"])


if __name__ == "__main__":
    main()
