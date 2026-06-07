"""
Diagnostic for the fold-3 / fold-4 generalization gap.

Three parts:
  A. Per-fold composition (class, defect subtype, brightness).
  B. LEAKAGE CHECK -- does the same physical coil (session_id + position)
     appear in multiple folds? Near-duplicate exposures of one coil split
     across train/val inflate CV and hurt held-out generalization.
  C. Model errors on the held-out fold 4: run production model.pt on each
     fold-4 crop, list every misclassification, save an annotated montage.

Run:
    python -m src.training.diagnose
"""

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.config_loader import load_config
from src.inference.predictor import Predictor
from src.training.dataset import LABEL_NAMES

cfg       = load_config()
MANIFEST  = ROOT / cfg["paths"]["manifest"]
CROPS_DIR = ROOT / cfg["paths"]["crops_dir"]
REPORTS   = ROOT / "reports"
COMP_FOLD = cfg["training"]["comparison_fold"]
N_FOLDS   = cfg["training"]["n_folds"]


def load_manifest():
    with open(MANIFEST, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def crop_path(row):
    return CROPS_DIR / row["label"] / f"{Path(row['filepath']).stem}.npy"


def brightness(row):
    arr = np.load(crop_path(row))
    return float(arr.mean())


# ---------------------------------------------------------------------------
# Part A -- per-fold composition
# ---------------------------------------------------------------------------

def part_a(rows):
    print("=" * 66)
    print("A. Per-fold composition")
    print("=" * 66)
    print(f"{'Fold':<6}{'Pass':>6}{'Dent':>6}{'Loose':>7}{'Total':>7}"
          f"{'Brightness(mean+/-std)':>26}")
    for f in range(N_FOLDS):
        fr = [r for r in rows if int(r["fold"]) == f]
        cnt = Counter(r["label"] for r in fr)
        br = np.array([brightness(r) for r in fr])
        tag = "  <- comparison" if f == COMP_FOLD else ""
        print(f"{f:<6}{cnt['Pass']:>6}{cnt['Dent']:>6}{cnt['Loose']:>7}"
              f"{len(fr):>7}{f'{br.mean():>14.1f} +/- {br.std():.1f}':>26}{tag}")


# ---------------------------------------------------------------------------
# Part B -- leakage check (same coil across folds)
# ---------------------------------------------------------------------------

def part_b(rows):
    print("\n" + "=" * 66)
    print("B. Leakage check -- same coil (session_id + position) across folds")
    print("=" * 66)

    # Group images by (session, position) = candidate unique physical coil
    coils = defaultdict(list)
    for r in rows:
        coils[(r["session_id"], r["position"])].append(r)

    n_coils      = len(coils)
    n_multi      = sum(1 for v in coils.values() if len(v) > 1)
    spanning     = {k: v for k, v in coils.items()
                    if len({int(r["fold"]) for r in v}) > 1}
    n_spanning   = len(spanning)

    print(f"  Total images              : {len(rows)}")
    print(f"  Unique (session,position) : {n_coils}")
    print(f"  ...with >1 image          : {n_multi}  "
          f"(avg {len(rows)/n_coils:.1f} images per coil)")
    print(f"  ...spanning >1 fold       : {n_spanning}  "
          f"<-- LEAKAGE if >0")

    # How many images are involved in cross-fold leakage
    leaked_imgs = sum(len(v) for v in spanning.values())
    print(f"  Images involved in leak   : {leaked_imgs} "
          f"({100*leaked_imgs/len(rows):.0f}% of dataset)")

    # Focus: for the fold-3 CV split (train={0,1,2}, val=3) and the final
    # split (train={0,1,2,3}, val=4), how many val coils also appear in train?
    for val_fold, train_folds, name in [
        (3, {0, 1, 2},       "Fold-3 CV split (train 0-2, val 3)"),
        (4, {0, 1, 2, 3},    "Fold-4 final split (train 0-3, val 4)"),
    ]:
        val_coils = {k for k, v in coils.items()
                     if any(int(r["fold"]) == val_fold for r in v)}
        leaked = 0
        for k in val_coils:
            folds_of_coil = {int(r["fold"]) for r in coils[k]}
            if folds_of_coil & train_folds:
                leaked += 1
        print(f"\n  {name}:")
        print(f"    val coils total      : {len(val_coils)}")
        print(f"    val coils also in tr : {leaked}  "
              f"({100*leaked/max(len(val_coils),1):.0f}% leaked)")

    # Defect-only leakage (the classes we care about for recall)
    defect_rows = [r for r in rows if r["label"] != "Pass"]
    defect_coils = defaultdict(list)
    for r in defect_rows:
        defect_coils[(r["session_id"], r["position"])].append(r)
    defect_spanning = sum(1 for v in defect_coils.values()
                          if len({int(r["fold"]) for r in v}) > 1)
    print(f"\n  Defect coils (Dent/Loose): {len(defect_coils)} unique, "
          f"{defect_spanning} span >1 fold")

    return spanning


# ---------------------------------------------------------------------------
# Part C -- model.pt errors on held-out fold 4
# ---------------------------------------------------------------------------

def part_c(rows):
    print("\n" + "=" * 66)
    print(f"C. Production model errors on held-out fold {COMP_FOLD}")
    print("=" * 66)

    predictor = Predictor()
    f4 = [r for r in rows if int(r["fold"]) == COMP_FOLD]

    records = []
    for r in rows_progress(f4):
        crop = np.load(crop_path(r))
        probs = predictor._probs_pytorch(crop)   # exact training preprocessing
        pred  = LABEL_NAMES[int(probs.argmax())]
        records.append({
            "row": r, "true": r["label"], "pred": pred,
            "probs": probs, "crop": crop,
        })

    # confusion
    print(f"\n  Confusion (rows=true, cols=pred):")
    print(f"    {'':>8}" + "".join(f"{n:>8}" for n in LABEL_NAMES))
    for t in LABEL_NAMES:
        line = f"    {t:>8}"
        for p in LABEL_NAMES:
            c = sum(1 for x in records if x["true"] == t and x["pred"] == p)
            line += f"{c:>8}"
        print(line)

    # critical errors: defect predicted as Pass (missed defects)
    missed = [x for x in records
              if x["true"] != "Pass" and x["pred"] == "Pass"]
    false_rej = [x for x in records
                 if x["true"] == "Pass" and x["pred"] != "Pass"]
    subtype_conf = [x for x in records
                    if x["true"] != "Pass" and x["pred"] != "Pass"
                    and x["true"] != x["pred"]]

    print(f"\n  MISSED DEFECTS (true=Dent/Loose, pred=Pass): {len(missed)}  "
          f"<-- critical")
    for x in missed:
        r = x["row"]
        print(f"    {r['label']:>5}  {Path(r['filepath']).name[:46]:<48} "
              f"P(Pass)={x['probs'][0]:.2f}")
    print(f"\n  False rejects (true=Pass, pred=defect): {len(false_rej)}")
    print(f"  Subtype confusion (Dent<->Loose):        {len(subtype_conf)}")

    # montage of missed defects (most important to eyeball)
    if missed:
        save_error_montage(missed, REPORTS / "diagnose_fold4_missed.png",
                           "Fold 4 MISSED DEFECTS (predicted Pass)")
        print(f"\n  Montage -> reports/diagnose_fold4_missed.png")

    return records, missed


def rows_progress(items):
    for i, it in enumerate(items, 1):
        if i % 40 == 0 or i == len(items):
            print(f"    scoring fold-4 image {i}/{len(items)}")
        yield it


def save_error_montage(errors, path, title):
    n = len(errors)
    cols = min(4, n)
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols,
                             figsize=(cols * 3.2, rows_n * 1.9))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, x in zip(axes, errors):
        ax.imshow(x["crop"])
        p = x["probs"]
        ax.set_title(f"true={x['true']} pred={x['pred']}\n"
                     f"P={p[0]:.2f}/{p[1]:.2f}/{p[2]:.2f}", fontsize=7)
    fig.suptitle(title + "  (P = Pass/Dent/Loose)", fontsize=10)
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    rows = load_manifest()
    part_a(rows)
    part_b(rows)
    part_c(rows)
    print("\n" + "=" * 66)
    print("Diagnosis complete.")
    print("=" * 66)


if __name__ == "__main__":
    main()
