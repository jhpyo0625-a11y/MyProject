"""
T1.4 — Stratified 5-fold split.

Adds a 'fold' column (0-4) to data/manifest.csv.
Fold 4 is reserved as the comparison fold for the model approval gate
and is never used in training (see TASKS.md T7.3).
Folds 0-3 rotate as train/validation during CV.
"""

import sys
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.config_loader import load_config

from sklearn.model_selection import StratifiedKFold

cfg      = load_config()
MANIFEST = ROOT / cfg["paths"]["manifest"]
N_FOLDS  = cfg["training"]["n_folds"]
SEED     = cfg["training"]["random_seed"]


def main():
    with open(MANIFEST, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    labels  = [r["label"] for r in rows]
    indices = list(range(len(rows)))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for row in rows:
        row["fold"] = ""

    for fold_idx, (_, val_idx) in enumerate(skf.split(indices, labels)):
        for i in val_idx:
            rows[i]["fold"] = str(fold_idx)

    # write back
    fieldnames = list(rows[0].keys())
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # summary
    from collections import Counter
    print(f"Stratified {N_FOLDS}-fold split written to: {MANIFEST}\n")
    print(f"{'Fold':<6} {'Pass':>6} {'Dent':>6} {'Loose':>6} {'Total':>7}")
    print("-" * 32)
    for fold in range(N_FOLDS):
        fold_rows = [r for r in rows if r["fold"] == str(fold)]
        cnt = Counter(r["label"] for r in fold_rows)
        tag = " <- comparison fold (never trained on)" if fold == N_FOLDS - 1 else ""
        print(f"  {fold}    {cnt['Pass']:>4}   {cnt['Dent']:>4}   {cnt['Loose']:>4}   {len(fold_rows):>5}{tag}")


if __name__ == "__main__":
    main()
