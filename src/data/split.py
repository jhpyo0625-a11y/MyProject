"""
T1.4 — Stratified 5-fold split.

Adds a 'fold' column (0-4) to data/manifest.csv.
Fold 4 is reserved as the comparison fold for the model approval gate
and is never used in training (see TASKS.md T7.3).
Folds 0-3 rotate as train/validation during CV.
"""

import csv
import os
import sys
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.config_loader import load_config

from sklearn.model_selection import StratifiedKFold

cfg       = load_config()
MANIFEST  = ROOT / cfg["paths"]["manifest"]
N_FOLDS   = cfg["training"]["n_folds"]
SEED      = cfg["training"]["random_seed"]
COMP_FOLD = cfg["training"]["comparison_fold"]


def _incremental_fold(stem: str) -> int:
    """Deterministic fold in [0, COMP_FOLD) for a row appended after the initial
    split. MUST match src.labeling.ingestion._assign_fold so the two entry
    points never disagree, and never assigns the held-out comparison fold."""
    return zlib.crc32(stem.encode("utf-8")) % COMP_FOLD


def assign_folds(rows: list) -> tuple:
    """Return (rows, (mode, n)) with a 'fold' set on each row.

    - Fresh manifest (no folds yet): stratified K-fold over labels (0..N-1).
    - Already-folded manifest: only fill rows MISSING a fold, via the
      incremental scheme. Existing assignments -- crucially the comparison fold
      -- are left untouched, so re-running this after ingestion can't reshuffle
      folds or leak images into the held-out fold.
    """
    if any((r.get("fold") or "") != "" for r in rows):
        new = [r for r in rows if not (r.get("fold") or "")]
        for r in new:
            r["fold"] = str(_incremental_fold(Path(r["filepath"]).stem))
        return rows, ("incremental", len(new))

    labels = [r["label"] for r in rows]
    for r in rows:
        r["fold"] = ""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold_idx, (_, val_idx) in enumerate(skf.split(range(len(rows)), labels)):
        for i in val_idx:
            rows[i]["fold"] = str(fold_idx)
    return rows, ("stratified", N_FOLDS)


def main():
    with open(MANIFEST, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows, (mode, n) = assign_folds(rows)

    # atomic write: temp file then os.replace, so an interrupt can't corrupt
    # the manifest.
    fieldnames = list(rows[0].keys())
    tmp = MANIFEST.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, MANIFEST)

    # summary
    if mode == "incremental":
        print(f"Incremental: assigned folds to {n} new row(s) "
              f"(folds 0-{COMP_FOLD-1}, never the comparison fold); "
              f"existing assignments preserved.\n")
    else:
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
