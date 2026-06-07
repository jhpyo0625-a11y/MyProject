"""
T1.4 — Stratified, coil-GROUPED 5-fold split.

Adds a 'fold' column (0-4) to data/manifest.csv.
Fold 4 is reserved as the comparison fold for the model approval gate
and is never used in training (see TASKS.md T7.3).
Folds 0-3 rotate as train/validation during CV.

Grouping: all images of one physical coil (session_id + position) are kept in
the SAME fold. A coil is imaged up to 3x; plain image-level stratification put
those near-duplicate exposures in different folds, leaking train info into
validation and inflating CV. StratifiedGroupKFold prevents that while keeping
classes balanced.

Run:
    python -m src.data.split             # initial / incremental
    python -m src.data.split --regroup   # force a full re-split by coil
"""

import argparse
import csv
import os
import sys
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.config_loader import load_config

from sklearn.model_selection import StratifiedGroupKFold

cfg       = load_config()
MANIFEST  = ROOT / cfg["paths"]["manifest"]
N_FOLDS   = cfg["training"]["n_folds"]
SEED      = cfg["training"]["random_seed"]
COMP_FOLD = cfg["training"]["comparison_fold"]


def coil_id(row: dict) -> str:
    """Physical-coil identity used as the grouping key (matches diagnose.py)."""
    return f"{row.get('session_id', '')}|{row.get('position', '')}"


def _incremental_fold(coil: str) -> int:
    """Deterministic fold in [0, COMP_FOLD) for a coil appended after the initial
    split. Hashing the COIL (not the filename) keeps all of a coil's images in
    one fold. MUST match src.labeling.ingestion._assign_fold, and never assigns
    the held-out comparison fold."""
    return zlib.crc32(coil.encode("utf-8")) % COMP_FOLD


def assign_folds(rows: list, regroup: bool = False) -> tuple:
    """Return (rows, (mode, n)) with a 'fold' set on each row.

    - Fresh manifest (no folds) or regroup=True: stratified GROUP K-fold, so
      every image of a coil shares a fold and no coil spans the train/val line.
    - Already-folded manifest (regroup=False): only fill rows MISSING a fold,
      via the coil-stable incremental scheme. Existing assignments -- crucially
      the comparison fold -- are left untouched.
    """
    if any((r.get("fold") or "") != "" for r in rows) and not regroup:
        new = [r for r in rows if not (r.get("fold") or "")]
        for r in new:
            r["fold"] = str(_incremental_fold(coil_id(r)))
        return rows, ("incremental", len(new))

    labels = [r["label"] for r in rows]
    groups = [coil_id(r) for r in rows]
    for r in rows:
        r["fold"] = ""
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold_idx, (_, val_idx) in enumerate(sgkf.split(range(len(rows)), labels, groups)):
        for i in val_idx:
            rows[i]["fold"] = str(fold_idx)
    return rows, ("regrouped" if regroup else "grouped", N_FOLDS)


def main():
    ap = argparse.ArgumentParser(description="Coil-grouped stratified fold split")
    ap.add_argument("--regroup", action="store_true",
                    help="force a full re-split by coil, overwriting existing folds")
    args = ap.parse_args()

    with open(MANIFEST, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows, (mode, n) = assign_folds(rows, regroup=args.regroup)

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
        print(f"Coil-grouped stratified {N_FOLDS}-fold split written to: "
              f"{MANIFEST}\n")
    print(f"{'Fold':<6} {'Pass':>6} {'Dent':>6} {'Loose':>6} {'Total':>7}")
    print("-" * 32)
    for fold in range(N_FOLDS):
        fold_rows = [r for r in rows if r["fold"] == str(fold)]
        cnt = Counter(r["label"] for r in fold_rows)
        tag = " <- comparison fold (never trained on)" if fold == N_FOLDS - 1 else ""
        print(f"  {fold}    {cnt['Pass']:>4}   {cnt['Dent']:>4}   {cnt['Loose']:>4}   {len(fold_rows):>5}{tag}")

    # leakage check: a coil must never span >1 fold
    coils = {}
    for r in rows:
        coils.setdefault(coil_id(r), set()).add(r["fold"])
    spanning = sum(1 for folds in coils.values() if len(folds) > 1)
    print(f"\nCoils spanning >1 fold (leakage): {spanning}  "
          f"({'OK' if spanning == 0 else 'LEAK!'})")


if __name__ == "__main__":
    main()
