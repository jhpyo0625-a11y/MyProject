"""
T2.4 -- Generate augmented embeddings for minority classes (Dent, Loose).

Reads existing index.csv, applies N augmented versions of each original
minority-class crop, extracts embeddings, and appends rows to index.csv.

Rules:
  - Only Dent and Loose originals are augmented (Pass is already the majority).
  - Augmented embeddings are NEVER generated for fold 4 (comparison fold).
  - Each augmented file is named: {stem}_aug{i}.npy
  - is_augmented=True in index.csv marks these rows for training code to handle.

Run after build_cache.py.
"""

import sys
import csv
import random
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.config_loader import load_config
from src.features.extractor import CoilEmbedder
from src.data.augment import augment_from_config

cfg           = load_config()
CROPS_DIR     = ROOT / cfg["paths"]["crops_dir"]
EMB_DIR       = ROOT / cfg["paths"]["embeddings_dir"]
INDEX_CSV     = EMB_DIR / "index.csv"
N_COPIES      = cfg["augmentation"]["minority_copies"]
COMPARISON_FOLD = str(cfg["training"]["comparison_fold"])
MINORITY_CLASSES = ["Dent", "Loose"]
SEED          = cfg["training"]["random_seed"]


def load_index() -> list[dict]:
    with open(INDEX_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    print(f"Loading backbone...")
    embedder = CoilEmbedder.from_config()
    print(f"Backbone : {embedder.backbone_name}  |  dim={embedder.embedding_dim}")
    print(f"Augmented copies per minority image: {N_COPIES}")
    print(f"Skipping fold {COMPARISON_FOLD} (comparison fold)\n")

    index = load_index()
    existing_stems = {r["stem"] for r in index if r["is_augmented"] == "True"}

    candidates = [
        r for r in index
        if r["label"] in MINORITY_CLASSES
        and r["is_augmented"] == "False"
        and r["fold"] != COMPARISON_FOLD
    ]

    print(f"Minority originals eligible for augmentation: {len(candidates)}")
    for cls in MINORITY_CLASSES:
        n = sum(1 for r in candidates if r["label"] == cls)
        print(f"  {cls}: {n} originals -> {n * N_COPIES} augmented")
    print()

    new_rows = []
    ok = skipped = errors = 0
    master_rng = random.Random(SEED)
    t_start = time.perf_counter()

    for row in candidates:
        stem  = row["stem"]
        cls   = row["label"]
        fold  = row["fold"]
        crop_path = CROPS_DIR / cls / f"{stem}.npy"

        if not crop_path.exists():
            print(f"  WARN: crop not found for {cls}/{stem}")
            errors += 1
            continue

        arr = np.load(crop_path)

        for aug_idx in range(1, N_COPIES + 1):
            aug_stem = f"{stem}_aug{aug_idx}"
            dst      = EMB_DIR / cls / f"{aug_stem}.npy"

            if dst.exists() or aug_stem in existing_stems:
                skipped += 1
                new_rows.append({
                    "stem": aug_stem, "label": cls, "fold": fold,
                    "embedding_path": str(dst.relative_to(ROOT)),
                    "is_augmented": "True", "aug_idx": str(aug_idx),
                })
                continue

            try:
                aug_rng = random.Random(master_rng.randint(0, 2**31))
                aug_arr = augment_from_config(arr, cfg, rng=aug_rng)
                emb     = embedder.embed(aug_arr)
                np.save(dst, emb)
                ok += 1
                new_rows.append({
                    "stem": aug_stem, "label": cls, "fold": fold,
                    "embedding_path": str(dst.relative_to(ROOT)),
                    "is_augmented": "True", "aug_idx": str(aug_idx),
                })
            except Exception as exc:
                errors += 1
                print(f"  ERROR {cls}/{aug_stem}: {exc}")

    # append to index
    fieldnames = ["stem", "label", "fold", "embedding_path", "is_augmented", "aug_idx"]
    all_rows = index + new_rows
    import shutil
    tmp = INDEX_CSV.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    shutil.move(tmp, INDEX_CSV)

    elapsed = time.perf_counter() - t_start
    print(f"Done.  {ok} new | {skipped} cached | {errors} errors  ({elapsed:.1f}s)")

    # summary by class
    aug_rows = [r for r in all_rows if r["is_augmented"] == "True"]
    orig_rows = [r for r in all_rows if r["is_augmented"] == "False"]
    print(f"\nIndex totals after augmentation:")
    for cls in ["Pass", "Dent", "Loose"]:
        n_orig = sum(1 for r in orig_rows if r["label"] == cls)
        n_aug  = sum(1 for r in aug_rows  if r["label"] == cls)
        print(f"  {cls:<6}: {n_orig} originals + {n_aug} augmented = {n_orig+n_aug} total")


if __name__ == "__main__":
    main()
