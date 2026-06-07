"""
T2.2 -- Build the embedding cache.

Runs every crop through the frozen backbone and saves the resulting
embedding vector. Writes:
  data/embeddings/{label}/{stem}.npy  -- float32 (D,) per image
  data/embeddings/index.csv           -- manifest of all embeddings
  data/embeddings/backbone_info.json  -- backbone name + embedding dim

Skips images whose .npy already exists (safe to re-run after adding images).
"""

import sys
import csv
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.config_loader import load_config
from src.features.extractor import CoilEmbedder

cfg         = load_config()
CROPS_DIR   = ROOT / cfg["paths"]["crops_dir"]
EMB_DIR     = ROOT / cfg["paths"]["embeddings_dir"]
MANIFEST    = ROOT / cfg["paths"]["manifest"]
INDEX_CSV   = EMB_DIR / "index.csv"
BACKBONE_INFO = EMB_DIR / "backbone_info.json"

CLASSES = ["Pass", "Dent", "Loose"]


def load_manifest_dict() -> dict[str, dict]:
    """Return {stem: row} from manifest.csv for fast lookup."""
    with open(MANIFEST, encoding="utf-8") as f:
        return {Path(r["filepath"]).stem: r for r in csv.DictReader(f)}


def main():
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    for cls in CLASSES:
        (EMB_DIR / cls).mkdir(exist_ok=True)

    print("Loading backbone...")
    embedder    = CoilEmbedder.from_config()
    manifest    = load_manifest_dict()

    print(f"Backbone : {embedder.backbone_name}")
    print(f"Emb dim  : {embedder.embedding_dim}")

    # benchmark
    dummy = np.random.randint(0, 255, (192, 448, 3), dtype=np.uint8)
    lat   = embedder.benchmark(dummy, n=5)
    print(f"Latency  : {lat:.1f} ms/image (CPU)\n")

    index_rows = []
    total = ok = skipped = errors = 0

    t_start = time.perf_counter()

    for cls in CLASSES:
        crop_paths = sorted((CROPS_DIR / cls).glob("*.npy"))
        total += len(crop_paths)

        for crop_path in crop_paths:
            stem    = crop_path.stem
            dst     = EMB_DIR / cls / f"{stem}.npy"
            meta    = manifest.get(stem, {})
            fold    = meta.get("fold", "")

            if dst.exists():
                skipped += 1
                emb_dim = embedder.embedding_dim
                index_rows.append({
                    "stem":         stem,
                    "label":        cls,
                    "fold":         fold,
                    "embedding_path": str(dst.relative_to(ROOT)),
                    "is_augmented": "False",
                    "aug_idx":      "0",
                })
                continue

            try:
                arr = np.load(crop_path)
                emb = embedder.embed(arr)
                np.save(dst, emb)
                ok += 1
                index_rows.append({
                    "stem":         stem,
                    "label":        cls,
                    "fold":         fold,
                    "embedding_path": str(dst.relative_to(ROOT)),
                    "is_augmented": "False",
                    "aug_idx":      "0",
                })
            except Exception as exc:
                errors += 1
                print(f"  ERROR {cls}/{stem}: {exc}")

        elapsed = time.perf_counter() - t_start
        processed = ok + skipped
        print(f"  {cls}: {len(crop_paths)} crops  "
              f"[{ok} new | {skipped} cached | {errors} err]  "
              f"elapsed {elapsed:.1f}s")

    # write index
    fieldnames = ["stem", "label", "fold", "embedding_path", "is_augmented", "aug_idx"]
    with open(INDEX_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    # write backbone info for training code
    with open(BACKBONE_INFO, "w", encoding="utf-8") as f:
        json.dump({"backbone": embedder.backbone_name,
                   "embedding_dim": embedder.embedding_dim}, f, indent=2)

    total_elapsed = time.perf_counter() - t_start
    print(f"\nDone.  {ok} new | {skipped} cached | {errors} errors")
    print(f"Total time: {total_elapsed:.1f}s  ({total_elapsed/max(ok,1)*1000:.0f} ms/image avg)")
    print(f"Index  -> {INDEX_CSV}  ({len(index_rows)} rows)")
    print(f"Info   -> {BACKBONE_INFO}")


if __name__ == "__main__":
    main()
