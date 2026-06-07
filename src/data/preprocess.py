"""
T1.3 — Preprocessing pipeline.

For every image in the manifest:
  1. Load raw BMP
  2. Apply fixed crop  (from config.yaml)
  3. Resize to target dimensions (448 x 192)
  4. Save as uint8 numpy array → data/crops/{label}/{stem}.npy

Normalization (ImageNet mean/std) is deferred to the feature extractor so that
the same crop can be used for augmentation and visualization.
"""

import sys
import csv
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.config_loader import load_config

cfg = load_config()

MANIFEST  = ROOT / cfg["paths"]["manifest"]
CROPS_DIR = ROOT / cfg["paths"]["crops_dir"]
crop_c    = cfg["preprocessing"]["crop"]
CROP      = (crop_c["x_min"], crop_c["y_min"], crop_c["x_max"], crop_c["y_max"])
OUT_W     = cfg["preprocessing"]["resize"]["width"]
OUT_H     = cfg["preprocessing"]["resize"]["height"]


def preprocess_image(path: Path) -> np.ndarray:
    img  = Image.open(path).convert("RGB")
    img  = img.crop(CROP)                                # fixed-box crop
    img  = img.resize((OUT_W, OUT_H), Image.LANCZOS)     # resize
    return np.asarray(img, dtype=np.uint8)               # (H, W, 3)


def main():
    with open(MANIFEST, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total   = len(rows)
    ok      = 0
    errors  = []

    for label in set(r["label"] for r in rows):
        (CROPS_DIR / label).mkdir(parents=True, exist_ok=True)

    print(f"Preprocessing {total} images → {CROPS_DIR}")
    print(f"  Crop:   x=[{CROP[0]},{CROP[2]}]  y=[{CROP[1]},{CROP[3]}]")
    print(f"  Resize: {OUT_W} x {OUT_H} px\n")

    for i, row in enumerate(rows, 1):
        src  = Path(row["filepath"])
        stem = src.stem
        dst  = CROPS_DIR / row["label"] / f"{stem}.npy"

        if dst.exists():          # skip if already cached
            ok += 1
            if i % 100 == 0:
                print(f"  [{i}/{total}] skipping (cached)")
            continue

        try:
            arr = preprocess_image(src)
            np.save(dst, arr)
            ok += 1
        except Exception as e:
            errors.append((str(src), str(e)))

        if i % 50 == 0 or i == total:
            print(f"  [{i}/{total}]  errors so far: {len(errors)}")

    print(f"\nDone.  Saved: {ok}/{total}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for path, msg in errors:
            print(f"  {path}: {msg}")
    else:
        print("No errors.")

    # verify output shapes
    print("\nSpot-checking output shapes:")
    for label in ["Pass", "Dent", "Loose"]:
        p = next((CROPS_DIR / label).glob("*.npy"), None)
        if p:
            arr = np.load(p)
            print(f"  {label}: {arr.shape}  dtype={arr.dtype}")


if __name__ == "__main__":
    main()
