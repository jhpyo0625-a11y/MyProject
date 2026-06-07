"""
Unit tests for the Phase 1 data pipeline: fold assignment, preprocessing
output contract, and the crop-edge validation logic.

Run:
    python scripts/test_data_pipeline.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.split import assign_folds, N_FOLDS, COMP_FOLD
from src.data.preprocess import preprocess_image, OUT_W, OUT_H, CROP
from validate_crop import check_crop_edges   # noqa: E402  (scripts/ on path)


def _rows(n_per_class):
    rows = []
    for cls in ("Pass", "Dent", "Loose"):
        for i in range(n_per_class):
            rows.append({"filepath": f"data/raw/{cls}/{cls}_{i}.bmp", "label": cls})
    return rows


def test_assign_folds_fresh():
    rows, (mode, n) = assign_folds(_rows(10))
    assert mode == "stratified"
    folds = {int(r["fold"]) for r in rows}
    assert folds == set(range(N_FOLDS))            # all folds populated, incl. comp fold
    assert COMP_FOLD in folds
    # every class is spread across more than one fold (stratified, not clumped)
    for cls in ("Pass", "Dent", "Loose"):
        cf = {r["fold"] for r in rows if r["label"] == cls}
        assert len(cf) > 1, (cls, cf)
    print(f"  ok: fresh split populates folds 0..{N_FOLDS-1}, classes spread")


def test_assign_folds_incremental_preserves_and_never_comp():
    rows, _ = assign_folds(_rows(10))
    before = {r["filepath"]: r["fold"] for r in rows}
    # append two new, unfolded rows
    new = [{"filepath": "data/raw/Dent/late_1.bmp", "label": "Dent"},
           {"filepath": "data/raw/Loose/late_2.bmp", "label": "Loose"}]
    rows2, (mode, n) = assign_folds(rows + new)
    assert mode == "incremental" and n == 2
    # existing assignments untouched
    for r in rows2:
        if r["filepath"] in before:
            assert r["fold"] == before[r["filepath"]]
    # new rows never land in the held-out comparison fold
    for r in new:
        assert 0 <= int(r["fold"]) < COMP_FOLD, r
    # idempotent: re-running assigns nothing new
    _, (mode3, n3) = assign_folds(rows2)
    assert mode3 == "incremental" and n3 == 0
    print("  ok: incremental preserves existing folds, never comp fold, idempotent")


def test_preprocess_image_contract():
    tmp = Path(tempfile.mkdtemp(prefix="prep_")) / "frame.bmp"
    Image.new("RGB", (2448, 2048), (123, 123, 123)).save(tmp)
    arr = preprocess_image(tmp)
    assert arr.shape == (OUT_H, OUT_W, 3), arr.shape
    assert arr.dtype == np.uint8
    print(f"  ok: preprocess_image -> {arr.shape} uint8")


def test_check_crop_edges():
    w, h = CROP[2] + 100, CROP[3] + 100        # large enough to contain the box
    dark = check_crop_edges(Image.new("RGB", (w, h), (0, 0, 0)))
    assert dark["warnings"] == [], dark["warnings"]
    bright = check_crop_edges(Image.new("RGB", (w, h), (255, 255, 255)))
    assert len(bright["warnings"]) == 4, bright["warnings"]   # all edges warn
    print("  ok: check_crop_edges silent on dark edges, warns on bright edges")


if __name__ == "__main__":
    print("Data-pipeline unit tests...")
    test_assign_folds_fresh()
    test_assign_folds_incremental_preserves_and_never_comp()
    test_preprocess_image_contract()
    test_check_crop_edges()
    print("ALL PASS")
