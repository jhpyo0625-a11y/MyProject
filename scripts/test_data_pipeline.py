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


def _rows(coils_per_class=12, imgs_per_coil=2):
    """Synthetic manifest: each physical coil has multiple images so the test
    can verify no coil is split across folds."""
    rows = []
    for cls in ("Pass", "Dent", "Loose"):
        for c in range(coils_per_class):
            for k in range(imgs_per_coil):
                rows.append({"filepath": f"data/raw/{cls}/{cls}_{c}_{k}.bmp",
                             "label": cls,
                             "session_id": f"{cls}_sess",
                             "position": f"{cls}_{c}"})    # coil = (session, position)
    return rows


def _coil(r):
    return (r["session_id"], r["position"])


def test_assign_folds_fresh_grouped():
    rows, (mode, n) = assign_folds(_rows())
    assert mode == "grouped"
    folds = {int(r["fold"]) for r in rows}
    assert folds == set(range(N_FOLDS)) and COMP_FOLD in folds
    # THE leakage guarantee: no physical coil spans more than one fold
    coil_folds = {}
    for r in rows:
        coil_folds.setdefault(_coil(r), set()).add(r["fold"])
    assert all(len(fs) == 1 for fs in coil_folds.values()), "a coil spans folds!"
    for cls in ("Pass", "Dent", "Loose"):
        assert len({r["fold"] for r in rows if r["label"] == cls}) > 1
    print(f"  ok: grouped split populates folds 0..{N_FOLDS-1}, NO coil spans folds")


def test_assign_folds_incremental_coil_stable():
    rows, _ = assign_folds(_rows())
    before = {r["filepath"]: r["fold"] for r in rows}
    # two images of ONE new coil
    new = [{"filepath": "data/raw/Dent/late_0.bmp", "label": "Dent",
            "session_id": "late_sess", "position": "late_0"},
           {"filepath": "data/raw/Dent/late_1.bmp", "label": "Dent",
            "session_id": "late_sess", "position": "late_0"}]
    rows2, (mode, n) = assign_folds(rows + new)
    assert mode == "incremental" and n == 2
    for r in rows2:                                   # existing untouched
        if r["filepath"] in before:
            assert r["fold"] == before[r["filepath"]]
    new_folds = {r["fold"] for r in new}
    assert len(new_folds) == 1                        # both images -> same fold (coil-stable)
    assert 0 <= int(new[0]["fold"]) < COMP_FOLD       # new coil never in comp fold
    _, (mode3, n3) = assign_folds(rows2)
    assert mode3 == "incremental" and n3 == 0         # idempotent
    print("  ok: incremental keeps a coil's images together, never comp fold, idempotent")


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
    test_assign_folds_fresh_grouped()
    test_assign_folds_incremental_coil_stable()
    test_preprocess_image_contract()
    test_check_crop_edges()
    print("ALL PASS")
