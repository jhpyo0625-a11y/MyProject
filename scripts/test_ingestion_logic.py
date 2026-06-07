"""
Unit tests for the label-ingestion write path (fold assignment, manifest dedup,
and the commit ordering that keeps the retrain trigger honest).

Run:
    python scripts/test_ingestion_logic.py
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.labeling.ingestion as ing
from src.labeling.ingestion import (
    _assign_fold, _append_manifest, commit_label, COMP_FOLD, MANIFEST_FIELDS,
)
from src.labeling.label_store import LabelStore


def test_assign_fold():
    for i in range(100):
        f = _assign_fold(f"coil_{i}")
        assert 0 <= f < COMP_FOLD, f      # never the held-out comparison fold
    assert _assign_fold("coil_7") == _assign_fold("coil_7")   # deterministic
    print(f"  ok: _assign_fold deterministic + always in 0..{COMP_FOLD-1}")


def test_append_manifest_dedup():
    tmp = Path(tempfile.mkdtemp(prefix="manif_"))
    ing.MANIFEST = tmp / "manifest.csv"
    row = {k: "" for k in MANIFEST_FIELDS}
    row.update(filepath="data/raw/Dent/x.bmp", label="Dent", fold=1)
    _append_manifest(row)
    _append_manifest(row)            # duplicate filepath -> must be skipped
    lines = ing.MANIFEST.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2, lines    # header + exactly one data row
    assert lines[0].startswith("filepath")
    print("  ok: _append_manifest dedups by filepath, single header")


def test_commit_label_ordering():
    tmp = Path(tempfile.mkdtemp(prefix="commit_"))
    ing.RAW_DIR = tmp / "raw"
    ing.MANIFEST = tmp / "manifest.csv"
    store = LabelStore(db_path=tmp / "labels.db")
    src = tmp / "src.bmp"
    src.write_bytes(b"x")

    def _boom(*a, **k):
        raise OSError("simulated disk-full during copy")

    orig = ing.shutil.copy2
    ing.shutil.copy2 = _boom
    try:
        raised = False
        try:
            commit_label(src, "Dent", store)
        except OSError:
            raised = True
    finally:
        ing.shutil.copy2 = orig

    assert raised
    # DB confirmation runs LAST, so a failed copy must NOT count the label
    assert store.count_confirmed() == 0
    assert not ing.MANIFEST.exists()      # manifest never written either
    print("  ok: commit_label leaves no counted label when the copy fails")


if __name__ == "__main__":
    print("Ingestion write-path unit tests...")
    test_assign_fold()
    test_append_manifest_dedup()
    test_commit_label_ordering()
    print("ALL PASS")
