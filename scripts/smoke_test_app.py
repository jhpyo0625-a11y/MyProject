"""
Headless smoke test for the desktop app.

Runs Qt in offscreen mode so it works without a display:
  1. Builds the full MainWindow with the real predictor + a temp label store.
  2. Runs single-image inference through the Inspect tab.
  3. Exercises the label-commit / ingestion path in a temp sandbox (temp DB,
     temp manifest, temp raw dir) so it never touches real project data.

Run:
    python scripts/smoke_test_app.py
"""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import glob
import sys
import tempfile
from pathlib import Path

import torch  # noqa: F401  -- must precede PyQt5 on Windows (OpenMP DLL clash)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt5.QtWidgets import QApplication

from src.data.config_loader import load_config
from src.inference.predictor import Predictor
from src.labeling.label_store import LabelStore


def test_gui(pred, cfg, bmp):
    from src.app.main import MainWindow
    store = LabelStore(db_path=Path(tempfile.gettempdir()) / "app_smoke.db")
    win = MainWindow(pred, store, cfg)
    tabs = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert tabs == ["Inspect", "Label", "Retrain"], tabs

    win.inspect.show_image(bmp)
    badge = win.inspect.badge.text()
    assert badge in ("PASS", "REVIEW", "FAIL"), badge
    win.retrain.refresh()
    print(f"  GUI ok: tabs={tabs}  inspect badge={badge}  "
          f"meter={win.inspect.meter.value()}")


def test_ingestion(pred, cfg, bmp):
    import src.labeling.ingestion as ing
    tmp = Path(tempfile.mkdtemp(prefix="coil_ing_"))
    # redirect side-effect targets into the sandbox
    ing.RAW_DIR = tmp / "raw"
    ing.MANIFEST = tmp / "manifest.csv"
    store = LabelStore(db_path=tmp / "labels.db")

    bmp = Path(bmp)
    store.insert_pending(bmp, session_id="smoke")
    row = ing.commit_label(bmp, "Dent", store, operator_id="tester")

    assert store.count_confirmed() == 1
    assert (ing.RAW_DIR / "Dent" / bmp.name).exists(), "raw copy missing"
    assert ing.MANIFEST.exists(), "manifest not written"
    assert 0 <= int(row["fold"]) < cfg["training"]["comparison_fold"], row["fold"]
    manifest_text = ing.MANIFEST.read_text(encoding="utf-8")
    assert "Dent" in manifest_text and str(ing.RAW_DIR) in manifest_text
    print(f"  ingestion ok: confirmed=1  fold={row['fold']}  "
          f"copied+manifest appended")


if __name__ == "__main__":
    app = QApplication([])
    cfg = load_config()
    samples = sorted(glob.glob("Coil-image-Dataset/**/*.bmp", recursive=True))
    if not samples:
        print("SKIP: no sample BMPs under Coil-image-Dataset/ (dataset not present)")
        sys.exit(0)
    bmp = samples[0]
    print("Loading predictor...")
    pred = Predictor()
    print("Testing GUI (offscreen)...")
    test_gui(pred, cfg, bmp)
    print("Testing label commit / ingestion (sandboxed)...")
    test_ingestion(pred, cfg, bmp)
    print("SMOKE OK")
