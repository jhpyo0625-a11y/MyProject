r"""
Phase 5/6 -- Coil Defect Inspection desktop app.

QMainWindow with three tabs:
  - Inspect : single-image / batch inference (PaDiM band verdict)
  - Label   : human confirmation loop -> labels.db + data/raw + manifest
  - Retrain : data-flywheel status + deployed-model refit

Run:
    python src/app/main.py
    scripts\launch_app.bat      (Windows)
"""

import os
import sys
from pathlib import Path

# On Windows, PyQt5 and torch both ship libiomp5md.dll. If Qt is imported first
# it loads its OpenMP runtime and torch's c10.dll init then fails (WinError 1114).
# Import torch before any PyQt5 import, and allow the duplicate OpenMP runtime.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: F401  -- must precede PyQt5

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAction, QApplication, QDialog, QLabel, QMainWindow, QMessageBox,
    QPushButton, QTabWidget, QTextBrowser, QVBoxLayout,
)

from src.data.config_loader import load_config
from src.inference.predictor import Predictor
from src.labeling.label_store import LabelStore
from src.app.inspect_tab import InspectTab
from src.app.label_tab import LabelTab
from src.app.retrain_tab import RetrainTab


class MainWindow(QMainWindow):
    def __init__(self, predictor, store, cfg):
        super().__init__()
        self.predictor = predictor
        self.store = store
        self.cfg = cfg
        self.setWindowTitle("Coil Defect Inspection")
        self.resize(1180, 820)

        self.status = self.statusBar()
        self._model_label = QLabel()
        self.status.addPermanentWidget(self._model_label)
        self._set_model_label()

        self.tabs = QTabWidget()
        self.inspect = InspectTab(predictor, status_cb=self._status)
        self.label   = LabelTab(predictor, store, status_cb=self._status,
                                on_label_committed=self._on_label_committed)
        self.retrain = RetrainTab(store, cfg, status_cb=self._status)
        self.tabs.addTab(self.inspect, "Inspect")
        self.tabs.addTab(self.label, "Label")
        self.tabs.addTab(self.retrain, "Retrain")
        self.setCentralWidget(self.tabs)

        self._build_menu()
        self._status("Ready")

    def _build_menu(self):
        m_file = self.menuBar().addMenu("&File")
        a_img = QAction("Open Image...", self)
        a_img.triggered.connect(self._menu_open_image)
        a_dir = QAction("Open Folder...", self)
        a_dir.triggered.connect(self._menu_open_folder)
        a_quit = QAction("Quit", self)
        a_quit.triggered.connect(self.close)
        m_file.addAction(a_img)
        m_file.addAction(a_dir)
        m_file.addSeparator()
        m_file.addAction(a_quit)

        m_help = self.menuBar().addMenu("&Help")
        a_card = QAction("Model Card", self)
        a_card.triggered.connect(self._show_model_card)
        m_help.addAction(a_card)

    # ------------------------------------------------------------------
    def _menu_open_image(self):
        self.tabs.setCurrentWidget(self.inspect)
        self.inspect.open_image()

    def _menu_open_folder(self):
        self.tabs.setCurrentWidget(self.inspect)
        self.inspect.open_folder()

    def _show_model_card(self):
        card = ROOT / self.cfg["paths"]["production_dir"] / "padim_card.md"
        text = (card.read_text(encoding="utf-8") if card.exists()
                else "No model card found.")
        dlg = QDialog(self)
        dlg.setWindowTitle("Model Card")
        dlg.resize(620, 540)
        lay = QVBoxLayout(dlg)
        view = QTextBrowser()
        view.setMarkdown(text)
        lay.addWidget(view)
        close = QPushButton("Close")
        close.clicked.connect(dlg.accept)
        lay.addWidget(close)
        dlg.exec_()

    def _on_label_committed(self):
        self.retrain.refresh()

    def _set_model_label(self):
        thr = getattr(self.predictor, "_pd_t_flag", None)
        kind = getattr(self.predictor, "_type", "?")
        self._model_label.setText(f"model: {kind}"
                                  + (f"  (flag>={thr:.1f})" if thr else ""))

    def _status(self, msg):
        self.status.showMessage(msg)


def main():
    app = QApplication(sys.argv)
    cfg = load_config()

    splash = QLabel("Loading model...")
    splash.setAlignment(Qt.AlignCenter)
    splash.resize(360, 120)
    splash.setStyleSheet("font-size:18px;")
    splash.show()
    app.processEvents()

    try:
        predictor = Predictor()
    except Exception as e:
        splash.close()
        QMessageBox.critical(None, "Startup error",
                             f"Could not load a model:\n{e}\n\n"
                             "Run: python -m src.training.anomaly")
        return 1
    store = LabelStore()
    splash.close()

    win = MainWindow(predictor, store, cfg)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
