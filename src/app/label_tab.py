"""
Phase 6 -- Label tab.

The human side of the loop. Operator points at a folder of new BMPs; the app
queues the ones not already labeled, shows each coil cropped and large with the
PaDiM model's band/score pre-filled, and the operator confirms Pass / Dent /
Loose (or Skip). Each confirmation writes to labels.db, copies the image into
data/raw/{label}/, and appends to manifest.csv -- growing the defect set that
is the only real lever on the recall ceiling.
"""

from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QShortcut,
    QVBoxLayout, QWidget,
)

from src.app.imaging import load_crop_pixmap
from src.labeling import ingestion

SUGGEST = {"AUTO-PASS": "Pass", "REVIEW": None, "AUTO-FLAG": None}
LABELS  = ("Pass", "Dent", "Loose")


class LabelTab(QWidget):
    def __init__(self, predictor, store, status_cb=None, on_label_committed=None):
        super().__init__()
        self.predictor = predictor
        self.store = store
        self.status_cb = status_cb or (lambda *_: None)
        self.on_label_committed = on_label_committed or (lambda: None)
        self.queue = []
        self.idx = 0
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self.btn_load = QPushButton("Load Folder")
        self.btn_load.clicked.connect(self.load_folder)
        top.addWidget(self.btn_load)
        top.addWidget(QLabel("Operator:"))
        self.operator = QLineEdit("operator")
        self.operator.setFixedWidth(140)
        top.addWidget(self.operator)
        top.addStretch(1)
        self.counter = QLabel("0 labeled today")
        top.addWidget(self.counter)
        root.addLayout(top)

        self.image_label = QLabel("Load a folder of unlabeled BMPs to begin")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(720, 380)
        self.image_label.setStyleSheet(
            "background:#101417; color:#888; border:1px solid #2a2f33;")
        root.addWidget(self.image_label, stretch=3)

        self.model_lbl = QLabel("model: --")
        self.model_lbl.setAlignment(Qt.AlignCenter)
        self.model_lbl.setStyleSheet("font-size:15px;")
        root.addWidget(self.model_lbl)

        self.queue_lbl = QLabel("queue: --")
        self.queue_lbl.setAlignment(Qt.AlignCenter)
        self.queue_lbl.setStyleSheet("color:#999;")
        root.addWidget(self.queue_lbl)

        btns = QHBoxLayout()
        self.label_btns = {}
        for name, key in (("Pass", "P"), ("Dent", "D"), ("Loose", "L")):
            b = QPushButton(f"✓ {name}  ({key})")
            b.setMinimumHeight(48)
            b.clicked.connect(lambda _, n=name: self.confirm(n))
            btns.addWidget(b)
            self.label_btns[name] = b
        self.btn_skip = QPushButton("Skip  (→)")
        self.btn_skip.setMinimumHeight(48)
        self.btn_skip.clicked.connect(self.skip)
        btns.addWidget(self.btn_skip)
        root.addLayout(btns)

        # keyboard shortcuts
        QShortcut(QKeySequence("P"), self, activated=lambda: self.confirm("Pass"))
        QShortcut(QKeySequence("D"), self, activated=lambda: self.confirm("Dent"))
        QShortcut(QKeySequence("L"), self, activated=lambda: self.confirm("Loose"))
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self.skip)

        self._set_enabled(False)
        self._refresh_counter()

    # ------------------------------------------------------------------
    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open folder of BMPs")
        if not folder:
            return
        new = ingestion.scan_folder(folder, self.store)
        if not new:
            self.status_cb("No new unlabeled .bmp files in that folder")
            return
        sid = datetime.now().date().isoformat()
        for p in new:
            self.store.insert_pending(p, session_id=sid)
        self.queue = new
        self.idx = 0
        self.status_cb(f"Queued {len(new)} new images for labeling")
        self.show_current()

    def show_current(self):
        if self.idx >= len(self.queue):
            self._set_enabled(False)
            self.image_label.setText("Queue complete — load another folder")
            self.image_label.setPixmap(self._blank())
            self.model_lbl.setText("model: --")
            self.queue_lbl.setText("queue: done")
            return
        path = self.queue[self.idx]
        self._set_enabled(True)
        pix = load_crop_pixmap(path, self.predictor.crop)
        self.image_label.setPixmap(pix.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        try:
            r = self.predictor.predict(path)
            self._current_pred = r
            sugg = SUGGEST.get(r["band"])
            self.model_lbl.setText(
                f"model: <b>{r['band']}</b>  (score {r.get('anomaly_score','?')})"
                + (f"  — suggests <b>{sugg}</b>" if sugg
                   else "  — operator picks subtype"))
            self._highlight(sugg)
        except Exception as e:
            self._current_pred = None
            self.model_lbl.setText(f"model: (failed: {e})")
            self._highlight(None)

        self.queue_lbl.setText(
            f"queue: {self.idx + 1} / {len(self.queue)}   "
            f"({Path(path).name})")

    def confirm(self, label):
        if self.idx >= len(self.queue) or not self.label_btns[label].isEnabled():
            return
        path = self.queue[self.idx]
        try:
            ingestion.commit_label(path, label, self.store,
                                   operator_id=self.operator.text().strip() or None)
        except Exception as e:
            self.status_cb(f"Failed to save label: {e}")
            return
        self.status_cb(f"Labeled {Path(path).name} = {label}")
        self._refresh_counter()
        self.on_label_committed()
        self.idx += 1
        self.show_current()

    def skip(self):
        if self.idx >= len(self.queue):
            return
        self.store.skip(self.queue[self.idx])
        self.idx += 1
        self.show_current()

    # ------------------------------------------------------------------
    def _highlight(self, name):
        for n, b in self.label_btns.items():
            if n == name:
                b.setStyleSheet("border:2px solid #2db84d; font-weight:bold;")
            else:
                b.setStyleSheet("")

    def _set_enabled(self, on):
        for b in self.label_btns.values():
            b.setEnabled(on)
        self.btn_skip.setEnabled(on)

    def _refresh_counter(self):
        self.counter.setText(f"{self.store.count_labeled_today()} labeled today")

    def _blank(self):
        from PyQt5.QtGui import QPixmap
        return QPixmap()
