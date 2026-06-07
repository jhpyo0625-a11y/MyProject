"""
Retrain tab (Phase 7 preview).

Surfaces the human-assist data flywheel: how many labels have been confirmed,
the breakdown by class, and whether the retrain threshold
(config: retraining.trigger_count) has been crossed. "Retrain Now" runs the
real deployed-model refit -- preprocess new crops, then re-fit the PaDiM
detector on Pass coils -- in a worker thread, streaming logs.

The formal trigger module + old-vs-new comparison + approval gate
(src/retraining/, fold-4 comparison) is the full Phase 7 task; this tab is the
operator-facing entry point to it.
"""

import subprocess
import sys
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

ROOT = Path(__file__).resolve().parents[2]


class RetrainWorker(QThread):
    line = pyqtSignal(str)
    done = pyqtSignal(int)

    STEPS = [
        ("Preprocessing new crops", [sys.executable, "-u", "-m", "src.data.preprocess"]),
        ("Refitting PaDiM detector", [sys.executable, "-u", "-m", "src.training.anomaly"]),
    ]

    def run(self):
        rc = 0
        for title, cmd in self.STEPS:
            self.line.emit(f"\n=== {title} ===")
            try:
                proc = subprocess.Popen(
                    cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                    errors="replace", bufsize=1)
            except Exception as e:
                self.line.emit(f"failed to launch: {e}")
                rc = 1
                break
            for ln in proc.stdout:
                if "HF_TOKEN" in ln or "unauthenticated" in ln:
                    continue
                self.line.emit(ln.rstrip())
            proc.wait()
            if proc.returncode != 0:
                rc = proc.returncode
                self.line.emit(f"step exited with code {rc}")
                break
        self.done.emit(rc)


class RetrainTab(QWidget):
    def __init__(self, store, cfg, status_cb=None):
        super().__init__()
        self.store = store
        self.trigger_count = int(cfg["retraining"]["trigger_count"])
        self.status_cb = status_cb or (lambda *_: None)
        self._worker = None
        self._build()
        self.refresh()

    def _build(self):
        root = QVBoxLayout(self)

        self.banner = QLabel()
        self.banner.setFixedHeight(44)
        root.addWidget(self.banner)

        self.stats = QLabel()
        self.stats.setStyleSheet("font-size:14px;")
        root.addWidget(self.stats)

        row = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_retrain = QPushButton("Retrain Now (PaDiM refit)")
        self.btn_retrain.clicked.connect(self.retrain)
        row.addWidget(self.btn_refresh)
        row.addWidget(self.btn_retrain)
        row.addStretch(1)
        root.addLayout(row)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet(
            "font-family:Consolas,monospace; font-size:12px; "
            "background:#0c0f11; color:#ccc;")
        root.addWidget(self.log, stretch=1)

    def refresh(self):
        confirmed = self.store.count_confirmed()
        unlabeled = self.store.get_unlabeled_count()
        counts = self.store.label_counts()
        self.stats.setText(
            f"Confirmed labels: <b>{confirmed}</b>  "
            f"(Pass={counts.get('Pass',0)}  Dent={counts.get('Dent',0)}  "
            f"Loose={counts.get('Loose',0)})<br>"
            f"Unlabeled in queue: {unlabeled}  |  "
            f"retrain threshold: {self.trigger_count}")
        if confirmed >= self.trigger_count:
            self.banner.setText(
                f"  {confirmed} labels available — retrain recommended")
            self.banner.setStyleSheet(
                "background:#b8860b; color:white; font-weight:bold; "
                "border-radius:6px; padding:4px;")
        else:
            self.banner.setText(
                f"  {confirmed}/{self.trigger_count} labels toward next retrain")
            self.banner.setStyleSheet(
                "background:#23282c; color:#bbb; border-radius:6px; padding:4px;")

    def retrain(self):
        self.btn_retrain.setEnabled(False)
        self.log.clear()
        self.log.appendPlainText(
            "Retraining the deployed PaDiM detector "
            "(preprocess new crops -> refit on Pass coils)...")
        self.status_cb("Retraining...")
        self._worker = RetrainWorker()
        self._worker.line.connect(self._append)
        self._worker.done.connect(self._finished)
        self._worker.start()

    def _append(self, ln):
        self.log.appendPlainText(ln)
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _finished(self, rc):
        self.btn_retrain.setEnabled(True)
        msg = ("Retrain complete — restart the app to load the new detector."
               if rc == 0 else f"Retrain failed (exit {rc}) — see log.")
        self._append("\n" + msg)
        self.status_cb(msg)
        self.refresh()
