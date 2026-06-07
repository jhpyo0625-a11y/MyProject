"""
Phase 7 -- Retrain tab (gated retraining pipeline).

The full human-in-the-loop flow:
  1. Status banner: labels confirmed since the last deployed retrain vs the
     trigger threshold (config: retraining.trigger_count).
  2. "Retrain Now" -> builds a CANDIDATE (preprocess + PaDiM refit) in a worker,
     streaming logs. Never touches production directly.
  3. Auto-runs the comparison: candidate vs current production on the locked
     fold 4, rendered inline.
  4. Approval gate: "Approve & Deploy" (archive old -> promote candidate ->
     reset counter) or "Reject" (discard candidate).
  5. Version history with one-click rollback to any archived model.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPlainTextEdit,
    QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QTextBrowser,
    QVBoxLayout, QWidget,
)

ROOT = Path(__file__).resolve().parents[2]

from src.retraining import trigger, promote, compare


class RetrainWorker(QThread):
    line = pyqtSignal(str)
    done = pyqtSignal(int)

    def run(self):
        try:
            proc = subprocess.Popen(
                [sys.executable, "-u", "-m", "src.retraining.retrain"],
                cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1)
        except Exception as e:
            self.line.emit(f"failed to launch retrain: {e}")
            self.done.emit(1)
            return
        for ln in proc.stdout:
            if "HF_TOKEN" in ln or "unauthenticated" in ln:
                continue
            self.line.emit(ln.rstrip())
        proc.wait()
        self.done.emit(proc.returncode)


class CompareWorker(QThread):
    done = pyqtSignal(dict)
    fail = pyqtSignal(str)

    def run(self):
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.done.emit(compare.compare(ts))
        except Exception as e:
            self.fail.emit(str(e))


class RetrainTab(QWidget):
    def __init__(self, store, cfg, status_cb=None):
        super().__init__()
        self.store = store
        self.cfg = cfg
        self.trigger_count = int(cfg["retraining"]["trigger_count"])
        self.status_cb = status_cb or (lambda *_: None)
        self._rworker = None
        self._cworker = None
        self._build()
        self.refresh()

    def _build(self):
        root = QVBoxLayout(self)

        self.banner = QLabel()
        self.banner.setFixedHeight(40)
        root.addWidget(self.banner)

        self.stats = QLabel()
        root.addWidget(self.stats)

        row = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_retrain = QPushButton("Retrain Now")
        self.btn_retrain.clicked.connect(self.retrain)
        row.addWidget(self.btn_refresh)
        row.addWidget(self.btn_retrain)
        row.addStretch(1)
        root.addLayout(row)

        split = QSplitter(Qt.Vertical)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet(
            "font-family:Consolas,monospace; font-size:12px; "
            "background:#0c0f11; color:#ccc;")
        split.addWidget(self.log)
        self.report = QTextBrowser()
        self.report.setStyleSheet("font-size:13px;")
        split.addWidget(self.report)
        split.setSizes([220, 260])
        root.addWidget(split, stretch=1)

        gate = QHBoxLayout()
        self.btn_approve = QPushButton("Approve & Deploy")
        self.btn_approve.setMinimumHeight(40)
        self.btn_approve.setStyleSheet("background:#1b7f3b; color:white; font-weight:bold;")
        self.btn_approve.clicked.connect(self.approve)
        self.btn_reject = QPushButton("Reject")
        self.btn_reject.setMinimumHeight(40)
        self.btn_reject.setStyleSheet("background:#9b2222; color:white; font-weight:bold;")
        self.btn_reject.clicked.connect(self.reject)
        gate.addWidget(self.btn_approve)
        gate.addWidget(self.btn_reject)
        root.addLayout(gate)
        self._show_gate(False)

        root.addWidget(QLabel("Version history"))
        self.history = QTableWidget(0, 3)
        self.history.setHorizontalHeaderLabels(
            ["Version", "CV recall@20%FR", "fold4 defects leaked"])
        self.history.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self.history.setMaximumHeight(150)
        root.addWidget(self.history)
        self.btn_rollback = QPushButton("Rollback to selected version")
        self.btn_rollback.clicked.connect(self.rollback)
        root.addWidget(self.btn_rollback)

    # ------------------------------------------------------------------
    def refresh(self):
        st = trigger.status(self.store, self.trigger_count)
        counts = self.store.label_counts()
        self.stats.setText(
            f"Confirmed labels: <b>{st['confirmed_total']}</b>  "
            f"(Pass={counts.get('Pass',0)}  Dent={counts.get('Dent',0)}  "
            f"Loose={counts.get('Loose',0)})<br>"
            f"Since last retrain: {st['since_last_retrain']}  |  "
            f"threshold: {st['threshold']}  |  "
            f"last retrain: {st['last_retrain_at'] or 'never'}")
        if st["recommended"]:
            self.banner.setText(
                f"  {st['since_last_retrain']} new labels since last retrain "
                f"— retrain recommended")
            self.banner.setStyleSheet(
                "background:#b8860b; color:white; font-weight:bold; "
                "border-radius:6px; padding:4px;")
        else:
            self.banner.setText(
                f"  {st['since_last_retrain']}/{st['threshold']} new labels "
                f"toward next retrain")
            self.banner.setStyleSheet(
                "background:#23282c; color:#bbb; border-radius:6px; padding:4px;")
        self._load_history()

    def _load_history(self):
        versions = promote.list_versions()
        self.history.setRowCount(0)
        for v in versions:
            m = v.get("metrics", {})
            r = self.history.rowCount()
            self.history.insertRow(r)
            self.history.setItem(r, 0, QTableWidgetItem(v["version"]))
            rec = m.get("cv_recall_at_20fr")
            leak = m.get("fold4_autopass_defect_miss")
            self.history.setItem(r, 1, QTableWidgetItem(
                f"{rec:.1%}" if isinstance(rec, (int, float)) else "?"))
            self.history.setItem(r, 2, QTableWidgetItem(
                f"{leak:.1%}" if isinstance(leak, (int, float)) else "?"))

    # ------------------------------------------------------------------
    def retrain(self):
        self.btn_retrain.setEnabled(False)
        self._show_gate(False)
        self.report.clear()
        self.log.clear()
        self.log.appendPlainText("Building candidate model (does not touch "
                                 "production)...")
        self.status_cb("Retraining (building candidate)...")
        self._rworker = RetrainWorker()
        self._rworker.line.connect(self._append)
        self._rworker.done.connect(self._retrain_done)
        self._rworker.start()

    def _retrain_done(self, rc):
        if rc != 0:
            self._append(f"\nRetrain failed (exit {rc}).")
            self.status_cb("Retrain failed")
            self.btn_retrain.setEnabled(True)
            return
        self._append("\nCandidate built. Comparing against production on fold 4...")
        self.status_cb("Comparing candidate vs production...")
        self._cworker = CompareWorker()
        self._cworker.done.connect(self._compare_done)
        self._cworker.fail.connect(self._compare_fail)
        self._cworker.start()

    def _compare_done(self, result):
        self.report.setMarkdown(result["markdown"])
        rec = "recommended" if result["candidate_better"] else "NOT clearly better"
        self._append(f"\nComparison done — candidate is {rec}.")
        self._append(f"Report: {result['report_path']}")
        self.status_cb(f"Candidate ready for review — {rec}")
        self._show_gate(True)
        self.btn_retrain.setEnabled(True)

    def _compare_fail(self, msg):
        self._append(f"\nComparison failed: {msg}")
        self.status_cb("Comparison failed")
        self.btn_retrain.setEnabled(True)

    # ------------------------------------------------------------------
    def approve(self):
        if QMessageBox.question(
                self, "Approve & Deploy",
                "Promote the candidate to production?\nThe current model will "
                "be archived and can be rolled back.") != QMessageBox.Yes:
            return
        try:
            res = promote.approve_candidate(store=self.store)
        except Exception as e:
            QMessageBox.critical(self, "Approve failed", str(e))
            return
        self._show_gate(False)
        QMessageBox.information(
            self, "Deployed",
            f"Candidate promoted to production.\nPrevious model archived as "
            f"{res['archived_as']}.\n\nRestart the app to load the new detector.")
        self.status_cb("Candidate deployed — restart app to load it")
        self.refresh()

    def reject(self):
        try:
            promote.reject_candidate()
        except Exception as e:
            QMessageBox.critical(self, "Reject failed", str(e))
            return
        self._show_gate(False)
        self.report.clear()
        self.status_cb("Candidate rejected — production unchanged")
        self.refresh()

    def rollback(self):
        rows = self.history.selectionModel().selectedRows()
        if not rows:
            self.status_cb("Select a version to roll back to")
            return
        version = self.history.item(rows[0].row(), 0).text()
        if QMessageBox.question(
                self, "Rollback",
                f"Restore archived version {version} to production?"
                ) != QMessageBox.Yes:
            return
        try:
            promote.rollback(version)
        except Exception as e:
            QMessageBox.critical(self, "Rollback failed", str(e))
            return
        QMessageBox.information(
            self, "Rolled back",
            f"Production restored to {version}.\nRestart the app to load it.")
        self.status_cb(f"Rolled back to {version}")
        self.refresh()

    # ------------------------------------------------------------------
    def _append(self, ln):
        self.log.appendPlainText(ln)
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _show_gate(self, on):
        self.btn_approve.setVisible(on)
        self.btn_reject.setVisible(on)
