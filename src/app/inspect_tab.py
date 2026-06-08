"""
Phase 5 -- Inspect tab.

Single-image and batch inference on raw BMPs using the deployed PaDiM detector
(via the single Predictor entry point). The supervised 3-class probability bars
from the original spec are replaced by the anomaly model's signal: an anomaly
score and a 3-band verdict (AUTO-PASS / REVIEW / AUTO-FLAG), since the system
runs as human-assist triage, not a >=95% auto-pass gate.
"""

import csv
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog, QHBoxLayout, QHeaderView, QLabel, QProgressBar, QPushButton,
    QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from src.app.imaging import (
    load_full_with_box, load_full_with_heatmap, load_clean_coil,
)

# verdict -> (label text, background color)
VERDICT_STYLE = {
    "Pass":   ("PASS",   "#1b7f3b"),
    "Review": ("REVIEW", "#b8860b"),
    "Fail":   ("FAIL",   "#9b2222"),
}
BAND_LABEL = {"AUTO-PASS": "AUTO-PASS", "REVIEW": "REVIEW", "AUTO-FLAG": "AUTO-FLAG"}


class BatchWorker(QThread):
    progress = pyqtSignal(int, int, str, dict)
    done     = pyqtSignal(list)

    def __init__(self, predictor, paths):
        super().__init__()
        self.predictor = predictor
        self.paths = paths

    def run(self):
        results = []
        n = len(self.paths)
        for i, p in enumerate(self.paths, 1):
            try:
                r = self.predictor.predict(p)
            except Exception as e:                       # keep batch alive
                r = {"backend": "padim", "decision": "Review",
                     "band": "REVIEW", "anomaly_score": -1,
                     "latency_ms": 0, "error": str(e)}
            results.append((p, r))
            self.progress.emit(i, n, str(p), r)
        self.done.emit(results)


class InspectTab(QWidget):
    def __init__(self, predictor, status_cb=None):
        super().__init__()
        self.predictor = predictor
        self.status_cb = status_cb or (lambda *_: None)
        self._batch_results = []
        self._worker = None
        self._current = None          # (path, result) currently displayed
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        root = QVBoxLayout(self)

        # --- controls ---
        controls = QHBoxLayout()
        self.btn_image  = QPushButton("Open Image")
        self.btn_folder = QPushButton("Open Folder")
        self.btn_export = QPushButton("Export CSV")
        self.btn_export.setEnabled(False)
        self.btn_image.clicked.connect(self.open_image)
        self.btn_folder.clicked.connect(self.open_folder)
        self.btn_export.clicked.connect(self.export_csv)
        for b in (self.btn_image, self.btn_folder, self.btn_export):
            controls.addWidget(b)
        self.btn_clean = QPushButton("Clean coil")
        self.btn_clean.setCheckable(True)
        self.btn_clean.setToolTip(
            "Isolate the coil: dim the PCB background so the winding is easy to read")
        self.btn_clean.toggled.connect(self._on_toggle_view)
        controls.addWidget(self.btn_clean)
        self.btn_heatmap = QPushButton("Defect heatmap")
        self.btn_heatmap.setCheckable(True)
        self.btn_heatmap.setToolTip(
            "Overlay the PaDiM anomaly heatmap to show where the coil was flagged")
        self.btn_heatmap.toggled.connect(self._on_toggle_view)
        controls.addWidget(self.btn_heatmap)
        controls.addStretch(1)
        root.addLayout(controls)

        # --- image | result panel ---
        split = QSplitter(Qt.Horizontal)

        self.image_label = QLabel("Open an image to inspect")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(560, 380)
        self.image_label.setStyleSheet(
            "background:#101417; color:#888; border:1px solid #2a2f33;")
        split.addWidget(self.image_label)

        panel = QWidget()
        pl = QVBoxLayout(panel)
        self.badge = QLabel("--")
        self.badge.setAlignment(Qt.AlignCenter)
        self.badge.setFixedHeight(96)
        self.badge.setStyleSheet(
            "font-size:34px; font-weight:bold; color:#ddd; "
            "background:#222; border-radius:8px;")
        pl.addWidget(self.badge)

        self.band_lbl = QLabel("band: --")
        self.band_lbl.setStyleSheet("font-size:15px;")
        pl.addWidget(self.band_lbl)

        self.score_lbl = QLabel("anomaly score: --")
        pl.addWidget(self.score_lbl)

        self.meter = QProgressBar()
        self.meter.setRange(0, 100)
        self.meter.setFormat("severity %p%")
        pl.addWidget(self.meter)

        self.thr_lbl = QLabel("thresholds: --")
        self.thr_lbl.setStyleSheet("color:#999;")
        pl.addWidget(self.thr_lbl)

        self.latency_lbl = QLabel("latency: --")
        self.latency_lbl.setStyleSheet("color:#999;")
        pl.addWidget(self.latency_lbl)
        pl.addStretch(1)
        split.addWidget(panel)
        split.setSizes([620, 360])
        root.addWidget(split, stretch=3)

        # --- batch progress + table ---
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["File", "Band", "Decision", "Score", "ms"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        root.addWidget(self.table, stretch=2)

    # ------------------------------------------------------------------
    # Single image
    # ------------------------------------------------------------------
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open coil image", "", "BMP images (*.bmp);;All files (*)")
        if path:
            self.show_image(path)

    def show_image(self, path):
        self._display(path)

    def _display(self, path):
        """Localize (one forward pass) and render the verdict panel + image.
        Used for single-image open and batch row selection."""
        try:
            r = self.predictor.localize(path)
        except Exception as e:
            self.status_cb(f"Inference failed: {e}")
            return
        self._current = (path, r)
        self._render_result(r)
        self._render_image(path, r)
        self.status_cb(f"Inspected {Path(path).name}  ->  {r.get('band', '--')}")

    def _render_image(self, path, r=None):
        if self.btn_clean.isChecked():
            pix = load_clean_coil(path, self.predictor.crop)
        elif (self.btn_heatmap.isChecked() and r is not None
              and r.get("amap") is not None):
            pix = load_full_with_heatmap(
                path, self.predictor.crop, r["amap"], r.get("peak"),
                t_flag=r.get("thresholds", {}).get("t_flag"))
        else:
            pix = load_full_with_box(path, self.predictor.crop)
        self.image_label.setPixmap(pix.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _on_toggle_view(self, checked):
        # the two views are mutually exclusive
        if checked:
            other = self.btn_heatmap if self.sender() is self.btn_clean else self.btn_clean
            if other.isChecked():
                other.blockSignals(True)
                other.setChecked(False)
                other.blockSignals(False)
        if self._current:
            self._render_image(*self._current)
        elif checked:
            self.status_cb("Open an image first")

    def _render_result(self, r):
        decision = r["decision"]
        text, color = VERDICT_STYLE.get(decision, ("--", "#222"))
        self.badge.setText(text)
        self.badge.setStyleSheet(
            f"font-size:34px; font-weight:bold; color:white; "
            f"background:{color}; border-radius:8px;")
        self.band_lbl.setText(f"band: <b>{r.get('band','--')}</b>")
        self.score_lbl.setText(f"anomaly score: <b>{r.get('anomaly_score','--')}</b>")
        self.meter.setValue(int(round(r.get("score_norm", 0) * 100)))
        thr = r.get("thresholds", {})
        if thr:
            self.thr_lbl.setText(
                f"thresholds: auto-pass &lt; {thr['t_low']}  |  "
                f"auto-flag &ge; {thr['t_flag']}")
        self.latency_lbl.setText(f"latency: {r.get('latency_ms','--')} ms")

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------
    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open folder of BMPs")
        if not folder:
            return
        paths = sorted(Path(folder).glob("*.bmp"))
        if not paths:
            self.status_cb("No .bmp files found in folder")
            return
        self.table.setRowCount(0)
        self._batch_results = []
        self.btn_export.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(paths))
        self.progress.setValue(0)
        self._set_buttons(False)

        self._worker = BatchWorker(self.predictor, paths)
        self._worker.progress.connect(self._on_batch_progress)
        self._worker.done.connect(self._on_batch_done)
        self._worker.start()

    def _on_batch_progress(self, i, n, path, r):
        self.progress.setValue(i)
        row = self.table.rowCount()
        self.table.insertRow(row)
        _, color = VERDICT_STYLE.get(r["decision"], ("--", "#222"))
        cells = [Path(path).name, r.get("band", "--"), r["decision"],
                 str(r.get("anomaly_score", "--")), str(r.get("latency_ms", "--"))]
        for c, val in enumerate(cells):
            item = QTableWidgetItem(val)
            if c == 1:
                item.setForeground(Qt.white)
                item.setBackground(_qcolor(color))
            self.table.setItem(row, c, item)
        self.status_cb(f"Batch {i}/{n}")

    def _on_batch_done(self, results):
        self._batch_results = results
        self.progress.setVisible(False)
        self.btn_export.setEnabled(True)
        self._set_buttons(True)
        bands = {}
        for _, r in results:
            bands[r["band"]] = bands.get(r["band"], 0) + 1
        summary = "  ".join(f"{k}={v}" for k, v in sorted(bands.items()))
        self.status_cb(f"Batch done: {len(results)} images  [{summary}]")

    def _on_row_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if rows and self._batch_results:
            idx = rows[0].row()
            if idx < len(self._batch_results):
                path, _ = self._batch_results[idx]
                # re-localize the selected row so the heatmap (which the scalar
                # batch result doesn't carry) is available on demand
                self._display(str(path))

    def export_csv(self):
        if not self._batch_results:
            return
        out, _ = QFileDialog.getSaveFileName(
            self, "Export results", "inspection_results.csv", "CSV (*.csv)")
        if not out:
            return
        fields = ["filepath", "band", "decision", "pass_fail",
                  "anomaly_score", "score_norm", "latency_ms", "timestamp"]
        ts = datetime.now().isoformat(timespec="seconds")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for path, r in self._batch_results:
                w.writerow({
                    "filepath":      str(path),
                    "band":          r.get("band", ""),
                    "decision":      r.get("decision", ""),
                    "pass_fail":     r.get("pass_fail", ""),
                    "anomaly_score": r.get("anomaly_score", ""),
                    "score_norm":    r.get("score_norm", ""),
                    "latency_ms":    r.get("latency_ms", ""),
                    "timestamp":     ts,
                })
        self.status_cb(f"Exported {len(self._batch_results)} rows -> {Path(out).name}")

    def _set_buttons(self, on):
        self.btn_image.setEnabled(on)
        self.btn_folder.setEnabled(on)


def _qcolor(hexstr):
    from PyQt5.QtGui import QColor
    return QColor(hexstr)
