"""
T6.1 -- SQLite label store for human-confirmed coil labels.

This is the heart of the human-assist loop: the PaDiM detector flags/triages,
the operator confirms the true class, and every confirmation lands here. Those
confirmations are the only thing that will raise the ~55% recall ceiling, so
this store is also the data source for the Phase 7 retraining trigger.

Schema (one row per image):
    id, filepath, model_prediction, model_confidence,
    human_label, operator_id, labeled_at, session_id, status

    status: 'pending'  -- queued, not yet decided
            'labeled'  -- human_label set
            'skipped'  -- deferred by operator (still counts as unlabeled)

Usage:
    store = LabelStore()                       # data/labels.db from config
    store.insert_pending(path, "REVIEW", 0.43, session_id="2026-06-08")
    store.confirm_label(path, "Dent", operator_id="line1")
    store.get_unlabeled_count()
    store.export_to_csv("labels_export.csv")
"""

import csv
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.config_loader import load_config

VALID_LABELS = ("Pass", "Dent", "Loose")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS labels (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath         TEXT UNIQUE NOT NULL,
    model_prediction TEXT,
    model_confidence REAL,
    human_label      TEXT,
    operator_id      TEXT,
    labeled_at       TEXT,
    session_id       TEXT,
    status           TEXT NOT NULL DEFAULT 'pending'
);
"""


class LabelStore:
    def __init__(self, db_path=None, config_path=None):
        if db_path is None:
            cfg = load_config(config_path)
            db_path = ROOT / cfg["paths"]["labels_db"]
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self):
        # One connection per operation -> safe across the GUI + worker threads.
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def insert_pending(self, filepath, model_prediction=None,
                       model_confidence=None, session_id=None) -> bool:
        """Queue an image for labeling. No-op if already present. Returns True
        if a new row was created."""
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO labels "
                "(filepath, model_prediction, model_confidence, session_id, status) "
                "VALUES (?, ?, ?, ?, 'pending')",
                (str(filepath), model_prediction,
                 None if model_confidence is None else float(model_confidence),
                 session_id),
            )
            return cur.rowcount > 0

    def confirm_label(self, filepath, human_label, operator_id=None) -> None:
        """Record the operator's decision for an image."""
        if human_label not in VALID_LABELS:
            raise ValueError(f"human_label must be one of {VALID_LABELS}, "
                             f"got {human_label!r}")
        with self._connect() as conn:
            # Upsert: the image may not have been pre-queued (e.g. drag-drop).
            conn.execute(
                "INSERT OR IGNORE INTO labels (filepath, status) VALUES (?, 'pending')",
                (str(filepath),),
            )
            conn.execute(
                "UPDATE labels SET human_label=?, operator_id=?, labeled_at=?, "
                "status='labeled' WHERE filepath=?",
                (human_label, operator_id, datetime.now().isoformat(timespec="seconds"),
                 str(filepath)),
            )

    def skip(self, filepath) -> None:
        """Defer an image without labeling it (stays unlabeled)."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO labels (filepath, status) VALUES (?, 'skipped')",
                (str(filepath),),
            )
            conn.execute(
                "UPDATE labels SET status='skipped' WHERE filepath=? "
                "AND human_label IS NULL",
                (str(filepath),),
            )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def exists(self, filepath) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM labels WHERE filepath=?",
                               (str(filepath),)).fetchone()
            return row is not None

    def get_unlabeled_count(self) -> int:
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM labels WHERE human_label IS NULL"
            ).fetchone()[0]

    def count_confirmed(self) -> int:
        """Total images with a human label (drives the retrain trigger)."""
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM labels WHERE human_label IS NOT NULL"
            ).fetchone()[0]

    def count_labeled_today(self) -> int:
        today = datetime.now().date().isoformat()
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM labels WHERE human_label IS NOT NULL "
                "AND labeled_at LIKE ?", (today + "%",)).fetchone()[0]

    def label_counts(self) -> dict:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT human_label, COUNT(*) c FROM labels "
                "WHERE human_label IS NOT NULL GROUP BY human_label").fetchall()
        return {r["human_label"]: r["c"] for r in rows}

    def all_rows(self) -> list:
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM labels ORDER BY id").fetchall()]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_to_csv(self, out_path) -> int:
        rows = self.all_rows()
        fields = ["id", "filepath", "model_prediction", "model_confidence",
                  "human_label", "operator_id", "labeled_at", "session_id", "status"]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        return len(rows)


if __name__ == "__main__":
    # Smoke test against a temp DB.
    import tempfile
    tmp = Path(tempfile.gettempdir()) / "labelstore_smoke.db"
    if tmp.exists():
        tmp.unlink()
    s = LabelStore(db_path=tmp)
    assert s.insert_pending("a.bmp", "AUTO-FLAG", 0.91, "sess1") is True
    assert s.insert_pending("a.bmp") is False          # dup ignored
    s.insert_pending("b.bmp", "REVIEW", 0.4, "sess1")
    assert s.get_unlabeled_count() == 2
    s.confirm_label("a.bmp", "Dent", "op1")
    assert s.get_unlabeled_count() == 1
    assert s.count_confirmed() == 1
    s.skip("b.bmp")
    assert s.get_unlabeled_count() == 1                 # skipped still unlabeled
    s.confirm_label("c.bmp", "Loose", "op1")            # never pre-queued
    assert s.count_confirmed() == 2
    n = s.export_to_csv(Path(tempfile.gettempdir()) / "labelstore_smoke.csv")
    print(f"OK -- {n} rows, counts={s.label_counts()}, "
          f"unlabeled={s.get_unlabeled_count()}")
    try:
        tmp.unlink()
    except PermissionError:
        pass   # Windows may hold the sqlite handle briefly; temp file, ignore
