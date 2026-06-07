"""
T6.3 -- Unlabeled-image ingestion + label commit helpers.

- scan_folder: find .bmp files in a folder not already in the label store.
- watch_folder: the configured drop folder camera software writes to.
- commit_label: the write side of the human-assist loop. When an operator
  confirms a class, the image is recorded in labels.db, copied into the curated
  data/raw/{label}/ store, and appended to manifest.csv so the next retrain
  sees it. New labels are assigned to a CV fold in 0-3 -- never fold 4, the
  held-out comparison fold used by the retraining approval gate.
"""

import csv
import shutil
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.config_loader import load_config
from src.data.build_manifest import parse_filename

_cfg       = load_config()
RAW_DIR    = ROOT / _cfg["paths"]["raw_dir"]
MANIFEST   = ROOT / _cfg["paths"]["manifest"]
COMP_FOLD  = _cfg["training"]["comparison_fold"]
MANIFEST_FIELDS = ["filepath", "label", "session_id", "sensor",
                   "position", "layer", "param", "fold"]


def scan_folder(folder, store) -> list:
    """Return sorted .bmp paths in `folder` not yet in the label store."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    out = []
    for p in sorted(folder.glob("*.bmp")):
        if not store.exists(p):
            out.append(p)
    return out


def watch_folder_path():
    """Configured drop folder (paths.watch_folder), or None if unset."""
    wf = _cfg["paths"].get("watch_folder") or ""
    return Path(wf) if wf else None


def _assign_fold(stem: str) -> int:
    """Deterministic CV fold in [0, COMP_FOLD) from the filename -- stable so a
    re-imported image always lands in the same fold. Never the comparison fold."""
    h = zlib.crc32(stem.encode("utf-8"))
    return h % COMP_FOLD


def commit_label(src_path, human_label, store, operator_id=None) -> dict:
    """Confirm a label end-to-end: curated copy + manifest append + DB record.

    The recoverable filesystem work (copy, manifest) runs FIRST and the DB
    confirmation LAST, so a failure (disk full, unparseable name) leaves the
    label simply not-done rather than counted-but-missing-from-the-manifest --
    which would otherwise over-count the Phase 7 retrain trigger.

    Returns the manifest row that was written. Idempotent on the manifest
    (won't duplicate a filepath already present)."""
    src_path = Path(src_path)

    # 1) copy into the curated raw store (keep original untouched)
    dest_dir = RAW_DIR / human_label
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src_path.name
    if not dest.exists():
        shutil.copy2(src_path, dest)

    # 2) append to manifest (skip if this filepath is already present)
    parsed = parse_filename(src_path.name)
    row = {
        "filepath":   str(dest),
        "label":      human_label,
        "session_id": parsed.get("session", ""),
        "sensor":     parsed.get("sensor", ""),
        "position":   parsed.get("position", ""),
        "layer":      parsed.get("layer", ""),
        "param":      parsed.get("param") or "",
        "fold":       _assign_fold(src_path.stem),
    }
    _append_manifest(row)

    # 3) record the human decision LAST -- only count it once the data is durable
    store.confirm_label(src_path, human_label, operator_id=operator_id)
    return row


def _append_manifest(row: dict) -> None:
    existing = set()
    write_header = True
    if MANIFEST.exists():
        write_header = False
        with open(MANIFEST, encoding="utf-8") as f:
            existing = {r["filepath"] for r in csv.DictReader(f)}
    if row["filepath"] in existing:
        return
    with open(MANIFEST, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)
