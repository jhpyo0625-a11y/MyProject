"""
T6.3 -- Unlabeled-image ingestion + label commit helpers.

- scan_folder: find .bmp files in a folder not already in the label store.
- watch_folder: the configured drop folder camera software writes to.
- commit_label: the write side of the human-assist loop. When an operator
  confirms a class, the image is recorded in labels.db, copied into the curated
  data/raw/{label}/ store, and appended to manifest.csv so the next retrain
  sees it. Folds are assigned by physical coil (session_id + position): a new
  image of a coil already in the manifest joins that coil's fold, and a new
  coil is hashed into folds 0-(COMP_FOLD-1) -- never fold 4, the held-out
  comparison fold. Keeping a coil's images in one fold prevents the leakage
  that plain per-image assignment caused.
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


def _coil(session: str, position: str) -> str:
    """Physical-coil identity -- the grouping key (matches src.data.split.coil_id)."""
    return f"{session}|{position}"


def _assign_fold(session: str, position: str) -> int:
    """Deterministic CV fold in [0, COMP_FOLD) for a NEW coil, hashing the coil
    identity so all of its images share a fold. Matches
    src.data.split._incremental_fold. Never the held-out comparison fold."""
    return zlib.crc32(_coil(session, position).encode("utf-8")) % COMP_FOLD


def _existing_coil_fold(session: str, position: str):
    """Fold of this coil if it's already in the manifest, else None. Reusing it
    keeps every image of a physical coil in one fold across retrains (no leak).
    A coil already in the comparison fold keeps growing it -- leakage-freedom is
    preferred over a perfectly frozen yardstick; only NEW coils are barred from
    fold 4."""
    if not MANIFEST.exists():
        return None
    with open(MANIFEST, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("session_id", ""), r.get("position", "")) == (session, position) \
                    and (r.get("fold") or "") != "":
                return r["fold"]
    return None


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
    session  = parsed.get("session", "")
    position = parsed.get("position", "")
    fold = _existing_coil_fold(session, position)
    if fold is None:                       # new coil -> hash into folds 0..COMP_FOLD-1
        fold = str(_assign_fold(session, position))
    row = {
        "filepath":   str(dest),
        "label":      human_label,
        "session_id": session,
        "sensor":     parsed.get("sensor", ""),
        "position":   position,
        "layer":      parsed.get("layer", ""),
        "param":      parsed.get("param") or "",
        "fold":       fold,
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
