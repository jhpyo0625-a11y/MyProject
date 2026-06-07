"""
T7.4 / T7.5 -- Approval gate, model versioning, and rollback.

- approve_candidate : archive the current production model, promote the
  candidate to production, reset the retrain counter.
- reject_candidate  : discard the candidate, keep production untouched.
- list_versions     : archived snapshots with their metrics (history view).
- rollback          : restore an archived version to production (after first
  archiving the current one, so rollback is itself reversible).

Each production model is a small set of files (padim.pkl, padim_card.md,
metrics.json). Archives live in models/archive/{YYYYMMDD_HHMMSS}/ and are
gitignored.
"""

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.config_loader import load_config
from src.retraining import trigger

cfg         = load_config()
PROD_DIR    = ROOT / cfg["paths"]["production_dir"]
ARCHIVE_DIR = ROOT / cfg["paths"]["archive_dir"]
MODELS_DIR  = PROD_DIR.parent
CANDIDATE_DIR = MODELS_DIR / "candidate"

SNAPSHOT_FILES = ("padim.pkl", "padim_card.md", "metrics.json")


def _now():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _copy_snapshot(src_dir: Path, dst_dir: Path) -> list:
    """Plain copy into a fresh dir (archiving). No atomicity needed -- the
    destination is a new, otherwise-empty archive folder."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in SNAPSHOT_FILES:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
            copied.append(name)
    return copied


def _promote_snapshot(src_dir: Path, dst_dir: Path) -> list:
    """Replace the live production model atomically per file.

    Stage every snapshot file as a `.new` temp first; if any copy fails,
    production is left untouched. Only once all temps exist do we os.replace
    them into place -- fast metadata swaps, so production is never left with a
    half-written individual file.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    try:
        for name in SNAPSHOT_FILES:
            src = src_dir / name
            if not src.exists():
                continue
            tmp = dst_dir / (name + ".new")
            shutil.copy2(src, tmp)
            staged.append((tmp, dst_dir / name))
    except Exception:
        for tmp, _ in staged:
            tmp.unlink(missing_ok=True)
        raise
    for tmp, final in staged:
        os.replace(tmp, final)
    return [final.name for _, final in staged]


def _unique_version(version: str) -> str:
    """Avoid clobbering an existing archive when two ops land in the same second
    (e.g. approve then an immediate rollback). Append _2, _3, ... as needed."""
    if not (ARCHIVE_DIR / version).exists():
        return version
    i = 2
    while (ARCHIVE_DIR / f"{version}_{i}").exists():
        i += 1
    return f"{version}_{i}"


def archive_current_production(version=None) -> str:
    """Snapshot the live production model into models/archive/{version}/."""
    if (PROD_DIR / "padim.pkl").exists():
        version = _unique_version(version or _now())
        _copy_snapshot(PROD_DIR, ARCHIVE_DIR / version)
        return version
    return version or _now()


def approve_candidate(store=None) -> dict:
    """Promote models/candidate/ -> models/production/, archiving the old one."""
    if not (CANDIDATE_DIR / "padim.pkl").exists():
        raise FileNotFoundError("no candidate to approve -- run retrain first")
    archived = archive_current_production()
    _promote_snapshot(CANDIDATE_DIR, PROD_DIR)
    shutil.rmtree(CANDIDATE_DIR, ignore_errors=True)
    state = trigger.mark_retrained(store) if store is not None else None
    return {"promoted": True, "archived_as": archived, "retrain_state": state}


def reject_candidate() -> dict:
    existed = CANDIDATE_DIR.exists()
    shutil.rmtree(CANDIDATE_DIR, ignore_errors=True)
    return {"rejected": existed}


def list_versions() -> list:
    """Archived versions, newest first, with metrics if present."""
    if not ARCHIVE_DIR.exists():
        return []
    out = []
    for d in sorted(ARCHIVE_DIR.iterdir(), reverse=True):
        if not (d / "padim.pkl").exists():
            continue
        metrics = {}
        mp = d / "metrics.json"
        if mp.exists():
            try:
                metrics = json.loads(mp.read_text(encoding="utf-8"))
            except Exception:
                metrics = {}
        out.append({"version": d.name, "metrics": metrics})
    return out


def rollback(version: str) -> dict:
    """Restore an archived version to production (archiving the current first)."""
    if Path(version).name != version:
        raise ValueError(f"invalid version {version!r} -- must be a direct "
                         "child of the archive dir")
    src = ARCHIVE_DIR / version
    if not (src / "padim.pkl").exists():
        raise FileNotFoundError(f"no archived model at {src}")
    prev = archive_current_production()
    _promote_snapshot(src, PROD_DIR)
    return {"rolled_back_to": version, "previous_archived_as": prev}


if __name__ == "__main__":
    print("Archived versions:")
    for v in list_versions():
        m = v["metrics"]
        print(f"  {v['version']}  cv_recall@20FR="
              f"{m.get('cv_recall_at_20fr', '?')}  "
              f"fold4_leak={m.get('fold4_autopass_defect_miss', '?')}")
