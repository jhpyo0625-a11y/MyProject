"""
T7.1 -- Count-based retrain trigger.

Tracks how many labels have been confirmed *since the last deployed retrain*
(not just total), so the "retrain recommended" banner reflects genuinely new
data. State is a tiny JSON file; the count baseline is advanced when a candidate
is approved & deployed (see promote.approve).
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.config_loader import load_config

_cfg       = load_config()
STATE_PATH = ROOT / _cfg["paths"]["production_dir"] / "retrain_state.json"
TRIGGER_COUNT = int(_cfg["retraining"]["trigger_count"])


def _default_state() -> dict:
    return {"last_retrain_confirmed_count": 0, "last_retrain_at": None}


def read_state() -> dict:
    if not STATE_PATH.exists():
        return _default_state()
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # A truncated/corrupt state file must not brick the Retrain tab; fall
        # back to the default (baseline 0) so the banner still renders.
        return _default_state()


def _write_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_PATH)   # atomic swap; no half-written state file


def labels_since_last_retrain(store) -> int:
    baseline = read_state()["last_retrain_confirmed_count"]
    return max(0, store.count_confirmed() - baseline)


def should_trigger(store, threshold: int = TRIGGER_COUNT) -> bool:
    return labels_since_last_retrain(store) >= threshold


def mark_retrained(store) -> dict:
    """Advance the baseline to the current confirmed-label count. Call this when
    a candidate is approved & deployed, so the counter resets to 0."""
    state = {
        "last_retrain_confirmed_count": store.count_confirmed(),
        "last_retrain_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_state(state)
    return state


def status(store, threshold: int = TRIGGER_COUNT) -> dict:
    since = labels_since_last_retrain(store)
    return {
        "confirmed_total": store.count_confirmed(),
        "since_last_retrain": since,
        "threshold": threshold,
        "recommended": since >= threshold,
        "last_retrain_at": read_state()["last_retrain_at"],
    }


if __name__ == "__main__":
    from src.labeling.label_store import LabelStore
    print(json.dumps(status(LabelStore()), indent=2))
