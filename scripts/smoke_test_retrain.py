"""
Headless smoke test for the Phase 7 retraining pipeline.

Exercises the fast, file-level logic without re-running a full retrain:
  - trigger: since-last-retrain counting + mark_retrained reset
  - promote: approve (archive + promote + counter reset), list_versions,
    rollback, reject -- all against temp dirs with fake snapshot files.

The heavy paths (retrain.py building a candidate, compare.py scoring fold 4)
are validated separately by running them directly:
    python -m src.retraining.retrain
    python -m src.retraining.compare
"""

import json
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.retraining.promote as pr
import src.retraining.trigger as tg
from src.labeling.label_store import LabelStore


def _mk(d, tag):
    d.mkdir(parents=True, exist_ok=True)
    (d / "padim.pkl").write_bytes(b"x")
    (d / "padim_card.md").write_text("card " + tag, encoding="utf-8")
    (d / "metrics.json").write_text(
        json.dumps({"cv_recall_at_20fr": 0.55, "tag": tag}), encoding="utf-8")


def test_trigger():
    tmp = Path(tempfile.mkdtemp(prefix="trig_"))
    tg.STATE_PATH = tmp / "retrain_state.json"
    store = LabelStore(db_path=tmp / "labels.db")
    store.confirm_label("a.bmp", "Dent")
    store.confirm_label("b.bmp", "Loose")
    assert tg.labels_since_last_retrain(store) == 2
    assert not tg.should_trigger(store, threshold=5)
    tg.mark_retrained(store)
    assert tg.labels_since_last_retrain(store) == 0   # baseline advanced
    store.confirm_label("c.bmp", "Pass")
    assert tg.labels_since_last_retrain(store) == 1
    print("  trigger ok: counting + mark_retrained reset")


def test_corrupt_state():
    # A truncated/corrupt state file must not brick the Retrain tab.
    tmp = Path(tempfile.mkdtemp(prefix="corrupt_"))
    tg.STATE_PATH = tmp / "retrain_state.json"
    tg.STATE_PATH.write_text("{ broken", encoding="utf-8")
    store = LabelStore(db_path=tmp / "labels.db")
    store.confirm_label("a.bmp", "Dent")
    st = tg.status(store, threshold=5)        # must not raise
    assert st["since_last_retrain"] == 1      # fell back to default baseline 0
    assert st["last_retrain_at"] is None
    print("  trigger ok: survives corrupt state file")


def test_baseline_is_approval_time():
    # Policy pin: mark_retrained snapshots the count at APPROVAL time, so labels
    # confirmed during the review window are folded into the baseline (treated
    # as included). Intentional + documented; this test makes it explicit.
    tmp = Path(tempfile.mkdtemp(prefix="policy_"))
    tg.STATE_PATH = tmp / "retrain_state.json"
    store = LabelStore(db_path=tmp / "labels.db")
    for i in range(5):
        store.confirm_label(f"b{i}.bmp", "Pass")   # candidate "built" at count==5
    store.confirm_label("late1.bmp", "Pass")        # arrive during review
    store.confirm_label("late2.bmp", "Pass")
    tg.mark_retrained(store)                         # approval time, count==7
    assert tg.labels_since_last_retrain(store) == 0
    print("  trigger ok: baseline advances to approval-time count (policy pinned)")


def test_retrain_survives_mlflow_failure():
    # A failing mlflow log must not discard an already-built candidate.
    # Stub the heavy deps (preprocess + PaDiM fit) so this stays a fast unit:
    # inject a fake src.training.anomaly so retrain's lazy import never loads torch.
    import src.retraining.retrain as rt
    fake_metrics = {"cv_recall_at_20fr": 0.55, "fold4_auto_pass": 0.18,
                    "fold4_autopass_defect_miss": 0.0, "n_pass_train": 100}
    fake_an = types.ModuleType("src.training.anomaly")
    fake_an.run_padim = lambda out_dir, verbose=True: fake_metrics
    sys.modules["src.training.anomaly"] = fake_an
    rt._run_preprocess = lambda: None
    def boom(_m):
        raise RuntimeError("simulated tracking-store failure")
    rt._log_mlflow = boom
    out = rt.retrain()                              # must not raise
    assert out is fake_metrics
    sys.modules.pop("src.training.anomaly", None)
    print("  retrain ok: survives mlflow logging failure")


def test_promote():
    tmp = Path(tempfile.mkdtemp(prefix="prom_"))
    pr.PROD_DIR = tmp / "production"
    pr.ARCHIVE_DIR = tmp / "archive"
    pr.CANDIDATE_DIR = tmp / "candidate"
    tg.STATE_PATH = tmp / "production" / "retrain_state.json"

    _mk(pr.PROD_DIR, "v1")
    _mk(pr.CANDIDATE_DIR, "v2")

    class FakeStore:
        def count_confirmed(self):
            return 60

    res = pr.approve_candidate(store=FakeStore())
    assert res["promoted"]
    assert json.loads((pr.PROD_DIR / "metrics.json").read_text())["tag"] == "v2"
    assert not pr.CANDIDATE_DIR.exists()
    assert json.loads(tg.STATE_PATH.read_text())["last_retrain_confirmed_count"] == 60

    vers = pr.list_versions()
    assert len(vers) == 1 and vers[0]["metrics"]["tag"] == "v1"

    pr.rollback(vers[0]["version"])
    assert json.loads((pr.PROD_DIR / "metrics.json").read_text())["tag"] == "v1"

    _mk(pr.CANDIDATE_DIR, "v3")
    assert pr.reject_candidate()["rejected"]
    assert not pr.CANDIDATE_DIR.exists()
    print("  promote ok: approve -> archive/promote/reset, list, rollback, reject")


if __name__ == "__main__":
    print("Testing trigger...")
    test_trigger()
    test_corrupt_state()
    test_baseline_is_approval_time()
    print("Testing retrain resilience...")
    test_retrain_survives_mlflow_failure()
    print("Testing promote/versioning/rollback...")
    test_promote()
    print("SMOKE OK")
