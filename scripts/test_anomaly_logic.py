"""
Pure-function unit tests for the PaDiM anomaly math and the human-assist bands.

The critical one is test_score_path_equivalence: the deployed thresholds (t_low,
t_flag) are only meaningful if the inference-time score (predictor._score_padim's
per-image einsum) matches the training-time score (PaDiM.score's batched einsum
+ max-over-patches). A future tweak to either path could silently shift scores
and invalidate the thresholds -- this pins the contract without loading a model.

Run:
    python scripts/test_anomaly_logic.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.anomaly import PaDiM, _band_split
from src.training.finetune import recall_at_fr


def _random_padim(L, R, seed=0):
    rng = np.random.RandomState(seed)
    mean = rng.randn(L, R).astype(np.float32)
    inv = np.empty((L, R, R), dtype=np.float32)
    for l in range(L):
        A = rng.randn(R, R).astype(np.float32)
        inv[l] = A @ A.T + np.eye(R, dtype=np.float32)   # SPD
    det = PaDiM()
    det.mean, det.inv = mean, inv
    return det, mean, inv


def test_score_path_equivalence():
    L, R = 6, 4
    det, mean, inv = _random_padim(L, R)
    X = np.random.RandomState(1).randn(L, R).astype(np.float32)   # one image

    # training-time: PaDiM.score on a (1, L, R) batch
    train_score = float(det.score(X[None])[0])

    # inference-time: predictor._score_padim's per-image form
    d = X - mean
    m = np.einsum("lr,lrs,ls->l", d, inv, d)
    infer_score = float(np.sqrt(np.maximum(m, 0.0)).max())

    assert abs(train_score - infer_score) < 1e-4, (train_score, infer_score)
    print("  ok: training PaDiM.score == inference per-image score")


def test_band_split():
    s = _band_split(np.array([1.0, 2.0, 3.0, 100.0]),
                    np.array([0, 0, 1, 1]), t_low=2.0, t_flag=50.0)
    assert s["auto_pass"] == 0.25      # only score 1 is < 2
    assert s["review"] == 0.5          # scores 2 and 3
    assert s["flag"] == 0.25           # only score 100 is >= 50
    assert s["autopass_defect_miss"] == 0.0    # no defect auto-passed
    assert s["flag_defect_recall"] == 0.5      # 1 of 2 defects auto-flagged
    print("  ok: _band_split workload + critical auto-pass-miss")


def test_recall_at_fr_zero_budget():
    rec, thr, fr = recall_at_fr(np.array([1.0, 2.0, 3.0, 4.0]),
                                np.array([0, 0, 1, 1]), fr_budget=0.0)
    assert rec == 1.0 and fr == 0.0 and thr == 3.0   # both defects, zero false-reject
    print("  ok: recall_at_fr honors a 0% false-reject budget")


if __name__ == "__main__":
    print("Anomaly-logic unit tests...")
    test_score_path_equivalence()
    test_band_split()
    test_recall_at_fr_zero_budget()
    print("ALL PASS")
