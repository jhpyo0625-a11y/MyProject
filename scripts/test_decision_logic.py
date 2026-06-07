"""
Pure-function unit tests for the pass/fail decision logic and threshold math.

Unlike scripts/smoke_test_predictor.py (which needs a real serialized model +
data on disk), these exercise the gating logic in isolation -- no torch weights,
no images -- so they run fast and catch regressions in the safety-critical
"never silently auto-pass" rules.

Run:
    python scripts/test_decision_logic.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.training.train as train
from src.training.train import binary_metrics, p_fail_from_proba, tune_threshold
from src.inference.predictor import Predictor


def test_p_fail_from_proba():
    proba = np.array([[0.7, 0.2, 0.1]])
    # Pass encoded as class 0 -> column 0
    assert abs(p_fail_from_proba(proba, [0, 1, 2])[0] - 0.3) < 1e-9
    # If Pass sits in a different column, p_fail must track that column
    assert abs(p_fail_from_proba(proba, [1, 0, 2])[0] - 0.8) < 1e-9
    print("  ok: p_fail_from_proba tracks the Pass column")


def test_binary_metrics():
    y_bin = np.array([0, 0, 1, 1])
    pred  = np.array([0, 1, 1, 0])
    m = binary_metrics(y_bin, pred)
    assert abs(m["fail_recall"]    - 0.5) < 1e-9   # caught 1 of 2 defects
    assert abs(m["fail_precision"] - 0.5) < 1e-9   # 1 of 2 flagged were real
    assert abs(m["false_reject"]   - 0.5) < 1e-9   # 1 of 2 good coils rejected
    print("  ok: binary_metrics recall/precision/false-reject")


def test_tune_threshold_reachable():
    proba = np.array([[0.9, 0.05, 0.05]] * 10 + [[0.1, 0.45, 0.45]] * 10)
    y_bin = np.array([0] * 10 + [1] * 10)
    threshold, p_fail, stats = tune_threshold(proba, y_bin, [0, 1, 2])
    # Perfectly separable -> target recall met with zero false-rejects
    assert stats["achieved_recall"] >= train.TARGET_RECALL
    assert stats["false_reject"] == 0.0
    assert 0.1 <= threshold <= 0.9
    print("  ok: tune_threshold hits target recall on separable data")


def test_tune_threshold_unreachable_fallback():
    proba = np.array([[0.9, 0.05, 0.05]] * 10 + [[0.1, 0.45, 0.45]] * 10)
    y_bin = np.array([0] * 10 + [1] * 10)
    p_fail = p_fail_from_proba(proba, [0, 1, 2])
    expected_thr0 = float(precision_recall_curve(y_bin, p_fail)[2][0])

    saved = train.TARGET_RECALL
    train.TARGET_RECALL = 1.01           # impossible -> force the fallback branch
    try:
        threshold, _, _ = tune_threshold(proba, y_bin, [0, 1, 2])
    finally:
        train.TARGET_RECALL = saved
    assert abs(threshold - expected_thr0) < 1e-9
    print("  ok: tune_threshold falls back to thr[0] when target unreachable")


def test_decide_supervised():
    d = Predictor._decide_supervised
    assert d(0.7, 0.5, 0.1) == ("Fail", "Fail")     # above band
    assert d(0.3, 0.5, 0.1) == ("Pass", "Pass")     # below band
    assert d(0.5, 0.5, 0.1) == ("Review", "Fail")   # in band -> never auto-pass
    assert d(0.6, 0.5, 0.1) == ("Fail", "Fail")     # boundary thr+delta
    assert d(0.4, 0.5, 0.1) == ("Pass", "Pass")     # boundary thr-delta
    print("  ok: supervised banding, Review never auto-passes")


def test_decide_padim():
    d = Predictor._decide_padim
    assert d(35.0, 10.0, 30.0) == ("AUTO-FLAG", "Fail")
    assert d(5.0,  10.0, 30.0) == ("AUTO-PASS", "Pass")
    assert d(20.0, 10.0, 30.0) == ("REVIEW",    "Review")
    assert d(30.0, 10.0, 30.0) == ("AUTO-FLAG", "Fail")    # boundary == t_flag
    assert d(10.0, 10.0, 30.0) == ("REVIEW",    "Review")  # boundary == t_low
    print("  ok: PaDiM 3-band thresholds")


if __name__ == "__main__":
    print("Decision-logic unit tests...")
    test_p_fail_from_proba()
    test_binary_metrics()
    test_tune_threshold_reachable()
    test_tune_threshold_unreachable_fallback()
    test_decide_supervised()
    test_decide_padim()
    print("ALL PASS")
