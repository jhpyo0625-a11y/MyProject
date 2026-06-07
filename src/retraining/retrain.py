"""
T7.2 -- Retrain script (PaDiM).

Adapted from the original sklearn-classifier spec to the deployed anomaly
detector. A retrain here means:
  1. Re-run preprocessing on any new images (preprocess.py is skip-if-exists,
     so only newly-labeled crops are built).
  2. Re-fit the PaDiM detector on the current Pass coils (folds 0-3) and derive
     the 3-band policy from OOF -- writing a CANDIDATE into models/candidate/,
     never straight to production.
  3. Log the run to MLflow.

The candidate is then compared against production on the locked fold 4
(compare.py) and only promoted on human approval (promote.py). Fold 4 never
receives new labels, so it stays a stable yardstick across retrains.

Run:
    python -m src.retraining.retrain
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.config_loader import load_config

cfg          = load_config()
MODELS_DIR   = (ROOT / cfg["paths"]["production_dir"]).parent
CANDIDATE_DIR = MODELS_DIR / "candidate"


def _run_preprocess() -> None:
    print("\n=== [1/2] Preprocess (build crops for any new images) ===",
          flush=True)
    proc = subprocess.run(
        [sys.executable, "-u", "-m", "src.data.preprocess"],
        cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"preprocess failed (exit {proc.returncode})")


def retrain() -> dict:
    """Build a candidate PaDiM model. Returns its metrics dict."""
    _run_preprocess()

    print("\n=== [2/2] Fit candidate PaDiM -> models/candidate/ ===", flush=True)
    # Import here so preprocess's heavy deps load only once, after step 1.
    from src.training.anomaly import run_padim

    metrics = run_padim(out_dir=CANDIDATE_DIR, verbose=True)

    # Logging is best-effort: a candidate is already built and written, so a
    # tracking-store failure must not discard it.
    try:
        _log_mlflow(metrics)
    except Exception as e:
        print(f"  (mlflow logging skipped: {e})")
    print(f"\nCANDIDATE READY -> {CANDIDATE_DIR}")
    print(f"  CV recall@20%FR        : {metrics['cv_recall_at_20fr']:.1%}")
    print(f"  fold4 auto-pass        : {metrics['fold4_auto_pass']:.0%}")
    print(f"  fold4 defects leaked   : {metrics['fold4_autopass_defect_miss']:.1%}")
    return metrics


def _log_mlflow(metrics: dict) -> None:
    # Guard the whole body, not just the import: a locked/again-busy SQLite
    # tracking DB or disk error here is non-fatal to the retrain.
    try:
        import mlflow
        db = ROOT / cfg["paths"]["experiments"] / "mlflow.db"
        db.parent.mkdir(exist_ok=True)
        mlflow.set_tracking_uri("sqlite:///" + str(db).replace("\\", "/"))
        mlflow.set_experiment("coil-defect-detection")
        with mlflow.start_run(run_name="retrain-padim-candidate"):
            mlflow.log_params({"approach": "padim_retrain",
                               "n_pass_train": metrics["n_pass_train"]})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                if isinstance(v, (int, float))})
    except Exception as e:
        print(f"  (mlflow logging skipped: {e})")


if __name__ == "__main__":
    retrain()
