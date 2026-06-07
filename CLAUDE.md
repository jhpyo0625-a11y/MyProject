# Coil Defect Inspection — Claude Context

Classify PCB-mounted induction coil images as **Pass / Dent / Loose** (+ binary Pass/Fail).  
Target: ≥95% defect recall. CPU-only. Windows line-side PC.

## Key architectural decisions (non-obvious)

- **Fixed crop, not learned ROI.** Coil center is nearly fixed across all images (~±45 px jitter). Crop box `x∈[300,2000], y∈[610,1480]` in full 2448×2048 frame. Locked in `config.yaml`.
- **Frozen backbone → cached embeddings → sklearn classifier.** The CNN never trains. Embeddings are extracted once and saved to `data/embeddings/`. Retraining = re-fit the classifier only (seconds on CPU).
- **False negative is the costlier error.** When uncertain → route to human review, never auto-pass. Threshold + review band in `config.yaml` under `decision.*`.
- **Fold 4 is the comparison fold.** Never used in training. Used only for old-vs-new model comparison in the approval gate.

## Directory layout

```
src/
  data/         # manifest, crop, augment, split
  features/     # backbone extractor, embedding cache
  training/     # train, evaluate, threshold tuning
  inference/    # predictor.py  ← single inference entry point
  labeling/     # label_store.py (SQLite), ingestion
  retraining/   # trigger, retrain, compare, approval
  app/          # PyQt5 desktop app (3 tabs: Inspect / Label / Retrain)
data/
  raw/          # original BMPs (do not modify)
  crops/        # preprocessed crops (.pt)
  embeddings/   # cached embeddings (.npy) + index.csv
  labels.db     # SQLite — human-confirmed labels
  manifest.csv  # filepath, label, fold, session metadata
models/
  production/   # active model (classifier.pkl + config snapshot)
  archive/      # versioned past models
experiments/    # MLflow tracking store
config.yaml     # all thresholds, paths, trigger counts — no hardcoded values
```

## Common commands

```bash
# Setup
pip install -r requirements.txt
python scripts/verify_env.py

# Data pipeline (run in order on first setup)
python src/data/build_manifest.py
python scripts/validate_crop.py          # must pass before proceeding
python src/data/preprocess.py
python src/features/build_cache.py

# Training & evaluation
python src/training/train.py
python src/training/evaluate.py
python src/training/threshold.py

# MLflow UI
mlflow ui --backend-store-uri experiments/

# Quick inference (CLI)
python scripts/run_inference.py --image path/to/image.bmp

# Launch desktop app
python src/app/main.py
# or on Windows:
scripts\launch_app.bat
```

## Style

- Config over constants: paths, thresholds, trigger counts → `config.yaml`, never hardcoded.
- `src/inference/predictor.py` is the single inference entry point. Don't bypass it.
- Scripts that a user runs directly → `scripts/`. Library code → `src/`.
- No comments unless the WHY is non-obvious.

## Reference docs (read on demand, not every session)

- `PROJECT_SPEC.md` — full architecture spec, open items, preprocessing details
- `TASKS.md` — full task breakdown with build order and deliverables
