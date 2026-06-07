# Coil Defect Inspection

![Python](https://img.shields.io/badge/python-3.14-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20CPU--only-lightgrey)
![UI](https://img.shields.io/badge/UI-PyQt5-41cd52)
![Model](https://img.shields.io/badge/model-PaDiM%20%C2%B7%20human--assist-orange)
![Unit tests](https://img.shields.io/badge/unit%20tests-4%20suites%20passing-brightgreen)
![Defect recall](https://img.shields.io/badge/CV%20recall%20%4020%25%20FR-49.7%25%20(honest)-yellow)
![License](https://img.shields.io/badge/license-proprietary-red)

Classify PCB-mounted induction coil images as **Pass / Dent / Loose** (+ a binary
Pass/Fail gate), CPU-only on a Windows line-side PC.

The ≥95% auto-pass target proved unreachable on the current dataset (74 Dent /
110 Loose distinct defects) — four approaches across supervised and anomaly
paradigms all converged near a **~50% leakage-free recall ceiling**. The system
is therefore deployed as **human-assist triage**: a frozen-backbone PaDiM anomaly
detector ranks each coil into AUTO-PASS / REVIEW / AUTO-FLAG, a human confirms,
and every confirmation feeds a gated retraining loop. It never silently auto-passes.

```
MyProject/
├── config.yaml                 # all paths, thresholds, crop box, trigger counts
├── CLAUDE.md · PROJECT_SPEC.md · TASKS.md
│
├── src/
│   ├── data/                   # pipeline: manifest, fixed crop, coil-grouped split
│   ├── features/               # frozen backbone + cached embeddings
│   ├── training/               # train · finetune · anomaly (PaDiM) · diagnose
│   ├── inference/predictor.py  # single inference entry point (PaDiM backend)
│   ├── labeling/               # SQLite label store + ingestion (human-assist loop)
│   ├── retraining/             # trigger · retrain · compare · promote (gated)
│   └── app/                    # PyQt5 desktop app — Inspect / Label / Retrain tabs
│
├── scripts/
│   ├── validate_crop.py · eda.py            # data QA gates
│   ├── run_inference.py · benchmark_inference.py
│   ├── smoke_test_*.py                      # integration smokes
│   └── test_*.py                            # unit suites (data / ingestion / anomaly / decision)
│
├── data/        # manifest.csv + embedding index (crops/raw/labels.db gitignored)
├── models/      # production card + metrics (padim.pkl gitignored, regenerable)
└── reports/     # EDA, crop validation, confusion matrices, latency
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/verify_env.py

# data pipeline (first setup, in order)
python src/data/build_manifest.py
python scripts/validate_crop.py          # must pass before proceeding
python src/data/preprocess.py
python src/features/build_cache.py

# fit the deployed detector
python -m src.training.anomaly

# desktop app
python src/app/main.py            # or scripts\launch_app.bat on Windows

# CLI inference
python scripts/run_inference.py --image path/to/image.bmp
```

## Key design decisions

- **Fixed crop, not learned ROI** — the coil center is nearly fixed; crop box is locked in `config.yaml`.
- **Frozen backbone → cached embeddings → light head** — the CNN never trains; retraining re-fits only the detector (seconds on CPU).
- **False negatives are the costly error** — when uncertain, route to human review, never auto-pass.
- **Coil-grouped folds** — all images of one physical coil stay in one fold (no train/val leakage); fold 4 is the held-out comparison fold for the retraining approval gate.

See `PROJECT_SPEC.md` for the full architecture and `CLAUDE.md` for working context.
