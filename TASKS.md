# TASKS.md — Coil Defect Inspection: Development Task Breakdown

**Lead Developer reference:** read alongside `PROJECT_SPEC.md`.  
**Go-live bar:** ≥ 95% defect recall (Dent + Loose combined) on 5-fold CV.  
**Target platform:** Windows PC, CPU-only, Python 3.11, Desktop GUI app.  
**Stack:** PyTorch + timm · scikit-learn · Pillow/OpenCV · MLflow · Tkinter or PyQt5 · SQLite.

---

## Dependency Map (read before starting any phase)

```
Phase 0 (Setup)
  └─► Phase 1 (Data Validation)
        └─► Phase 2 (Feature Extraction)
              └─► Phase 3 (Model Training)
                    └─► Phase 4 (Inference Engine)
                          ├─► Phase 5 (Desktop App — Inference tab)
                          └─► Phase 6 (Labeling Tool tab)
                                └─► Phase 7 (Retraining Pipeline)
                                      └─► Phase 8 (Monitoring & Drift)

Phase 0 also feeds: Phase 8 (MLflow setup used from first training run onward)
```

---

## Phase 0 — Project Setup
*Estimated effort: 0.5 day*

### T0.1 — Repository & directory structure
Create the canonical folder layout for the project:

```
MyProject/
├── data/
│   ├── raw/                    # symlink or copy of Coil-image-Dataset
│   ├── crops/                  # preprocessed coil crops (output of T1.3)
│   ├── embeddings/             # cached .npy embedding files (output of T2.2)
│   └── labels.db               # SQLite label store (output of T6.2)
├── models/
│   ├── production/             # active production model
│   └── archive/                # versioned past models
├── src/
│   ├── data/                   # preprocessing, crop, dataset manifest
│   ├── features/               # backbone extractor, embedding cache
│   ├── training/               # classifier training, CV, threshold tuning
│   ├── inference/              # inference wrapper, decision logic
│   ├── labeling/               # label storage, ingestion helpers
│   ├── retraining/             # retrain trigger, pipeline, comparison
│   └── app/                    # desktop GUI (all tabs)
├── experiments/                # MLflow tracking directory
├── reports/                    # confusion matrices, evaluation CSVs
├── PROJECT_SPEC.md
├── TASKS.md
├── requirements.txt
└── config.yaml                 # thresholds, paths, trigger counts — all configurable
```

**Deliverable:** empty directory tree committed to git, `config.yaml` with placeholder values.

---

### T0.2 — Environment & dependency lockfile
- Create `requirements.txt` pinning: `torch`, `torchvision`, `timm`, `scikit-learn`, `Pillow`, `opencv-python`, `numpy`, `pandas`, `mlflow`, `PyYAML`, `pyqt5` (or `tk` is built-in).
- Verify install on the target Windows machine.
- Add a `setup.py install --check` or a `scripts/verify_env.py` that imports each dep and prints versions.

**Deliverable:** `requirements.txt` + `scripts/verify_env.py` that passes cleanly.

---

### T0.3 — Dataset manifest CSV
Build `data/manifest.csv` with one row per image:

| column | description |
|--------|-------------|
| `filepath` | absolute path to raw BMP |
| `label` | `Pass`, `Dent`, or `Loose` |
| `session_id` | timestamp prefix from filename (e.g. `250825_152739`) |
| `position` | position number from filename |
| `layer` | layer number from filename |
| `split` | `train` or `test` (assigned in T1.4) |

Parse filenames with a regex; derive labels from parent folder name.  
**Deliverable:** `data/manifest.csv`, a script `src/data/build_manifest.py`, printed class counts.

---

## Phase 1 — Data Validation & Preprocessing
*Estimated effort: 1.5 days*

### T1.1 — Crop box validation
**Goal:** confirm the fixed crop box `x ∈ [300, 2000], y ∈ [620, 1320]` (from spec §5) contains the full coil + margin across all 817 images, not just the three sample images it was measured on.

- Write `scripts/validate_crop.py`:
  - Load a random sample of 30–50 images across all three classes.
  - Apply copper-color segmentation (HSV threshold on the coil's warm copper hue) to find the coil bounding box in each.
  - Assert the bounding box fits inside the crop window with ≥ 40 px margin on every side.
  - Save a montage of 9 random crops to `reports/crop_validation.png` for visual inspection.
- If any image fails the assertion: record it, adjust the crop window, re-run.
- Lock the final crop coordinates into `config.yaml`.

**Deliverable:** `reports/crop_validation.png`, confirmed crop coords in `config.yaml`.  
**Blocks:** everything downstream (crops must be correct before embedding).

---

### T1.2 — EDA (Exploratory Data Analysis)
**Goal:** understand the data distribution before modeling — catch outliers, brightness variance, and any label anomalies.

- Write `scripts/eda.py` producing:
  - Per-class image count (confirm against §2 table).
  - Per-class brightness mean & std histogram (exposure varies — this quantifies how much).
  - 5 random thumbnails per class saved to `reports/eda_samples_{class}.png`.
  - Per-class filename timestamp distribution (are sessions balanced across classes?).
  - List of any images with unusual stats (too dark, too bright, wrong size).

**Deliverable:** `reports/eda_report.md` with plots, list of flagged images.

---

### T1.3 — Preprocessing pipeline
Write `src/data/preprocess.py`:
1. Load raw BMP.
2. Apply fixed crop (coords from T1.1).
3. Resize to `448 × 192` px (preserves wide coil aspect ratio; ≥384 px on long axis as spec requires).
4. Convert to float32, normalize with ImageNet mean/std (for ImageNet-pretrained backbone).
5. Save cropped + normalized tensor to `data/crops/{label}/{original_stem}.pt` (or `.npy`).

- Process all 817 images; log any failures.
- Confirm output dimensions are consistent.

**Deliverable:** `data/crops/` populated, `src/data/preprocess.py`.

---

### T1.4 — Train / test split assignment
**Strategy:** CV-only (no frozen holdout per user decision). BUT we still need a stable evaluation set for the retraining approval gate (§7 of the spec). Resolution: use the **last fold** of the 5-fold split as a quasi-test set for model comparison — same images every time, never seen during the training folds.

- Write `src/data/split.py`:
  - Stratified 5-fold split on `manifest.csv`, stratified by `label`.
  - Assign fold index (0–4) to each row; add `fold` column to manifest.
  - Fold 4 = "comparison fold" (used by the retraining approval gate in T7.3).
  - Folds 0–3 = training / validation rotation.

**Deliverable:** `data/manifest.csv` updated with `fold` column.

---

## Phase 2 — Feature Extraction
*Estimated effort: 1 day*

### T2.1 — Frozen backbone extractor
Write `src/features/extractor.py`:
- Load one of: `resnet18`, `efficientnet_b0`, `mobilenetv3_small_100` from `timm` with pretrained ImageNet weights.
- Set to `eval()` mode; freeze all parameters.
- Forward pass: crop tensor → global average pooling → 1-D embedding vector.
- Output embedding dimension: ResNet18=512, EfficientNet-B0=1280, MobileNetV3=576.

The backbone is a **hyperparameter** — make it selectable via `config.yaml` so swapping is one line.

**Deliverable:** `src/features/extractor.py` with a `extract_embedding(image_tensor) -> np.ndarray` function.

---

### T2.2 — Embedding cache
Write `src/features/build_cache.py`:
- Iterate all preprocessed crops in `data/crops/`.
- Extract embedding for each; save to `data/embeddings/{label}/{stem}.npy`.
- Also write `data/embeddings/index.csv`: `stem, label, fold, embedding_path`.
- Log total time (CPU benchmark — should be fast; ResNet18 forward on a 448×192 crop is ~10ms on CPU).

**Deliverable:** `data/embeddings/` fully populated, `index.csv`.

---

### T2.3 — Augmentation pipeline
Write `src/data/augment.py` using `torchvision.transforms`:
- `ColorJitter(brightness=0.3, contrast=0.3)` — handles exposure variation.
- `RandomHorizontalFlip(p=0.5)` + `RandomVerticalFlip(p=0.5)`.
- `RandomAffine(degrees=5, translate=(0.03, 0.03))` — ±5° rotation, ±50 px translate.
- **Do not add:** heavy zoom, elastic distortion, or crop (spec §6).

Augmentation is applied **before** embedding extraction (not at inference time). Each augmented variant is a new image with its own cached embedding.

**Deliverable:** `src/data/augment.py`, a smoke-test script showing before/after side-by-side.

---

### T2.4 — Augmented embedding generation for minority classes
Apply the augmentation pipeline to **Dent** and **Loose** only (not Pass — the majority class doesn't need it):
- Generate N augmented versions per minority image. Start with N=5 (bringing Dent to ~370 effective samples). Tune N in T3.6 if needed.
- Cache augmented embeddings alongside originals; mark `is_augmented=True` in index.csv.
- Augmented images are **only used in training folds**, never in the evaluation fold (fold 4).

**Deliverable:** Augmented embeddings in cache, `index.csv` updated.

---

## Phase 3 — Model Training & Evaluation
*Estimated effort: 2 days*

### T3.1 — Baseline classifier
Write `src/training/train.py`:
- Load embeddings from cache via `index.csv`.
- Fit `LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0)` on folds 0–3 (training folds).
- Use stratified 4-fold CV within those four folds to select `C` via grid search.
- Save trained model to `models/production/classifier.pkl` + metadata JSON.

**Deliverable:** trained `classifier.pkl`, training script.

---

### T3.2 — Evaluation reporting
Write `src/training/evaluate.py`:
- Run 5-fold stratified CV (folds 0–4), report per-fold and mean metrics.
- **Primary:** per-class recall for Dent, Loose, and the combined Fail recall.
- **Also report:** precision per class, F1, full 3×3 confusion matrix, binary Pass/Fail confusion matrix.
- Save confusion matrix heatmap PNG to `reports/confusion_matrix.png`.
- Save all metrics to `reports/metrics.csv`.
- **Pass/fail check:** print clearly whether ≥95% Fail recall is achieved.

**Deliverable:** `reports/confusion_matrix.png`, `reports/metrics.csv`, clear pass/fail verdict printed.

---

### T3.3 — Threshold tuning & review band
Write `src/training/threshold.py`:
- From 5-fold CV, collect predicted probabilities for every image.
- Plot precision–recall curve for the binary Pass/Fail decision.
- Find the probability threshold that achieves ≥95% Fail recall; record resulting false-reject rate.
- Define the **review band**: images with `P(Fail) ∈ [threshold − δ, threshold + δ]` are routed to human review. Start with δ=0.05; tune to keep review queue manageable.
- Save final thresholds to `config.yaml` under `decision.pass_threshold`, `decision.review_band_delta`.

**Deliverable:** `reports/precision_recall_curve.png`, thresholds written to `config.yaml`.

---

### T3.4 — MLflow experiment tracking
- Initialize MLflow with `experiments/` as tracking URI.
- Every training run (T3.1 + any iteration in T3.5–T3.6) logs:
  - Parameters: backbone name, classifier type, `C`, augment N, fold config.
  - Metrics: per-class recall, F1, false-reject rate, threshold.
  - Artifacts: `confusion_matrix.png`, `classifier.pkl`, `config.yaml` snapshot.
- Wrap `train.py` and `evaluate.py` in an `mlflow.start_run()` context.

**Deliverable:** MLflow UI accessible at `localhost:5000` via `mlflow ui --backend-store-uri experiments/`.

---

### T3.5 — Backbone comparison (if baseline passes ≥95%)
If the baseline hits the bar, skip this and move on. If not:
- Swap backbone in `config.yaml`; re-run T2.1–T2.2 for the new backbone; re-run T3.1–T3.3.
- Compare: ResNet18 vs EfficientNet-B0 vs MobileNetV3.
- Log all runs to MLflow; pick the best by Fail recall.

---

### T3.6 — Classifier comparison (if backbone swap doesn't fix it)
- Try: `SVC(kernel='rbf', class_weight='balanced')`, `GradientBoostingClassifier`, small 2-layer MLP.
- Also tune augmentation N (try N=8, N=10 for Dent).
- Keep logging to MLflow.

---

### T3.7 — Model serialization
Once a model hits ≥95% Fail recall:
- Save `models/production/classifier.pkl`, `models/production/backbone_name.txt`, `models/production/config_snapshot.yaml`, `models/production/thresholds.json`.
- Write `models/production/model_card.md`: backbone, classifier, CV metrics, threshold, date, training image count.

**Deliverable:** `models/production/` fully populated, `model_card.md`.

---

## Phase 4 — Inference Engine
*Estimated effort: 0.5 day*

### T4.1 — Inference wrapper
Write `src/inference/predictor.py` — the single entry point for all inference:

```python
predictor = Predictor(config_path="config.yaml")
result = predictor.predict(image_path)
# result = {
#   "label": "Dent",           # 3-class prediction
#   "pass_fail": "Fail",       # binary gate
#   "decision": "Review",      # Pass / Fail / Review
#   "probabilities": {"Pass": 0.12, "Dent": 0.71, "Loose": 0.17},
#   "latency_ms": 83.4
# }
```

Internals: load image → fixed crop → resize → normalize → backbone forward → classifier predict_proba → threshold logic.

**Deliverable:** `src/inference/predictor.py`, a `scripts/run_inference.py` CLI wrapper for quick testing.

---

### T4.2 — Latency benchmark
Run `predictor.predict()` on 20 random images on the target CPU; report:
- Mean / p95 / max latency in ms.
- Must be ≤ 1 s/image (spec §13-E). Log result to `reports/latency_benchmark.txt`.

---

## Phase 5 — Desktop App (Inference Tab)
*Estimated effort: 2 days*

Single window with **three tabs**: Inference (Phase 5), Labeling (Phase 6), Retraining (Phase 7). Use PyQt5 (richer widgets than Tkinter; easier to add plots later).

### T5.1 — App skeleton & navigation
- Main `QMainWindow` with `QTabWidget` (Inspect / Label / Retrain).
- Status bar showing model version + last-loaded image.
- Menu: File → Open Image, File → Open Folder, Help → Model Card.

**Deliverable:** launches, tabs switch, no crashes.

---

### T5.2 — Single image inference UI
- **Left panel:** image viewer (shows the full raw image, with the crop box drawn as an overlay rectangle).
- **Right panel:**
  - Large verdict badge: green `PASS`, red `FAIL`, yellow `REVIEW`.
  - Probability bars for Pass / Dent / Loose (three horizontal `QProgressBar` elements, color-coded).
  - Prediction latency shown.
- Operator clicks "Open Image" → file picker → result renders within 1 s.

---

### T5.3 — Batch folder inference UI
- "Open Folder" → processes all BMP files in the folder; shows a scrollable results table (filename, verdict, probabilities).
- Progress bar during batch processing.
- "Export to CSV" button saves the table.

---

### T5.4 — Results export
- Export button saves: `filepath, label, pass_fail, decision, P_Pass, P_Dent, P_Loose, latency_ms, timestamp`.
- CSV format, UTF-8.

---

## Phase 6 — Labeling Tool (Label Tab)
*Estimated effort: 1.5 days*

### T6.1 — Label storage (SQLite)
Write `src/labeling/label_store.py`:
- SQLite DB at `data/labels.db`.
- Table `labels`: `id, filepath, model_prediction, model_confidence, human_label, operator_id, labeled_at, session_id`.
- Methods: `insert_pending()`, `confirm_label()`, `get_unlabeled_count()`, `export_to_csv()`.

---

### T6.2 — Labeling UI (Label tab)
Flow: operator points to a folder of new unlabeled images → app queues them → shows one at a time with model prediction pre-filled.

**Layout:**
- Image viewer (cropped coil shown large).
- Model prediction + confidence shown prominently.
- Three buttons: `✓ Pass`, `✓ Dent`, `✓ Loose` (pre-selects to model's prediction; operator can override).
- "Skip" button (defers image, doesn't label).
- Progress counter: "12 / 47 labeled today".
- Keyboard shortcuts: `P` = Pass, `D` = Dent, `L` = Loose, `→` = next.

On confirm: write to `labels.db`, move image to `data/raw/{label}/` (or symlink), update manifest.

---

### T6.3 — Unlabeled image ingestion
- "Load Folder" button opens a folder picker.
- Scans folder for `.bmp` files not already in `labels.db`.
- Queues them for labeling in the Label tab.
- Also supports a **watched folder** (configurable path in `config.yaml` under `ingestion.watch_folder`): app polls on startup for any new files dropped there. This supports the case where camera software auto-drops files.

---

## Phase 7 — Retraining Pipeline (Retrain Tab)
*Estimated effort: 2 days*

### T7.1 — Count-based trigger
Write `src/retraining/trigger.py`:
- Reads `labels.db`, counts confirmed labels since last retrain.
- Threshold from `config.yaml` (`retraining.trigger_count`, default: 50).
- If count ≥ threshold: show a notification banner in the app ("50 new labels available — retrain recommended").
- Also: "Retrain Now" button available at all times (manual trigger).

---

### T7.2 — Retrain script
Write `src/retraining/retrain.py`, callable from the UI or CLI:
1. Pull all confirmed labels from `labels.db` + original 817 images.
2. Re-run preprocessing (T1.3) on any new images.
3. Re-extract embeddings (T2.1–T2.2) for new images.
4. Re-augment minority classes (T2.4) if class counts have changed.
5. Rebuild manifest + fold assignments (T0.3, T1.4).
6. Train classifier (T3.1) and evaluate (T3.2–T3.3).
7. Log entire run to MLflow.
8. Output: candidate model in `models/candidate/`.

---

### T7.3 — Model comparison report
Write `src/retraining/compare.py`:
- Evaluate **current production model** and **candidate model** both on fold 4 (the comparison fold, locked since T1.4).
- Generate a side-by-side comparison table: per-class recall, false-reject rate, confusion matrices.
- Save to `reports/model_comparison_{timestamp}.md`.
- Show in the Retrain tab as a formatted table.

---

### T7.4 — Human approval gate (Retrain tab UI)
In the Retrain tab:
- "Retrain" button triggers T7.2 (async, shows progress bar).
- After completion: shows comparison report inline.
- Two large buttons: `Approve & Deploy` (promotes candidate to production) and `Reject` (deletes candidate, keeps current).
- On approval: copies `models/candidate/` → `models/production/`, archives old production model to `models/archive/{timestamp}/`, updates `model_card.md`, shows toast notification.

---

### T7.5 — Model versioning
- Each model version stored in `models/archive/{YYYYMMDD_HHMMSS}/` with its full snapshot (pkl, config, card).
- `models/production/` always contains the active model.
- The Retrain tab shows a version history list with metrics for each past version.
- "Rollback to version X" button restores an archived model to production.

---

## Phase 8 — Monitoring & Drift Detection
*Estimated effort: 1 day*

### T8.1 — Prediction logging
Every inference call appended to `data/prediction_log.csv`:
`timestamp, filepath, label, pass_fail, decision, P_Pass, P_Dent, P_Loose, latency_ms, model_version`.

---

### T8.2 — Basic drift detection
Write `src/monitoring/drift.py`:
- Weekly (or on-demand): compare brightness distribution of recent 100 images vs training set distribution.
- Use KS-test on per-image mean brightness.
- If p < 0.05: show a warning banner in the app ("Image distribution may have shifted — check camera/lighting").
- Log drift check result to MLflow.

---

### T8.3 — Dashboard (stretch goal)
If time allows: a simple "Stats" panel in the app showing:
- Rolling pass/fail rate over the last N images.
- Per-class prediction frequency.
- Average confidence score trend.

---

## Phase 9 — Packaging & Documentation
*Estimated effort: 0.5 day*

### T9.1 — Install script
- `scripts/install.bat`: creates a virtualenv, installs `requirements.txt`, verifies env.
- `scripts/launch_app.bat`: activates env, runs `python src/app/main.py`.

### T9.2 — User guide
`docs/USER_GUIDE.md` covering:
- How to inspect a single image / batch.
- How to label new images.
- How to trigger and approve a retrain.
- How to roll back a model.

### T9.3 — Model card (auto-generated at each retrain)
Template in `src/retraining/model_card_template.md`; filled in by `retrain.py` with actual metrics.

---

## Effort Summary

| Phase | Description | Est. Days |
|-------|-------------|-----------|
| 0 | Setup | 0.5 |
| 1 | Data Validation & Preprocessing | 1.5 |
| 2 | Feature Extraction | 1.0 |
| 3 | Model Training & Evaluation | 2.0 |
| 4 | Inference Engine | 0.5 |
| 5 | Desktop App (Inference tab) | 2.0 |
| 6 | Labeling Tool (Label tab) | 1.5 |
| 7 | Retraining Pipeline (Retrain tab) | 2.0 |
| 8 | Monitoring & Drift | 1.0 |
| 9 | Packaging & Docs | 0.5 |
| **Total** | | **~12.5 days** |

---

## Build Order (sequenced by dependency)

```
T0.1 → T0.2 → T0.3
T0.3 → T1.1 (crop validation) ← must pass before anything downstream
T1.1 → T1.2 (EDA, parallel with T1.3)
T1.1 → T1.3 → T1.4
T1.4 → T2.1 → T2.2 → T2.3 → T2.4
T2.4 → T3.1 → T3.2 → T3.3 → T3.4
       T3.5, T3.6 only if T3.2 fails the 95% recall bar
T3.3 → T3.7 (serialize winner)
T3.7 → T4.1 → T4.2
T4.1 → T5.1 → T5.2 → T5.3 → T5.4
T4.1 → T6.1 → T6.2 → T6.3
T6.1 → T7.1 → T7.2 → T7.3 → T7.4 → T7.5
T7.2 → T8.1 → T8.2
T5+T6+T7 → T9.1 → T9.2 → T9.3
```

---

## Open Decisions (confirm before implementation begins)

| # | Item | Current default |
|---|------|-----------------|
| A | Retraining trigger count | 50 confirmed labels |
| B | New image ingestion | Manual folder browse + optional watch folder in config |
| C | Heatmap (Grad-CAM) — go-live requirement or stretch? | Stretch goal only |
| D | Session grouping in CV (can one session have mixed Pass/Fail)? | Treat each image independently |
| E | Operator ID tracking in labeling tool? | Optional field, not required |
