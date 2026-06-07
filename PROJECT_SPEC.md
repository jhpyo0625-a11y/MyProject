# PROJECT_SPEC.md — Coil Winding Defect Inspection

**Goal:** Classify images of PCB-mounted induction-antenna coils as **Pass / Dent / Loose**, with a derived **Pass / Fail** gate, running on **CPU-only** hardware and tuned so that a missed defect (false negative) is the error we most aggressively avoid.

This spec reflects decisions made so far plus findings from a hands-on analysis of the raw 2448×2048 images. Items still needing your confirmation are collected in **§13 Open Items** — none of them block starting the build.

---

## 1. Problem Definition

- **Primary output:** 3-class label — `Pass`, `Dent`, `Loose`.
- **Derived output:** binary `Pass` / `Fail`, where `Fail = Dent OR Loose`. This gives the line a simple gate while preserving the defect type for analysis and operator feedback.
- **Defect localization:** not required for v1. A Grad-CAM-style heatmap is a stretch goal (see §9). Bounding-box localization is out of scope.
- **Decision policy:** when the model is uncertain, **flag for human review** rather than auto-pass. "When in doubt, review" beats "when in doubt, pass," because a shipped defect is the costliest error.

---

## 2. Data

| Class | Count* | Notes |
|-------|--------|-------|
| Pass  | ~633   | majority class |
| Dent  | 74     | minority — drives class-imbalance handling |
| Loose | 110    | minority |
| **Total** | **817** | |

\* From earlier folder counts — **confirm exact numbers** before training.

**Image properties (verified from raw samples):**
- Resolution **2448 × 2048**, RGB, BMP, ~15 MB each.
- Scene: an oval copper coil wound around a central SMD component, on an orange PCB substrate with gold traces/pads, gray inspection-table surround. Partial neighbor assemblies are visible top/bottom.
- **Brightness/exposure varies noticeably between images** — must be handled by normalization + augmentation (§5–6).

**Filename metadata:** filenames encode `timestamp_sensor_position-layer` (e.g. `250825_152739_A35W_10-2`). This is structured provenance but carries **no class label** — labels come from the folder the image lives in, and new images arrive unlabeled (§10).

---

## 3. Key Finding — Coil Position Is Fixed

A direct measurement on the Pass/Dent/Loose raw samples (copper-color segmentation + centroid + a blended overlay of all three) showed:

- Coil **center varies by only 43 px (1.8%) horizontally and 8 px (0.4%) vertically**.
- Coil **width is constant** (1510–1531 px); scale does not change.
- In a stacked overlay the coil winding stays sharp; only the surrounding PCB ghosts.

**Consequence:** the ROI stage is a **fixed crop**, not a learned/geometric detector. This is the simplest and most robust choice and removes an entire failure mode. *(Validate the crop box on a larger sample before locking it — three images confirm the principle, not the exact pixel bounds.)*

---

## 4. Pipeline Architecture (CPU-optimized)

```
raw 2448×2048 image
      │
  [Stage 0]  Fixed-crop ROI  → coil + small margin
      │
  [Stage 1]  Frozen pretrained CNN backbone  → feature embedding   (cached to disk)
      │
  [Stage 2]  Lightweight classifier  → {Pass, Dent, Loose} + probabilities
      │
  Threshold / decision logic  → Pass / Fail / Review
```

Rationale: with ~800 images and no GPU, fine-tuning a large CNN is the wrong tool. Using a **frozen** backbone as a feature extractor means the expensive forward pass happens **once per image and is cached**; the classifier on top trains in seconds and retrains just as fast when new data arrives. This is fast, data-efficient, and CPU-friendly.

---

## 5. Preprocessing

- **Fixed crop:** centered on the coil (~`(1135, 970)` in full-res coordinates), generous enough to absorb the ~±45 px jitter and include a small margin of surrounding PCB. **Starting box:** `x ∈ [300, 2000]`, `y ∈ [620, 1320]` (≈1700×700 px). Confirm/tune on more images.
- **Resize:** to a fixed input that **preserves the coil's wide aspect ratio and keeps winding detail** (e.g. ~448×192, or pad-to-square at ≥384 px). **Do not** downsample below ~384 px on the long axis — the Loose signal is fine winding spacing and will blur away.
- **Color:** keep **RGB**. Color cleanly separates the copper coil from background and costs nothing; even though the main Loose/Dent cues turn out to be geometric/textural, there's no reason to discard color.
- **Normalize** with the backbone's expected mean/std (ImageNet stats if using an ImageNet-pretrained model). Normalization also absorbs the per-image brightness variation.

---

## 6. Model

**Stage 1 — backbone (frozen):** a small ImageNet-pretrained CNN — `ResNet18`, `EfficientNet-B0`, or `MobileNetV3`. Extract embeddings once and cache them (`.npy`/`.pt`) keyed by filename.

**Stage 2 — classifier (the only trained part):** `LinearSVC` / logistic regression / a small MLP / gradient-boosted trees on the cached embeddings. Start with the simplest (logistic regression or SVM) and only escalate if metrics demand it.

**Class imbalance (Dent = 74):** use `class_weight="balanced"` (or equivalent) in the classifier. Optionally oversample the minority classes' embeddings.

**Augmentation** (applied before feature extraction; mild on purpose — the coil has fixed position/scale/orientation, so unrealistic transforms can mask the very defects we detect):
- brightness/contrast jitter — **important**, exposure varies between shots;
- small translation (±~50 px) to mimic the measured jitter;
- horizontal + vertical flip (defects are position-agnostic);
- small rotation (≤±5°).
- **Avoid** heavy elastic/zoom distortion — it can erase Dent shape and Loose spacing cues.

> Note: with frozen features, augmentation means re-extracting embeddings for augmented copies. Since extraction is one cheap forward pass each, this is fine on CPU; cache them too.

---

## 7. Evaluation

- **Validation scheme:** **stratified k-fold cross-validation** (e.g. 5-fold), not a single split — 74 Dent samples are too few to waste on one holdout. Group by inspection session if a session can contain multiple images of the same coil, to avoid leakage (depends on §13-A).
- **Primary metric:** **recall on the Fail class** (and per-class recall for Dent/Loose) — directly tied to "missing a defect is worst."
- **Report alongside:** precision / false-reject rate at the chosen operating point (so the cost of high recall is visible), full 3-class confusion matrix, and binary Pass/Fail confusion matrix.
- **Operating point:** tune the Pass/Fail probability threshold to hit a target defect recall, then report the resulting false-reject rate. Add an **uncertain → review** band around the threshold.

---

## 8. Class Imbalance — Summary

Handled in three places: balanced class weighting in the classifier, optional minority oversampling, and a recall-first threshold. Track per-class recall every run; Dent is the class to watch.

---

## 9. Explainability

- **Default:** report per-class probabilities + the Pass/Fail/Review decision. Low-cost and often sufficient.
- **Trade-off:** the frozen-features + classifier design makes Grad-CAM heatmaps **indirect** (the classifier isn't the CNN). If operator-visible heatmaps become a hard requirement, switch Stage 1+2 to a **single small CNN fine-tuned end-to-end** — slower to train on CPU but feasible at this data size, and it restores clean Grad-CAM. **Decision needed:** is a heatmap a go-live requirement? (§13-G)

---

## 10. Labeling Workflow (for new, unlabeled images)

New production images have no label. Recommended tooling: a lightweight **"model predicts → human confirms"** UI — the model proposes a class + confidence, a human accepts or corrects, and the confirmed label + image flow into the training pool. This is faster than blind folder-sorting and improves as the model improves. Exact form depends on §13-B/C.

---

## 11. Retraining

- **Trigger:** accumulate confirmed labels until a **count threshold** is reached (since labeling is manual), plus a **manual "retrain now"** button. Not time-scheduled on auto-labeled data.
- **Human approval gate:** **required** — a new model is evaluated against the current one on a held-out set and is only promoted after a human approves. No silent auto-deploy.
- **Location:** wherever training runs today; can move to a server later (§13-J).

---

## 12. Deployment & Stack (defaults — confirm in §13)

- **Inference target:** line-side **Windows PC**, CPU. Small backbone forward pass on the cropped coil runs in tens of milliseconds; total well under 1 s/image.
- **Consumption:** desktop app or a local REST endpoint (default).
- **Connectivity:** assume **on-prem / offline-capable** (typical for a line).
- **Packaging:** plain Python (or a packaged `.exe`); Docker optional, only if you have the infra.
- **Stack:** **PyTorch** + `timm` (backbones) + `scikit-learn` (Stage-2 classifier) + `Pillow`/`OpenCV` (image I/O & crop) + `numpy`. Add **MLflow** for run tracking; **DVC** optional for data/version control.
- **Python:** assume 3.11 on Windows.

---

## 13. Open Items (defaults assumed — confirm or override)

| # | Question | Default assumed |
|---|----------|-----------------|
| A | Can one inspection session contain both good and bad coils? (affects CV grouping) | Treat each image independently; group by session in CV if sessions repeat a coil |
| B | What generates the label — operator visual check or downstream electrical/QC test? | Operator visual check |
| C | How long does labeling take, and who does it? | Build a predict→confirm UI to minimize effort |
| D | Any known mislabels in the current 817? | Assume clean; audit lowest-confidence predictions after first training |
| E | Max acceptable inference latency per image? | ≤1 s |
| F | Minimum accuracy/F1 bar to ship? | Target high defect recall; report achieved precision |
| G | Is an operator-visible heatmap a go-live requirement? | No (use probabilities; revisit per §9) |
| H | How is the model consumed — desktop app / local API / CLI / MES-SCADA? | Desktop app or local API |
| I | Internet-connected or air-gapped? | On-prem / offline-capable |
| J | Where does the retraining pipeline live? | Local training machine |
| K | Docker required? | Optional |
| L | Existing tools (MLflow / DVC / DB)? | None; add MLflow |
| M | Python version + OS of deployment target? | Windows, Python 3.11 |

---

## 14. Suggested Build Order

1. **Crop validation** — overlay 20–30 raw images, confirm the fixed crop box contains the coil with margin in all of them.
2. **Dataset prep** — apply crop + resize to all 817 images; verify class counts.
3. **Feature cache** — run the frozen backbone over the dataset, cache embeddings.
4. **Baseline classifier** — train logistic regression / SVM with balanced weights; 5-fold CV; record per-class recall + confusion matrices.
5. **Threshold tuning** — set the Pass/Fail operating point for target defect recall; define the review band.
6. **Iterate** — try alternate backbones / classifier / augmentation only if the baseline misses the bar.
7. **Packaging** — wrap as the chosen inference target; add the predict→confirm labeling UI.
8. **Retraining loop** — count-based trigger + human approval gate.
