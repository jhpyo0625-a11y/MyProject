# Model Card -- Coil Defect Inspection (PaDiM, human-assist)

**Approach:** Unsupervised anomaly detection (PaDiM). Trains on Pass coils
ONLY -- no defect labels needed. Backbone: frozen ImageNet resnet18,
layers (1, 2) (fine patch grid). Image score = MAX patch Mahalanobis.

**Why human-assist, not auto-pass:** Four approaches across supervised and
anomaly paradigms all hit ~50-55% Fail-recall @ 20% false-reject -- the
dataset (74 Dent / 110 Loose) cannot support the >=95% auto-recall target.
This detector is deployed as a triage/ranking aid; humans confirm; each
confirmation grows the defect set.

## Decision policy (3-band)

| Band | Rule | Action |
|---|---|---|
| AUTO-FLAG | score >= 73.014 | clear defect -- human confirms before scrap |
| REVIEW    | 23.379 <= score < 73.014 | human looks |
| AUTO-PASS | score < 23.379 | clearly normal -- offloads easy bulk |

Thresholds derived from OOF (folds 0-3); defects only used to place bands.

## Workload (held-out fold 4)

- AUTO-PASS 18% / REVIEW 77% / AUTO-FLAG 5%
- **Defects auto-passed (critical, must stay ~0): 0.0%**
- Defects auto-flagged: 19%

NOT a >=95% auto-recall gate. Honest CV Fail-recall @20% FR ~= 55%.
