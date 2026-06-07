# Model Card -- Coil Defect Inspection v3 (fine-tuned, honest eval)

**Approach:** End-to-end fine-tuning (EfficientNet-B0, last 3 blocks)
**Input:** 448x224 px  |  **Crop:** full coil (no arc clipping)
**Epoch selection:** macro-F1 (not raw recall -- avoids degenerate states)
**MLflow run:** `bad0f877f53a4a139716dc348eafa4e7`

## CV operating points (OOF, folds 0-3)

Mean macro-F1: **0.516**

| FR budget | Fail recall | actual FR |
|---|---|---|
| <= 5% | 29.3% | 5% |
| <= 10% | 37.4% | 10% |
| <= 15% | 46.9% | 15% |
| <= 20% | 53.7% | 20% |
| <= 30% | 62.6% | 30% |

Deployed operating point: threshold 0.5542 -> Fail recall
**53.7%** at 20% false-reject (budget <= 20%).

## Fold 4 held-out check

macro-F1: 0.632

| FR budget | Fail recall | actual FR |
|---|---|---|
| <= 5% | 59.5% | 5% |
| <= 10% | 70.3% | 10% |
| <= 15% | 81.1% | 14% |
| <= 20% | 86.5% | 20% |
| <= 30% | 91.9% | 29% |

## Verdict: BELOW TARGET -- see next steps (target 95% recall @ <= 20% FR)
