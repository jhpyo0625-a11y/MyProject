# Model Card -- Coil Defect Inspection v2 (fine-tuned)

**Approach:** End-to-end fine-tuning (EfficientNet-B0, last 3 blocks)
**Input:** 224x96 px (resized from 448x192 crops)
**MLflow run:** `383c5d955939482ab0a7143a101fde5b`

## CV Performance (OOF, folds 0-3)

| Metric | Value |
|--------|-------|
| Fail recall (target >=95%) | **91.2%** |
| Fail recall @ tuned threshold | 95.2% |
| False-reject @ tuned threshold | 93.3% |
| Pass/Fail threshold | 0.4290 |
| Review band | +/-0.05 |

## Fold 4 Independent Check

| Metric | Value |
|--------|-------|
| Fail recall | 64.9% |
| False-reject | 31.7% |

## Verdict: FAIL -- see next steps
