# Kermany / Guangzhou External Validation Artifacts

This directory contains the published artifacts from MedVision's frozen-checkpoint external evaluation on the Kermany pediatric chest X-ray cohort.

## Experiment snapshot

- Checkpoint: densenet121_weighted_bce.pt
- Architecture: DenseNet121
- Images evaluated: 5,856
- Pneumonia: 4,273
- Normal: 1,583
- Quality exclusions: 0
- Threshold: 0.5, fixed before evaluation
- Retraining: No
- External threshold optimization: No

### Primary result

| Metric | Result |
|---|---:|
| ROC-AUC | **0.3944** |
| Average Precision | **0.6696** |
| Sensitivity @ 0.5 | **99.91%** |
| Specificity @ 0.5 | **0.00%** |
| F1 @ 0.5 | **84.33%** |

The result demonstrates substantial external-domain failure: all 1,583 Normal images were classified as Pneumonia at the fixed 0.5 threshold.

## Published files

- external_validation.json — canonical metrics and protocol metadata.
- external_*.svg — repository-rendered ROC, PR, confusion-matrix and reliability figures.
- POSTHOC_ANALYSIS.md / posthoc_analysis.json — bootstrap uncertainty and descriptive threshold analysis.
- threshold_sensitivity.csv — numerical threshold sweep; not used to select a threshold.
- prediction_summary.csv — class-wise probability distribution summary.
- MANIFEST.json — source and artifact manifest.
- SHA256SUMS.txt — checksums of the raw artifacts produced by the Kaggle run.

The full per-image predictions, quality audit and NumPy arrays remain in the downloadable Kaggle result package supplied with this experiment; the medical images themselves are not included in GitHub.

For the full protocol and limitations, see [External Validation](../../../docs/EXTERNAL_VALIDATION.md).
