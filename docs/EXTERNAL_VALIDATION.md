# External Validation Protocol

## Why it matters

Internal test performance does not establish generalization to a different institution, scanner, acquisition protocol, population, or disease-prevalence distribution.

MedVision now includes both a **Kaggle execution notebook** and a lightweight evaluator for independent prediction arrays. The notebook is designed to keep the large medical-image dataset in Kaggle rather than requiring a local download.

## Current status

**Execution notebook: implemented. External evaluation completed.** The canonical result is preserved under `reports/external_validation/kermany/`.

The repository intentionally does not contain the external medical images. The evaluated checkpoint is also not committed to the repository. The Kaggle notebook expects the checkpoint to be attached as a Kaggle Input and evaluates the external cohort without retraining.

## Recommended experiment: Kermany / Guangzhou cohort

The notebook at `notebooks/external_validation_kermany.ipynb` evaluates the complete available Kermany cohort attached to the Kaggle notebook.

The source cohort contains two labels: `NORMAL` and `PNEUMONIA`. Therefore this experiment validates **only MedVision's Pneumonia output**. It does not validate the other seven MedVision labels.

### Frozen-model protocol

1. Attach `paultimothymooney/chest-xray-pneumonia` to the Kaggle Notebook.
2. Attach the exact MedVision checkpoint used by the API, preferably `densenet121_weighted_bce.pt`.
3. Run the notebook on the complete available cohort.
4. Audit image validity before inference.
5. Load the checkpoint with `pretrained=False` and the stored state dict.
6. Do not retrain on the external images.
7. Do not optimize the classification threshold on the external labels.
8. Generate Pneumonia probabilities for every valid image.
9. Report ROC-AUC and PR-AUC as the primary threshold-independent metrics.
10. Report sensitivity, specificity, precision, F1 and the confusion matrix at the fixed 0.5 threshold as secondary operating-point metrics.
11. Save aggregate metrics, predictions, quality audit and figures as small output artifacts.

The notebook deliberately does **not** upload or copy the external X-rays into the repository.

## Data provenance

The Kermany pediatric pneumonia cohort is associated with Guangzhou Women and Children's Medical Center. Because it is a binary pediatric pneumonia dataset, it introduces a meaningful population/domain shift relative to a general multi-label chest-X-ray model. That shift should be reported as a limitation, not hidden.


## Completed Kermany / Guangzhou evaluation

The frozen DenseNet121 + Weighted BCE checkpoint was evaluated on all **5,856 images** discovered under the attached Kermany cohort's train, test and validation directories.

| Metric | Result |
|---|---:|
| Images evaluated | **5,856** |
| Normal / Pneumonia | **1,583 / 4,273** |
| Quality exclusions | **0** |
| ROC-AUC | **0.3944** |
| Average Precision | **0.6696** |
| Accuracy @ 0.5 | **72.90%** |
| Precision @ 0.5 | **72.95%** |
| Sensitivity @ 0.5 | **99.91%** |
| Specificity @ 0.5 | **0.00%** |
| F1 @ 0.5 | **84.33%** |
| Mean Pneumonia probability | **0.9938** |
| Median Pneumonia probability | **0.9980** |

Confusion matrix at the fixed threshold:

~~~text
                 Predicted
              Normal  Pneumonia
Actual Normal      0       1583
       Pneumonia   4       4269
~~~

The checkpoint therefore shows **substantial external-domain failure** on this pediatric cohort. It correctly identifies almost every Pneumonia image but labels every Normal image as Pneumonia at the fixed threshold. The mean and median Pneumonia probabilities are both near 1.0, indicating severe external overconfidence.

The result is retained as a negative/generalization finding rather than hidden or threshold-tuned away.

### Descriptive uncertainty analysis

A separate 2,000-replicate IID bootstrap produced the following percentile 95% intervals:

| Metric | Estimate | 95% bootstrap interval |
|---|---:|---:|
| ROC-AUC | 0.3944 | 0.3777–0.4099 |
| Average Precision | 0.6696 | 0.6540–0.6855 |
| Sensitivity @ 0.5 | 0.9991 | 0.9979–0.9998 |
| Specificity @ 0.5 | 0.0000 | 0.0000–0.0000 |
| F1 @ 0.5 | 0.8433 | 0.8357–0.8504 |

This analysis was performed after the primary frozen evaluation and was not used to choose a threshold.

## Local evaluator

```bash
python -m src.evaluation.external_validation \\
  --predictions artifacts/external_predictions.npy \\
  --labels artifacts/external_labels.npy \\
  --dataset-name "External Dataset Name" \\
  --output reports/external_validation.json
```

Expected array shape: `(N, 8)` with label order: `Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax`.

## What not to do

- report an external result from a dataset used for model development;
- tune thresholds on the external set;
- silently change the label mapping;
- treat `NORMAL` as absence of all seven other thoracic findings;
- compare metrics without documenting prevalence and cohort differences;
- describe external validation as clinical validation.

## Acceptance criteria

- dataset name/version;
- access/provenance statement;
- inclusion/exclusion criteria;
- sample size;
- label mapping;
- preprocessing;
- frozen model/checkpoint identifier;
- ROC-AUC and PR-AUC;
- sensitivity/specificity/F1 at a pre-specified threshold;
- confusion matrix;
- confidence/reliability analysis;
- limitations and domain-shift observations.

## Output contract

```text
external_validation.json
external_predictions.csv
external_quality_audit.csv
external_pneumonia_probabilities.npy
external_pneumonia_labels.npy
external_roc_curve.png
external_pr_curve.png
external_confusion_matrix.png
external_reliability.png
POSTHOC_ANALYSIS.md
posthoc_analysis.json
threshold_sensitivity.csv
prediction_summary.csv
MANIFEST.json
SHA256SUMS.txt
```

These are intentionally small enough to transfer back into the MedVision repository without copying the underlying X-ray corpus.