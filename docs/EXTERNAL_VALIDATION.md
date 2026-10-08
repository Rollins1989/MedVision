# External Validation Protocol

## Why it matters

Internal test performance does not establish generalization to a different institution, scanner, acquisition protocol, population, or disease-prevalence distribution.

MedVision now includes both a **Kaggle execution notebook** and a lightweight evaluator for independent prediction arrays. The notebook is designed to keep the large medical-image dataset in Kaggle rather than requiring a local download.

## Current status

**Execution notebook: implemented. External metric results: pending execution.**

The repository intentionally does not contain the external medical images or a trained checkpoint. The Kaggle notebook expects the checkpoint to be attached as a Kaggle Input and evaluates the external cohort without retraining.

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
MANIFEST.json
```

These are intentionally small enough to transfer back into the MedVision repository without copying the underlying X-ray corpus.