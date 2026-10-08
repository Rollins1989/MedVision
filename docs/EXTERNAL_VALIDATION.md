# External Validation Protocol

## Why it matters

Internal test performance does not establish generalization to a different institution, scanner, acquisition protocol, population, or disease-prevalence distribution.

MedVision now includes an external-validation harness, but **does not bundle a third-party clinical dataset**. This is intentional: external medical datasets can have access, licensing, provenance, and label-definition requirements.

## Current status

**Harness: implemented. External dataset results: not yet reported.**

No external-validation metric should be added to the README until a separately obtained dataset has been evaluated.

## Protocol

1. Obtain an independent dataset with appropriate access/usage rights.
2. Define an explicit label mapping to MedVision's eight labels.
3. Do not use the external dataset to retrain the model.
4. Do not tune thresholds on the external dataset.
5. Generate model probabilities using the frozen MedVision checkpoint.
6. Run `src/evaluation/external_validation.py`.
7. Report macro ROC-AUC, macro PR-AUC, and per-label results.
8. Record dataset provenance, sample count, label mapping, preprocessing, checkpoint SHA, and evaluation date.

## Label contract

The evaluator expects:

```text
Atelectasis
Cardiomegaly
Effusion
Infiltration
Mass
Nodule
Pneumonia
Pneumothorax
```

The external labels must represent the same clinical concepts closely enough for a defensible comparison. Ambiguous mappings should be excluded or explicitly documented.

## Example

```bash
python -m src.evaluation.external_validation \
  --predictions artifacts/external_predictions.npy \
  --labels artifacts/external_labels.npy \
  --dataset-name "External Dataset Name" \
  --output reports/external_validation.json
```

Expected array shape:

```text
(N, 8)
```

## What not to do

Do not:

- report an external result from a dataset that was used for model development;
- tune thresholds on the external set;
- silently change the label mapping;
- compare metrics without documenting prevalence and cohort differences;
- describe external validation as clinical validation.

## Acceptance criteria

A future external-validation result should include:

- dataset name/version;
- access/provenance statement;
- inclusion/exclusion criteria;
- sample size;
- label mapping;
- preprocessing;
- frozen model/checkpoint identifier;
- per-label ROC-AUC;
- per-label PR-AUC;
- macro averages;
- limitations and domain-shift observations.
