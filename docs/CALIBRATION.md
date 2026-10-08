# Calibration Experiment v2

## Motivation

The original temperature-scaling experiment produced an ECE change from **0.2394 to 0.2735**, so it did not improve the recorded calibration.

Calibration v2 therefore tests more than one post-hoc strategy and enforces a cleaner data boundary.

## Methods

The new evaluator compares:

1. **Raw sigmoid probabilities**
2. **Global temperature scaling**
3. **Per-label sigmoid calibration**

The calibrators must be fit on a calibration split that is separate from the model-training data and final test data.

This separation is important because fitting a calibrator on training predictions can produce optimistic probability estimates.

## Metrics

The experiment records:

- Macro expected calibration error (ECE)
- Flattened binary log loss
- Learned temperature
- Per-label sigmoid parameters

ECE is calculated per label and then macro-averaged.

## Run

Prepare:

- `logits.npy`: raw model logits, shape `(N, 8)`
- `labels.npy`: binary labels, shape `(N, 8)`

Then:

```bash
python -m src.evaluation.calibration \
  --logits artifacts/calibration_logits.npy \
  --labels artifacts/calibration_labels.npy \
  --output reports/calibration_v2.json
```

## Interpretation rule

A method is not called “better calibrated” simply because it is more complex. The preferred method should improve calibration metrics on data that was not used to fit it, while preserving a clear record of the baseline.

For small calibration datasets, isotonic regression should be treated cautiously because non-parametric calibration can overfit.

## Current status

The **implementation is complete**, but a new `calibration_v2.json` result should only be generated from the actual held-out calibration predictions. The repository does not fabricate that result.
