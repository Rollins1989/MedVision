# MedVision Failure Analysis

## Purpose

A professional medical-AI evaluation should explain **where the model fails and what trade-offs produce those failures**, not only report its strongest metric.

This analysis is based only on the experiments currently stored in the repository. It does not invent per-image error counts that were not recorded.

## 1. Class-imbalance operating-point failure

The clearest failure mode is the trade-off introduced by Weighted BCE.

| Configuration | F1 | Sensitivity | Specificity |
|---|---:|---:|---:|
| DenseNet121 + BCE | 0.3687 | 0.3085 | 0.9972 |
| DenseNet121 + Focal | 0.2928 | 0.2103 | 0.9956 |
| **DenseNet121 + Weighted BCE** | **0.6656** | **0.9765** | **0.8080** |

Weighted BCE dramatically improves sensitivity and F1, but specificity drops from 0.9972 under standard BCE to 0.8080.

### Interpretation

The model is much less conservative about positive findings. In an imbalanced multi-label problem this can be desirable for sensitivity-oriented screening research, but it also means more false-positive predictions.

**Failure mode:** optimizing for sensitivity can make the model insufficiently specific.

**Engineering implication:** threshold selection must be treated as an operating-point decision rather than using a single metric to declare a model “best.”

## 2. ROC-AUC vs F1 disagreement

DenseNet121 + Focal obtains the highest ROC-AUC (0.9394), yet its F1 (0.2928) and sensitivity (0.2103) are much lower than Weighted BCE.

This shows why ROC-AUC alone is insufficient for selecting the deployed configuration in this project.

**Failure mode:** a model can rank positive examples well while performing poorly at the selected classification threshold.

**Next step:** evaluate per-label precision-recall curves and validation-derived thresholds.

## 3. Weak-performing architectures

EfficientNet-B0 and ViT-B/32 underperform the DenseNet121 configurations on the current benchmark.

| Model | ROC-AUC | PR-AUC | F1 | Latency |
|---|---:|---:|---:|---:|
| DenseNet121 + Weighted BCE | 0.9354 | 0.7689 | 0.6656 | 28.5 ms |
| EfficientNet-B0 + BCE | 0.7914 | 0.4876 | 0.1292 | **9.7 ms** |
| ViT-B/32 + BCE | 0.7946 | 0.4274 | 0.0938 | 107.2 ms |

### Interpretation

EfficientNet-B0 demonstrates a speed/performance trade-off: it is much faster but substantially weaker on the recorded predictive metrics.

ViT-B/32 is slower in this CPU-oriented benchmark and does not show a compensating performance advantage.

**Failure mode:** a newer or lighter architecture is not automatically a better medical-imaging model under a fixed experimental budget.

## 4. Calibration failure

Temperature scaling was evaluated on the selected DenseNet121 Weighted BCE checkpoint.

- ECE before: 0.2394
- ECE after: 0.2735
- Change: -0.0341

The calibration procedure therefore **worsened** the recorded ECE.

**Failure mode:** a standard calibration technique can fail when its assumptions, calibration split, label structure, or fitting procedure do not match the task.

**Next step:** evaluate per-label temperature scaling and isotonic regression on a dedicated calibration split.

## 5. Explainability failure

Grad-CAM faithfulness was evaluated on 45 image-label pairs.

- Overall faithful: 75.56%
- Mean faithfulness margin: 0.1044

Per-label results:

| Label | Faithful |
|---|---:|
| Infiltration | 100.0% |
| Mass | 100.0% |
| Atelectasis | 83.3% |
| Pneumonia | 83.3% |
| Cardiomegaly | 66.7% |
| Nodule | 66.7% |
| Pneumothorax | 66.7% |
| Effusion | 50.0% |

### Interpretation

The explanation method is not uniformly faithful across labels. In particular, Effusion has the weakest result in this evaluated sample.

**Failure mode:** visually plausible heatmaps can fail to correspond consistently to the model's predictive signal.

**Engineering implication:** explanations should be presented as model-attribution evidence, not as medically validated lesion localization.

## 6. Dataset-scale/generalization failure

The data audit reports 454 total images, with 430 valid images after quality checks.

For a medical-imaging model, this is a small experimental scale.

The project currently does not demonstrate external validation on an independent hospital/dataset distribution.

**Failure mode:** benchmark performance may not transfer to a new population, scanner, acquisition protocol, prevalence profile, or institution.

**Highest-priority mitigation:** independent external validation.

## 7. What is not yet measured

The following should not be claimed until measured:

- Per-label confusion matrices.
- Per-label precision/recall from a fixed held-out test set.
- External-dataset performance.
- Subgroup performance.
- Robustness to acquisition/domain shift.
- Clinical utility.
- Calibration improvement.
- Clinically meaningful localization accuracy.

The absence of these measurements is itself an important limitation.

## Recommended next experiments

1. Tune **per-label thresholds on validation data only**.
2. Add precision-recall curves and per-label confusion matrices.
3. Create a dedicated calibration split.
4. Compare per-label temperature scaling with isotonic regression.
5. Add external validation.
6. Evaluate domain-shift robustness.
7. Inspect false positives and false negatives by label once prediction-level test artifacts are available.
