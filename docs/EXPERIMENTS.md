# MedVision Experiment Registry

This document turns the project from a collection of model runs into an explicit experimental program.

## Research questions

### RQ1 — Does backbone choice materially affect multi-label chest X-ray performance?

**Comparison:** ResNet50, DenseNet121, EfficientNet-B0, ViT-B/32.

**Primary evidence:** ROC-AUC, PR-AUC, F1, sensitivity, specificity, latency.

**Finding:** DenseNet121 configurations clearly outperform the EfficientNet-B0 and ViT-B/32 rows in the current benchmark. The exact trade-off depends on the metric.

### RQ2 — How does loss design affect class-imbalanced performance?

**Comparison:** BCE vs Weighted BCE vs Focal on DenseNet121.

**Finding:** Weighted BCE produces much higher F1 and sensitivity, while Focal achieves the highest ROC-AUC. Weighted BCE also lowers specificity, demonstrating a meaningful operating-point trade-off.

### RQ3 — Does calibration improve confidence quality?

**Method:** Temperature scaling on the selected DenseNet121 Weighted BCE checkpoint.

**Finding:** No. ECE changed from 0.2394 to 0.2735, so the tested calibration procedure worsened the recorded ECE.

**Next experiment:** Compare per-label temperature scaling and isotonic regression using a strictly held-out calibration split.

### RQ4 — Are Grad-CAM explanations faithful to the model's prediction signal?

**Method:** Mask the highest-attribution region and compare the probability change against a randomly masked region.

**Current evidence:** 45 image-label evaluations; 75.56% were classified as faithful under the project's faithfulness criterion.

**Finding:** Faithfulness is label-dependent. Effusion was the weakest evaluated label at 50% faithful cases; Infiltration and Mass were strongest at 100%.

### RQ5 — Can a single global threshold represent all labels fairly?

**Status:** **Open.**

The current project should evaluate per-label thresholds on the validation set before finalizing an operating point. Threshold optimization must use validation data only and must not tune against the test set.

## Benchmark table

| Model | Loss | ROC-AUC | PR-AUC | F1 | Sensitivity | Specificity | Latency |
|---|---|---:|---:|---:|---:|---:|---:|
| DenseNet121 | BCE | 0.9380 | 0.8287 | 0.3687 | 0.3085 | 0.9972 | 30.2 ms |
| DenseNet121 | Focal | 0.9394 | 0.8086 | 0.2928 | 0.2103 | 0.9956 | 30.1 ms |
| DenseNet121 | Weighted BCE | 0.9354 | 0.7689 | 0.6656 | 0.9765 | 0.8080 | 28.5 ms |
| ResNet50 | BCE | 0.8455 | 0.6561 | 0.2396 | 0.1954 | 0.9958 | 31.1 ms |
| EfficientNet-B0 | BCE | 0.7914 | 0.4876 | 0.1292 | 0.1032 | 1.0000 | 9.7 ms |
| ViT-B/32 | BCE | 0.7946 | 0.4274 | 0.0938 | 0.1071 | 0.9933 | 107.2 ms |

## Decision log

| Decision | Evidence | Decision |
|---|---|---|
| Use patient-level split | Medical imaging leakage risk | Keep |
| Compare multiple backbones | Architecture may dominate performance | Keep |
| Compare imbalance-aware losses | Strong class imbalance | Keep |
| Default to Weighted BCE checkpoint | Highest F1/sensitivity in current table | Keep, with specificity caveat |
| Claim calibration success | ECE worsened | **Do not claim** |
| Claim Grad-CAM is always faithful | 75.56% overall faithfulness | **Do not claim** |
| Use external validation as next major upgrade | Current generalization evidence is limited | Prioritize |

## Reproducibility notes

Each experiment should record:

- Repository commit SHA.
- Dataset/version identifier.
- Split seed.
- Backbone.
- Loss.
- Input resolution.
- Batch size.
- Learning rate.
- Weight decay.
- Number of epochs / early stopping.
- Pretrained weight source.
- Evaluation command.
- Output metrics.

The existing reports provide the current benchmark snapshot; future runs should extend this registry rather than overwrite previous results.
