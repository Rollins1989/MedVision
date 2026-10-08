# MedVision Model Card

## Model summary

**Model:** DenseNet121 + Weighted Binary Cross-Entropy  
**Task:** Multi-label chest X-ray classification  
**Output:** 8 thoracic finding probabilities  
**Input:** Chest X-ray image  
**Explainability:** Grad-CAM for CNN backbones  
**Primary checkpoint:** `models/densenet121_weighted_bce.pt`

MedVision is a research/educational prototype for studying medical computer vision, class imbalance, uncertainty, and visual explanation. It is **not a clinical diagnostic system**.

## Intended use

The model is intended for:

- ML/medical-AI experimentation.
- Demonstrating patient-level dataset splitting.
- Comparing CNN and Vision Transformer backbones.
- Studying class-imbalance strategies.
- Evaluating confidence/uncertainty and Grad-CAM explanations.
- Demonstrating a deployable inference API.

## Out-of-scope use

Do **not** use this model to:

- Diagnose or rule out disease in a patient.
- Make treatment decisions.
- Replace a radiologist or other qualified clinician.
- Triage patients without independent clinical validation.
- Claim clinical performance from this small experimental dataset.

## Architecture

The selected deployment checkpoint uses:

- DenseNet121 ImageNet-pretrained backbone.
- Dropout classification head.
- 8 independent sigmoid outputs.
- Weighted BCE loss with capped per-label positive weights.
- Patient-level train/validation/test separation.
- Grad-CAM for CNN visual explanations.

Other evaluated backbones include ResNet50, EfficientNet-B0, ViT-B/16, and ViT-B/32.

## Evaluation

The repository's current benchmark table is:

| Backbone | Loss | ROC-AUC | PR-AUC | F1 | Sensitivity | Specificity | Latency |
|---|---|---:|---:|---:|---:|---:|---:|
| DenseNet121 | BCE | 0.9380 | 0.8287 | 0.3687 | 0.3085 | 0.9972 | 30.2 ms |
| **DenseNet121** | **Weighted BCE** | **0.9354** | **0.7689** | **0.6656** | **0.9765** | **0.8080** | **28.5 ms** |
| DenseNet121 | Focal | **0.9394** | 0.8086 | 0.2928 | 0.2103 | 0.9956 | 30.1 ms |
| ResNet50 | BCE | 0.8455 | 0.6561 | 0.2396 | 0.1954 | 0.9958 | 31.1 ms |
| EfficientNet-B0 | BCE | 0.7914 | 0.4876 | 0.1292 | 0.1032 | 1.0000 | **9.7 ms** |
| ViT-B/32 | BCE | 0.7946 | 0.4274 | 0.0938 | 0.1071 | 0.9933 | 107.2 ms |

### How to interpret the results

There is no single universally best row.

- DenseNet121 + Focal has the highest ROC-AUC (0.9394).
- DenseNet121 + Weighted BCE has substantially higher F1 and sensitivity.
- Weighted BCE also reduces specificity to 0.8080, showing the cost of aggressively addressing class imbalance.
- EfficientNet-B0 is fastest in this benchmark but has materially weaker predictive performance.
- ViT-B/32 is slower in this CPU-oriented experiment and does not outperform the DenseNet configuration.

For that reason, the deployment default is described as a **high-sensitivity research configuration**, not as the globally best classifier.

## Calibration

Temperature scaling was evaluated on the selected checkpoint.

- ECE before calibration: **0.2394**
- ECE after calibration: **0.2735**
- Change: **-0.0341 improvement**

The negative improvement means the tested temperature-scaling configuration **made ECE worse**. The result is retained as a documented negative experiment rather than presented as a calibration improvement.

Further work should compare per-label calibration, isotonic regression, and validation-set calibration procedures.

## Explainability

Grad-CAM is implemented for CNN backbones. A faithfulness evaluation was performed on 45 image-label cases:

- Mean faithfulness margin: **0.1044**
- Faithful cases: **75.56%**

Performance varies by label. Effusion showed 50% faithful cases in the evaluated sample, while Infiltration and Mass reached 100%.

These results support treating Grad-CAM as an **evidence visualization**, not proof that the model is reasoning clinically correctly.

Vision Transformer explainability is not currently implemented with Grad-CAM; attention-based methods would be more appropriate for those architectures.

## Limitations

1. The evaluated dataset is small for a medical-imaging model.
2. External validation on an independent dataset has not yet been demonstrated.
3. Results may not generalize across hospitals, scanners, demographics, acquisition protocols, or disease prevalence.
4. The current benchmark is not a clinical validation study.
5. Threshold selection and calibration require further systematic evaluation.
6. Grad-CAM faithfulness is imperfect and label-dependent.
7. The model should not be interpreted as a medical device.

## Reproducibility

Training configuration, model factory, patient-level split logic, evaluation reports, API tests, and CI workflows are version controlled in this repository.

See:

- [DATA_CARD.md](DATA_CARD.md)
- [EXPERIMENTS.md](EXPERIMENTS.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
