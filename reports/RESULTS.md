# MedVision Results

## Model comparison

![Model comparison](figures/model_comparison.svg)

The benchmark demonstrates a clear trade-off between predictive quality, sensitivity, specificity, and inference latency.

### Key observations

- DenseNet121 + Focal has the highest ROC-AUC: **0.9394**.
- DenseNet121 + Weighted BCE has the highest F1: **0.6656** and sensitivity: **0.9765**.
- Weighted BCE specificity is **0.8080**, so the sensitivity gain comes with substantially more false positives.
- EfficientNet-B0 has the lowest latency: **9.7 ms**, but materially weaker predictive metrics.
- ViT-B/32 is the slowest evaluated model at **107.2 ms** and does not outperform DenseNet121 in this benchmark.

## Explainability faithfulness

![Faithfulness by label](figures/faithfulness_by_label.svg)

The overall faithfulness rate was **75.56% across 45 evaluated image-label pairs**. The result is heterogeneous across labels, which is why Grad-CAM should be treated as an interpretability aid rather than proof of causal reasoning.

## Calibration

The selected checkpoint's ECE changed from **0.2394** to **0.2735** after the tested temperature-scaling procedure.

**Conclusion:** the tested calibration method did not improve calibration. This negative result is retained as part of the experiment record.


## External validation — Kermany / Guangzhou cohort

A frozen DenseNet121 + Weighted BCE checkpoint was evaluated on all **5,856 images** present in the attached Kermany cohort. No retraining or external threshold optimization was performed.

| Metric | External result |
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

At the fixed 0.5 threshold:

~~~text
                 Predicted
              Normal  Pneumonia
Actual Normal      0       1583
       Pneumonia   4       4269
~~~

The model labels all 1,583 Normal images as Pneumonia. This combination of **99.91% sensitivity and 0% specificity** is not a useful clinical operating point. The near-one probability distribution also indicates severe external overconfidence.

The ROC-AUC of 0.3944 shows that the score ranking does not transfer well to this cohort. The result is therefore interpreted as a **domain-shift/failure-analysis finding**, not as clinical validation.

![External ROC](external_validation/kermany/external_roc_curve.svg)

![External reliability](external_validation/kermany/external_reliability.svg)

See [External Validation](../docs/EXTERNAL_VALIDATION.md) and [Post-hoc analysis](external_validation/kermany/POSTHOC_ANALYSIS.md) for protocol details and uncertainty analysis.

## Recommended reading order

1. [Architecture](../docs/ARCHITECTURE.md)
2. [Data Card](../docs/DATA_CARD.md)
3. [Model Card](../docs/MODEL_CARD.md)
4. [Experiment Registry](../docs/EXPERIMENTS.md)
5. [Experiment table](experiment_table.md)
