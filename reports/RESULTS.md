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

## Recommended reading order

1. [Architecture](../docs/ARCHITECTURE.md)
2. [Data Card](../docs/DATA_CARD.md)
3. [Model Card](../docs/MODEL_CARD.md)
4. [Experiment Registry](../docs/EXPERIMENTS.md)
5. [Experiment table](experiment_table.md)
