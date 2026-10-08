# Post-hoc external validation analysis

This document supplements the primary frozen-checkpoint evaluation. It does **not** replace the prespecified 0.5 operating point and was not used to tune or select the deployed threshold.

## Bootstrap uncertainty

A 2,000-replicate IID bootstrap over the 5,856 evaluated images was used only to provide descriptive percentile 95% intervals.

| Metric | Estimate | 95% bootstrap interval |
|---|---:|---:|
| ROC-AUC | 0.3944 | 0.3777–0.4099 |
| Average Precision | 0.6696 | 0.6540–0.6855 |
| Sensitivity @ 0.5 | 0.9991 | 0.9979–0.9998 |
| Specificity @ 0.5 | 0.0000 | 0.0000–0.0000 |
| F1 @ 0.5 | 0.8433 | 0.8357–0.8504 |

## Descriptive threshold sweep

The sweep below illustrates the frozen score distribution under alternative thresholds. **No threshold was selected from this sweep.** The primary external-validation operating point remains 0.5.

| Threshold | Sensitivity | Specificity | Precision | F1 |
|---:|---:|---:|---:|---:|
| 0.5000 | 0.9991 | 0.0000 | 0.7295 | 0.8433 |
| 0.7500 | 0.9965 | 0.0000 | 0.7290 | 0.8420 |
| 0.9000 | 0.9888 | 0.0006 | 0.7276 | 0.8383 |
| 0.9500 | 0.9761 | 0.0038 | 0.7256 | 0.8325 |
| 0.9800 | 0.9469 | 0.0107 | 0.7210 | 0.8186 |
| 0.9900 | 0.8991 | 0.0328 | 0.7151 | 0.7966 |
| 0.9950 | 0.7868 | 0.1213 | 0.7073 | 0.7450 |
| 0.9970 | 0.6319 | 0.2495 | 0.6944 | 0.6617 |
| 0.9980 | 0.4559 | 0.3948 | 0.6703 | 0.5427 |
| 0.9990 | 0.1828 | 0.6974 | 0.6198 | 0.2823 |
| 0.9995 | 0.0393 | 0.9299 | 0.6022 | 0.0738 |
| 0.9999 | 0.0002 | 1.0000 | 1.0000 | 0.0005 |

### Interpretation

The model assigns extremely high Pneumonia probabilities to both classes. Raising the threshold can recover specificity only by sacrificing substantial sensitivity. This is consistent with severe external overconfidence and domain shift rather than a simple threshold-selection problem.

This analysis is descriptive and must not be interpreted as evidence for a clinically appropriate operating threshold.
