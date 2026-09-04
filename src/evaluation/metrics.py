"""
src/evaluation/metrics.py
---------------------------
Clinically-relevant evaluation for multi-label classification. Deliberately
does NOT lead with accuracy -- with 8 sparse findings (most images negative
for most labels), a model predicting "all negative" scores extremely high
accuracy while being clinically useless. ROC-AUC, PR-AUC, sensitivity
(recall on the positive/disease class), and specificity are what's reported
here, per-label and macro-averaged.
"""
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class LabelMetrics:
    label: str
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    sensitivity: float  # = recall on positive class
    specificity: float
    confusion_matrix: list  # [[TN, FP], [FN, TP]]
    support_positive: int


@dataclass
class EvaluationResult:
    per_label: list = field(default_factory=list)
    macro_roc_auc: float = 0.0
    macro_pr_auc: float = 0.0
    macro_f1: float = 0.0
    macro_sensitivity: float = 0.0
    macro_specificity: float = 0.0
    mean_inference_latency_ms: float = 0.0

    def to_dict(self):
        return {
            "macro_roc_auc": round(self.macro_roc_auc, 4),
            "macro_pr_auc": round(self.macro_pr_auc, 4),
            "macro_f1": round(self.macro_f1, 4),
            "macro_sensitivity": round(self.macro_sensitivity, 4),
            "macro_specificity": round(self.macro_specificity, 4),
            "mean_inference_latency_ms": round(self.mean_inference_latency_ms, 2),
            "per_label": [
                {
                    "label": m.label, "roc_auc": round(m.roc_auc, 4), "pr_auc": round(m.pr_auc, 4),
                    "f1": round(m.f1, 4), "precision": round(m.precision, 4),
                    "sensitivity": round(m.sensitivity, 4), "specificity": round(m.specificity, 4),
                    "confusion_matrix": m.confusion_matrix, "support_positive": m.support_positive,
                }
                for m in self.per_label
            ],
        }


def evaluate_predictions(
    y_true: np.ndarray, y_prob: np.ndarray, label_names: list[str], threshold: float = 0.5,
) -> EvaluationResult:
    """y_true, y_prob: (N, C) arrays. Returns per-label + macro metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    result = EvaluationResult()

    for i, label in enumerate(label_names):
        yt, yp, ypred = y_true[:, i], y_prob[:, i], y_pred[:, i]
        n_pos = int(yt.sum())

        if n_pos == 0 or n_pos == len(yt):
            # ROC-AUC/PR-AUC undefined with only one class present in y_true
            roc_auc, pr_auc = float("nan"), float("nan")
        else:
            roc_auc = roc_auc_score(yt, yp)
            pr_auc = average_precision_score(yt, yp)

        f1 = f1_score(yt, ypred, zero_division=0)
        precision = precision_score(yt, ypred, zero_division=0)
        sensitivity = recall_score(yt, ypred, zero_division=0)  # TP / (TP + FN)

        cm = confusion_matrix(yt, ypred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

        result.per_label.append(LabelMetrics(
            label=label, roc_auc=roc_auc, pr_auc=pr_auc, f1=f1, precision=precision,
            sensitivity=sensitivity, specificity=specificity,
            confusion_matrix=cm.tolist(), support_positive=n_pos,
        ))

    valid_roc = [m.roc_auc for m in result.per_label if not np.isnan(m.roc_auc)]
    valid_pr = [m.pr_auc for m in result.per_label if not np.isnan(m.pr_auc)]
    result.macro_roc_auc = float(np.mean(valid_roc)) if valid_roc else float("nan")
    result.macro_pr_auc = float(np.mean(valid_pr)) if valid_pr else float("nan")
    result.macro_f1 = float(np.mean([m.f1 for m in result.per_label]))
    result.macro_sensitivity = float(np.mean([m.sensitivity for m in result.per_label]))
    valid_spec = [m.specificity for m in result.per_label if not np.isnan(m.specificity)]
    result.macro_specificity = float(np.mean(valid_spec)) if valid_spec else float("nan")

    return result


def measure_inference_latency(model, sample_input, device, n_warmup: int = 5, n_runs: int = 20) -> float:
    """Returns mean single-image inference latency in milliseconds."""
    model.eval()
    sample_input = sample_input.to(device)
    import torch
    with torch.no_grad():
        for _ in range(n_warmup):
            model(sample_input)

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            model(sample_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    return float(np.mean(times))
