"""
src/uncertainty/calibration.py
----------------------------------
A model's raw sigmoid output isn't automatically a trustworthy
probability -- a model can be systematically over- or under-confident.
Temperature scaling (Guo et al., 2017) fixes this post-hoc: divide logits
by a single learned scalar T (fit on validation data, doesn't touch model
weights or accuracy/ranking at all) so that predicted probabilities better
match empirical outcome frequencies.

Also implements Expected Calibration Error (ECE) and the data for a
reliability diagram, before and after calibration, so the improvement is
measured, not asserted.
"""
import numpy as np
import torch
import torch.nn as nn


class TemperatureScaler(nn.Module):
    """Wraps a trained model; forward() returns temperature-scaled logits.
    Only `self.temperature` is ever trained -- the underlying model is
    frozen."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, lr: float = 0.01, max_iter: int = 100) -> float:
        """logits, labels: (N, C) tensors of RAW (unscaled) logits and
        binary targets, collected once on the validation set (no need to
        re-run the network at every optimization step)."""
        nll_criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.item())


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> tuple[float, dict]:
    """Flattened across all labels (multi-label ECE, standard approach:
    treat every (image, label) pair as one binary calibration sample).
    Returns (ece, reliability_diagram_data)."""
    y_true = y_true.flatten()
    y_prob = y_prob.flatten()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    ece = 0.0
    bins_data = []
    n = len(y_true)

    for b in range(n_bins):
        mask = bin_indices == b
        count = mask.sum()
        if count == 0:
            bins_data.append({"bin_lower": round(float(bin_edges[b]), 2), "bin_upper": round(float(bin_edges[b+1]), 2),
                               "count": 0, "avg_confidence": None, "avg_accuracy": None})
            continue
        avg_confidence = y_prob[mask].mean()
        avg_accuracy = y_true[mask].mean()
        gap = abs(avg_confidence - avg_accuracy)
        ece += (count / n) * gap
        bins_data.append({
            "bin_lower": round(float(bin_edges[b]), 2), "bin_upper": round(float(bin_edges[b+1]), 2),
            "count": int(count), "avg_confidence": round(float(avg_confidence), 4),
            "avg_accuracy": round(float(avg_accuracy), 4),
        })

    return float(ece), {"n_bins": n_bins, "bins": bins_data}
