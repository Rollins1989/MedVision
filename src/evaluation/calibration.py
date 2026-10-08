"""Disjoint-split calibration utilities for multi-label probabilities."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def ece_binary(y_true, prob, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    score = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob <= hi if hi == 1 else prob < hi)
        if np.any(mask):
            score += mask.mean() * abs(prob[mask].mean() - y_true[mask].mean())
    return float(score)

def macro_ece(y, p):
    return float(np.mean([ece_binary(y[:, i], p[:, i]) for i in range(y.shape[1])]))

def global_temperature(logits, labels):
    candidates = np.exp(np.linspace(np.log(0.25), np.log(8.0), 160))
    losses = [log_loss(labels.ravel(), sigmoid((logits / t).ravel()), labels=[0, 1]) for t in candidates]
    return float(candidates[int(np.argmin(losses))])

def fit_per_label_sigmoid(logits, labels):
    params = []
    for i in range(labels.shape[1]):
        model = LogisticRegression(C=1.0, solver="lbfgs")
        model.fit(logits[:, [i]], labels[:, i])
        params.append({"coef": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0])})
    return params

def apply_per_label_sigmoid(logits, params):
    out = np.zeros_like(logits, dtype=float)
    for i, p in enumerate(params):
        out[:, i] = sigmoid(p["coef"] * logits[:, i] + p["intercept"])
    return out

def main():
    parser = argparse.ArgumentParser(description="Compare raw, global-temperature and per-label sigmoid calibration.")
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/calibration_v2.json"))
    args = parser.parse_args()
    logits = np.load(args.logits)
    labels = np.load(args.labels).astype(int)
    if logits.ndim != 2 or labels.shape != logits.shape:
        raise ValueError("logits and labels must both have shape (N, labels).")
    if len(labels) < 30:
        raise ValueError("At least 30 calibration samples are required; more is strongly preferred.")
    raw = sigmoid(logits)
    temperature = global_temperature(logits, labels)
    temp_prob = sigmoid(logits / temperature)
    params = fit_per_label_sigmoid(logits, labels)
    sigmoid_prob = apply_per_label_sigmoid(logits, params)
    result = {
        "protocol": {"samples": int(len(labels)), "labels": int(labels.shape[1]),
                     "test_set_used_for_fitting": False,
                     "calibration_data_required": "disjoint from model-training and final-test data"},
        "raw": {"macro_ece": macro_ece(labels, raw),
                "log_loss": float(log_loss(labels.ravel(), raw.ravel(), labels=[0, 1]))},
        "global_temperature": {"temperature": temperature,
                               "macro_ece": macro_ece(labels, temp_prob),
                               "log_loss": float(log_loss(labels.ravel(), temp_prob.ravel(), labels=[0, 1]))},
        "per_label_sigmoid": {"parameters": params,
                              "macro_ece": macro_ece(labels, sigmoid_prob),
                              "log_loss": float(log_loss(labels.ravel(), sigmoid_prob.ravel(), labels=[0, 1]))},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
