"""Evaluate MedVision on a separately obtained external dataset.

Third-party medical datasets are intentionally not bundled. Supply model
probabilities and labels as .npy files with shape (N, 8), in MedVision label order.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

LABELS = ["Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule","Pneumonia","Pneumothorax"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/external_validation.json"))
    args = parser.parse_args()
    p, y = np.load(args.predictions), np.load(args.labels).astype(int)
    if p.shape != y.shape or p.ndim != 2 or p.shape[1] != 8:
        raise ValueError("predictions and labels must both have shape (N, 8).")
    per_label, roc_values, pr_values = {}, [], []
    for i, label in enumerate(LABELS):
        if np.unique(y[:, i]).size < 2:
            per_label[label] = {"roc_auc": None, "pr_auc": None, "status": "skipped: one class present"}
            continue
        roc, pr = float(roc_auc_score(y[:, i], p[:, i])), float(average_precision_score(y[:, i], p[:, i]))
        per_label[label] = {"roc_auc": roc, "pr_auc": pr, "status": "evaluated"}
        roc_values.append(roc); pr_values.append(pr)
    result = {"dataset": args.dataset_name, "n_samples": int(y.shape[0]),
              "threshold_tuning_on_external_set": False,
              "macro_roc_auc": float(np.mean(roc_values)) if roc_values else None,
              "macro_pr_auc": float(np.mean(pr_values)) if pr_values else None,
              "per_label": per_label}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
