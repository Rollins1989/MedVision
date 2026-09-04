"""
src/uncertainty/run_calibration_eval.py
-------------------------------------------
Fits temperature scaling on the validation set, then reports ECE before
and after calibration on the held-out test set. Real numbers, computed
from the actual model -- not fabricated.

Usage:
    python3 -m src.uncertainty.run_calibration_eval --checkpoint models/densenet121_weighted_bce.pt
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ChestXrayDataset
from src.data.split import patient_level_split
from src.models.factory import build_model
from src.preprocessing.transforms import build_transforms
from src.uncertainty.calibration import TemperatureScaler, expected_calibration_error

ROOT = Path(__file__).resolve().parent.parent.parent


@torch.no_grad()
def collect_logits(model, loader):
    model.eval()
    all_logits, all_labels = [], []
    for images, labels, _ in loader:
        logits = model(images)
        all_logits.append(logits)
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/densenet121_weighted_bce.pt")
    parser.add_argument("--labels_csv", type=str, default="data/processed/labels_clean.csv")
    parser.add_argument("--images_dir", type=str, default="data/raw/images")
    parser.add_argument("--out", type=str, default="reports/calibration_report.json")
    args = parser.parse_args()

    ckpt = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model_name"], num_labels=ckpt["num_labels"], pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    df = pd.read_csv(ROOT / args.labels_csv)
    train_df, val_df, test_df = patient_level_split(df, seed=42)

    tf = build_transforms(ckpt["image_size"], train=False)
    images_dir = ROOT / args.images_dir
    val_loader = DataLoader(ChestXrayDataset(val_df, images_dir, transform=tf), batch_size=16, shuffle=False)
    test_loader = DataLoader(ChestXrayDataset(test_df, images_dir, transform=tf), batch_size=16, shuffle=False)

    val_logits, val_labels = collect_logits(model, val_loader)
    scaler = TemperatureScaler(model)
    temperature = scaler.fit(val_logits, val_labels)
    print(f"Fitted temperature: {temperature:.4f}")

    test_logits, test_labels = collect_logits(model, test_loader)
    y_true = test_labels.numpy()

    probs_before = torch.sigmoid(test_logits).numpy()
    probs_after = torch.sigmoid(test_logits / temperature).numpy()

    ece_before, diagram_before = expected_calibration_error(y_true, probs_before)
    ece_after, diagram_after = expected_calibration_error(y_true, probs_after)

    report = {
        "checkpoint": args.checkpoint,
        "fitted_temperature": round(temperature, 4),
        "ece_before_calibration": round(ece_before, 4),
        "ece_after_calibration": round(ece_after, 4),
        "improvement": round(ece_before - ece_after, 4),
        "reliability_diagram_before": diagram_before,
        "reliability_diagram_after": diagram_after,
    }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"ECE before calibration: {ece_before:.4f}")
    print(f"ECE after calibration:  {ece_after:.4f}")
    print(f"Improvement: {ece_before - ece_after:+.4f}")
    print(f"\nFull report -> {out_path}")


if __name__ == "__main__":
    main()
