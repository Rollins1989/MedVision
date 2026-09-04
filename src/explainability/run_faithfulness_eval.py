"""
src/explainability/run_faithfulness_eval.py
-----------------------------------------------
Runs the faithfulness test across a sample of test-set positives for each
label and reports the aggregate faithfulness margin. Writes
reports/faithfulness_report.json.

Usage:
    python3 -m src.explainability.run_faithfulness_eval --checkpoint models/densenet121_weighted_bce.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data.split import patient_level_split
from src.data.synthetic_generator import LABELS
from src.explainability.faithfulness import evaluate_faithfulness
from src.models.factory import build_model
from src.preprocessing.transforms import build_transforms

ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/densenet121_weighted_bce.pt")
    parser.add_argument("--labels_csv", type=str, default="data/processed/labels_clean.csv")
    parser.add_argument("--images_dir", type=str, default="data/raw/images")
    parser.add_argument("--n_per_label", type=int, default=6)
    parser.add_argument("--k_percent", type=float, default=0.2)
    parser.add_argument("--out", type=str, default="reports/faithfulness_report.json")
    args = parser.parse_args()

    ckpt = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model_name"], num_labels=ckpt["num_labels"], pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    df = pd.read_csv(ROOT / args.labels_csv)
    _, _, test_df = patient_level_split(df, seed=42)

    tf = build_transforms(ckpt["image_size"], train=False)
    images_dir = ROOT / args.images_dir

    all_results = []
    per_label_margins = {lbl: [] for lbl in LABELS}

    for label in LABELS:
        label_idx = LABELS.index(label)
        positives = test_df[test_df[label] == 1]
        sample = positives.sample(n=min(args.n_per_label, len(positives)), random_state=0) if len(positives) else positives

        for _, row in sample.iterrows():
            img = Image.open(images_dir / row["image_id"]).convert("L")
            tensor = tf(img).unsqueeze(0)
            result = evaluate_faithfulness(model, ckpt["model_name"], tensor, label_idx, label, k_percent=args.k_percent)
            all_results.append({"image_id": row["image_id"], **result.__dict__})
            per_label_margins[label].append(result.faithfulness_margin)

    summary = {}
    for label, margins in per_label_margins.items():
        if margins:
            summary[label] = {
                "n_evaluated": len(margins),
                "mean_faithfulness_margin": round(float(np.mean(margins)), 4),
                "pct_faithful": round(float(np.mean([m > 0 for m in margins])), 4),
            }

    overall_margins = [r["faithfulness_margin"] for r in all_results]
    report = {
        "checkpoint": args.checkpoint,
        "k_percent_masked": args.k_percent,
        "n_total_evaluated": len(all_results),
        "overall_mean_faithfulness_margin": round(float(np.mean(overall_margins)), 4) if overall_margins else None,
        "overall_pct_faithful": round(float(np.mean([m > 0 for m in overall_margins])), 4) if overall_margins else None,
        "per_label": summary,
        "detail": all_results,
    }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Evaluated {len(all_results)} (image, label) pairs across {len(LABELS)} labels")
    print(f"Overall mean faithfulness margin: {report['overall_mean_faithfulness_margin']}")
    print(f"Overall %% of cases where masking the Grad-CAM region dropped the prediction "
          f"MORE than masking a random region: {report['overall_pct_faithful']:.1%}")
    print("\nPer-label breakdown:")
    for label, s in summary.items():
        print(f"  {label:15s} margin={s['mean_faithfulness_margin']:+.4f}  faithful={s['pct_faithful']:.1%}  (n={s['n_evaluated']})")
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
