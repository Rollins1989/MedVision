"""
src/evaluation/build_experiment_table.py
------------------------------------------
Reads every *_metrics.json in models/ and produces the architecture
comparison table (README-ready markdown + CSV).

Usage:
    python3 -m src.evaluation.build_experiment_table
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


def build_table(models_dir: Path = ROOT / "models", pattern: str = "*_metrics.json") -> pd.DataFrame:
    rows = []
    for path in sorted(models_dir.glob(pattern)):
        with open(path) as f:
            data = json.load(f)
        tm = data["test_metrics"]
        rows.append({
            "run": data["run_name"],
            "model": data["model"],
            "loss": data["loss"],
            "roc_auc": tm["macro_roc_auc"],
            "pr_auc": tm["macro_pr_auc"],
            "f1": tm["macro_f1"],
            "sensitivity": tm["macro_sensitivity"],
            "specificity": tm["macro_specificity"],
            "latency_ms": tm["mean_inference_latency_ms"],
            "training_time_sec": data["training_time_sec"],
        })
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    header = "| Model | Loss | ROC-AUC | PR-AUC | F1 | Sensitivity | Specificity | Latency (ms) |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['model']} | {r['loss']} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} | "
            f"{r['f1']:.4f} | {r['sensitivity']:.4f} | {r['specificity']:.4f} | {r['latency_ms']:.1f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    df = build_table()
    print(df.to_string(index=False))
    out_csv = ROOT / "reports" / "experiment_table.csv"
    out_md = ROOT / "reports" / "experiment_table.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    with open(out_md, "w") as f:
        f.write(to_markdown(df))
    print(f"\nWrote {out_csv} and {out_md}")
