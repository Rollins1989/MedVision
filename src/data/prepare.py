"""
src/data/prepare.py
----------------------
Runs the quality checker and writes a cleaned labels CSV (bad rows
dropped) to data/processed/. This is the step that turns "raw dataset with
known problems" into "what the training pipeline actually consumes" --
kept as an explicit, auditable step rather than silently filtering inside
the Dataset class, so the quality report and the training data are always
in sync and inspectable.

Usage:
    python3 -m src.data.prepare
"""
from pathlib import Path

import pandas as pd

from src.data.quality_checks import run_quality_checks

ROOT = Path(__file__).resolve().parent.parent.parent


def prepare_clean_labels(
    images_dir: Path = ROOT / "data" / "raw" / "images",
    labels_csv: Path = ROOT / "data" / "raw" / "labels.csv",
    out_csv: Path = ROOT / "data" / "processed" / "labels_clean.csv",
    report_path: Path = ROOT / "reports" / "data_quality_report.json",
) -> pd.DataFrame:
    report = run_quality_checks(images_dir, labels_csv)
    print(report.summary_text())

    report_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    bad_files = set(report.corrupted) | set(report.blank_images) | \
        {d["file"] for d in report.bad_dimensions} | \
        {d["file"] for d in report.bad_channels} | \
        {d["file"] for d in report.extreme_brightness} | \
        {d["dropped"] for d in report.duplicates}

    df = pd.read_csv(labels_csv)
    clean_df = df[~df["image_id"].isin(bad_files)].reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(out_csv, index=False)
    print(f"\nDropped {len(df) - len(clean_df)} bad rows -> {len(clean_df)} clean rows")
    print(f"Clean labels written to {out_csv}")
    return clean_df


if __name__ == "__main__":
    prepare_clean_labels()
