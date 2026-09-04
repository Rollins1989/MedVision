"""
src/data/quality_checks.py
---------------------------
Automated dataset quality report: corrupted files, duplicates (perceptual
hash), blank/near-blank images, unexpected dimensions/channels, missing
labels. Run before training so silently-broken data doesn't corrupt a
training run.

Usage:
    python3 -m src.data.quality_checks --images data/raw/images --labels data/raw/labels.csv
"""
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

EXPECTED_MODES = {"L", "RGB"}  # grayscale or RGB accepted; RGBA/CMYK etc rejected
MIN_DIM = 32
BLANK_STD_THRESHOLD = 2.0  # near-zero pixel std => effectively blank
EXTREME_BRIGHTNESS_LOW = 5
EXTREME_BRIGHTNESS_HIGH = 250


@dataclass
class QualityReport:
    total_images: int = 0
    valid_images: int = 0
    corrupted: list = field(default_factory=list)
    duplicates: list = field(default_factory=list)  # list of (kept, dropped) pairs
    blank_images: list = field(default_factory=list)
    extreme_brightness: list = field(default_factory=list)
    bad_dimensions: list = field(default_factory=list)
    bad_channels: list = field(default_factory=list)
    missing_labels: list = field(default_factory=list)

    def to_dict(self):
        return {
            "total_images": self.total_images,
            "valid_images": self.valid_images,
            "corrupted_count": len(self.corrupted),
            "duplicate_count": len(self.duplicates),
            "blank_count": len(self.blank_images),
            "extreme_brightness_count": len(self.extreme_brightness),
            "bad_dimensions_count": len(self.bad_dimensions),
            "bad_channels_count": len(self.bad_channels),
            "missing_labels_count": len(self.missing_labels),
            "corrupted": self.corrupted,
            "duplicates": self.duplicates,
            "blank_images": self.blank_images,
            "extreme_brightness": self.extreme_brightness,
            "bad_dimensions": self.bad_dimensions,
            "bad_channels": self.bad_channels,
            "missing_labels": self.missing_labels,
        }

    def summary_text(self) -> str:
        lines = [
            "Dataset Quality Report",
            "=======================",
            f"Total images:        {self.total_images}",
            f"Valid images:        {self.valid_images}",
            f"Corrupted:           {len(self.corrupted)}",
            f"Duplicates:          {len(self.duplicates)}",
            f"Blank images:        {len(self.blank_images)}",
            f"Extreme brightness:  {len(self.extreme_brightness)}",
            f"Bad dimensions:      {len(self.bad_dimensions)}",
            f"Bad channel count:   {len(self.bad_channels)}",
            f"Missing labels:      {len(self.missing_labels)}",
        ]
        return "\n".join(lines)


def run_quality_checks(image_dir: Path, labels_csv: Path, label_cols: list[str] | None = None) -> QualityReport:
    df = pd.read_csv(labels_csv)
    report = QualityReport(total_images=len(df))

    if label_cols is None:
        label_cols = [c for c in df.columns if c not in ("image_id", "patient_id")]

    # missing labels: any row where all label columns are NaN
    missing = df[df[label_cols].isna().all(axis=1)]
    report.missing_labels = missing["image_id"].tolist()

    hashes: dict[str, str] = {}  # perceptual hash -> first filename that had it
    problem_files: set[str] = set()

    for _, row in df.iterrows():
        fname = row["image_id"]
        fpath = image_dir / fname

        if not fpath.exists():
            report.corrupted.append(fname)
            problem_files.add(fname)
            continue

        try:
            img = Image.open(fpath)
            img.load()  # force full decode -- catches truncated files
        except (UnidentifiedImageError, OSError, ValueError):
            report.corrupted.append(fname)
            problem_files.add(fname)
            continue

        # channel / mode check
        if img.mode not in EXPECTED_MODES:
            report.bad_channels.append({"file": fname, "mode": img.mode})
            problem_files.add(fname)
            continue

        # dimension check
        w, h = img.size
        aspect = max(w, h) / max(1, min(w, h))
        if w < MIN_DIM or h < MIN_DIM or aspect > 3.0:
            report.bad_dimensions.append({"file": fname, "size": [w, h]})
            problem_files.add(fname)
            continue

        arr = np.array(img.convert("L"), dtype=np.float32)

        # blank image check
        if arr.std() < BLANK_STD_THRESHOLD:
            report.blank_images.append(fname)
            problem_files.add(fname)
            continue

        # extreme brightness check
        mean_brightness = arr.mean()
        if mean_brightness < EXTREME_BRIGHTNESS_LOW or mean_brightness > EXTREME_BRIGHTNESS_HIGH:
            report.extreme_brightness.append({"file": fname, "mean_brightness": round(float(mean_brightness), 2)})
            problem_files.add(fname)
            continue

        # duplicate check (perceptual hash -- catches near-identical, not just byte-identical)
        phash = str(imagehash.phash(img))
        if phash in hashes:
            report.duplicates.append({"kept": hashes[phash], "dropped": fname})
            problem_files.add(fname)
            continue
        hashes[phash] = fname

    report.valid_images = report.total_images - len(problem_files)
    return report


def main():
    parser = argparse.ArgumentParser(description="Run dataset quality checks")
    parser.add_argument("--images", type=Path, default=Path("data/raw/images"))
    parser.add_argument("--labels", type=Path, default=Path("data/raw/labels.csv"))
    parser.add_argument("--out", type=Path, default=Path("reports/data_quality_report.json"))
    args = parser.parse_args()

    report = run_quality_checks(args.images, args.labels)
    print(report.summary_text())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()
