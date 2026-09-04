"""
src/data/synthetic_generator.py
--------------------------------
Generates a small synthetic chest-X-ray-like dataset that mimics the
*structure* of NIH ChestX-ray14 (patient IDs, multi-label findings,
duplicates, corrupted files, blank images) so the rest of this pipeline
-- quality checks, patient-level splitting, training, evaluation,
Grad-CAM, calibration -- can run against something real on disk and be
verified end-to-end.

IMPORTANT: this is NOT real medical imaging data and the images have no
diagnostic meaning. It exists purely so this sandbox can demonstrate a
working pipeline without a 45GB download of ChestX-ray14/CheXpert, which
requires an account-gated host (Box/Kaggle) unreachable from this
environment. See README "Using the real dataset" for exact steps to point
this same codebase at the real ChestX-ray14 data.

The generator injects, on purpose:
  - a handful of corrupted files
  - a handful of exact-duplicate images (different filenames)
  - a couple of blank (all-black) images
  - a few images with unexpected channel count / dimensions
so `src/data/quality_checks.py` has real problems to find.
"""
import hashlib
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
]

IMG_SIZE = 128
RNG_SEED = 42


def _draw_lung_template(rng: np.random.Generator) -> Image.Image:
    """A crude procedural 'chest X-ray-like' base image: two lung fields,
    a spine line, ribs -- enough structure for a CNN to find correlated
    patterns, not remotely radiologically accurate."""
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=int(rng.integers(15, 35)))
    draw = ImageDraw.Draw(img)

    # spine
    draw.line([(IMG_SIZE // 2, 5), (IMG_SIZE // 2, IMG_SIZE - 5)],
              fill=int(rng.integers(60, 90)), width=3)

    # two lung fields (ellipses), slightly randomized
    for side in (-1, 1):
        cx = IMG_SIZE // 2 + side * IMG_SIZE // 4
        cy = IMG_SIZE // 2
        rx = IMG_SIZE // 5 + int(rng.integers(-4, 4))
        ry = IMG_SIZE // 2 - 10 + int(rng.integers(-4, 4))
        draw.ellipse(
            [cx - rx, cy - ry, cx + rx, cy + ry],
            fill=int(rng.integers(90, 130)),
        )

    # ribs
    for i in range(6):
        y = 15 + i * 15 + int(rng.integers(-2, 2))
        draw.arc([10, y - 40, IMG_SIZE - 10, y + 60], start=200, end=340,
                  fill=int(rng.integers(40, 60)), width=1)

    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    return img


def _inject_finding_signal(img: Image.Image, active_labels: list[str],
                            rng: np.random.Generator) -> Image.Image:
    """Overlay a crude, label-specific visual signal so a model has
    something learnable to correlate with each finding. Not medically
    meaningful -- purely a synthetic proxy."""
    draw = ImageDraw.Draw(img)
    for label in active_labels:
        if label == "Cardiomegaly":
            # enlarged bright blob near the spine (proxy for heart size)
            cx, cy = IMG_SIZE // 2, IMG_SIZE // 2 + 15
            r = 22 + int(rng.integers(0, 6))
            draw.ellipse([cx - r, cy - r // 1.5, cx + r, cy + r // 1.5],
                         fill=int(rng.integers(150, 190)))
        elif label in ("Nodule", "Mass"):
            size = 6 if label == "Nodule" else 16
            cx = int(rng.integers(30, IMG_SIZE - 30))
            cy = int(rng.integers(30, IMG_SIZE - 30))
            draw.ellipse([cx - size, cy - size, cx + size, cy + size],
                         fill=int(rng.integers(180, 220)))
        elif label == "Pneumonia" or label == "Infiltration":
            # patchy bright cloud in one lung field
            side = rng.choice([-1, 1])
            cx = IMG_SIZE // 2 + side * IMG_SIZE // 4
            cy = int(rng.integers(40, IMG_SIZE - 40))
            for _ in range(8):
                dx, dy = rng.integers(-15, 15), rng.integers(-15, 15)
                r = int(rng.integers(4, 10))
                draw.ellipse([cx + dx - r, cy + dy - r, cx + dx + r, cy + dy + r],
                             fill=int(rng.integers(140, 180)))
        elif label == "Effusion":
            # bright band at the base of a lung
            side = rng.choice([-1, 1])
            cx = IMG_SIZE // 2 + side * IMG_SIZE // 4
            draw.rectangle([cx - 18, IMG_SIZE - 35, cx + 18, IMG_SIZE - 15],
                            fill=int(rng.integers(150, 190)))
        elif label == "Pneumothorax":
            # dark band at lung apex (proxy for absent lung markings)
            side = rng.choice([-1, 1])
            cx = IMG_SIZE // 2 + side * IMG_SIZE // 4
            draw.rectangle([cx - 15, 12, cx + 15, 30], fill=int(rng.integers(0, 10)))
        elif label == "Atelectasis":
            side = rng.choice([-1, 1])
            cx = IMG_SIZE // 2 + side * IMG_SIZE // 4
            draw.line([(cx - 12, 40), (cx + 12, IMG_SIZE - 40)],
                      fill=int(rng.integers(150, 190)), width=4)
    return img.filter(ImageFilter.GaussianBlur(radius=0.8))


def generate_dataset(
    out_dir: Path,
    n_patients: int = 220,
    max_images_per_patient: int = 3,
    seed: int = RNG_SEED,
    n_corrupted: int = 6,
    n_duplicates: int = 10,
    n_blank: int = 3,
    n_bad_dims: int = 3,
) -> pd.DataFrame:
    """Generates images into out_dir/images/ and returns the labels
    DataFrame (also written to out_dir/labels.csv). Deliberately injects
    data-quality problems for src/data/quality_checks.py to catch."""
    rng = np.random.default_rng(seed)
    random.seed(seed)

    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    records = []
    filenames_for_dupe = []

    patient_ids = [f"P{str(i).zfill(5)}" for i in range(n_patients)]

    for patient_id in patient_ids:
        n_images = rng.integers(1, max_images_per_patient + 1)
        # a patient's disease profile is somewhat consistent across their images
        base_active = [lbl for lbl in LABELS if rng.random() < 0.12]

        for _ in range(n_images):
            active = list(base_active)
            # small per-image variation
            if rng.random() < 0.08:
                extra = rng.choice(LABELS)
                if extra not in active:
                    active.append(extra)

            img = _draw_lung_template(rng)
            img = _inject_finding_signal(img, active, rng)

            fname = f"{patient_id}_{hashlib.md5(str(rng.random()).encode()).hexdigest()[:8]}.png"
            fpath = img_dir / fname
            img.save(fpath)
            filenames_for_dupe.append(fname)

            row = {"image_id": fname, "patient_id": patient_id}
            for lbl in LABELS:
                row[lbl] = 1 if lbl in active else 0
            records.append(row)

    df = pd.DataFrame(records)

    # --- inject quality problems -------------------------------------
    # 1. corrupted files: truncate/garble a few existing images
    corrupt_targets = rng.choice(df["image_id"].values, size=n_corrupted, replace=False)
    for fname in corrupt_targets:
        with open(img_dir / fname, "wb") as f:
            f.write(b"NOT_A_REAL_PNG_FILE_" + os.urandom(50))

    # 2. exact duplicates: copy an existing image under a new filename,
    #    reusing the same patient's labels
    dup_source_rows = df.sample(n=n_duplicates, random_state=seed).to_dict("records")
    for row in dup_source_rows:
        src = img_dir / row["image_id"]
        if not src.exists():
            continue
        new_name = f"DUP_{row['image_id']}"
        try:
            Image.open(src).save(img_dir / new_name)
        except Exception:
            continue
        new_row = dict(row)
        new_row["image_id"] = new_name
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # 3. blank (all-black) images
    for i in range(n_blank):
        fname = f"BLANK_{i}.png"
        Image.new("L", (IMG_SIZE, IMG_SIZE), color=0).save(img_dir / fname)
        df = pd.concat([df, pd.DataFrame([{
            "image_id": fname, "patient_id": f"P_BLANK_{i}",
            **{lbl: 0 for lbl in LABELS},
        }])], ignore_index=True)

    # 4. unexpected dimensions / channel count
    for i in range(n_bad_dims):
        fname = f"BADDIM_{i}.png"
        arr = (np.random.rand(40, 200, 4) * 255).astype(np.uint8)  # RGBA, odd aspect ratio
        Image.fromarray(arr, mode="RGBA").save(img_dir / fname)
        df = pd.concat([df, pd.DataFrame([{
            "image_id": fname, "patient_id": f"P_BADDIM_{i}",
            **{lbl: 0 for lbl in LABELS},
        }])], ignore_index=True)

    df.to_csv(out_dir / "labels.csv", index=False)
    return df


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
    df = generate_dataset(out)
    print(f"Generated {len(df)} image records for {df['patient_id'].nunique()} patients")
    print(f"Wrote images -> {out / 'images'}")
    print(f"Wrote labels -> {out / 'labels.csv'}")
    print(df[LABELS].sum())
