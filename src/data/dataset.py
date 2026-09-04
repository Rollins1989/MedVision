"""
src/data/dataset.py
--------------------
PyTorch Dataset for multi-label chest X-ray classification.
"""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.synthetic_generator import LABELS


class ChestXrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: Path, transform=None, label_cols: list[str] | None = None):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.label_cols = label_cols or LABELS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_dir / row["image_id"]
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(row[self.label_cols].values.astype("float32"))
        return image, labels, row["image_id"]
