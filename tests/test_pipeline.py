"""
tests/test_pipeline.py
--------------------------
Unit tests for the data / model / loss layers, independent of the API.
"""
from pathlib import Path

import pandas as pd
import torch

from src.data.quality_checks import run_quality_checks
from src.data.split import patient_level_split
from src.models.factory import DEFAULT_IMAGE_SIZE, SUPPORTED_MODELS, build_model
from src.training.losses import FocalLoss, WeightedBCELoss, build_loss, compute_pos_weights

ROOT = Path(__file__).resolve().parent.parent


def test_quality_checks_catch_injected_problems():
    report = run_quality_checks(ROOT / "data" / "raw" / "images", ROOT / "data" / "raw" / "labels.csv")
    assert report.total_images > 0
    assert len(report.corrupted) > 0
    assert len(report.blank_images) > 0
    assert report.valid_images < report.total_images


def test_patient_level_split_no_overlap():
    df = pd.read_csv(ROOT / "data" / "processed" / "labels_clean.csv")
    train_df, val_df, test_df = patient_level_split(df, seed=42)

    train_p = set(train_df["patient_id"])
    val_p = set(val_df["patient_id"])
    test_p = set(test_df["patient_id"])

    assert train_p.isdisjoint(val_p)
    assert train_p.isdisjoint(test_p)
    assert val_p.isdisjoint(test_p)
    assert len(train_df) + len(val_df) + len(test_df) == len(df)


def test_patient_level_split_reproducible_with_seed():
    df = pd.read_csv(ROOT / "data" / "processed" / "labels_clean.csv")
    train1, _, _ = patient_level_split(df, seed=42)
    train2, _, _ = patient_level_split(df, seed=42)
    assert set(train1["image_id"]) == set(train2["image_id"])


def test_all_supported_models_build_and_forward():
    for name in SUPPORTED_MODELS:
        size = DEFAULT_IMAGE_SIZE[name]
        model = build_model(name, num_labels=8, pretrained=False)
        model.eval()
        x = torch.randn(1, 3, size, size)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 8)


def test_compute_pos_weights_shape_and_range():
    labels = torch.tensor([[1, 0, 0], [0, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=torch.float32)
    weights = compute_pos_weights(labels)
    assert weights.shape == (3,)
    assert (weights >= 0).all()


def test_weighted_bce_loss_computes():
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    logits = torch.randn(2, 2)
    loss_fn = build_loss("weighted_bce", label_matrix=labels)
    loss = loss_fn(logits, labels)
    assert loss.item() >= 0
    assert isinstance(loss_fn, WeightedBCELoss)


def test_focal_loss_computes_and_is_lower_for_confident_correct_predictions():
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    targets = torch.tensor([[1.0]])

    confident_correct_logits = torch.tensor([[5.0]])   # sigmoid ~0.99, matches target=1
    unconfident_logits = torch.tensor([[0.0]])          # sigmoid = 0.5

    loss_confident = loss_fn(confident_correct_logits, targets)
    loss_unconfident = loss_fn(unconfident_logits, targets)

    # Focal loss should assign much lower loss to the easy, confident-correct example
    assert loss_confident.item() < loss_unconfident.item()


def test_bce_loss_builds():
    loss_fn = build_loss("bce")
    logits = torch.randn(4, 8)
    targets = torch.randint(0, 2, (4, 8)).float()
    loss = loss_fn(logits, targets)
    assert loss.item() >= 0
