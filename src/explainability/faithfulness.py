"""
src/explainability/faithfulness.py
-------------------------------------
Sanity check for Grad-CAM: does masking the region Grad-CAM says is
important actually change the model's prediction? A heatmap that "looks"
medically plausible but doesn't causally affect the prediction is not
trustworthy -- this is a perturbation-based faithfulness test (in the
spirit of RISE / deletion metrics from the explainability literature).

Procedure per image:
  1. Get baseline prediction P(label) and Grad-CAM heatmap.
  2. Mask the top-k% highest-activation pixels (replace with the image's
     mean value) -> re-run inference -> P_masked(label).
  3. Mask a random region of the same size (control) -> P_random(label).
  4. Faithfulness score = (P(label) - P_masked(label)) -- how much masking
     the "important" region actually drops the prediction. A faithful
     explanation should drop the prediction MORE than masking a random
     region of the same size does.
"""
from dataclasses import dataclass

import numpy as np
import torch

from src.explainability.gradcam import GradCAM


@dataclass
class FaithfulnessResult:
    label: str
    baseline_prob: float
    masked_top_region_prob: float
    masked_random_region_prob: float
    drop_from_top_region: float
    drop_from_random_region: float
    faithfulness_margin: float  # drop_from_top_region - drop_from_random_region; >0 means explanation is meaningfully more informative than chance


def _mask_top_k_percent(image_tensor: torch.Tensor, cam: np.ndarray, k_percent: float, fill_value: float) -> torch.Tensor:
    flat = cam.flatten()
    threshold = np.quantile(flat, 1 - k_percent)
    mask = (cam >= threshold).astype(np.float32)
    mask_t = torch.tensor(mask, device=image_tensor.device).unsqueeze(0).unsqueeze(0)
    masked = image_tensor.clone()
    masked = masked * (1 - mask_t) + fill_value * mask_t
    return masked


def _mask_random_region(image_tensor: torch.Tensor, k_percent: float, fill_value: float, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    h, w = image_tensor.shape[-2:]
    mask = np.zeros((h, w), dtype=np.float32)
    region_area = int(h * w * k_percent)
    region_side = int(np.sqrt(region_area))
    y0 = rng.integers(0, max(1, h - region_side))
    x0 = rng.integers(0, max(1, w - region_side))
    mask[y0:y0 + region_side, x0:x0 + region_side] = 1.0
    mask_t = torch.tensor(mask, device=image_tensor.device).unsqueeze(0).unsqueeze(0)
    masked = image_tensor.clone()
    masked = masked * (1 - mask_t) + fill_value * mask_t
    return masked


def evaluate_faithfulness(
    model, model_name: str, image_tensor: torch.Tensor, label_idx: int, label_name: str,
    k_percent: float = 0.2, seed: int = 0,
) -> FaithfulnessResult:
    """image_tensor: (1, 3, H, W), already normalized."""
    gradcam = GradCAM(model, model_name)
    cam, baseline_prob = gradcam.generate(image_tensor, label_idx)

    fill_value = float(image_tensor.mean().item())

    masked_top = _mask_top_k_percent(image_tensor, cam, k_percent, fill_value)
    masked_random = _mask_random_region(image_tensor, k_percent, fill_value, seed)

    model.eval()
    with torch.no_grad():
        prob_top = torch.sigmoid(model(masked_top))[0, label_idx].item()
        prob_random = torch.sigmoid(model(masked_random))[0, label_idx].item()

    drop_top = baseline_prob - prob_top
    drop_random = baseline_prob - prob_random

    return FaithfulnessResult(
        label=label_name, baseline_prob=round(baseline_prob, 4),
        masked_top_region_prob=round(prob_top, 4), masked_random_region_prob=round(prob_random, 4),
        drop_from_top_region=round(drop_top, 4), drop_from_random_region=round(drop_random, 4),
        faithfulness_margin=round(drop_top - drop_random, 4),
    )
