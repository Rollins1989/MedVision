"""
src/training/losses.py
------------------------
Three loss functions for multi-label classification under class imbalance,
so the project can run the "does loss choice matter for minority-class
performance" experiment instead of just asserting it does.

- BCEWithLogitsLoss: the naive baseline. Every finding (common or rare)
  contributes equally to the loss, so the model can get most of its loss
  reduction "for free" by just being right about common negatives, and
  under-fit the rare positive findings.
- Weighted BCE: multiplies the positive-class term by pos_weight[c] =
  (negatives_c / positives_c) per label, so getting a rare-label positive
  wrong costs proportionally more.
- Focal Loss: down-weights *already easy, well-classified* examples
  (regardless of class) via the (1-p_t)^gamma term, forcing gradient to
  concentrate on hard examples -- which, empirically, correlates heavily
  with minority-class examples in imbalanced medical imaging data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_pos_weights(label_matrix: torch.Tensor, cap: float = 20.0) -> torch.Tensor:
    """label_matrix: (N, C) binary tensor. Returns per-class pos_weight for
    BCEWithLogitsLoss, capped so a near-zero-positive-count label doesn't
    produce an exploding weight."""
    n_pos = label_matrix.sum(dim=0).clamp(min=1)
    n_neg = label_matrix.shape[0] - n_pos
    weights = (n_neg / n_pos).clamp(max=cap)
    return weights


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)


class FocalLoss(nn.Module):
    """Multi-label focal loss (Lin et al., 2017), applied per-label then
    averaged. alpha balances positive/negative class weight; gamma controls
    how strongly easy examples are down-weighted (gamma=0 reduces to
    plain weighted BCE)."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * ce_loss
        return loss.mean()


def build_loss(name: str, label_matrix: torch.Tensor | None = None, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    elif name == "weighted_bce":
        assert label_matrix is not None, "weighted_bce needs the training label matrix to compute pos_weight"
        return WeightedBCELoss(compute_pos_weights(label_matrix))
    elif name == "focal":
        return FocalLoss(alpha=kwargs.get("alpha", 0.25), gamma=kwargs.get("gamma", 2.0))
    else:
        raise ValueError(f"Unknown loss '{name}'. Choose from: bce, weighted_bce, focal")
