"""
src/uncertainty/mc_dropout.py
--------------------------------
Monte Carlo Dropout (Gal & Ghahramani, 2016): keep dropout ACTIVE at
inference time and run N stochastic forward passes. The spread across
those N predictions approximates the model's epistemic uncertainty --
how much the prediction would change if we perturbed the network's
internal representation slightly. A softmax/sigmoid probability alone
conflates "confident and correct" with "confident and wrong"; MC Dropout
gives a second, independent signal.
"""
import numpy as np
import torch
import torch.nn as nn


def _enable_dropout(model: nn.Module):
    """Sets only Dropout layers to train mode, leaving BatchNorm etc. in
    eval mode (we still want BN running stats, just want dropout noise)."""
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, image_tensor: torch.Tensor, n_samples: int = 20) -> dict:
    """image_tensor: (1, 3, H, W). Returns per-label mean probability and
    uncertainty (std across MC samples) as numpy arrays of shape (C,)."""
    model.eval()
    _enable_dropout(model)  # re-enable dropout only, after eval() disabled everything

    samples = []
    for _ in range(n_samples):
        logits = model(image_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        samples.append(probs)

    samples = np.stack(samples, axis=0)  # (n_samples, C)
    mean_probs = samples.mean(axis=0)
    std_probs = samples.std(axis=0)  # uncertainty per label

    model.eval()  # restore full eval mode for any caller downstream
    return {"mean_probability": mean_probs, "uncertainty": std_probs, "samples": samples}


def uncertainty_tier(std: float, low_thresh: float = 0.05, high_thresh: float = 0.15) -> str:
    if std < low_thresh:
        return "Low"
    elif std < high_thresh:
        return "Medium"
    return "High"
