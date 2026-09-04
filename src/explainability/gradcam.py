"""
src/explainability/gradcam.py
-------------------------------
Grad-CAM (Selvaraju et al., 2017) for the CNN backbones (ResNet50,
DenseNet121, EfficientNet-B0). Implemented directly with forward/backward
hooks rather than a third-party library, so it's inspectable end to end.

Not defined for ViT -- there's no single spatial conv feature map to hook,
and Grad-CAM's premise (weight each channel of a conv feature map by its
gradient) doesn't transfer directly to attention. Attention-rollout or
attention-map visualization would be the ViT-appropriate technique; out of
scope here (see README, item 9 of the upgrade list explicitly says the
goal is answering "does the model focus on relevant regions", not
collecting explainability algorithms).
"""
import numpy as np
import torch
import torch.nn.functional as F

from src.models.factory import get_target_layer_for_gradcam


class GradCAM:
    def __init__(self, model: torch.nn.Module, model_name: str):
        self.model = model
        self.model_name = model_name
        self.target_layer = get_target_layer_for_gradcam(model, model_name)
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor: torch.Tensor, label_idx: int) -> np.ndarray:
        """image_tensor: (1, 3, H, W). Returns a (H, W) heatmap in [0, 1],
        upsampled to the input resolution."""
        self.model.eval()
        image_tensor = image_tensor.clone().requires_grad_(True)

        logits = self.model(image_tensor)
        score = logits[0, label_idx]

        self.model.zero_grad()
        score.backward()

        # global-average-pool the gradients per channel -> channel importance weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, float(torch.sigmoid(score).item())


def overlay_heatmap(original_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """original_rgb: (H, W, 3) uint8. cam: (H, W) float in [0,1].
    Returns an (H, W, 3) uint8 overlay image using a simple red-heat
    colormap (no matplotlib dependency needed at inference time)."""
    heat = np.zeros((*cam.shape, 3), dtype=np.float32)
    heat[..., 0] = cam                     # red channel <- activation
    heat[..., 1] = np.clip(cam - 0.5, 0, 1) * 2  # a little yellow at peak activation
    heat = (heat * 255).astype(np.uint8)

    overlay = (original_rgb.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha)
    return np.clip(overlay, 0, 255).astype(np.uint8)
