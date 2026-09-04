"""
app/services/inference.py
-----------------------------
Loads the trained checkpoint once at startup and exposes a single
`InferenceService` used by the API routes. Handles: image validation,
basic image-quality gating, preprocessing, prediction, MC-Dropout
uncertainty, temperature-scaled calibration, and (optionally) Grad-CAM.
"""
import base64
import hashlib
import io
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from src.explainability.gradcam import GradCAM, overlay_heatmap
from src.models.factory import build_model
from src.preprocessing.transforms import build_transforms
from src.uncertainty.mc_dropout import mc_dropout_predict, uncertainty_tier

logger = logging.getLogger("medvision.inference")

MODEL_VERSION = "1.0.0"
MIN_DIM = 32
MAX_DIM = 4096
BLANK_STD_THRESHOLD = 2.0


class ImageValidationError(Exception):
    """Raised when an uploaded image fails validation or quality gating."""


class InferenceService:
    def __init__(self, checkpoint_path: Path, temperature: float = 1.0, mc_samples: int = 15):
        logger.info("Loading model checkpoint: %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.model_name = ckpt["model_name"]
        self.label_names = ckpt["label_names"]
        self.image_size = ckpt["image_size"]
        self.val_roc_auc = ckpt.get("val_macro_roc_auc")
        self.temperature = temperature
        self.mc_samples = mc_samples

        self.model = build_model(self.model_name, num_labels=ckpt["num_labels"], pretrained=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.transform = build_transforms(self.image_size, train=False)
        self.gradcam = GradCAM(self.model, self.model_name) if self.model_name != "vit_b_32" else None

        logger.info("Model loaded: %s (val ROC-AUC=%s)", self.model_name, self.val_roc_auc)

    # -------------------------------------------------------------
    # Validation / quality gating
    # -------------------------------------------------------------
    def validate_and_load(self, image_bytes: bytes) -> Image.Image:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.load()
        except (UnidentifiedImageError, OSError, ValueError):
            raise ImageValidationError("File is not a valid, decodable image.")

        if img.mode not in ("L", "RGB", "RGBA"):
            raise ImageValidationError(f"Unsupported image mode '{img.mode}'.")

        w, h = img.size
        if w < MIN_DIM or h < MIN_DIM:
            raise ImageValidationError(f"Image too small ({w}x{h}); minimum is {MIN_DIM}x{MIN_DIM}.")
        if w > MAX_DIM or h > MAX_DIM:
            raise ImageValidationError(f"Image too large ({w}x{h}); maximum is {MAX_DIM}x{MAX_DIM}.")

        aspect = max(w, h) / max(1, min(w, h))
        if aspect > 3.0:
            raise ImageValidationError(f"Unexpected aspect ratio ({w}x{h}); doesn't look like a chest X-ray.")

        arr = np.array(img.convert("L"), dtype=np.float32)
        if arr.std() < BLANK_STD_THRESHOLD:
            raise ImageValidationError("Image appears blank (near-zero pixel variance).")

        return img.convert("L")

    # -------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------
    def predict(self, image_bytes: bytes, with_explanation_for: str | None = None) -> dict:
        start = time.perf_counter()

        img = self.validate_and_load(image_bytes)
        tensor = self.transform(img).unsqueeze(0)

        mc_result = mc_dropout_predict(self.model, tensor, n_samples=self.mc_samples)
        mean_probs = mc_result["mean_probability"]
        uncertainties = mc_result["uncertainty"]

        with torch.no_grad():
            raw_logits = self.model(tensor)
            calibrated_probs = torch.sigmoid(raw_logits / self.temperature).numpy()[0]

        findings = []
        for i, label in enumerate(self.label_names):
            findings.append({
                "label": label,
                "probability": float(mean_probs[i]),
                "uncertainty": float(uncertainties[i]),
                "uncertainty_tier": uncertainty_tier(float(uncertainties[i])),
                "calibrated_probability": float(calibrated_probs[i]),
            })
        findings.sort(key=lambda f: f["probability"], reverse=True)
        top = findings[0]

        explanation = {"method": "Grad-CAM", "available": False, "target_label": None, "heatmap_png_base64": None}
        target_label = with_explanation_for or top["label"]
        if self.gradcam is not None and target_label in self.label_names:
            label_idx = self.label_names.index(target_label)
            cam, _ = self.gradcam.generate(tensor, label_idx)
            original_rgb = np.array(img.convert("RGB").resize((self.image_size, self.image_size)))
            overlay = overlay_heatmap(original_rgb, cam)
            buf = io.BytesIO()
            Image.fromarray(overlay).save(buf, format="PNG")
            explanation = {
                "method": "Grad-CAM", "available": True, "target_label": target_label,
                "heatmap_png_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
            }

        latency_ms = (time.perf_counter() - start) * 1000
        image_hash = hashlib.sha256(image_bytes).hexdigest()

        return {
            "top_finding": top["label"],
            "top_probability": top["probability"],
            "findings": findings,
            "model_name": self.model_name,
            "model_version": MODEL_VERSION,
            "latency_ms": latency_ms,
            "explanation": explanation,
            "image_hash": image_hash,
        }
