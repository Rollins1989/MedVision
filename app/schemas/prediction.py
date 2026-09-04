"""
app/schemas/prediction.py
---------------------------
Pydantic request/response models for the inference API.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class FindingProbability(BaseModel):
    label: str
    probability: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, description="MC-Dropout std across stochastic forward passes")
    uncertainty_tier: str  # Low / Medium / High
    calibrated_probability: float = Field(ge=0.0, le=1.0)


class ExplanationInfo(BaseModel):
    method: str = "Grad-CAM"
    available: bool
    target_label: Optional[str] = None
    heatmap_png_base64: Optional[str] = Field(
        default=None, description="Base64-encoded PNG of the original image with the Grad-CAM heatmap overlaid"
    )


class PredictionResponse(BaseModel):
    prediction_id: int
    top_finding: str
    top_probability: float
    findings: list[FindingProbability]
    model_name: str
    model_version: str
    latency_ms: float
    explanation: ExplanationInfo
    disclaimer: str = (
        "This system is a research/educational decision-support prototype "
        "and is not intended for clinical diagnosis."
    )


class PredictionRecordOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    image_hash: str
    model_version: str
    top_finding: str
    top_probability: float
    latency_ms: float
    created_at: datetime


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    labels: list[str]
    image_size: int
    test_macro_roc_auc: Optional[float] = None
    test_macro_pr_auc: Optional[float] = None
    trained_with_loss: Optional[str] = None
    disclaimer: str = (
        "This system is a research/educational decision-support prototype "
        "and is not intended for clinical diagnosis."
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
