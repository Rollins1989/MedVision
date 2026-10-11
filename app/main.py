"""FastAPI routes for the MedVision chest X-ray inference service.

The model is loaded once during application startup. This API is intended for
research and educational use, not for clinical diagnosis.
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import Base, PredictionRecord, engine, get_db
from app.schemas.prediction import (
    ExplanationInfo,
    FindingProbability,
    HealthResponse,
    ModelInfoResponse,
    PredictionRecordOut,
    PredictionResponse,
)
from app.services.inference import ImageValidationError, InferenceService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("medvision")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = ROOT / "models" / "densenet121_weighted_bce.pt"
CHECKPOINT_PATH = Path(os.getenv("MODEL_CHECKPOINT", str(DEFAULT_CHECKPOINT)))

_service: InferenceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prepare the database and load the model before serving requests."""
    global _service

    Base.metadata.create_all(bind=engine)

    if CHECKPOINT_PATH.exists():
        _service = InferenceService(CHECKPOINT_PATH)
        logger.info("Model loaded from %s", CHECKPOINT_PATH)
    else:
        logger.warning(
            "No checkpoint found at %s; prediction endpoints will return 503.",
            CHECKPOINT_PATH,
        )

    yield


app = FastAPI(
    title="MedVision API",
    description=(
        "An explainable medical-imaging research prototype for multi-label "
        "chest X-ray analysis. It is not intended for clinical diagnosis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# The bundled frontend is served by this API in local/demo deployments.
# Restrict allowed origins before exposing a multi-user deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["frontend"], include_in_schema=False)
def serve_frontend() -> FileResponse:
    """Serve the bundled web interface."""
    return FileResponse(ROOT / "frontend" / "index.html")


def get_service() -> InferenceService:
    """Return the loaded inference service or report that it is unavailable."""
    if _service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )
    return _service


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=_service is not None)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
def model_info(
    service: InferenceService = Depends(get_service),
) -> ModelInfoResponse:
    metrics_path = ROOT / "models" / f"{CHECKPOINT_PATH.stem}_metrics.json"
    test_roc_auc = None
    test_pr_auc = None
    loss_name = None

    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8") as metrics_file:
            metrics: dict[str, Any] = json.load(metrics_file)

        test_metrics = metrics["test_metrics"]
        test_roc_auc = test_metrics["macro_roc_auc"]
        test_pr_auc = test_metrics["macro_pr_auc"]
        loss_name = metrics["loss"]

    return ModelInfoResponse(
        model_name=service.model_name,
        model_version="1.0.0",
        labels=service.label_names,
        image_size=service.image_size,
        test_macro_roc_auc=test_roc_auc,
        test_macro_pr_auc=test_pr_auc,
        trained_with_loss=loss_name,
    )


def _persist_and_respond(
    result: dict[str, Any], db: Session
) -> PredictionResponse:
    """Store an inference result and convert it to the public response schema."""
    record = PredictionRecord(
        image_hash=result["image_hash"],
        model_name=result["model_name"],
        model_version=result["model_version"],
        top_finding=result["top_finding"],
        top_probability=result["top_probability"],
        latency_ms=result["latency_ms"],
        findings_json=json.dumps(result["findings"]),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    logger.info(
        "Prediction id=%s top_finding=%s prob=%.3f latency=%.1fms",
        record.id,
        result["top_finding"],
        result["top_probability"],
        result["latency_ms"],
    )

    return PredictionResponse(
        prediction_id=record.id,
        top_finding=result["top_finding"],
        top_probability=result["top_probability"],
        findings=[
            FindingProbability(**finding) for finding in result["findings"]
        ],
        model_name=result["model_name"],
        model_version=result["model_version"],
        latency_ms=result["latency_ms"],
        explanation=ExplanationInfo(**result["explanation"]),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(
    file: UploadFile = File(...),
    service: InferenceService = Depends(get_service),
    db: Session = Depends(get_db),
) -> PredictionResponse:
    image_bytes = await file.read()

    try:
        result = service.predict(image_bytes)
    except ImageValidationError as exc:
        logger.warning(
            "Image validation failed for upload %r: %s", file.filename, exc
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return _persist_and_respond(result, db)


@app.post("/explain", response_model=PredictionResponse, tags=["inference"])
async def explain(
    file: UploadFile = File(...),
    label: str | None = Query(
        default=None,
        description=(
            "Finding to explain with Grad-CAM. Defaults to the top prediction."
        ),
    ),
    service: InferenceService = Depends(get_service),
    db: Session = Depends(get_db),
) -> PredictionResponse:
    image_bytes = await file.read()

    if label is not None and label not in service.label_names:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown label {label!r}. Choose from: {service.label_names}",
        )

    try:
        result = service.predict(image_bytes, with_explanation_for=label)
    except ImageValidationError as exc:
        logger.warning(
            "Image validation failed for upload %r: %s", file.filename, exc
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return _persist_and_respond(result, db)


@app.get(
    "/prediction/{prediction_id}",
    response_model=PredictionRecordOut,
    tags=["inference"],
)
def get_prediction(
    prediction_id: int,
    db: Session = Depends(get_db),
) -> PredictionRecordOut:
    record = (
        db.query(PredictionRecord)
        .filter(PredictionRecord.id == prediction_id)
        .first()
    )
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found",
        )
    return record
