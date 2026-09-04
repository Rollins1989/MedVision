"""
app/main.py
--------------
MedVision FastAPI application.

    GET  /health
    GET  /model-info
    POST /predict
    POST /explain
    GET  /prediction/{id}

Loads the trained checkpoint once at startup (see app/services/inference.py)
and logs every prediction + every validation/processing failure with
Python's `logging` module -- exactly what you'd tail in production to
answer "why did this request fail" or "what's our prediction distribution
looking like today."
"""
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("medvision")

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = Path(
    __import__("os").getenv("MODEL_CHECKPOINT", str(ROOT / "models" / "densenet121_weighted_bce.pt"))
)

_service: InferenceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    Base.metadata.create_all(bind=engine)
    if CHECKPOINT_PATH.exists():
        _service = InferenceService(CHECKPOINT_PATH)
        logger.info("MedVision API started, model loaded from %s", CHECKPOINT_PATH)
    else:
        logger.warning("No checkpoint found at %s -- API will start but /predict will 503.", CHECKPOINT_PATH)
    yield


app = FastAPI(
    title="MedVision API",
    description=(
        "An end-to-end explainable medical imaging AI system for multi-label chest X-ray "
        "analysis. Research/educational prototype -- NOT for clinical diagnosis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Permissive CORS for this demo (single-user local/dev deployment serving
# its own bundled frontend). A real multi-tenant deployment should scope
# this to specific origins.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/", tags=["frontend"], include_in_schema=False)
def serve_frontend():
    return FileResponse(ROOT / "frontend" / "index.html")


def get_service() -> InferenceService:
    if _service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded.")
    return _service


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health_check():
    return HealthResponse(status="ok", model_loaded=_service is not None)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
def model_info(service: InferenceService = Depends(get_service)):
    metrics_path = ROOT / "models" / f"{CHECKPOINT_PATH.stem}_metrics.json"
    test_roc_auc, test_pr_auc, loss_name = None, None, None
    if metrics_path.exists():
        with open(metrics_path) as f:
            data = json.load(f)
        test_roc_auc = data["test_metrics"]["macro_roc_auc"]
        test_pr_auc = data["test_metrics"]["macro_pr_auc"]
        loss_name = data["loss"]

    return ModelInfoResponse(
        model_name=service.model_name, model_version="1.0.0", labels=service.label_names,
        image_size=service.image_size, test_macro_roc_auc=test_roc_auc,
        test_macro_pr_auc=test_pr_auc, trained_with_loss=loss_name,
    )


def _persist_and_respond(result: dict, db: Session) -> PredictionResponse:
    record = PredictionRecord(
        image_hash=result["image_hash"], model_name=result["model_name"],
        model_version=result["model_version"], top_finding=result["top_finding"],
        top_probability=result["top_probability"], latency_ms=result["latency_ms"],
        findings_json=json.dumps(result["findings"]),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    logger.info(
        "Prediction id=%s top_finding=%s prob=%.3f latency=%.1fms",
        record.id, result["top_finding"], result["top_probability"], result["latency_ms"],
    )

    return PredictionResponse(
        prediction_id=record.id, top_finding=result["top_finding"], top_probability=result["top_probability"],
        findings=[FindingProbability(**f) for f in result["findings"]],
        model_name=result["model_name"], model_version=result["model_version"],
        latency_ms=result["latency_ms"], explanation=ExplanationInfo(**result["explanation"]),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(
    file: UploadFile = File(...),
    service: InferenceService = Depends(get_service),
    db: Session = Depends(get_db),
):
    image_bytes = await file.read()
    try:
        result = service.predict(image_bytes)
    except ImageValidationError as e:
        logger.warning("Image validation failed for upload '%s': %s", file.filename, e)
        raise HTTPException(status_code=422, detail=str(e))

    return _persist_and_respond(result, db)


@app.post("/explain", response_model=PredictionResponse, tags=["inference"])
async def explain(
    file: UploadFile = File(...),
    label: str = Query(default=None, description="Which finding to generate Grad-CAM for (defaults to the top prediction)"),
    service: InferenceService = Depends(get_service),
    db: Session = Depends(get_db),
):
    image_bytes = await file.read()
    if label is not None and label not in service.label_names:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown label '{label}'. Choose from: {service.label_names}",
        )
    try:
        result = service.predict(image_bytes, with_explanation_for=label)
    except ImageValidationError as e:
        logger.warning("Image validation failed for upload '%s': %s", file.filename, e)
        raise HTTPException(status_code=422, detail=str(e))

    return _persist_and_respond(result, db)


@app.get("/prediction/{prediction_id}", response_model=PredictionRecordOut, tags=["inference"])
def get_prediction(prediction_id: int, db: Session = Depends(get_db)):
    record = db.query(PredictionRecord).filter(PredictionRecord.id == prediction_id).first()
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")
    return record
