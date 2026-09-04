"""
tests/conftest.py
--------------------
Shared fixtures: a TestClient wired to an isolated in-memory SQLite DB
(never touches the real medvision.db), and paths to real sample images
from the synthetic dataset for upload tests.
"""
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

ROOT = Path(__file__).resolve().parent.parent

# Point the app at a real trained checkpoint before importing it.
os.environ.setdefault("MODEL_CHECKPOINT", str(ROOT / "models" / "densenet121_weighted_bce.pt"))

from app.database import Base, get_db  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def sample_image_path() -> Path:
    """A real, valid synthetic X-ray from the demo dataset."""
    import pandas as pd
    df = pd.read_csv(ROOT / "data" / "processed" / "labels_clean.csv")
    return ROOT / "data" / "raw" / "images" / df.iloc[0]["image_id"]


@pytest.fixture()
def blank_image_path() -> Path:
    return ROOT / "data" / "raw" / "images" / "BLANK_0.png"
