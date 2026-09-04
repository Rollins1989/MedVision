"""
app/database.py
------------------
SQLAlchemy engine/session + the PredictionRecord model. Stores prediction
metadata for observability -- NOT patient data. This demo only ever
touches synthetic/public-dataset images; no real patient data should ever
be pointed at this table (see README).
"""
import os
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./medvision.db")
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    image_hash: Mapped[str] = mapped_column(String(64), index=True)  # sha256 of uploaded bytes, NOT the image itself
    model_name: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(32))
    top_finding: Mapped[str] = mapped_column(String(64))
    top_probability: Mapped[float] = mapped_column(Float)
    latency_ms: Mapped[float] = mapped_column(Float)
    findings_json: Mapped[str] = mapped_column(String)  # full per-label JSON blob
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
