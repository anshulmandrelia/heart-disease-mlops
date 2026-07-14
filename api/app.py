"""Production FastAPI service for heart disease inference."""
from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.predict import load_model, predict_records
from src.utils import get_logger

LOGGER = get_logger(__name__)
REQUESTS = Counter("heart_disease_prediction_requests_total", "Prediction requests", ["endpoint", "status"])
LATENCY = Histogram("heart_disease_prediction_latency_seconds", "Prediction latency", ["endpoint"])


class PatientFeatures(BaseModel):
    """UCI Cleveland feature contract used by prediction endpoints."""
    model_config = ConfigDict(extra="forbid")
    age: float = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=0, le=3)
    trestbps: float = Field(ge=50, le=300)
    chol: float = Field(ge=50, le=700)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(ge=30, le=250)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0, le=10)
    slope: int = Field(ge=0, le=2)
    ca: int = Field(ge=0, le=4)
    thal: int = Field(ge=0, le=3)


class BatchRequest(BaseModel):
    """Bounded batch payload to protect serving capacity."""
    records: List[PatientFeatures] = Field(min_length=1, max_length=1000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the immutable inference model once at service startup."""
    app.state.model = load_model(os.getenv("MODEL_PATH", "model/heart_disease_pipeline.joblib"))
    LOGGER.info("Model loaded successfully")
    yield


app = FastAPI(title="Heart Disease Prediction API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def request_logging(request: Request, call_next: Any) -> Response:
    """Add correlation IDs and structured latency logs to every request."""
    started = time.perf_counter()
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    try:
        response = await call_next(request)
    except Exception as exc:
        LOGGER.exception("Unhandled request error", extra={"request_id": request_id, "error": str(exc)})
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
    response.headers["X-Request-ID"] = request_id
    LOGGER.info("HTTP request completed", extra={"request_id": request_id, "latency_ms": round((time.perf_counter()-started)*1000, 2)})
    return response


@app.get("/", tags=["service"])
def root() -> Dict[str, str]:
    """Return API metadata."""
    return {"service": "heart-disease-prediction", "version": app.version, "docs": "/docs"}


@app.get("/health", tags=["service"])
def health() -> Dict[str, str]:
    """Return readiness status for container orchestrators."""
    return {"status": "healthy", "model": "loaded" if getattr(app.state, "model", None) else "unavailable"}


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Expose Prometheus scrape metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _infer(
    records: List[PatientFeatures], endpoint: str
) -> List[Dict[str, Union[float, int, str]]]:
    started = time.perf_counter()
    try:
        predictions = predict_records(app.state.model, [record.model_dump() for record in records])
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        for item in predictions:
            item["response_time_ms"] = elapsed
        REQUESTS.labels(endpoint=endpoint, status="success").inc()
        LATENCY.labels(endpoint=endpoint).observe(elapsed / 1000)
        LOGGER.info("Prediction completed", extra={"prediction": predictions[0]["prediction"], "latency_ms": elapsed})
        return predictions
    except Exception as exc:
        REQUESTS.labels(endpoint=endpoint, status="error").inc()
        LOGGER.exception("Prediction failed", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


@app.post("/predict", tags=["prediction"])
def predict(record: PatientFeatures) -> Dict[str, Union[float, int, str]]:
    """Return one disease prediction with calibrated classifier probability."""
    return _infer([record], "/predict")[0]


@app.post("/batch_predict", tags=["prediction"])
def batch_predict(
    payload: BatchRequest,
) -> Dict[str, List[Dict[str, Union[float, int, str]]]]:
    """Return predictions for up to 1,000 patients in one request."""
    return {"predictions": _infer(payload.records, "/batch_predict")}
