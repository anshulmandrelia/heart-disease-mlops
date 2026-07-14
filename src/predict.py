"""Reusable batch and single-record inference utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.preprocess import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS

FEATURE_COLUMNS = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS


def load_model(model_path: Path | str = "model/heart_disease_pipeline.joblib") -> Any:
    """Load a joblib pipeline, raising a clear error when training has not run."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run `python -m src.train` first.")
    return joblib.load(path)


def predict_records(model: Any, records: list[dict[str, float | int]]) -> list[dict[str, float | int | str]]:
    """Predict validated records and return labels, probabilities, and confidence values."""
    frame = pd.DataFrame(records)
    missing = set(FEATURE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Records are missing feature fields: {sorted(missing)}")
    frame = frame[FEATURE_COLUMNS]
    probabilities = model.predict_proba(frame)[:, 1]
    predictions = model.predict(frame)
    return [{
        "prediction": int(prediction),
        "prediction_label": "heart_disease" if prediction else "no_heart_disease",
        "probability": float(probability),
        "confidence_score": float(max(probability, 1 - probability)),
    } for prediction, probability in zip(predictions, probabilities)]

