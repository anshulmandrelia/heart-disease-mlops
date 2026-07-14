"""Dataset acquisition and sklearn preprocessing pipeline construction."""
from __future__ import annotations

import shutil
import ssl
import urllib.request
from pathlib import Path

import certifi
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import get_logger, project_path

LOGGER = get_logger(__name__)
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "target",
]
CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERICAL_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def download_dataset(destination: Path | str = "data/heart_disease.csv") -> Path:
    """Download, validate, and persist the UCI Cleveland data if it is absent."""
    output = project_path(str(destination))
    if output.exists():
        LOGGER.info("Using existing dataset: %s", output)
        return output
    temporary = output.with_suffix(".download")
    try:
        LOGGER.info("Downloading UCI Heart Disease dataset")
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(DATA_URL, context=ssl_context) as response:  # noqa: S310
            with temporary.open("wb") as file:
                shutil.copyfileobj(response, file)
        frame = pd.read_csv(temporary, names=COLUMNS, na_values="?")
        if frame.empty or list(frame.columns) != COLUMNS:
            raise ValueError("Downloaded file is not a valid UCI Cleveland dataset")
        frame["target"] = (pd.to_numeric(frame["target"], errors="coerce") > 0).astype(int)
        frame.to_csv(output, index=False)
        return output
    finally:
        if temporary.exists():
            temporary.unlink()


def load_dataset(path: Path | str = "data/heart_disease.csv") -> pd.DataFrame:
    """Load the local dataset, downloading it automatically when needed."""
    dataset_path = download_dataset(path)
    frame = pd.read_csv(dataset_path, na_values=["?", ""])
    for column in COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def build_preprocessor() -> ColumnTransformer:
    """Build leakage-safe feature transformation for numerical and categorical data."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERICAL_COLUMNS),
        ("categorical", categorical, CATEGORICAL_COLUMNS),
    ], remainder="drop", verbose_feature_names_out=False)


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix and binary target with schema validation."""
    missing = set(COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return frame.drop(columns="target"), frame["target"].astype(int)
