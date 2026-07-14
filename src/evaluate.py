"""Evaluation and EDA plot generation functions."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.preprocess import NUMERICAL_COLUMNS

sns.set_theme(style="whitegrid")


def _save(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return str(path)


def create_eda_plots(frame: pd.DataFrame, output_dir: Path | str = "screenshots") -> list[str]:
    """Create and save a comprehensive deterministic EDA chart collection."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    frame[NUMERICAL_COLUMNS].hist(figsize=(12, 8), bins=20)
    paths.append(_save(out / "eda_histograms.png"))
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=frame[NUMERICAL_COLUMNS])
    plt.xticks(rotation=30)
    paths.append(_save(out / "eda_boxplots.png"))
    plt.figure(figsize=(12, 9))
    sns.heatmap(frame.corr(numeric_only=True), cmap="coolwarm", center=0)
    paths.append(_save(out / "eda_correlation_heatmap.png"))
    pair_columns = NUMERICAL_COLUMNS + ["target"]
    grid = sns.pairplot(frame[pair_columns].dropna(), hue="target", corner=True, diag_kind="hist")
    grid.figure.savefig(out / "eda_pairplot.png", dpi=130, bbox_inches="tight")
    plt.close("all")
    paths.append(str(out / "eda_pairplot.png"))
    plt.figure(figsize=(5, 4))
    sns.countplot(data=frame, x="target", hue="target", legend=False)
    plt.title("Target class distribution")
    paths.append(_save(out / "eda_class_distribution.png"))
    zscore = ((frame[NUMERICAL_COLUMNS] - frame[NUMERICAL_COLUMNS].mean()) / frame[NUMERICAL_COLUMNS].std())
    outliers = (zscore.abs() > 3).sum().sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=outliers.index, y=outliers.values)
    plt.xticks(rotation=30)
    plt.title("Outliers (absolute z-score > 3)")
    paths.append(_save(out / "eda_outliers.png"))
    return paths


def evaluate_model(
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path | str = "screenshots",
) -> tuple[dict[str, float], list[str]]:
    """Calculate classification metrics and save diagnostic plots."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    paths: list[str] = []
    _, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, predictions)).plot(ax=ax)
    paths.append(_save(out / "confusion_matrix.png"))
    _, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_test, probabilities, ax=ax)
    paths.append(_save(out / "roc_curve.png"))
    _, ax = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_predictions(y_test, probabilities, ax=ax)
    paths.append(_save(out / "precision_recall_curve.png"))
    return metrics, paths


def create_feature_importance_plot(
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path | str = "screenshots",
) -> str:
    """Save model-agnostic permutation feature importance for the selected pipeline."""
    result = permutation_importance(
        model, x_test, y_test, n_repeats=15, random_state=42, scoring="roc_auc"
    )
    importance = pd.Series(result.importances_mean, index=x_test.columns).sort_values()
    plt.figure(figsize=(8, 5))
    importance.plot.barh()
    plt.title("Permutation feature importance (ROC-AUC)")
    return _save(Path(output_dir) / "feature_importance.png")
