"""Train, compare, register, and persist heart-disease classifiers."""
from __future__ import annotations

import argparse
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.evaluate import create_eda_plots, create_feature_importance_plot, evaluate_model
from src.preprocess import build_preprocessor, load_dataset, split_features_target
from src.utils import get_logger, load_config, project_path

LOGGER = get_logger(__name__)


def model_candidates(random_state: int) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    """Return supported estimators and intentionally compact production search grids."""
    return {
        "logistic_regression": (LogisticRegression(max_iter=3000, random_state=random_state),
                                {"classifier__C": [0.1, 1.0, 10.0]}),
        "random_forest": (RandomForestClassifier(random_state=random_state, n_jobs=-1),
                          {"classifier__n_estimators": [200, 400], "classifier__max_depth": [None, 5, 10],
                           "classifier__min_samples_leaf": [1, 3]}),
        "xgboost": (XGBClassifier(random_state=random_state, eval_metric="logloss", n_jobs=1),
                    {"classifier__n_estimators": [100, 250], "classifier__max_depth": [3, 5],
                     "classifier__learning_rate": [0.03, 0.1]}),
        "svm": (SVC(probability=True, random_state=random_state),
                {"classifier__C": [0.1, 1.0, 10.0], "classifier__gamma": ["scale", "auto"]}),
    }


def train(config_path: str | None = None) -> dict[str, Any]:
    """Run EDA, 5-fold tuning, evaluation, MLflow registration, and model persistence."""
    config = load_config(config_path)
    random_state = int(config["project"]["random_state"])
    frame = load_dataset(config["data"]["raw_path"])
    x, y = split_features_target(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=float(config["project"]["test_size"]), stratify=y, random_state=random_state
    )
    plot_paths = create_eda_plots(frame)
    mlflow.set_tracking_uri(config["training"]["tracking_uri"])
    mlflow.set_experiment(config["training"]["experiment_name"])
    results: dict[str, float] = {}
    best_name, best_search = "", None
    with mlflow.start_run(run_name="model-selection") as run:
        mlflow.log_params({"test_size": config["project"]["test_size"], "cv_folds": config["training"]["cv_folds"]})
        for name, (classifier, grid) in model_candidates(random_state).items():
            pipeline = Pipeline([("preprocessor", build_preprocessor()), ("classifier", classifier)])
            search = GridSearchCV(pipeline, grid, cv=int(config["training"]["cv_folds"]),
                                  scoring=config["training"]["scoring"], n_jobs=-1, refit=True)
            search.fit(x_train, y_train)
            results[name] = float(search.best_score_)
            mlflow.log_metric(f"cv_roc_auc_{name}", search.best_score_)
            mlflow.log_params({f"{name}.{key}": value for key, value in search.best_params_.items()})
            if best_search is None or search.best_score_ > best_search.best_score_:
                best_name, best_search = name, search
        assert best_search is not None
        metrics, diagnostic_paths = evaluate_model(best_search.best_estimator_, x_test, y_test)
        importance_path = create_feature_importance_plot(best_search.best_estimator_, x_test, y_test)
        mlflow.log_metrics(metrics)
        mlflow.log_param("best_model", best_name)
        for path in plot_paths + diagnostic_paths + [importance_path]:
            mlflow.log_artifact(path, artifact_path="plots")
        signature = infer_signature(x_train, best_search.best_estimator_.predict(x_train))
        model_info = mlflow.sklearn.log_model(
            sk_model=best_search.best_estimator_, artifact_path="model", signature=signature,
            input_example=x_train.head(2), registered_model_name="HeartDiseasePredictor",
        )
        model_path = project_path(config["training"]["model_path"])
        joblib.dump(best_search.best_estimator_, model_path)
        summary = {
            "run_id": run.info.run_id, "model_uri": model_info.model_uri, "best_model": best_name,
            "cv_scores": results, "metrics": metrics, "model_path": str(model_path),
        }
        project_path("model/training_summary.json").write_text(pd.Series(summary).to_json(indent=2), encoding="utf-8")
        LOGGER.info("Training completed with %s", best_name)
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the heart disease prediction model")
    parser.add_argument("--config", default=None, help="Path to YAML configuration")
    print(train(parser.parse_args().config))
