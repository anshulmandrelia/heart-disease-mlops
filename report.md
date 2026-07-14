# Heart Disease Prediction using MLOps — Technical Report

## Introduction

This project operationalizes a binary classifier for the UCI Cleveland Heart Disease dataset. Its goal is to demonstrate reproducible machine-learning delivery, not medical diagnosis. UCI’s original `num` target has five classes; this implementation converts any non-zero value to disease present.

## Problem statement

Given routinely collected patient attributes (age, sex, chest-pain type, blood pressure, cholesterol, ECG outcomes, exercise indicators, and thallium test fields), estimate the likelihood of heart disease. The solution must be reproducible, observable, deployable, and safe to operate.

## EDA

The training workflow saves dataset histograms, numerical boxplots, a correlation heatmap, pairplot, class distribution, and a z-score outlier count chart in `screenshots/`. Missing-value treatment is intentional: the UCI `ca` and `thal` fields may contain `?`, represented as nulls before imputation. EDA artifacts are also attached to the MLflow run for auditability.

## Feature engineering

The feature contract is split into numerical and categorical columns. A single sklearn `ColumnTransformer` applies median imputation and standardization to numerical values, and most-frequent imputation plus unknown-safe one-hot encoding to categorical values. Keeping this in the fitted pipeline prevents train/serve skew and leakage.

## Model comparison

Logistic Regression, Random Forest, XGBoost, and probability-enabled SVM are each optimized using five-fold `GridSearchCV`, ROC-AUC scoring, and stratified holdout evaluation. Selection is automatic and based on the best cross-validation ROC-AUC. The chosen pipeline is evaluated on accuracy, precision, recall, F1, and ROC-AUC, with confusion-matrix, ROC, precision-recall, and permutation-importance graphics persisted.

## MLflow

MLflow uses a SQLite tracking store by default. Each run records configuration, model-specific search parameters, model comparison scores, final holdout metrics, all plots, an input example and signature, and the sklearn model format. The winning artifact is registered as `HeartDiseasePredictor`; joblib provides a direct low-latency serving artifact.

## Docker

The Docker image uses Python 3.11 slim, a non-root user, no pip cache, a health check, and only production-serving source folders plus the pre-trained model. The build intentionally follows training, ensuring the model artifact is present in the container.

## CI/CD

GitHub Actions installs pinned dependencies, lints source/API/tests, executes pytest, runs a full training job, and builds the Docker image. A non-zero test, lint, training, or image-build status fails the workflow.

## Deployment

FastAPI validates all input fields and returns predictions through documented Swagger endpoints. Kubernetes manifests define a two-replica deployment, resource requests/limits, readiness/liveness probes, a ClusterIP service, and NGINX ingress routing. Request IDs and JSON logs make requests traceable across systems.

## Monitoring

Prometheus receives `/metrics` data for total prediction requests (by endpoint and outcome) and latency histograms. The Grafana dashboard provides request rate, p95 latency, and error-rate panels. These operational measures should be paired with real-world data-drift and model-performance monitoring before any broader deployment.

## Conclusion

The repository demonstrates a complete lifecycle: automated acquisition, reproducible transformations, measured model selection, registry tracking, packaged serving, automated verification, scalable deployment, and runtime observability. It is a strong educational baseline, while clinical adoption would require validated data governance, fairness assessment, calibration analysis, security review, and regulatory approval.

## Assignment validation checklist

| Requirement | Evidence | Status |
|---|---|---|
| Automatic UCI dataset acquisition | `src/preprocess.py`; secure HTTPS download tested during training | Complete |
| EDA and saved plots | `screenshots/eda_*.png` | Complete |
| Pipeline and ColumnTransformer | `src/preprocess.py` | Complete |
| Four tuned models and five-fold CV | `src/train.py` | Complete |
| Metrics and evaluation plots | `screenshots/confusion_matrix.png`, ROC and PR plots | Complete |
| MLflow tracking and registry | SQLite tracking database; `HeartDiseasePredictor` version 1 registered | Complete |
| Joblib serving model | `model/heart_disease_pipeline.joblib` | Complete |
| FastAPI, Swagger, batch inference | `api/app.py`; live request verified | Complete |
| Unit tests and linting | `pytest`: 5 passed; `flake8`: passed | Complete |
| Docker and CI configuration | `Dockerfile`, Compose file, `.github/workflows/ci.yml` | Complete |
| Kubernetes and ingress | `kubernetes/` manifests | Complete |
| Prometheus and Grafana | `/metrics`, `monitoring/` configuration | Complete |
| Assignment evidence and video script | `screenshots/README.md`, `VIDEO_SCRIPT.md` | Complete |

Docker and Kubernetes manifests are ready for their target runtimes. This workstation does not have Docker or a Kubernetes cluster installed, so container build/run and cluster apply require execution in those environments.
