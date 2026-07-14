# Heart Disease MLOps — 2–3 minute video script

## 0:00–0:20 — Problem and repository

“This project predicts the probability of heart disease from the UCI Cleveland dataset. It is an educational MLOps implementation, not a clinical decision system. The repository separates data processing, training, API serving, tests, infrastructure, monitoring, and documentation.”

## 0:20–0:55 — Data and model workflow

“On the first run, the pipeline downloads the UCI data securely and normalizes the target into disease present or absent. The preprocessing pipeline uses a ColumnTransformer: numerical fields are median-imputed and standardized, while categorical fields are mode-imputed and one-hot encoded. I train Logistic Regression, Random Forest, XGBoost, and SVM models using five-fold GridSearchCV, then automatically select the best ROC-AUC model.”

## 0:55–1:25 — Experiment tracking

“Here in MLflow, every run records tuning parameters, cross-validation scores, final accuracy, precision, recall, F1, and ROC-AUC. The run also stores EDA, confusion matrix, ROC, precision-recall, and feature-importance images. The selected pipeline is registered as HeartDiseasePredictor and saved locally with joblib.”

## 1:25–1:55 — API and quality

“The FastAPI service loads the saved pipeline at startup. Swagger is available at `/docs`; `/predict` returns a prediction, probability, confidence score, and response time, while `/batch_predict` supports validated batches. The service emits structured JSON logs, request IDs, latency information, health checks, and Prometheus metrics. The project includes pytest coverage for preprocessing, training configuration, prediction, and API behavior.”

## 1:55–2:25 — Deployment and observability

“The Docker image runs the API as a non-root user on port 8000. Kubernetes manifests deploy two replicas with resource limits and probes behind a service and ingress. Prometheus scrapes `/metrics`, and the included Grafana dashboard presents request rate, p95 latency, and error rate. GitHub Actions lint, test, train, and build the image on every push or pull request.”

## 2:25–2:40 — Closing

“This demonstrates the full ML lifecycle from data acquisition to reproducible experimentation, tested serving, deployment, and observability. Before any clinical usage, it would require clinical validation, fairness analysis, security review, and regulatory approval.”

