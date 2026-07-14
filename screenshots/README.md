# Screenshot evidence guide

This directory contains automatically generated data-science artifacts after `python -m src.train`:

- `eda_histograms.png`, `eda_boxplots.png`, `eda_correlation_heatmap.png`, `eda_pairplot.png`, `eda_class_distribution.png`, and `eda_outliers.png`
- `feature_importance.png`, `confusion_matrix.png`, `roc_curve.png`, and `precision_recall_curve.png`

Capture the following additional runtime evidence for an assignment submission:

1. Completed: `mlflow_experiment.png` and `mlflow_run_metrics.png` show the tracked experiment, completed run, and registered `HeartDiseasePredictor` model.
2. Completed: `swagger_docs.png` shows Swagger UI at `http://localhost:8000/docs`; the same local service returned a successful `POST /predict` during verification.
3. Completed: `prometheus_metrics.png` shows `heart_disease_prediction_requests_total` and the latency histogram after real prediction traffic.
4. Completed: `github_actions.png` shows successful CI workflow runs in the public repository.
5. Still required: Docker terminal output from a successful image build and running `/health` response.
6. Still required: Kubernetes output from `kubectl get deployment,service,ingress` after applying the manifests.
7. Still required: Grafana dashboard panels for request rate, p95 latency, and error rate.

Do not include patient-identifiable data in screenshots. The supplied request payload is synthetic.
