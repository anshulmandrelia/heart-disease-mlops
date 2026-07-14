# Screenshot evidence guide

This directory contains automatically generated data-science artifacts after `python -m src.train`:

- `eda_histograms.png`, `eda_boxplots.png`, `eda_correlation_heatmap.png`, `eda_pairplot.png`, `eda_class_distribution.png`, and `eda_outliers.png`
- `feature_importance.png`, `confusion_matrix.png`, `roc_curve.png`, and `precision_recall_curve.png`

Capture the following additional runtime evidence for an assignment submission:

1. MLflow experiment page showing the selected run, metrics, parameters, registered `HeartDiseasePredictor` model, and plots.
2. Swagger UI at `http://localhost:8000/docs`, including a successful `POST /predict` response.
3. `http://localhost:8000/metrics` showing the `heart_disease_prediction_requests_total` and latency histogram metrics.
4. Docker terminal output from a successful image build and running `/health` response.
5. Kubernetes output from `kubectl get deployment,service,ingress` after applying the manifests.
6. Grafana dashboard panels for request rate, p95 latency, and error rate.

Do not include patient-identifiable data in screenshots. The supplied request payload is synthetic.

