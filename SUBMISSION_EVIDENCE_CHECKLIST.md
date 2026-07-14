# Final submission evidence checklist

The source code is complete. Capture the following authentic evidence before submission; do not substitute configuration files for runtime proof.

| Item | Required capture | Command or location |
|---|---|---|
| MLflow | **Captured and embedded in the final DOCX:** `screenshots/mlflow_experiment.png`, `screenshots/mlflow_run_metrics.png` | `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000` |
| API | **Captured and embedded in the final DOCX:** `screenshots/swagger_docs.png`; successful prediction verified against the running API | `uvicorn api.app:app --port 8000`, then `http://localhost:8000/docs` |
| Metrics | **Captured and embedded in the final DOCX:** `screenshots/prometheus_metrics.png` | `http://localhost:8000/metrics` |
| CI | **Captured and embedded in the final DOCX:** `screenshots/github_actions.png` | GitHub repository → Actions → CI |
| Docker | **Verified in GitHub Actions:** image build, container start, `/health`, and `/predict`; artifact `docker-smoke-test-evidence` in the [successful run](https://github.com/anshulmandrelia/heart-disease-mlops/actions/runs/29347657740) | CI artifact: `docker-smoke-test-evidence` |
| Kubernetes | **Verified in GitHub Actions:** kind cluster created, manifests applied, rollout succeeded, `/health` and `/predict` hit through the Service; artifact `kubernetes-deployment-evidence` (deployment/service/ingress output) in the [successful run](https://github.com/anshulmandrelia/heart-disease-mlops/actions/runs/29347657740) | CI artifact: `kubernetes-deployment-evidence` |
| Grafana / Prometheus | **Verified in GitHub Actions:** docker-compose stack (API + Prometheus + Grafana) started, dashboard imported via Grafana API, Prometheus query for request count returned data; artifact `monitoring-stack-evidence` in the [successful run](https://github.com/anshulmandrelia/heart-disease-mlops/actions/runs/29347657740) | CI artifact: `monitoring-stack-evidence` |
| Video | 2–3 minute walkthrough of all stages | Follow `VIDEO_SCRIPT.md` |

## Submission package

1. Repository is pushed to GitHub with a fully green CI pipeline (all 18 steps, including Docker, Kubernetes, and monitoring verification).
2. Download the three CI evidence artifacts above and add screenshots/output into `Heart_Disease_MLOps_Assignment_Report.docx`, then export to PDF if the portal requires it.
3. Record and upload/share the short video.
4. Verify all links and commands from a clean clone before submitting.
