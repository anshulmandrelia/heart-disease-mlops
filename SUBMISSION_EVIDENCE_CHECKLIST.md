# Final submission evidence checklist

The source code is complete. Capture the following authentic evidence before submission; do not substitute configuration files for runtime proof.

| Item | Required capture | Command or location |
|---|---|---|
| MLflow | **Captured and embedded in the final DOCX:** `screenshots/mlflow_experiment.png`, `screenshots/mlflow_run_metrics.png` | `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000` |
| API | **Captured and embedded in the final DOCX:** `screenshots/swagger_docs.png`; successful prediction verified against the running API | `uvicorn api.app:app --port 8000`, then `http://localhost:8000/docs` |
| Metrics | **Captured and embedded in the final DOCX:** `screenshots/prometheus_metrics.png` | `http://localhost:8000/metrics` |
| CI | **Captured and embedded in the final DOCX:** `screenshots/github_actions.png` | GitHub repository → Actions → CI |
| Docker | **Verified in GitHub Actions run #5:** image build, container start, `/health`, and `/predict`; download `docker-smoke-test-evidence` from the [successful run](https://github.com/anshulmandrelia/heart-disease-mlops/actions/runs/29346351490) | CI artifact: `docker-smoke-test-evidence` |
| Kubernetes | deployed rollout plus deployment/service/ingress output | `kubectl apply -f kubernetes/` then `kubectl get deployment,service,ingress` |
| Grafana | request-rate, p95-latency, and error-rate panels | `docker compose -f docker/docker-compose.yml up --build` |
| Video | 2–3 minute walkthrough of all stages | Follow `VIDEO_SCRIPT.md` |

## Submission package

1. Push the repository to GitHub.
2. The evidence-filled `Heart_Disease_MLOps_Assignment_Report.docx` is ready. Add the remaining Docker/Kubernetes/Grafana proof, if available, then submit it as the required 10-page report.
3. Upload/share the short video.
4. Verify all links and commands from a clean clone before submitting.
