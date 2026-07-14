from fastapi.testclient import TestClient
import numpy as np

import api.app as application


class StubModel:
    def predict(self, frame):
        return [1] * len(frame)

    def predict_proba(self, frame):
        return np.array([[0.2, 0.8]] * len(frame))


def test_api_health_and_prediction(monkeypatch) -> None:
    monkeypatch.setattr(application, "load_model", lambda _path: StubModel())
    with TestClient(application.app) as client:
        assert client.get("/health").status_code == 200
        payload = {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0,
                   "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert response.json()["prediction"] == 1
