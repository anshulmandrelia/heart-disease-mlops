import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.predict import predict_records
from src.preprocess import COLUMNS, build_preprocessor, split_features_target


@pytest.fixture(scope="module")
def fitted_model() -> Pipeline:
    rows = [[55 + i % 10, i % 2, i % 4, 120 + i, 200 + i, i % 2, i % 3, 140 + i,
             i % 2, float(i % 4), i % 3, i % 4, 1 + i % 3, i % 2] for i in range(100)]
    frame = pd.DataFrame(rows, columns=COLUMNS)
    x, y = split_features_target(frame)
    model = Pipeline([("preprocessor", build_preprocessor()), ("classifier", LogisticRegression(max_iter=1000))])
    return model.fit(x.head(100), y.head(100))


def test_prediction_contract(fitted_model: Pipeline) -> None:
    record = {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0,
              "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}
    result = predict_records(fitted_model, [record])[0]
    assert result["prediction"] in (0, 1)
    assert 0 <= result["probability"] <= 1
