from src.train import model_candidates


def test_all_required_models_are_configured() -> None:
    assert set(model_candidates(42)) == {"logistic_regression", "random_forest", "xgboost", "svm"}

