import pandas as pd

from src.preprocess import COLUMNS, build_preprocessor, split_features_target


def test_preprocessor_transforms_missing_values() -> None:
    frame = pd.DataFrame([[55, 1, 2, 130, 250, 0, 1, 150, 0, 1.2, 1, None, 3, 1]], columns=COLUMNS)
    x, y = split_features_target(frame)
    transformed = build_preprocessor().fit_transform(x, y)
    assert transformed.shape[0] == 1
    assert transformed.shape[1] >= 13


def test_target_split_is_binary() -> None:
    frame = pd.DataFrame([[50, 0, 1, 120, 200, 0, 0, 160, 0, 0, 1, 0, 2, 0]], columns=COLUMNS)
    _, y = split_features_target(frame)
    assert y.tolist() == [0]
