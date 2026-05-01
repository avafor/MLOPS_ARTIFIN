import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))


import pytest
from prediction_core import predict, validate_input
class DummyModel:
    def predict(self, X):
        return [0]


class ShapeCheckingDummyModel:
    def predict(self, X):
        assert len(X) == 1
        assert len(X[0]) == 4
        return [1]


def test_valid_prediction():
    sample = [5.1, 3.5, 1.4, 0.2]
    model = DummyModel()

    result = predict(sample, model)

    assert result == [0]


def test_invalid_feature_length():
    sample = [5.1, 3.5, 1.4]

    with pytest.raises(ValueError):
        validate_input(sample)


def test_non_numeric_input():
    sample = [5.1, "wrong", 1.4, 0.2]

    with pytest.raises(ValueError):
        validate_input(sample)


def test_model_input_shape():
    sample = [5.1, 3.5, 1.4, 0.2]
    model = ShapeCheckingDummyModel()

    result = predict(sample, model)

    assert result == [1]