import sys
from zipfile import Path
sys.path.append(str(Path(__file__).parent))

def test_features_defined():
    import scripts.calculate_metrics as cm
    assert hasattr(cm, "FEATURES"), "FEATURES is not defined"


def test_features_structure():
    from scripts.calculate_metrics import FEATURES
    assert isinstance(FEATURES, list)
    assert len(FEATURES) > 0


def test_features_are_correct():
    from scripts.calculate_metrics import FEATURES

    expected = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]

    assert FEATURES == expected