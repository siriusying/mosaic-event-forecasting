import numpy as np

from mosaic_forecast.train import select_endpoint


def test_endpoint_uses_validation_only():
    target = np.array([[0.0, 0.0, 1.0], [2.0, 2.0, 2.0]])
    val = {"full": np.array([[0.1, 0.2, 0.9], [2.1, 2.2, 2.0]]), "structured": np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])}
    test = {"full": np.zeros_like(target), "structured": np.ones_like(target)}
    selected, _, _, scores = select_endpoint(val, test, target)
    assert selected == "full_gate"
    assert scores["full_gate"] < scores["hurdle_endpoint"]

    shuffled = {"full": test["structured"].copy(), "structured": test["full"].copy()}
    assert select_endpoint(val, shuffled, target)[0] == selected


def test_endpoint_prefers_hurdle_when_validation_supports_it():
    target = np.array([[2.0, 2.0, 2.0]])
    val = {"full": np.array([[0.0, 0.0, 0.0]]), "structured": np.array([[2.0, 2.0, 2.0]])}
    test = {"full": np.zeros((1, 3)), "structured": np.zeros((1, 3))}
    assert select_endpoint(val, test, target)[0] == "hurdle_endpoint"
