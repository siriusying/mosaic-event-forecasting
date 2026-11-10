import numpy as np

from mosaic_forecast.metrics import mae, rmse, row_mean, trend_accuracy


def test_point_metrics():
    assert rmse([0.0, 3.0], [0.0, 3.0]) == 0.0
    assert np.isclose(rmse([0.0, 0.0], [1.0, -1.0]), 1.0)
    assert np.isclose(mae([0.0, 2.0], [1.0, 0.0]), 1.5)
    assert np.isclose(row_mean(rmse, np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[1.0, 1.0], [1.0, 1.0]])), 0.5)


def test_trend_accuracy_uses_first_horizon():
    truth = np.array([[2.0, 0.0, 0.0]])
    pred = np.array([[1.5, 0.0, 0.0]])
    assert trend_accuracy(truth, pred, np.array([0.0])) == 1.0
