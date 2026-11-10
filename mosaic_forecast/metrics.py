"""Forecast metrics used in the paper."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np


def _array(values: Iterable[float]) -> np.ndarray:
    return np.nan_to_num(np.asarray(list(values), dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return float(np.mean(np.abs(_array(y_true) - _array(y_pred))))


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    diff = np.clip(_array(y_true) - _array(y_pred), -1e6, 1e6)
    return float(np.sqrt(np.mean(np.square(diff))))


def smape(y_true: Iterable[float], y_pred: Iterable[float], eps: float = 1e-8) -> float:
    truth, pred = _array(y_true), _array(y_pred)
    return float(np.mean(2.0 * np.abs(pred - truth) / (np.abs(truth) + np.abs(pred) + eps)))


def row_mean(metric_fn: Callable, y_true: Sequence, y_pred: Sequence) -> float:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if truth.size == 0:
        return 0.0
    if truth.ndim == 1:
        return float(metric_fn(truth, pred))
    return float(np.mean([metric_fn(t_row, p_row) for t_row, p_row in zip(truth, pred)]))


def trend_accuracy(y_true: Sequence, y_pred: Sequence, last_observed: Sequence, threshold: float = 0.5) -> float:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    history = np.asarray(last_observed, dtype=float).reshape(-1, 1)
    if truth.ndim == 1:
        truth, pred = truth.reshape(-1, 1), pred.reshape(-1, 1)

    def encode(delta: np.ndarray) -> np.ndarray:
        out = np.zeros_like(delta, dtype=int)
        out[delta > threshold] = 1
        out[delta < -threshold] = -1
        return out

    return float(np.mean(encode(truth[:, :1] - history) == encode(pred[:, :1] - history)))


def summarize(count_true: np.ndarray, count_pred: np.ndarray, history_last: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(count_true)),
        "rmse": rmse(count_true.flatten(), count_pred.flatten()),
        "rmse_row": row_mean(rmse, count_true, count_pred),
        "mae": mae(count_true.flatten(), count_pred.flatten()),
        "mae_row": row_mean(mae, count_true, count_pred),
        "smape": smape(count_true.flatten(), count_pred.flatten()),
        "trend_accuracy": trend_accuracy(count_true, count_pred, history_last),
    }
