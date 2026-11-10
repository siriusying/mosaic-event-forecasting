"""Prequential split-conformal intervals for rolling-origin forecasts."""

from __future__ import annotations

import numpy as np


def prequential_intervals(
    origins: list[str],
    y_pred: np.ndarray,
    y_true: np.ndarray,
    alpha: float = 0.1,
    min_calibration: int = 20,
) -> dict[str, object]:
    """Build a band for each origin from residuals of earlier origins.

    Each horizon keeps its own residual reservoir, matching the interval rule
    used in the paper. Rows from the same origin never enter the calibration
    set used for that origin.
    """

    order = sorted(set(origins), key=str)
    reservoir: list[list[float]] = [[] for _ in range(y_pred.shape[1])]
    lower = np.full(y_pred.shape, np.nan)
    upper = np.full(y_pred.shape, np.nan)
    evaluated = np.zeros(y_pred.shape, dtype=bool)
    for origin in order:
        mask = np.asarray([value == origin for value in origins])
        for horizon in range(y_pred.shape[1]):
            scores = reservoir[horizon]
            if len(scores) >= min_calibration:
                ordered_scores = np.sort(np.asarray(scores, dtype=float))
                rank = min(1.0, np.ceil((len(ordered_scores) + 1) * (1.0 - alpha)) / len(ordered_scores))
                radius = float(np.quantile(ordered_scores, rank, method="higher"))
                lower[mask, horizon] = y_pred[mask, horizon] - radius
                upper[mask, horizon] = y_pred[mask, horizon] + radius
                evaluated[mask, horizon] = True
            scores.extend(np.abs(y_pred[mask, horizon] - y_true[mask, horizon]).tolist())
    covered = (y_true >= lower) & (y_true <= upper) & evaluated
    return {
        "lower": lower,
        "upper": upper,
        "evaluated": evaluated,
        "coverage": float(np.mean(covered[evaluated])) if evaluated.any() else None,
        "mean_width": float(np.mean((upper - lower)[evaluated])) if evaluated.any() else None,
        "alpha": alpha,
        "min_calibration": min_calibration,
        "n_evaluated": int(evaluated.sum()),
    }
