"""Chronological and county-disjoint split definitions."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np
import pandas as pd


def ordered_months(months: Iterable[str]) -> list[str]:
    return sorted({str(month) for month in months})


def forward_split(months: Iterable[str], train_ratio: float = 0.7, val_ratio: float = 0.15) -> dict[str, list[str]]:
    ordered = ordered_months(months)
    if not ordered:
        return {"train": [], "val": [], "test": []}
    if len(ordered) == 1:
        return {"train": ordered, "val": [], "test": []}
    train_end = max(1, int(len(ordered) * train_ratio))
    val_end = min(max(train_end + 1, int(len(ordered) * (train_ratio + val_ratio))), len(ordered) - 1)
    return {"train": ordered[:train_end], "val": ordered[train_end:val_end], "test": ordered[val_end:]}


def assign_split(frame: pd.DataFrame, split: dict[str, list[str]], column: str = "forecast_origin") -> np.ndarray:
    mapping = {month: label for label, months in split.items() for month in months}
    return frame[column].astype(str).map(mapping).fillna("test").to_numpy()


def rolling_origins(months: Iterable[str], min_train_size: int = 6, horizon: int = 1) -> list[dict[str, list[str]]]:
    ordered = ordered_months(months)
    splits = []
    for train_end in range(min_train_size, len(ordered) - horizon):
        val_end = min(len(ordered) - horizon, train_end + horizon)
        splits.append(
            {
                "train": ordered[:train_end],
                "val": ordered[train_end:val_end],
                "test": ordered[val_end : val_end + horizon],
            }
        )
    return splits


def county_fold(county: str, n_folds: int = 5) -> int:
    digest = hashlib.sha256(str(county).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % n_folds


def county_folds(counties: Iterable[str], n_folds: int = 5) -> dict[str, list[str]]:
    folds: dict[str, list[str]] = {str(index): [] for index in range(n_folds)}
    for county in sorted({str(county) for county in counties}):
        folds[str(county_fold(county, n_folds))].append(county)
    return folds
