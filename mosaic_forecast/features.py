"""Feature selection, standardisation, priors and gate evidence."""

from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .metrics import rmse, row_mean

UNSAFE_COUNTY_COLUMNS = {"first_year_month", "last_year_month", "total_case_count_county"}

FEATURE_MODES = (
    "full",
    "leakage_safe",
    "leakage_safe_county_only",
    "leakage_safe_geosot",
    "history_only",
    "temporal_month_only",
    "spatial_county_only",
)


def resolve_feature_columns(feature_columns: Sequence[str], mode: str = "full") -> list[str]:
    columns = list(feature_columns)
    if mode == "full":
        return columns
    if mode.startswith("leakage_safe"):
        safe = [name for name in columns if name not in UNSAFE_COUNTY_COLUMNS and not name.endswith("_county")]
        if mode == "leakage_safe":
            return safe
        if mode == "leakage_safe_county_only":
            return [
                name
                for name in safe
                if not (name.startswith("fine_level_") or name.startswith("fine_location_parse_rate") or name.startswith("ust_"))
            ]
        return [name for name in safe if not (name.startswith("fine_location_parse_rate") or name.startswith("ust_"))]
    if mode == "history_only":
        prefixes = ("lag_total_case_count_", "rolling_mean_", "rolling_std_", "growth_diff_")
        keep = {"month_sin", "month_cos"}
        return [name for name in columns if name in keep or name.startswith(prefixes)]
    if mode == "temporal_month_only":
        return [
            name
            for name in columns
            if not (name.startswith("bucket_count_") or name.startswith("bucket_ratio_") or name.startswith("time_valid_rate") or name.startswith("ust_"))
        ]
    if mode == "spatial_county_only":
        return [
            name
            for name in columns
            if not (name.startswith("fine_level_") or name.startswith("fine_location_parse_rate") or name.startswith("ust_"))
        ]
    raise ValueError(f"unknown feature mode: {mode}")


def _vec(value: Any) -> np.ndarray:
    return np.asarray(json.loads(value) if isinstance(value, str) else value, dtype=np.float32)


def gate_evidence(history_counts: np.ndarray, feature_matrix: np.ndarray, feature_columns: Sequence[str]) -> np.ndarray:
    """Return support, quality and stability for each forecast origin."""

    history = np.nan_to_num(history_counts.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    support = np.mean(history > 0.0, axis=1)
    width = min(3, history.shape[1])
    recent_mean = history[:, -width:].mean(axis=1) if width else np.zeros(len(history))
    drift = np.abs(history[:, -1] - recent_mean) / (history.std(axis=1) + 1.0)
    stability = np.exp(-np.clip(drift, 0.0, 8.0))
    quality_indices = [
        index for index, name in enumerate(feature_columns) if "valid_rate" in str(name).lower() or "parse_rate" in str(name).lower()
    ]
    if quality_indices:
        values = np.nan_to_num(feature_matrix[:, :, quality_indices], nan=0.0, posinf=1.0, neginf=0.0)
        quality = np.clip(values, 0.0, 1.0).mean(axis=(1, 2))
    else:
        quality = np.ones(len(history), dtype=np.float32)
    return np.stack([support, quality, stability], axis=1).astype(np.float32)


def moving_average_prior(history_counts: np.ndarray, horizons: int, window: int = 3) -> np.ndarray:
    if history_counts.size == 0:
        return np.zeros((0, horizons), dtype=np.float32)
    width = min(window, history_counts.shape[1])
    values = history_counts[:, -width:].mean(axis=1)
    return np.repeat(values[:, None], horizons, axis=1).astype(np.float32)


def multiscale_prior(history_counts: np.ndarray, targets: np.ndarray, val_mask: np.ndarray) -> tuple[np.ndarray, str]:
    """Select averaging windows per horizon on validation rows only."""

    horizons = targets.shape[1]
    candidates = {
        "fixed_w2": [2] * horizons,
        "fixed_w3": [3] * horizons,
        "fixed_w4": [4] * horizons,
        "fixed_w6": [6] * horizons,
        "expanding_2_3_4": [min(2 + h, history_counts.shape[1]) for h in range(horizons)],
        "expanding_3_4_5": [min(3 + h, history_counts.shape[1]) for h in range(horizons)],
        "expanding_4_5_6": [min(4 + h, history_counts.shape[1]) for h in range(horizons)],
    }
    best_name, best_prior, best_score = "fixed_w3", None, float("inf")
    for name, windows in candidates.items():
        prior = np.stack([history_counts[:, -max(1, width):].mean(axis=1) for width in windows], axis=1)
        if not val_mask.any():
            raise ValueError("validation mask is empty; cannot select the prior")
        score = row_mean(rmse, targets[val_mask], prior[val_mask])
        if score < best_score:
            best_name, best_prior, best_score = name, prior, float(score)
    return np.asarray(best_prior, dtype=np.float32), best_name


def standardize(train_mask: np.ndarray, values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    safe = np.nan_to_num(values.astype(np.float64, copy=False), nan=0.0, posinf=1e12, neginf=-1e12)
    safe = np.clip(safe, -1e12, 1e12)
    train = safe[train_mask]
    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((safe - mean) / std).astype(np.float32)


def prepare_arrays(samples: pd.DataFrame, feature_columns: Sequence[str]) -> dict[str, np.ndarray]:
    raw_columns = list(json.loads(samples.iloc[0]["feature_columns"]))
    matrices = np.asarray([json.loads(value) for value in samples["history_feature_matrix"]], dtype=np.float32)
    counts = np.asarray([json.loads(value) for value in samples["history_total_case_count_seq"]], dtype=np.float32)
    targets = np.asarray([json.loads(value) for value in samples["target_total_case_count_seq"]], dtype=np.float32)
    types = np.asarray([json.loads(value) for value in samples["target_type_ratio_h1"]], dtype=np.float32)

    indices = [raw_columns.index(name) for name in feature_columns]
    matrices = matrices[:, :, indices]
    ust_indices = [index for index, name in enumerate(feature_columns) if name.startswith("ust_")]
    core_indices = [index for index in range(len(feature_columns)) if index not in ust_indices]
    features = matrices[:, :, core_indices] if core_indices else np.zeros((*matrices.shape[:2], 1), dtype=np.float32)
    ust = matrices[:, :, ust_indices] if ust_indices else np.zeros((matrices.shape[0], matrices.shape[1], 0), dtype=np.float32)

    return {
        "features": features,
        "ust_features": ust,
        "counts": counts,
        "targets": targets,
        "types": types,
        "history_last": counts[:, -1],
        "prior": moving_average_prior(counts, horizons=targets.shape[1]),
        "gate": gate_evidence(counts, matrices, feature_columns),
        "feature_columns": np.asarray(list(feature_columns), dtype=object),
    }
