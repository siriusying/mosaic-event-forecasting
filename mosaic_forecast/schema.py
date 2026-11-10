"""Sample schema for the released forecasting panels."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "region_county_id",
    "region_county_name",
    "forecast_origin",
    "history_start",
    "target_start",
    "history_length",
    "horizons",
    "feature_columns",
    "type_count_columns",
    "history_total_case_count_seq",
    "history_feature_matrix",
    "history_type_ratio_seq",
    "target_total_case_count_seq",
    "target_type_count_h1",
    "target_type_ratio_h1",
    "county_task_b_eligible",
    "task_b_eval_eligible",
    "dominant_type_h1",
)

JSON_LIST_COLUMNS: tuple[str, ...] = (
    "horizons",
    "feature_columns",
    "type_count_columns",
    "history_total_case_count_seq",
    "history_feature_matrix",
    "history_type_ratio_seq",
    "target_total_case_count_seq",
    "target_type_count_h1",
    "target_type_ratio_h1",
)


def _parse(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def validate_samples(frame: pd.DataFrame) -> dict[str, Any]:
    """Check column presence, array shapes and finite values.

    The function returns a report instead of raising so that users can inspect
    partial failures. ``report["ok"]`` is true only when every check passes.
    """

    errors: list[str] = []
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
        return {"ok": False, "rows": int(len(frame)), "errors": errors}

    if frame.empty:
        errors.append("empty frame")
        return {"ok": False, "rows": 0, "errors": errors}

    for name in JSON_LIST_COLUMNS:
        try:
            frame[name].map(_parse)
        except Exception as exc:  # pragma: no cover - reported, not raised
            errors.append(f"column {name} is not valid JSON: {exc}")

    first = frame.iloc[0]
    feature_names = list(_parse(first["feature_columns"]))
    horizons = list(_parse(first["horizons"]))
    history_length = int(first["history_length"])
    type_names = list(_parse(first["type_count_columns"]))
    shapes = {
        "history_total_case_count_seq": (history_length,),
        "target_total_case_count_seq": (len(horizons),),
        "history_feature_matrix": (history_length, len(feature_names)),
        "history_type_ratio_seq": (history_length, len(type_names)),
    }
    bad_shapes = []
    identical_columns = True
    for index, row in frame.iterrows():
        if list(_parse(row["feature_columns"])) != feature_names:
            identical_columns = False
        if list(_parse(row["horizons"])) != horizons:
            errors.append(f"row {index}: horizons differ from the first row")
            break
        for name, shape in shapes.items():
            array = np.asarray(_parse(row[name]), dtype=float)
            if array.shape != shape or not np.isfinite(array).all():
                bad_shapes.append((int(index), name, array.shape))
                if len(bad_shapes) >= 5:
                    break
        if len(bad_shapes) >= 5:
            break
    if bad_shapes:
        errors.append(f"array shape or finiteness failures (first five): {bad_shapes}")
    if not identical_columns:
        errors.append("feature_columns differ across rows")

    origins = sorted(frame["forecast_origin"].astype(str).unique().tolist())
    return {
        "ok": not errors,
        "rows": int(len(frame)),
        "counties": int(frame["region_county_id"].astype(str).nunique()),
        "origins": len(origins),
        "origin_start": origins[0] if origins else None,
        "origin_end": origins[-1] if origins else None,
        "history_length": history_length,
        "horizons": horizons,
        "n_features": len(feature_names),
        "n_types": len(type_names),
        "errors": errors,
    }


def json_schema() -> dict[str, Any]:
    """Return a JSON Schema document for one sample row."""

    string = {"type": "string"}
    integer = {"type": "integer"}
    properties = {
        "region_county_id": {**string, "description": "Stable county key: province|city|county."},
        "region_county_name": {**string, "description": "County display name."},
        "forecast_origin": {**string, "pattern": "^[0-9]{4}-[0-9]{2}$", "description": "Last month in the input history (YYYY-MM)."},
        "history_start": {**string, "description": "First month of the history window."},
        "target_start": {**string, "description": "First target month (origin + 1)."},
        "history_length": {**integer, "minimum": 1, "description": "Number of history months (6 in the paper)."},
        "horizons": {"type": "string", "description": "JSON array of forecast horizons, e.g. [1, 2, 3]."},
        "feature_columns": {"type": "string", "description": "JSON array naming the columns of history_feature_matrix."},
        "type_count_columns": {"type": "string", "description": "JSON array naming offense-type count columns."},
        "history_total_case_count_seq": {"type": "string", "description": "JSON array of monthly counts, length history_length."},
        "history_feature_matrix": {"type": "string", "description": "JSON array with shape history_length x n_features."},
        "history_type_ratio_seq": {"type": "string", "description": "JSON array with shape history_length x n_types."},
        "target_total_case_count_seq": {"type": "string", "description": "JSON array of counts for each horizon."},
        "target_type_count_h1": {"type": "string", "description": "JSON array of type counts at horizon 1."},
        "target_type_ratio_h1": {"type": "string", "description": "JSON array of type ratios at horizon 1."},
        "county_task_b_eligible": {**integer, "enum": [0, 1]},
        "task_b_eval_eligible": {**integer, "enum": [0, 1]},
        "dominant_type_h1": {**string, "description": "Most frequent offense type at horizon 1."},
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "MOSAIC forecast sample",
        "type": "object",
        "required": list(REQUIRED_COLUMNS),
        "properties": properties,
        "additionalProperties": True,
    }
