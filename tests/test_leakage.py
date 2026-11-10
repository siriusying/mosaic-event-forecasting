from pathlib import Path

import numpy as np

from mosaic_forecast.features import prepare_arrays, resolve_feature_columns
from mosaic_forecast.io import load_samples

ROOT = Path(__file__).resolve().parents[1]


def test_gate_evidence_is_bounded():
    samples = load_samples(ROOT / "data/sample/mosaic_sample_l6_h123.parquet")
    import json

    columns = resolve_feature_columns(json.loads(samples.iloc[0]["feature_columns"]), "leakage_safe")
    arrays = prepare_arrays(samples, columns)
    assert arrays["gate"].shape[1] == 3
    assert arrays["gate"].min() >= 0.0
    assert arrays["gate"].max() <= 1.0


def test_future_targets_do_not_change_features():
    samples = load_samples(ROOT / "data/sample/mosaic_sample_l6_h123.parquet")
    import json

    columns = resolve_feature_columns(json.loads(samples.iloc[0]["feature_columns"]), "leakage_safe")
    base = prepare_arrays(samples, columns)
    modified = samples.copy()
    modified["target_total_case_count_seq"] = modified["target_total_case_count_seq"].map(
        lambda value: json.dumps([item + 7.0 for item in json.loads(value)])
    )
    changed = prepare_arrays(modified, columns)
    assert np.array_equal(base["features"], changed["features"])
    assert np.array_equal(base["gate"], changed["gate"])
    assert not np.array_equal(base["targets"], changed["targets"])
