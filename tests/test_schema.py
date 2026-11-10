from pathlib import Path

import pandas as pd

from mosaic_forecast.io import load_samples
from mosaic_forecast.schema import validate_samples

ROOT = Path(__file__).resolve().parents[1]


def test_released_sample_matches_schema():
    samples = load_samples(ROOT / "data/sample/mosaic_sample_l6_h123.parquet")
    report = validate_samples(samples)
    assert report["ok"], report["errors"]
    assert report["rows"] == len(samples)
    assert report["history_length"] == 6
    assert report["horizons"] == [1, 2, 3]


def test_missing_column_is_reported():
    samples = load_samples(ROOT / "data/sample/mosaic_sample_l6_h123.parquet").drop(columns=["target_start"])
    report = validate_samples(samples)
    assert not report["ok"]
    assert "missing columns" in report["errors"][0]
