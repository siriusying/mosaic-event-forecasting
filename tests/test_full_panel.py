import json
from pathlib import Path

import pandas as pd

from mosaic_forecast.schema import validate_samples

ROOT = Path(__file__).resolve().parents[1]


def test_full_panel_shape():
    frame = pd.read_parquet(ROOT / "data/full_panel/forecast_samples_l6_h123.parquet")
    report = validate_samples(frame)
    assert report["ok"], report["errors"]
    assert report["rows"] == 2367
    assert report["counties"] == 40
    assert report["origins"] == 90


def test_released_sample_is_subset_of_full_panel():
    full = pd.read_parquet(ROOT / "data/full_panel/forecast_samples_l6_h123.parquet")
    sample = pd.read_parquet(ROOT / "data/sample/mosaic_sample_l6_h123.parquet")
    keys = set(zip(full["region_county_id"], full["forecast_origin"]))
    assert all(key in keys for key in zip(sample["region_county_id"], sample["forecast_origin"]))
    assert sample["region_county_id"].nunique() == 10


def test_full_panel_rebuilds_from_monthly_rows():
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from build_samples_from_panel import build

    panel = pd.read_csv(ROOT / "data/full_panel/county_month_panel.csv")
    rebuilt = build(panel, history=6, horizons=[1, 2, 3], counties=None)
    assert len(rebuilt) == 2367
    released = pd.read_parquet(ROOT / "data/full_panel/forecast_samples_l6_h123.parquet")
    key = ["region_county_id", "forecast_origin"]
    joined = rebuilt.merge(released, on=key, suffixes=("_new", "_old"))
    assert len(joined) == 2367
    for column in ("history_total_case_count_seq", "target_total_case_count_seq"):
        left = joined[column + "_new"].map(json.loads).to_list()
        right = joined[column + "_old"].map(json.loads).to_list()
        assert left == right
