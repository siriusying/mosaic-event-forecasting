import json
from pathlib import Path

import pandas as pd

from mosaic_forecast.splits import county_fold, forward_split, rolling_origins

ROOT = Path(__file__).resolve().parents[1]


def test_forward_split_matches_released_index():
    samples = pd.read_parquet(ROOT / "data/sample/mosaic_sample_l6_h123.parquet")
    split = forward_split(samples["forecast_origin"].astype(str))
    released = json.loads((ROOT / "data/splits/forward_split.json").read_text())
    assert split == released
    assert split["train"][-1] < split["val"][0] <= split["val"][-1] < split["test"][0]


def test_rolling_split_count():
    samples = pd.read_parquet(ROOT / "data/sample/mosaic_sample_l6_h123.parquet")
    splits = rolling_origins(samples["forecast_origin"].astype(str), min_train_size=6, horizon=1)
    assert len(splits) == 83
    assert splits[0]["test"] == ["2011-06"]


def test_county_fold_is_stable():
    key = "深圳市|罗湖区"
    assert county_fold(key) == county_fold(key)
    assert 0 <= county_fold(key) < 5
