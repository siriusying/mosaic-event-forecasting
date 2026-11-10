import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT / "scripts")


def test_panel_rebuild_matches_released_sample():
    import sys

    sys.path.insert(0, sys_path)
    from build_samples_from_panel import build

    panel = pd.read_csv(ROOT / "data/sample/county_month_panel.csv")
    rebuilt = build(panel, history=6, horizons=[1, 2, 3], counties=None)
    released = pd.read_parquet(ROOT / "data/sample/mosaic_sample_l6_h123.parquet")
    assert len(rebuilt) == len(released)
    key = ["region_county_id", "forecast_origin"]
    joined = rebuilt.merge(released, on=key, suffixes=("_new", "_old"), how="outer", indicator=True)
    assert (joined["_merge"] == "both").all()
    for column in ("history_total_case_count_seq", "target_total_case_count_seq"):
        left = np.array([json.loads(value) for value in joined[column + "_new"]])
        right = np.array([json.loads(value) for value in joined[column + "_old"]])
        assert np.allclose(left, right)
    left = np.array([json.loads(value) for value in joined["history_feature_matrix_new"]])
    right = np.array([json.loads(value) for value in joined["history_feature_matrix_old"]])
    assert np.allclose(left, right, rtol=1e-5, atol=1e-6)
