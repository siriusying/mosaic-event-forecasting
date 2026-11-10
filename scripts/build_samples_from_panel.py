#!/usr/bin/env python3
"""Build six-month forecast samples from a monthly county panel.

The panel holds one row per county and month with counts, quality rates and
spatial summaries derived from the source records. This script turns those
monthly rows into the windowed samples read by the training code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FEATURE_COLUMNS = [
    "bucket_count_00_04", "bucket_count_04_08", "bucket_count_08_12", "bucket_count_12_16",
    "bucket_count_16_20", "bucket_count_20_24", "bucket_ratio_00_04", "bucket_ratio_04_08",
    "bucket_ratio_08_12", "bucket_ratio_12_16", "bucket_ratio_16_20", "bucket_ratio_20_24",
    "fine_level_1_unique_count", "fine_level_2_unique_count", "fine_location_parse_rate",
    "fine_location_parse_rate_county", "growth_diff_1", "lag_total_case_count_1",
    "lag_total_case_count_2", "lag_total_case_count_3", "lag_total_case_count_4",
    "lag_total_case_count_5", "lag_total_case_count_6", "month_cos", "month_sin",
    "rolling_mean_3", "rolling_mean_6", "rolling_std_3", "rolling_std_6", "space_valid_rate",
    "space_valid_rate_county", "time_valid_rate", "time_valid_rate_county", "total_case_count_county",
    "type_drug_related_ratio", "type_fraud_ratio", "type_injury_assault_ratio", "type_other_ratio",
    "type_public_order_other_ratio", "type_robbery_ratio", "ust_centroid_mean", "ust_centroid_std",
    "ust_code_valid_rate", "ust_code_valid_rate_county", "ust_entropy", "ust_top1_code",
    "ust_top1_ratio", "ust_top2_code", "ust_top2_ratio", "ust_top3_code", "ust_top3_ratio",
    "ust_unique_count",
]

TYPE_COLUMNS = [
    "type_drug_related_count",
    "type_fraud_count",
    "type_injury_assault_count",
    "type_other_count",
    "type_public_order_other_count",
    "type_robbery_count",
]


def month_period(value: str) -> pd.Period:
    return pd.Period(str(value), freq="M")


def check_lag_consistency(counts: dict[pd.Period, float], period: pd.Period, row: pd.Series) -> None:
    for lag in range(1, 7):
        expected = counts.get(period - lag, 0.0)
        actual = float(row[f"lag_total_case_count_{lag}"])
        if not np.isclose(actual, expected, atol=1e-6):
            raise ValueError(f"lag {lag} at {period} is {actual}, expected {expected}")
    # Rolling means use the months that exist before ``period``; a county that
    # enters the panel in 2011-12 has a one-month window in 2012-01.
    previous = [counts[period - lag] for lag in range(3, 0, -1) if period - lag in counts]
    expected_mean = float(np.mean(previous)) if previous else 0.0
    if not np.isclose(float(row["rolling_mean_3"]), expected_mean, atol=1e-6):
        raise ValueError(f"rolling_mean_3 at {period} is inconsistent with the monthly counts")


def build(panel: pd.DataFrame, history: int, horizons: list[int], counties: list[str] | None) -> pd.DataFrame:
    rows = []
    for county, group in panel.groupby("region_county_id", sort=True):
        if counties is not None and county not in counties:
            continue
        group = group.sort_values("year_month")
        frame = group.set_index(group["year_month"].map(month_period))
        counts = {period: float(value) for period, value in frame["total_case_count"].items()}
        periods = sorted(counts)
        for index, origin in enumerate(periods):
            history_periods = [origin - (history - 1 - step) for step in range(history)]
            target_periods = [origin + horizon for horizon in horizons]
            if not all(period in counts for period in history_periods + target_periods):
                continue
            matrix = []
            ratios = []
            for period in history_periods:
                row = frame.loc[period]
                check_lag_consistency(counts, period, row)
                matrix.append([float(row[column]) for column in FEATURE_COLUMNS])
                total = max(1.0, float(row["total_case_count"]))
                ratios.append([float(row[column]) / total for column in TYPE_COLUMNS])
            target_counts = [counts[period] for period in target_periods]
            first_target = frame.loc[target_periods[0]]
            type_counts = [float(first_target[column]) for column in TYPE_COLUMNS]
            rows.append(
                {
                    "region_county_id": county,
                    "region_county_name": str(group.iloc[0]["region_county_name"]),
                    "forecast_origin": str(origin),
                    "history_start": str(history_periods[0]),
                    "target_start": str(target_periods[0]),
                    "history_length": history,
                    "horizons": json.dumps(horizons),
                    "feature_columns": json.dumps(FEATURE_COLUMNS),
                    "type_count_columns": json.dumps(TYPE_COLUMNS),
                    "history_total_case_count_seq": json.dumps([counts[period] for period in history_periods]),
                    "history_feature_matrix": json.dumps(matrix),
                    "history_type_ratio_seq": json.dumps(ratios),
                    "target_total_case_count_seq": json.dumps(target_counts),
                    "target_type_count_h1": json.dumps(type_counts),
                    "target_type_ratio_h1": json.dumps([value / max(1.0, sum(type_counts)) for value in type_counts]),
                    "county_task_b_eligible": 0,
                    "task_b_eval_eligible": 0,
                    "dominant_type_h1": TYPE_COLUMNS[int(np.argmax(type_counts))].replace("type_", "").replace("_count", ""),
                }
            )
    return pd.DataFrame(rows).sort_values(["region_county_id", "forecast_origin"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default=str(ROOT / "data/sample/county_month_panel.csv"))
    parser.add_argument("--output", default=str(ROOT / "data/sample/mosaic_sample_l6_h123.parquet"))
    parser.add_argument("--history", type=int, default=6)
    parser.add_argument("--horizons", default="1,2,3")
    parser.add_argument("--counties", default=None, help="Optional comma-separated county keys.")
    args = parser.parse_args()

    panel = pd.read_csv(args.panel)
    counties = [item for item in args.counties.split(",")] if args.counties else None
    horizons = [int(item) for item in args.horizons.split(",")]
    samples = build(panel, args.history, horizons, counties)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(args.output, index=False)
    print(json.dumps({"rows": len(samples), "counties": int(samples.region_county_id.nunique()), "origins": int(samples.forecast_origin.nunique()), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
