#!/usr/bin/env python3
"""Create the small released sample, split files and feature dictionary."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from mosaic_forecast.io import sha256_file, write_json
from mosaic_forecast.schema import json_schema
from mosaic_forecast.splits import county_folds, forward_split, rolling_origins


FEATURE_DEFINITIONS = {
    "lag_total_case_count_{k}": ("history", "Total count {k} month(s) before the row month; zero when absent."),
    "rolling_mean_3": ("history", "Mean count over the previous three months."),
    "rolling_mean_6": ("history", "Mean count over the previous six months."),
    "rolling_std_3": ("history", "Standard deviation over the previous three months."),
    "rolling_std_6": ("history", "Standard deviation over the previous six months."),
    "growth_diff_1": ("history", "One-month count difference."),
    "month_sin": ("seasonality", "Sine of the forecast-origin month."),
    "month_cos": ("seasonality", "Cosine of the forecast-origin month."),
    "bucket_count_{h}": ("seasonality", "Count of incidents in the {h} four-hour bucket."),
    "bucket_ratio_{h}": ("seasonality", "Share of incidents in the {h} four-hour bucket."),
    "time_valid_rate": ("quality", "Share of records with a usable incident time."),
    "space_valid_rate": ("quality", "Share of records with a usable incident location."),
    "fine_location_parse_rate": ("quality", "Share of locations parsed to county or finer."),
    "ust_code_valid_rate": ("quality", "Share of records with a valid UST code."),
    "{name}_county": ("quality", "County-period value of the same rate or count, when available."),
    "type_drug_related_ratio": ("composition", "Share of drug-related cases."),
    "type_fraud_ratio": ("composition", "Share of fraud cases."),
    "type_injury_assault_ratio": ("composition", "Share of injury and assault cases."),
    "type_robbery_ratio": ("composition", "Share of robbery cases."),
    "type_public_order_other_ratio": ("composition", "Share of public-order and other cases."),
    "type_other_ratio": ("composition", "Share of remaining cases."),
    "total_case_count_county": ("history", "County total over the complete panel; excluded in leakage-safe modes."),
    "first_year_month": ("provenance", "First month of the county panel; excluded in leakage-safe modes."),
    "last_year_month": ("provenance", "Last month of the county panel; excluded in leakage-safe modes."),
    "fine_level_1_unique_count": ("spatial", "Number of distinct coarse spatial cells."),
    "fine_level_2_unique_count": ("spatial", "Number of distinct fine spatial cells."),
    "ust_centroid_mean": ("spatial", "Mean UST centroid code."),
    "ust_centroid_std": ("spatial", "Standard deviation of UST centroid codes."),
    "ust_entropy": ("spatial", "Entropy of the UST code distribution."),
    "ust_top1_code": ("spatial", "Most frequent UST code."),
    "ust_top1_ratio": ("spatial", "Share of records with the most frequent UST code."),
    "ust_top2_code": ("spatial", "Second most frequent UST code."),
    "ust_top2_ratio": ("spatial", "Share of records with the second most frequent UST code."),
    "ust_top3_code": ("spatial", "Third most frequent UST code."),
    "ust_top3_ratio": ("spatial", "Share of records with the third most frequent UST code."),
    "ust_unique_count": ("spatial", "Number of distinct UST codes."),
}


def describe(name: str) -> tuple[str, str]:
    import re

    for pattern, (group, text) in FEATURE_DEFINITIONS.items():
        if "{" in pattern:
            regex = "^" + re.escape(pattern).replace(r"\{k\}", r"(\d+)").replace(r"\{h\}", r"([0-9_]+)").replace(r"\{name\}", r"(.+)") + "$"
            match = re.match(regex, name)
            if match:
                slot = match.group(1)
                if "_" in slot and len(slot) == 5:
                    start, end = slot.split("_")
                    slot = f"{start}:00-{end}:00"
                return group, text.format(**{"k": match.group(1), "h": slot, "name": name[:-7]})
        elif pattern == name:
            return group, text
    if name.endswith("_county"):
        return "quality", "County-period rate or count; excluded from leakage-safe feature sets."
    return "other", "Derived panel feature."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default=str(REPO / "results/forecasting_benchmark/forecast_samples_l6_h123.parquet"))
    parser.add_argument("--panel", default=str(REPO / "results/forecasting_benchmark/county_month_panel.csv"))
    parser.add_argument("--out-dir", default=str(ROOT))
    parser.add_argument("--counties", type=int, default=10)
    args = parser.parse_args()

    out = Path(args.out_dir)
    samples = pd.read_parquet(args.samples)
    coverage = samples.groupby("region_county_id")["forecast_origin"].nunique().sort_values(ascending=False)
    picks = sorted(coverage.head(args.counties).index.tolist())
    subset = samples.loc[samples["region_county_id"].isin(picks)].copy()
    subset = subset.sort_values(["region_county_id", "forecast_origin"]).reset_index(drop=True)
    sample_path = out / "data/sample/mosaic_sample_l6_h123.parquet"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(sample_path, index=False)

    panel = pd.read_csv(args.panel)
    tidy = panel.loc[panel["region_county_id"].isin(picks)].copy()
    tidy = tidy.sort_values(["region_county_id", "year_month"]).reset_index(drop=True)
    tidy.to_csv(out / "data/sample/county_month_panel.csv", index=False)

    split = forward_split(samples["forecast_origin"].astype(str))
    write_json(out / "data/splits/forward_split.json", split)
    origins = rolling_origins(samples["forecast_origin"].astype(str))
    with (out / "data/splits/rolling_origins.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "train_end", "val_month", "test_month"])
        for index, item in enumerate(origins):
            writer.writerow([index, item["train"][-1], item["val"][-1] if item["val"] else "", item["test"][-1] if item["test"] else ""])
    write_json(out / "data/splits/county_folds.json", county_folds(samples["region_county_id"].astype(str).unique(), 5))
    write_json(out / "data/splits/split_summary.json", {
        "forward": {key: len(value) for key, value in split.items()},
        "forward_first_last": {key: [value[0], value[-1]] for key, value in split.items() if value},
        "rolling_splits": len(origins),
        "rolling_rule": "train=prefix months, val=next month, test=following month",
        "county_fold_rule": "int(sha256(county_id)[:8], 16) % 5",
    })

    write_json(out / "data/schema/forecast_sample_schema.json", json_schema())
    feature_names = list(json.loads(samples.iloc[0]["feature_columns"]))
    with (out / "data/schema/feature_definitions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "group", "definition", "leakage_note"])
        for name in feature_names:
            group, text = describe(name)
            note = "computed from the origin history only"
            if name in {"total_case_count_county", "first_year_month", "last_year_month"} or name.endswith("_county"):
                note = "county-period statistic; excluded by leakage_safe feature modes"
            writer.writerow([name, group, text, note])
    print(json.dumps({"rows": len(subset), "counties": len(picks), "origins": int(subset["forecast_origin"].nunique()), "sha256": sha256_file(sample_path)}, indent=2))


if __name__ == "__main__":
    main()
