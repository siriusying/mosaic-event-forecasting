#!/usr/bin/env python3
"""Rolling-origin evaluation with the validation-only endpoint safeguard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mosaic_forecast.conformal import prequential_intervals
from mosaic_forecast.io import display_path, load_samples, sha256_file, write_json
from mosaic_forecast.metrics import mae, rmse, row_mean, smape
from mosaic_forecast.splits import rolling_origins
from mosaic_forecast.train import TrainConfig, run_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default=str(ROOT / "data/sample/mosaic_sample_l6_h123.parquet"))
    parser.add_argument("--output", default=str(ROOT / "runs/rolling"))
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--feature-mode", default="leakage_safe")
    parser.add_argument("--min-train", type=int, default=6)
    parser.add_argument("--max-splits", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(args.samples)
    splits = rolling_origins(samples["forecast_origin"].astype(str), min_train_size=args.min_train, horizon=1)
    if args.max_splits:
        splits = splits[: args.max_splits]
    config = TrainConfig(seed=args.seed, epochs=args.epochs, feature_mode=args.feature_mode, device=args.device)

    rows = []
    split_metrics = []
    available = set(samples["forecast_origin"].astype(str))
    for index, split in enumerate(splits):
        if not set(split["train"]) & available or not set(split["val"]) & available or not set(split["test"]) & available:
            continue
        result = run_forward(samples, split, config)
        mask = result["test_mask"]
        targets = result["arrays"]["targets"][mask]
        predictions = result["test_predictions"]
        origin = split["test"][0]
        split_metrics.append(
            {
                "split": index,
                "origin": origin,
                "n": int(mask.sum()),
                "rmse": rmse(targets.flatten(), predictions.flatten()),
                "endpoint": result["endpoint_selected_on_validation"],
            }
        )
        for row_index, sample in enumerate(samples.loc[mask].itertuples(index=False)):
            for horizon in range(targets.shape[1]):
                rows.append(
                    {
                        "split": index,
                        "forecast_origin": origin,
                        "region_county_id": sample.region_county_id,
                        "horizon": horizon + 1,
                        "y_true": float(targets[row_index, horizon]),
                        "y_pred": float(predictions[row_index, horizon]),
                    }
                )

    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "rolling_predictions.csv", index=False)

    horizons = int(frame["horizon"].nunique())
    frame = frame.sort_values(["split", "region_county_id", "horizon"]).reset_index(drop=True)
    grouped = frame.groupby(["split", "region_county_id"], as_index=False)["forecast_origin"].first()
    if len(frame) != len(grouped) * horizons:
        raise RuntimeError("each county-origin row must provide one value per horizon")
    y_pred = frame["y_pred"].to_numpy().reshape(len(grouped), horizons)
    y_true = frame["y_true"].to_numpy().reshape(len(grouped), horizons)
    origins = grouped["forecast_origin"].astype(str).tolist()
    band = prequential_intervals(origins, y_pred, y_true)

    summary = {
        "samples": display_path(args.samples),
        "samples_sha256": sha256_file(args.samples),
        "seed": args.seed,
        "epochs": args.epochs,
        "feature_mode": args.feature_mode,
        "n_splits": len(splits),
        "test": {
            "n": int(len(y_true)),
            "rmse": rmse(y_true.flatten(), y_pred.flatten()),
            "rmse_row": row_mean(rmse, y_true, y_pred),
            "mae": mae(y_true.flatten(), y_pred.flatten()),
            "smape": smape(y_true.flatten(), y_pred.flatten()),
        },
        "mean_split_rmse": float(np.mean([item["rmse"] for item in split_metrics])),
        "endpoint_counts": pd.Series([item["endpoint"] for item in split_metrics]).value_counts().to_dict(),
        "conformal": {key: value for key, value in band.items() if key not in {"lower", "upper", "evaluated"}},
        "splits": split_metrics,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("n_splits", "test", "mean_split_rmse", "endpoint_counts", "conformal")}, indent=2))


if __name__ == "__main__":
    main()
