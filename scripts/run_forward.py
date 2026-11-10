#!/usr/bin/env python3
"""Train MOSAIC on the chronological forward split and score the test months."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mosaic_forecast.io import display_path, load_samples, sha256_file, write_json
from mosaic_forecast.metrics import summarize
from mosaic_forecast.schema import validate_samples
from mosaic_forecast.splits import forward_split
from mosaic_forecast.train import TrainConfig, run_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default=str(ROOT / "data/sample/mosaic_sample_l6_h123.parquet"))
    parser.add_argument("--split", default=str(ROOT / "data/splits/forward_split.json"))
    parser.add_argument("--output", default=str(ROOT / "runs/forward"))
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--feature-mode", default="leakage_safe")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(args.samples)
    report = validate_samples(samples)
    if not report["ok"]:
        raise SystemExit(f"sample validation failed: {report['errors']}")

    split = json.loads(Path(args.split).read_text())
    config = TrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        feature_mode=args.feature_mode,
        device=args.device,
    )
    result = run_forward(samples, split, config)

    arrays = result["arrays"]
    test_mask = result["test_mask"]
    metrics = summarize(arrays["targets"][test_mask], result["test_predictions"], arrays["history_last"][test_mask])
    prior_metrics = summarize(arrays["targets"][test_mask], arrays["prior"][test_mask], arrays["history_last"][test_mask])
    flat_metrics = summarize(
        arrays["targets"][result["val_mask"]],
        result["val_full_gate"],
        arrays["history_last"][result["val_mask"]],
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    test_rows = samples.loc[test_mask].reset_index(drop=True)
    for index, row in test_rows.iterrows():
        for horizon in range(len(arrays["targets"][test_mask][index])):
            rows.append(
                {
                    "region_county_id": row["region_county_id"],
                    "forecast_origin": row["forecast_origin"],
                    "horizon": horizon + 1,
                    "y_true": float(arrays["targets"][test_mask][index][horizon]),
                    "y_pred": float(result["test_predictions"][index][horizon]),
                    "y_prior": float(arrays["prior"][test_mask][index][horizon]),
                }
            )
    pd.DataFrame(rows).to_csv(output / "predictions.csv", index=False)

    summary = {
        "samples": display_path(args.samples),
        "samples_sha256": sha256_file(args.samples),
        "seed": args.seed,
        "epochs": args.epochs,
        "feature_mode": args.feature_mode,
        "prior_selected_on_validation": result["prior_selected_on_validation"],
        "endpoint_selected_on_validation": result["endpoint_selected_on_validation"],
        "endpoint_validation_scores": result["endpoint_validation_scores"],
        "split_sizes": result["split_sizes"],
        "test": metrics,
        "validation_full_gate": flat_metrics,
        "test_prior": prior_metrics,
        "gate_test": result["gate_test"],
        "training": {key: value for key, value in result["training"].items() if key != "loss_history"},
        "feature_columns": result["feature_columns"],
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({"test": metrics, "endpoint": summary["endpoint_selected_on_validation"], "prior": summary["prior_selected_on_validation"]}, indent=2))


if __name__ == "__main__":
    main()
