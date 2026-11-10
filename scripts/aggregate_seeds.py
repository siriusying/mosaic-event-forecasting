#!/usr/bin/env python3
"""Aggregate per-seed summaries into mean and standard deviation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mosaic_forecast.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Directories containing summary.json")
    parser.add_argument("--output", default="runs/aggregate.json")
    parser.add_argument("--metric-set", choices=["test", "rolling"], default="test")
    args = parser.parse_args()

    records = []
    for run in args.runs:
        summary = json.loads((Path(run) / "summary.json").read_text())
        if args.metric_set == "test":
            metrics = summary["test"]
            record = {"run": str(run), "seed": summary.get("seed"), **{key: metrics[key] for key in ("rmse", "rmse_row", "mae", "trend_accuracy")}}
        else:
            metrics = summary["test"]
            record = {
                "run": str(run),
                "seed": summary.get("seed"),
                "rmse": metrics["rmse"],
                "rmse_row": metrics["rmse_row"],
                "mae": metrics["mae"],
                "mean_split_rmse": summary["mean_split_rmse"],
                "coverage": summary["conformal"]["coverage"],
            }
        records.append(record)

    keys = [key for key in records[0] if key not in {"run", "seed"}]
    aggregate = {
        "n_seeds": len(records),
        "seeds": [record["seed"] for record in records],
        "mean": {key: float(np.mean([record[key] for record in records])) for key in keys},
        "sd": {key: float(np.std([record[key] for record in records], ddof=1)) if len(records) > 1 else None for key in keys},
        "per_seed": records,
    }
    write_json(args.output, aggregate)
    print(json.dumps({"n_seeds": aggregate["n_seeds"], "mean": aggregate["mean"], "sd": aggregate["sd"]}, indent=2))


if __name__ == "__main__":
    main()
