#!/usr/bin/env python3
"""Score deterministic priors on the same split and target rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mosaic_forecast.io import display_path, load_samples, write_json
from mosaic_forecast.metrics import summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default=str(ROOT / "data/sample/mosaic_sample_l6_h123.parquet"))
    parser.add_argument("--split", default=str(ROOT / "data/splits/forward_split.json"))
    parser.add_argument("--output", default=str(ROOT / "runs/baselines_forward.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(args.samples)
    split = json.loads(Path(args.split).read_text())
    labels = samples["forecast_origin"].astype(str).to_numpy()
    train_mask = np.isin(labels, split["train"])
    val_mask = np.isin(labels, split["val"])
    test_mask = np.isin(labels, split["test"])

    history = np.asarray([json.loads(value) for value in samples["history_total_case_count_seq"]], dtype=float)
    targets = np.asarray([json.loads(value) for value in samples["target_total_case_count_seq"]], dtype=float)
    horizons = targets.shape[1]

    def window_prior(window: int) -> np.ndarray:
        width = min(window, history.shape[1])
        return np.repeat(history[:, -width:].mean(axis=1)[:, None], horizons, axis=1)

    candidates = {f"moving_average_{window}": window_prior(window) for window in (1, 3, 6, 12)}
    results = {}
    best_name = None
    best_score = float("inf")
    for name, prediction in candidates.items():
        val = summarize(targets[val_mask], prediction[val_mask], history[val_mask, -1])
        results[name] = {"validation": val, "test": summarize(targets[test_mask], prediction[test_mask], history[test_mask, -1])}
        if val["rmse_row"] < best_score:
            best_name, best_score = name, val["rmse_row"]

    payload = {
        "samples": display_path(args.samples),
        "split_sizes": {"train": int(train_mask.sum()), "val": int(val_mask.sum()), "test": int(test_mask.sum())},
        "selection": {"rule": "minimum validation row-mean RMSE", "selected": best_name},
        "results": results,
    }
    write_json(args.output, payload)
    print(json.dumps(payload["results"][best_name]["test"], indent=2))


if __name__ == "__main__":
    main()
