#!/usr/bin/env bash
set -euo pipefail

SAMPLES=${1:-data/full_panel/forecast_samples_l6_h123.parquet}
OUT=${2:-runs/full}
SEEDS=(11 23 37 51 73)

for seed in "${SEEDS[@]}"; do
  python scripts/run_forward.py --samples "$SAMPLES" --split data/splits/forward_split.json \
    --seed "$seed" --epochs 25 --output "$OUT/forward_seed$seed"
  python scripts/run_rolling.py --samples "$SAMPLES" --seed "$seed" --epochs 5 \
    --output "$OUT/rolling_seed$seed"
done

python scripts/aggregate_seeds.py "$OUT"/forward_seed* --output "$OUT/forward_aggregate.json"
python scripts/aggregate_seeds.py "$OUT"/rolling_seed* --metric-set rolling --output "$OUT/rolling_aggregate.json"
