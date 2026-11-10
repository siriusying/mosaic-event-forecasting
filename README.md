# MOSAIC: county-month event forecasting

This repository contains the model, training and evaluation code, the sample
schema, split indices and derived data needed to rerun the count forecasting
experiments reported in the submission *MOSAIC: ulti-scale Knowledge-State Assembly with Intensity Correction for Auditable Judicial Event Forecasting*.

The package runs on a small released sample (10 counties, 768 forecast rows)
and on the full 40-county panel (2,367 rows). Both use the same code path.

## Layout

| Path | Content |
|---|---|
| `mosaic_forecast/` | Model, features, training, metrics, conformal intervals |
| `scripts/run_forward.py` | Chronological forward split: train, validate, test |
| `scripts/run_rolling.py` | 83 rolling origins with per-origin endpoint selection |
| `scripts/run_baselines.py` | Moving-average priors on the same split |
| `scripts/build_samples_from_panel.py` | Rebuild windowed samples from the monthly panel |
| `scripts/make_sample.py` | Extract the released subset and split files |
| `scripts/verify_manifest.py` | Check every released file against its SHA256 |
| `data/sample/` | 768-row sample and its monthly panel |
| `data/full_panel/` | Complete 40-county derived panel used in the paper |
| `data/schema/` | JSON schema and feature dictionary |
| `data/splits/` | Forward split, rolling-origin index, county folds |
| `data/metadata/` | Dataset card, provenance, gate parameters |
| `results/reference/` | Replayed metrics for the released sample |
| `docs/` | Panel construction, reproducibility notes, model card |

## Install

Python 3.10 or newer. PyTorch CPU wheels are sufficient.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
```

## Run

Check the released files:

```
python scripts/verify_manifest.py
```

Train and score one forward split on the sample:

```
python scripts/run_forward.py --output runs/forward_sample
```

Run all rolling origins on the sample (83 fits, about two minutes on a laptop CPU):

```
python scripts/run_rolling.py --output runs/rolling_sample
```

Run the tests:

```
python -m pytest -q tests
```

## Reproducing the manuscript results

The manuscript uses the full 40-county sample, five seeds (11, 23, 37, 51, 73),
25 epochs for the forward split and 5 epochs per rolling fit. The derived full
panel is included at `data/full_panel/forecast_samples_l6_h123.parquet`. Run:

```bash
bash scripts/run_five_seeds.sh data/full_panel/forecast_samples_l6_h123.parquet runs/full
```

The script trains all five seeds and writes `forward_aggregate.json` and
`rolling_aggregate.json` with means and standard deviations.

Seed 37 with the released code gives the following values on the full sample:

| Protocol | RMSE | Row RMSE | MAE | Endpoint selection |
|---|---|---|---|---|
| Forward, 25 epochs | 0.6996 | 0.5054 | 0.4295 | full gate |
| Rolling, 83 origins, 5 epochs | 0.5863 | 0.4671 | 0.4026 | 61 full gate / 22 hurdle |

The manuscript reports five-seed means of 0.7067 (forward) and 0.5859
(rolling). The single-seed differences are within 0.001 RMSE for rolling and
0.005 for forward. Seed-37 summaries for the full panel and for the released
subset are stored in `results/reference/`, with predictions for the subset.

The forward run selects the averaging window for each horizon and the gate
endpoint on the validation months. The rolling run repeats both choices inside
every fit. Test targets are used only for scoring.

## Model

Each sample carries six months of history and three target months. Historical
columns are grouped into history, seasonality, quality, composition and spatial
channels. A two-layer Transformer (64 hidden units, four heads, dropout 0.1)
encodes the six-step sequence. Two heads predict whether a month has any event
and the positive count; their product is the hurdle estimate. A gate

    g = sigmoid(beta + sum_k softplus(w_k) * (e_k - 0.5))

mixes that estimate with a moving-average prior for evidence values support,
quality and stability. Because the slopes are non-negative, raising one
evidence value can only increase the weight of the hurdle branch. The endpoint
safeguard compares the full gate with the g=1 hurdle output on validation rows.

## Data and licence

The released panels contain aggregate counts and derived features only. They
contain no judgment text, case numbers or personal names. Source collection, access dates
and redistribution limits are documented in [DATA_SOURCES.md](DATA_SOURCES.md).
Code is released under the MIT licence; data files follow the terms of the
source collection described there.

## Publishing

The directory is a self-contained repository. To publish it:

```bash
cd mosaic-event-forecasting
git init
git add .
git commit -m "Release MOSAIC forecasting code and sample"
git remote add origin <repository-url>
git push -u origin main
```

`MANIFEST.json` covers every released file except itself and `SHA256SUMS`; run
`python scripts/verify_manifest.py` after any change and regenerate the
manifest with `--write` before committing.
