# Reproduction notes

## Environment used for the replayed values

- Python 3.13, numpy 2.x, pandas 3.x, pyarrow, PyTorch CPU build.
- 8 CPU threads, no GPU.
- Floating point is float32 in the model; the loss and metrics use float64 where stated.

`requirements.txt` lists the minimum versions. The paper's runs used PyTorch
2.11 with CUDA 12.8 on an RTX 5000 Ada. CPU and GPU runs agree to about
0.001 RMSE; the largest difference observed for the forward split was 0.0001
on seed 37 after matching the leakage-safe feature mode.

## Runtime

| Command | Rows | Time |
|---|---|---|
| `run_forward.py` on the released sample | 768 | about 10 s |
| `run_rolling.py` on the released sample | 768 | about 1.5 min |
| `run_forward.py` on the full sample | 2,367 | about 12 s |
| `run_rolling.py` on the full sample | 2,367 | under 10 min |

## What each script writes

`run_forward.py` writes `summary.json` and `predictions.csv`. The summary
records the selected prior window, the endpoint decision, validation scores,
split sizes, the feature list and the sample hash. `predictions.csv` has one
row per county, origin and horizon.

`run_rolling.py` writes `rolling_predictions.csv` and `summary.json` with
per-origin metrics, the endpoint counts, and the conformal audit.

`run_baselines.py` writes validation and test metrics for 1, 3, 6 and 12 month
moving averages and the validation-selected baseline.

## Differences from the manuscript pipeline

- The manuscript averages five seeds. Each script here runs one seed; repeat
  with the seeds listed in `README.md` and average the `summary.json` values.
- The conformal function in this package keeps one reservoir per horizon and
  uses a minimum calibration count of 20. The manuscript additionally falls
  back to validation residuals when a reservoir is short and averages coverage
  over five seeds. Single-seed coverage on the full sample is about 0.905; the
  five-seed manuscript value is 0.869. Treat the two as different estimators.
- The paper's rolling protocol uses 5 epochs per fit. Running with 25 epochs
  changes the endpoint selection and is not the reported protocol.
- The gate diagnostics are computed from the trained model at scoring time.
  No test target enters the prior choice, the endpoint choice or the gate.

## Not included

The external FBI, QPS, JUSTFAIR and Taiwan panels are source-held-out checks in
the manuscript. They need separate retrieval and are not part of this package.
The PEFT adapter branch and the next-month offense-type task are also outside
this release; Task B has five eligible forward-test rows and is reported as a
support-limited diagnostic in the paper.
