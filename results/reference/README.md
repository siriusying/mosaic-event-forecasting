# Reference runs on the released sample

These files were produced with the scripts in this repository and the released
10-county sample. They let a reader compare a fresh run without downloading the
full panel.

| File | Command |
|---|---|
| `forward_seed37/summary.json` | `python scripts/run_forward.py --seed 37 --epochs 25` |
| `forward_seed37/predictions.csv` | same run, test rows |
| `baselines_forward.json` | `python scripts/run_baselines.py` |
| `rolling_seed37/summary.json` | `python scripts/run_rolling.py --seed 37 --epochs 5` |
| `rolling_seed37/rolling_predictions.csv` | same run, all origins |
| `full_forward_seed37_summary.json` | Full 40-county panel, forward split, 25 epochs |
| `full_rolling_seed37_summary.json` | Full panel, 83 rolling origins, 5 epochs per fit |
| `reference_metrics.json` | Manuscript values and full-sample replays |

Sample forward split (83 test rows, seed 37, 25 epochs): RMSE 0.958, row RMSE
0.616, MAE 0.502. The validation-selected three-month moving average scores
RMSE 1.054 and row RMSE 0.675 on the same rows.

Sample rolling run (83 origins, 5 epochs per fit): RMSE 0.631, mean split RMSE
0.599, with 71 full-gate and 12 hurdle endpoint selections. Prequential
coverage is 0.917 at a nominal 0.90 level.

These values characterise the ten longest-history counties, which are larger
than the panel average. The full-sample replications used for the manuscript
are 0.6996 (forward) and 0.5863 (rolling) for seed 37.
