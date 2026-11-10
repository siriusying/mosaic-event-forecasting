# Model card

## Intended use

Short-horizon count forecasting for county-month administrative records of
judicial events, with an explicit record of which evidence drives the prior
correction. The output is an aggregate count. It is not a crime rate, an
individual risk score or a causal estimate.

## Inputs

Six monthly records per sample. Each month contributes the total count, offense
composition, time-of-day distribution, data-quality rates and coarse spatial
summaries. Three leakage-safe evidence values are computed from the history:
share of non-zero months, mean quality rate and recent stability.

## Architecture

- Five typed channels plus a residual path, each projected to 64 dimensions.
- Channel weights from a learned linear fusion of the six channel projections.
- Two Transformer encoder layers, four attention heads, dropout 0.1.
- Hurdle heads: event logit and positive log-count, multiplied to form the
  structured estimate.
- Monotone gate on three evidence values, mixing the structured estimate with a
  moving-average prior.
- Validation-only choice between the full gate and the g=1 hurdle endpoint.

The released leakage-safe model has 99,414 parameters. Training uses Adam with
learning rate 1e-3, full-batch updates, 25 epochs for the forward split and 5
epochs per rolling fit, keeping the checkpoint with the lowest validation loss.

## Outputs and metrics

Counts for horizons 1 to 3, scored by flattened RMSE, row-mean RMSE, MAE, sMAPE
and first-horizon trend accuracy. Rolling runs also report prequential interval
coverage.

## Limits

- Counts depend on publishing practice and case selection; the target is the
  number of normalized records, not population incidence.
- Incident months are parsed from text and are less complete than judgment
  dates. Only about 71% of the 2020--2025 records yield an incident month.
- The panel is one document collection. County-disjoint and temporal splits
  test within-source generalization.
- Interval coverage in the manuscript is 86.9% at a nominal 90%, below the
  pre-specified acceptance target. Intervals are drift diagnostics.
- External police and sentencing panels are transfer checks; the model was not
  trained on them and their counts are not pooled with the judicial panel.
