# Panel construction

The released files cover three stages: documents, a monthly county panel and
windowed samples. Each stage has a fixed input contract.

## Stage 1: records to monthly counts

Input fields: case number, court, judgment date, publication date, incident
time text, incident location text, offense category and document text.

Rules:

1. Keep criminal cases with a parseable incident month. Cases without an
   incident month are excluded from the count target and reported separately.
2. Resolve the county from the court name and the parsed incident location. Use
   the province and city hierarchy when the county name is ambiguous.
3. Count one record per case number and county-month. Duplicate case numbers
   are merged.
4. Record quality flags for usable incident time, usable location, fine
   location parsing and valid UST code.
5. Months with no record after county eligibility are stored as zero, with a
   `panel_observed_mask` flag that identifies zero counts.
6. County eligibility uses the support gate in the paper: at least six months
   with a positive count and at least 15 records in the six-month panel. The
   stricter 12/30 and looser 3/10 rules are sensitivity settings.

The monthly panel is the level at which lag, rolling and calendar features are
defined. Nothing in a row for month `t` reads month `t+1` or later.

## Stage 2: monthly panel to samples

For each county and forecast origin `t`:

- history is months `t-5` to `t`;
- targets are months `t+1` to `t+3`;
- `history_total_case_count_seq` holds the six monthly counts;
- `history_feature_matrix` holds six rows of 52 features;
- type counts and ratios are stored for the first target month.

Lag columns are counts shifted by 1 to 6 months. Rolling means use the months
that exist before the row month. `month_sin` and `month_cos` use the row month.
Quality, composition, bucket and spatial columns are copied from the monthly
panel row. `scripts/build_samples_from_panel.py` performs this step and checks
lag and rolling consistency against `total_case_count`.

## Stage 3: splits

The forward split orders forecast origins and assigns 70% to training, 15% to
validation and 15% to test. For 90 origins this gives 62, 14 and 14 months and
1,536, 507 and 324 rows. Rolling evaluation uses a training prefix, the next
month for validation and the following month for test, repeated over 83
origins. County folds use the first eight hex digits of the county key modulo
five.

## Leakage rules

- Feature construction reads only months up to the origin.
- Prior selection, endpoint selection and checkpoint selection use validation
  rows only.
- Columns labelled `*_county` and the two panel-boundary columns are excluded
  by the `leakage_safe` feature mode used for the manuscript.
- Conformal calibration uses residuals from earlier origins only.
