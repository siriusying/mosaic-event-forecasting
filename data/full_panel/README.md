# Full derived panel

`forecast_samples_l6_h123.parquet` holds the 2,367 windowed samples used for
the forward and rolling results in the manuscript: 40 counties, 90 forecast
origins, six history months and three target months. The forward split is
1,536 training, 507 validation and 324 test rows.

`county_month_panel.csv` holds the matching monthly rows with counts, quality
rates and spatial summaries. It is the input to
`scripts/build_samples_from_panel.py`.

Both files contain aggregate counts and derived features. They contain no
document text, case numbers or personal details. The source collection is
described in `DATA_SOURCES.md`.

SHA256 values are listed in `MANIFEST.json`.
