# Data sources

## Primary panel: Chinese criminal judgment documents

The county-month counts come from public criminal judgment documents published
on **China Judgments Online** (中国裁判文书网), operated by the Supreme People's
Court of the People's Republic of China: https://wenshu.court.gov.cn/.

Two release windows were used. The 2010--2018 panel was assembled from public
judgments covering that period. The 2020--2025 extension was assembled from a
separate public export whose records carry the source page URL, judgment date
and publication date. The export itself is not redistributed here. Judgment
dates, publication dates and case numbers were used to audit coverage and
detect duplicates. Incident month, offense type, county and time-of-day bucket
were parsed from the judgment text.

The derived panel keeps one row per county and calendar month with:

- the number of normalized criminal records;
- offense-type counts for six coarse categories;
- share of records with a usable incident time, location and UST code;
- counts in six four-hour buckets;
- coarse spatial summaries (unique cells and UST code frequencies).

Coordinates in the panel are administrative centroids. Two derived releases are
included: the complete 40-county panel used in the paper (`data/full_panel/`)
and a ten-county subset with the longest histories (`data/sample/`). Both hold
aggregate counts and derived features only; they contain no document text, case
identifiers, names or party details. Rebuilding the panel from documents
follows the rules in [docs/PANEL_CONSTRUCTION.md](docs/PANEL_CONSTRUCTION.md).
Access to China Judgments Online is subject to the site's terms of use, which
change over time; the authors do not redistribute the documents.

## External panels (not included)

These panels are used for transfer checks in the paper. They are not part of
this package and are not pooled with the primary panel.

| Source | Owner | Access | Used for |
|---|---|---|---|
| Crime Data Explorer, summarized state and agency endpoints | U.S. Federal Bureau of Investigation | https://cde.ucr.cjis.gov/LATEST/ | State panel (51 entities) and California county panel (58 counties) |
| Offences Against the Person by local government area | Queensland Police Service | https://www.data.qld.gov.au/ | 78 LGAs, 2001-01 to 2026-07 |
| JUSTFAIR 2.0 | Topaz et al. | https://github.com/chadtopaz/justfair-2.0; DOI 10.31235/osf.io/pg69n_v2 | 140,299 federal sentencing records, 87 districts |
| Crime statistics open data | Taiwan National Police Agency | https://data.gov.tw/ | 46 resources, 447,173 rows; provenance check only |

Retrieval windows and per-file hashes for the FBI panels are recorded in the
manuscript's result package. They are regenerated from the public API and are
not shipped here.

## Derived artefacts in this repository

| File | Derivation |
|---|---|
| `data/sample/county_month_panel.csv` | Monthly aggregation of source records for ten counties |
| `data/full_panel/county_month_panel.csv` | Monthly aggregation for all 40 counties in the paper |
| `data/full_panel/forecast_samples_l6_h123.parquet` | Full windowed panel, 2,367 rows |
| `data/sample/mosaic_sample_l6_h123.parquet` | Six-month windows built from the monthly panel |
| `data/splits/forward_split.json` | Chronological 70/15/15 split over forecast origins |
| `data/splits/rolling_origins.csv` | Train prefix, next month for validation, following month for test |
| `data/splits/county_folds.json` | `sha256(county_id)[:8] % 5` folds |
| `data/schema/feature_definitions.csv` | Column-level definition and leakage note |
| `data/metadata/provenance.json` | Input hashes, row counts and rebuild steps |

The full-sample artifact used for the manuscript has SHA256
`6d944ffe4f46bef610707e6351b44307624fe6197084c78e9e9d8e5095d0f648`.
`scripts/build_samples_from_panel.py` regenerates the released sample from the
monthly panel; a test checks that counts, targets and feature matrices match the
original artifact for the released counties.
