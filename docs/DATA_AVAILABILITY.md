# Data and code availability

**Code.** Model, training, evaluation and split code are in this repository
under the MIT licence. `README.md` lists the commands that regenerate the
forward and rolling results.

**Derived panel.** `data/full_panel/` contains the 40-county, 2,367-row derived
panel used for the reported count forecasts, together with its monthly county
panel. `data/sample/` contains a ten-county subset for quick runs. Both hold
aggregate counts, quality rates and spatial summaries. They contain no
judgment text, case numbers or personal details, and they are not covered by
the code licence; use is subject to the terms of the source collection.

**Source documents.** Criminal judgment documents are published on China
Judgments Online (https://wenshu.court.gov.cn/). The authors do not redistribute
the documents. `DATA_SOURCES.md` records the fields used, the two release
windows and the construction rules. The 2020--2025 export is identified by the
source URL stored with each record; the export itself is not redistributed.

**External panels.** FBI Crime Data Explorer, Queensland Police Service,
JUSTFAIR 2.0 and Taiwan National Police Agency data are public and are not
redistributed here. `DATA_SOURCES.md` gives their owners and access pages.

**Integrity.** `MANIFEST.json` lists the size and SHA256 of every released file.
Run `python scripts/verify_manifest.py` to check a copy. The full-sample
artifact used in the manuscript has SHA256
`6d944ffe4f46bef610707e6351b44307624fe6197084c78e9e9d8e5095d0f648`.
