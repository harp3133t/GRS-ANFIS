# External Data

This directory is intentionally excluded from git except for this note and
placeholder files.

Expected local snapshots:

- `vowel_openml_307.csv`
- `spambase_uci_94.csv`
- `bcwd_uci_15.csv`
- `gisette_train.data`
- `gisette_train.labels`
- `gisette_valid.data`
- `gisette_test.data`
- `gisette.param`

`data.py` uses local snapshots first when they are present. The Vowel, Spambase,
and BCWD loaders can fetch their source datasets when local snapshots are
missing, but Gisette requires the local files above.

To recreate these snapshots on a fresh clone, run:

```text
notebooks/reproducibility/00_download_datasets.ipynb
```

Source IDs:

- Vowel: OpenML data id `307` (the OpenML record cites the UCI Vowel source).
- Spambase: UCI id `94`.
- Breast Cancer Wisconsin (Original): UCI id `15`.
- Gisette: UCI id `170`, downloaded as raw archive files.
