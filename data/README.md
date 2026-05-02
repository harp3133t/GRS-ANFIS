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
