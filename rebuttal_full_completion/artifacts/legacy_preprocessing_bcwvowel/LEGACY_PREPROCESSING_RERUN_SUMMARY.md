# BCWD/Vowel Legacy-Preprocessing Rerun Summary

Date: 2026-04-30

## Preprocessing source

- Legacy/original-code root: `/home/harp3133t/Research/03_Research/GH-ANFIS_exp`
- BCWD feature count after preprocessing: 80
- Vowel feature count after preprocessing: 29

These replace the current E404 preprocessing dimensions for the affected rerun scope, where BCWD/Vowel had been reduced differently by the current loader.

## Completed reruns

### MAT-10 tabular baselines

Output directory:

`artifacts/legacy_preprocessing_bcwvowel/mat10_tabular_baselines/`

Completed rows:

- 2 datasets
- 6 baseline model families
- 5 folds each
- 60 fold-result rows

Source-of-truth files:

- `tabular_baselines_summary.csv`
- `tabular_baselines_folds.csv`
- `baseline_fairness_table.csv`
- `baseline_completion_status.csv`
- `run_config.json`

Mean F1 results:

| Dataset | n_features | RandomForest | HGB | XGBoost | LightGBM | CatBoost | EBM |
|---|---:|---:|---:|---:|---:|---:|---:|
| BCWD | 80 | 0.9467 | 0.9377 | 0.9295 | 0.9381 | 0.9459 | 0.9588 |
| Vowel | 29 | 0.9755 | 0.9143 | 0.9196 | 0.9390 | 0.9031 | 0.9027 |

### MAT-6 architecture ablation: BCWD

Output directory:

`artifacts/legacy_preprocessing_bcwvowel/mat6_architecture_bcwd/`

Completed rows:

- 7 architecture variants
- 5 folds each
- Primary/full/complementary component summaries where applicable
- 85 fold-result rows

Source-of-truth files:

- `summary_by_variant.csv`
- `fold_results.csv`
- `attribution_summary.csv`
- `run_config.json`

The run config records `n_features: 80`.

Selected full-component mean F1 values:

| Variant | Mean F1 |
|---|---:|
| Dense TSK-ANFIS, same total rules/epochs | 0.9462 |
| Full GRS-ANFIS | 0.9338 |
| Joint training, same epoch budget | 0.9304 |
| Product firing instead of HTSK | 0.9286 |
| No disjoint complementary routing | 0.9260 |
| No hard fine-tune, same epoch budget | 0.9129 |

### MAT-6 architecture ablation: Vowel

Output directory:

`artifacts/legacy_preprocessing_bcwvowel/mat6_architecture_vowel/`

Completed rows:

- 7 architecture variants
- 5 folds each
- Primary/full/complementary component summaries where applicable
- 85 fold-result rows

Source-of-truth files:

- `summary_by_variant.csv`
- `fold_results.csv`
- `attribution_summary.csv`
- `run_config.json`

The run config records `n_features: 29`.

Selected full-component mean F1 values:

| Variant | Mean F1 |
|---|---:|
| Dense TSK-ANFIS, same total rules/epochs | 0.9726 |
| No disjoint complementary routing | 0.9695 |
| Full GRS-ANFIS | 0.9656 |
| Product firing instead of HTSK | 0.9474 |
| Joint training, same epoch budget | 0.7881 |
| No hard fine-tune, same epoch budget | 0.7731 |

## Caveat

For both MAT-6 runs, the training and CSV/JSON artifact generation completed. The script exited only at the optional Markdown `REPORT.md` generation step because the environment lacks the optional Python package `tabulate`.

## Scope note

MAT-5 and MAT-8 were already based on the legacy `GH-ANFIS_exp` preprocessing artifacts, so their BCWD/Vowel feature counts already match the original-code dimensions. Other current-loader-based evidence tiers should be refreshed separately if they are to be included in the rebuttal package with BCWD/Vowel feature counts fully harmonized.
