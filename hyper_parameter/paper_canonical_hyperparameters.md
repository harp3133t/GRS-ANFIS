# Paper-Matching Canonical Hyperparameters

This file records the single hyperparameter source used to align the reproducibility code with the manuscript tables.

## Canonical source

Use the JSON files in this directory as the canonical source for experiments and manuscript setup tables. The reproducibility backend loads these files through `utils.py` rather than maintaining separate notebook-local hyperparameter blocks.

Current sources:

- ANFIS: `hyper_parameter/paper_ANFIS_HP.json`
- GA-ANFIS: `hyper_parameter/best_GA-ANFIS_HP.json`
- PSO-ANFIS: `hyper_parameter/best_PSO-ANFIS_HP.json`
- H-ANFIS: `hyper_parameter/best_H-ANFIS_HP.json`
- GRS-ANFIS: `hyper_parameter/best_GRS-ANFIS_HP.json`
- SVM/common protocol: `hyper_parameter/common_experiment_settings.json`
- Tree/boosting/EBM candidate grids: `hyper_parameter/tabular_baseline_candidates.json`

GA-ANFIS and PSO-ANFIS read `selected_features` from their JSON files. For one-hot encoded datasets such as BCWD, a raw selected feature name expands to the corresponding processed model-input columns.

The generated setup-table outputs are written by notebook 00 / `--task data-audit`:

- `output/reproducible_paper_tables/00_hyperparameter_ranges.csv`
- `output/reproducible_paper_tables/00_common_experiment_settings.csv`
- `output/reproducible_paper_tables/00_ga_pso_hyperparameters.csv`
- `output/reproducible_paper_tables/00_h_anfis_hyperparameters.csv`
- `output/reproducible_paper_tables/00_grs_anfis_hyperparameters.csv`
- `output/reproducible_paper_tables/00_ga_pso_selected_feature_resolution.csv`

Paper-matching result artifact:

- `output/maincode_with_ph_no_mi_then_mi100_summary.csv`
- Original matching run: archived provenance, now copied into `output/maincode_with_ph_no_mi_then_mi100_summary.csv` for self-contained verification

## GRS-ANFIS canonical values

| Dataset | lr_primary | lr_complementary | primary_rules | complementary_rules | mf_per_feature | epochs_stage1 | epochs_stage2 | lambda_primary_s1 | lambda_complementary_s2 | weight_decay | primary_hard_epochs | complementary_hard_epochs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Breast_Cancer_Wisconsin_(Original) | 0.1 | 0.005 | 6 | 11 | 2 | 80 | 40 | 0.01 | 0.0 | 1e-5 | 30 | 20 |
| Vowel | 0.1 | 0.005 | 6 | 11 | 2 | 80 | 40 | 0.01 | 0.0 | 1e-5 | 30 | 20 |
| Spambase | 0.1 | 0.01 | 7 | 11 | 3 | 60 | 50 | 0.001 | 0.001 | 1e-6 | 50 | 50 |
| Gisette | 0.01 | 0.01 | 7 | 11 | 2 | 30 | 30 | 0.01 | 0.001 | 1e-4 | 20 | 10 |

For all datasets, `primary_mask_threshold` and `complementary_mask_threshold` are `null`, and the hard-stage sparsity terms are `0.0`.

## Naming note

Older artifacts use `base` and `residual`; the current code uses `primary` and `complementary`. These are the same role pair:

- `base` -> `primary`
- `residual` -> `complementary`

## Baseline references

The main non-GRS settings are now file-backed:

- ANFIS: `hyper_parameter/paper_ANFIS_HP.json`
- SVM: `C=1.0`, `gamma='scale'`, `kernel='rbf'`
- GA/PSO-ANFIS: `hyper_parameter/best_GA-ANFIS_HP.json` and `hyper_parameter/best_PSO-ANFIS_HP.json`
- H/PH-ANFIS: `hyper_parameter/best_H-ANFIS_HP.json`
