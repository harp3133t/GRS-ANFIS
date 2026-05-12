# GRS-ANFIS Reproducibility Package

This repository contains the source code and notebook workflow used to reproduce the GRS-ANFIS manuscript tables. The notebooks are designed to regenerate manuscript values from direct experiment code, not from hand-entered table rows or legacy summary files.

## Repository Layout

- `model.py`, `learning.py`, `data.py`, `utils.py`: core model, training, data-loading, and experiment helpers.
- `experiments/reproducible_paper_tables.py`: table-oriented experiment backend called by the reproducibility notebooks.
- `experiments/compute_saved_model_interpretability.py`: checkpoint-based Nauck/HFSi interpretability recomputation.
- `notebooks/reproducibility/`: ordered notebook suite for data audit, model reruns, baseline comparisons, diagnostics, and manuscript table packaging.
- `hyper_parameter/`: canonical experiment settings and selected hyperparameters.
- `data/`: external dataset snapshots, intentionally excluded from GitHub except for `data/README.md`.
- `output/`: generated CSV tables, checkpoints, figures, and manuscript-ready table packages, intentionally excluded from GitHub.

## Setup

Create and activate a Python environment, then install the required packages:

```bash
pip install -r requirements.txt
pip install -r requirements-extras.txt
pip install -r notebooks/reproducibility/requirements_rebuttal_extras.txt
```

The shell wrapper `scripts/python_with_local_deps.sh` is used by the notebooks when they call the experiment backend. If needed, override the Python interpreter explicitly:

```bash
export GRS_ANFIS_PYTHON=/path/to/your/python
```

## Data

Dataset snapshots are not tracked in GitHub. On a fresh clone, run:

```text
notebooks/reproducibility/00_download_datasets.ipynb
```

This creates the files expected by `data.py` under `data/`:

- `vowel_openml_307.csv`
- `spambase_uci_94.csv`
- `bcwd_uci_15.csv`
- `gisette_train.data`
- `gisette_train.labels`
- `gisette_valid.data`
- `gisette_test.data`
- `gisette.param`

The sources are OpenML data id `307` for Vowel, UCI id `94` for Spambase, UCI id `15` for Breast Cancer Wisconsin (Original), and UCI id `170` raw archive files for Gisette.

## Reproduction Order

Run notebooks from the repository root or from `notebooks/reproducibility/`; each notebook locates the project root automatically.

1. `00_download_datasets.ipynb`
2. `00_environment_and_data_audit.ipynb`
3. `01_main_neuro_fuzzy_and_svm_experiments.ipynb`
4. `08_saved_model_interpretability_from_checkpoints.ipynb`
5. `02_tree_boosting_and_ebm_baselines.ipynb`
6. `04_ablation_robustness_and_htsk_analysis.ipynb`
7. `05_complementary_boundary_and_error_recovery.ipynb`
8. `06_interpretability_shap_lime_grs_rule_path.ipynb`
9. `07_statistical_tests_and_submission_tables.ipynb`
10. `03_reviewer_table_package_rebuild.ipynb`

All manuscript-oriented outputs are written to:

```text
output/reproducible_paper_tables/
```

## Manuscript Table Map

| Notebook | Manuscript table number | Responsibility |
|---|---:|---|
| `00_environment_and_data_audit.ipynb` | Table 1 | Benchmark dataset dimensions and task metadata |
| `01_main_neuro_fuzzy_and_svm_experiments.ipynb` | Tables 7-10 | ANFIS-family, SVM, and GRS-ANFIS rows for the Vowel, Spambase, Breast Cancer, and Gisette performance/interpretability tables |
| `02_tree_boosting_and_ebm_baselines.ipynb` | Tables 7-10; Table A8 | Modern tabular baseline rows in the main dataset tables and supplemental full-metric baseline summary |
| `03_reviewer_table_package_rebuild.ipynb` | Tables 1, 7-17, A1-A9 package | Builds manuscript-ready CSV tables from directly generated notebook outputs only |
| `04_ablation_robustness_and_htsk_analysis.ipynb` | Tables 13-14; Tables A1-A6 | Architecture, schedule, threshold, sparsity, HTSK/product, and correlated-stress diagnostics |
| `05_complementary_boundary_and_error_recovery.ipynb` | Table 12; Tables 15-17 | Q1 complementary correction and Breast Cancer case-level recovery tables |
| `06_interpretability_shap_lime_grs_rule_path.ipynb` | Table A7; supports Table 17 | SHAP/LIME/GRS rule-path explanation-form comparison and case-level rule-path support |
| `07_statistical_tests_and_submission_tables.ipynb` | Table A9; inventory | Paired statistics and generated-table inventory |

`08_saved_model_interpretability_from_checkpoints.ipynb` is a supporting notebook: it loads fold checkpoints from notebook 01 and recomputes the saved-model interpretability columns used in Tables 7-10.

## Reproducibility Policy

- Do not commit external dataset snapshots, generated output tables, model checkpoints, or serialized estimators.
- Do not paste manuscript numeric rows by hand.
- Do not use legacy summary tables as evidence inputs.
- Keep fold-level CSVs next to every summary CSV.
- If an optional dependency is missing, write an explicit skip/status file instead of falling back to legacy numbers.
- Use the canonical seed policy in the experiment backend: outer folds use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

