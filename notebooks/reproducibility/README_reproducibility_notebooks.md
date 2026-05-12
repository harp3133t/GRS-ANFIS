# Reproducibility Notebook Suite

This directory contains the notebook workflow for regenerating the manuscript tables from direct experiment code. The notebooks write outputs under `output/reproducible_paper_tables/` and avoid hard-coded manuscript numbers or silent imports from previous result tables.

## Before Running

Install the project requirements:

```bash
pip install -r requirements.txt
pip install -r requirements-extras.txt
pip install -r notebooks/reproducibility/requirements_rebuttal_extras.txt
```

External data files are not stored in GitHub. Run `00_download_datasets.ipynb` first on a fresh clone. It recreates the local snapshots expected by `data.py` from OpenML/UCI sources.

## Execution Order

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

Notebook 01 writes ANFIS-family/SVM fold artifacts to `output/reproducible_paper_tables/model_artifacts/01_main_neuro_fuzzy_and_svm_experiments/cv_weights/`. Notebook 08 loads those checkpoints and writes `08_saved_model_interpretability_{folds,summary}.csv`. Notebook 02 writes tree/boosting/EBM artifacts to `output/reproducible_paper_tables/model_artifacts/02_tree_boosting_and_ebm_baselines/`. Notebook 05 reads GRS-ANFIS artifacts from notebook 01's artifact directory.

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

`08_saved_model_interpretability_from_checkpoints.ipynb` is a supporting notebook for Tables 7-10. It recomputes saved-model Nauck/HFSi interpretability columns from notebook 01 checkpoints without retraining.

## Generated Outputs

- `00_*.csv`: dataset audit and hyperparameter/protocol audit outputs.
- `01_main_neuro_fuzzy_{folds,summary}.csv`: ANFIS-family, SVM, and GRS-ANFIS fold and summary results.
- `02_tabular_baseline_{folds,summary,skipped}.csv`: tabular baseline results and dependency skip records.
- `04_*.csv`: threshold, sparsity, schedule, architecture, HTSK/product, and correlated-stress diagnostics.
- `05_boundary/*.csv`: complementary-boundary and case-level recovery outputs.
- `06_interpretability/*.csv` and `figure_a1_top_shap_lime.{png,pdf}`: explanation-form comparison outputs.
- `07_generated_table_inventory.csv`: generated-table inventory and paired-statistics support.
- `manuscript_tables/*.csv`: final manuscript-ready package assembled by notebook 03.

## Policy

- Do not paste numeric table rows by hand.
- Do not read `rebuttal_full_completion/reviewer_response_work/full_completion_tables/*.csv` as evidence inputs.
- Keep fold-level CSVs next to every summary CSV.
- If a dependency is missing, write an explicit skip/status file rather than falling back to legacy numbers.
- `paper_table_map.csv` and `paper_table_map.md` are generated outputs documenting notebook-to-table provenance.
- Use the canonical seed policy: outer folds use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, and randomized model choices are seeded with `SEED=42` unless a notebook documents an exception.

