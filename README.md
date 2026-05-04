# GRS-ANFIS Reproducibility Code

This repository contains the source code and public Jupyter notebooks used to reproduce the GRS-ANFIS paper experiments. Dataset files, trained checkpoints, generated tables, figures, and reviewer-response artifacts are intentionally excluded.

## Repository Layout

```text
.
├── notebooks/
│   ├── 01_main_performance.ipynb
│   └── 02_additional_experiments.ipynb
├── hyper_parameter/
│   └── best_*.json
├── data/
│   └── README.md
├── model.py
├── learning.py
├── data.py
├── gh_config.py
├── gh_eval_utils.py
├── interpretability.py
├── feature_schema.py
├── utils.py
└── requirements.txt
```

## Notebooks

- `notebooks/01_main_performance.ipynb`: main performance workflows, including the full main experiment and 5-fold BCWD/Vowel/GH-ANFIS runs.
- `notebooks/02_additional_experiments.ipynb`: additional analyses, including complementary-boundary ablation, feature-count summaries, IF-THEN rule inspection, and BCWD interpretation cases.

Both notebooks are committed without execution outputs so that the GitHub repository contains code only.

## Data Policy

The datasets are not included in this repository. Place local dataset snapshots under `data/` only when running the notebooks locally. The `.gitignore` file excludes dataset snapshots and generated artifacts from version control.

See `data/README.md` for the expected local data files.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open one of the two notebooks and run the cells in order.

## Reproducibility Notes

- Hyperparameter JSON files under `hyper_parameter/` are versioned because they are experiment configuration, not generated results.
- Generated summaries, checkpoints, figures, and reviewer-response tables are excluded from git.
- If a notebook requires a dataset snapshot that is absent from `data/`, restore it locally before running the corresponding section.
