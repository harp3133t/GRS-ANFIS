#!/usr/bin/env python3
"""Build the complete reviewer-facing experiment table package.

The earlier reviewer table used placeholder coverage rows for several high-cost
Gisette analyses. After the completion reruns, this script fills those rows from
the new CSV artifacts and regenerates a single table package for the rebuttal.

Run from the Multica workdir:

    python scripts/build_full_experiment_tables.py

Optional DOCX/PDF conversion requires LibreOffice:

    python scripts/build_full_experiment_tables.py --convert-docs
"""

from __future__ import annotations

import argparse
import csv
import html
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


DISPLAY_DATASET = {
    "Breast_Cancer_Wisconsin_(Original)": "Breast Cancer",
    "Breast Cancer": "Breast Cancer",
    "BCWD": "Breast Cancer",
    "vowel": "Vowel",
    "Vowel": "Vowel",
    "Spambase": "Spambase",
    "Gisette": "Gisette",
}

VARIANT_ORDER = {
    "full_grs": 0,
    "primary_only_same_budget": 1,
    "joint_training_same_budget": 2,
    "stage1_unfreeze_refinement": 3,
    "sequential_freeze": 4,
    "joint_no_freeze": 5,
    "no_disjoint": 6,
    "no_hard_finetune_same_budget": 7,
    "dense_tsk_same_rules": 8,
    "prod_firing": 9,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=(
            "Workspace root containing artifacts/ and reviewer_response_work/. "
            "If omitted, the script auto-detects either the current directory "
            "or the parent directory of this script."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reviewer_response_work/full_completion_tables"),
        help="Directory for generated CSV/HTML/DOCX/PDF outputs.",
    )
    parser.add_argument(
        "--convert-docs",
        action="store_true",
        help="Also convert the generated HTML to DOCX and PDF with LibreOffice.",
    )
    return parser.parse_args()


def has_required_sources(path: Path) -> bool:
    return (
        (path / "artifacts/gisette_full_completion").exists()
        and (path / "artifacts/legacy_preprocessing_bcwvowel").exists()
    )


def infer_root() -> Path:
    candidates = [
        Path.cwd(),
        Path(__file__).resolve().parent.parent,
    ]
    for candidate in candidates:
        if has_required_sources(candidate):
            return candidate.resolve()
    return Path.cwd().resolve()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required source is missing: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def f4(value: str | float | int | None) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.4f}"


def f2(value: str | float | int | None) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.2f}"


def mean_std(row: dict[str, str], mean_col: str, std_col: str, decimals: int = 4) -> str:
    mean = row.get(mean_col, "")
    std = row.get(std_col, "")
    if mean == "":
        return ""
    if std == "":
        return f"{float(mean):.{decimals}f}"
    return f"{float(mean):.{decimals}f} +/- {float(std):.{decimals}f}"


def dataset_name(name: str) -> str:
    return DISPLAY_DATASET.get(name, name)


def row_by_value(rows: Iterable[dict[str, str]], column: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(column) == value:
            return row
    raise KeyError(f"No row where {column}={value}")


def build_threshold_table(root: Path) -> list[dict[str, str]]:
    rows = [
        {
            "dataset": "Breast Cancer",
            "tau_0.3_f1": "0.9415",
            "tau_0.4_f1": "0.9542",
            "tau_0.5_f1": "0.9524",
            "tau_0.6_f1": "0.9412",
            "tau_0.7_f1": "0.9267",
            "selected_at_0.5": "75.8",
            "mask_jaccard_at_0.5": "0.9045",
            "source": "MAT-5 legacy/revision robustness summary",
        },
        {
            "dataset": "Vowel",
            "tau_0.3_f1": "0.9394",
            "tau_0.4_f1": "0.9524",
            "tau_0.5_f1": "0.9606",
            "tau_0.6_f1": "0.9334",
            "tau_0.7_f1": "0.8868",
            "selected_at_0.5": "21.0",
            "mask_jaccard_at_0.5": "0.7909",
            "source": "MAT-5 legacy/revision robustness summary",
        },
        {
            "dataset": "Spambase",
            "tau_0.3_f1": "0.8842",
            "tau_0.4_f1": "0.9009",
            "tau_0.5_f1": "0.9120",
            "tau_0.6_f1": "0.8658",
            "tau_0.7_f1": "0.7941",
            "selected_at_0.5": "48.0",
            "mask_jaccard_at_0.5": "0.8056",
            "source": "MAT-5 legacy/revision robustness summary",
        },
    ]
    tau_csv = root / "artifacts/gisette_full_completion/mat5_threshold_sparsity_seed_rerun/tau_summary_by_dataset.csv"
    tau_rows = [r for r in read_csv(tau_csv) if r["dataset"] == "Gisette"]
    by_tau = {r["tau"]: r for r in tau_rows}
    row_05 = by_tau["0.5"]
    rows.append(
        {
            "dataset": "Gisette",
            "tau_0.3_f1": f4(by_tau["0.3"]["full_f1_mean"]),
            "tau_0.4_f1": f4(by_tau["0.4"]["full_f1_mean"]),
            "tau_0.5_f1": f4(row_05["full_f1_mean"]),
            "tau_0.6_f1": f4(by_tau["0.6"]["full_f1_mean"]),
            "tau_0.7_f1": f4(by_tau["0.7"]["full_f1_mean"]),
            "selected_at_0.5": f2(row_05["full_selected_mean"]),
            "mask_jaccard_at_0.5": f4(row_05["full_mask_jaccard_mean"]),
            "source": str(tau_csv),
        }
    )
    return rows


def build_lambda_table(root: Path) -> list[dict[str, str]]:
    rows = [
        {"dataset": "Breast Cancer", "lambda": "0", "full_acc": "0.9714 +/- 0.0101", "full_f1": "0.9567 +/- 0.0156", "selected_features": "69.2 +/- 3.2", "mask_jaccard": "0.8102", "pc_overlap": "0.0000", "source": "MAT-5 targeted rerun"},
        {"dataset": "Breast Cancer", "lambda": "0.01", "full_acc": "0.9771 +/- 0.0083", "full_f1": "0.9658 +/- 0.0126", "selected_features": "40.6 +/- 2.7", "mask_jaccard": "0.7334", "pc_overlap": "0.0000", "source": "MAT-5 targeted rerun"},
        {"dataset": "Breast Cancer", "lambda": "0.1", "full_acc": "0.9600 +/- 0.0035", "full_f1": "0.9402 +/- 0.0049", "selected_features": "13.2 +/- 1.3", "mask_jaccard": "0.6486", "pc_overlap": "0.0000", "source": "MAT-5 targeted rerun"},
        {"dataset": "Vowel", "lambda": "0", "full_acc": "0.9616 +/- 0.0118", "full_f1": "0.9614 +/- 0.0119", "selected_features": "26.8 +/- 1.3", "mask_jaccard": "0.9152", "pc_overlap": "0.0000", "source": "MAT-5 targeted rerun"},
        {"dataset": "Vowel", "lambda": "0.01", "full_acc": "0.9545 +/- 0.0090", "full_f1": "0.9544 +/- 0.0091", "selected_features": "21.4 +/- 3.1", "mask_jaccard": "0.7880", "pc_overlap": "0.0000", "source": "MAT-5 targeted rerun"},
        {"dataset": "Vowel", "lambda": "0.1", "full_acc": "0.9394 +/- 0.0146", "full_f1": "0.9387 +/- 0.0146", "selected_features": "10.0 +/- 0.0", "mask_jaccard": "1.0000", "pc_overlap": "0.0000", "source": "MAT-5 targeted rerun"},
        {"dataset": "Spambase", "lambda": "0", "full_acc": "0.9309 +/- 0.0023", "full_f1": "0.9117 +/- 0.0033", "selected_features": "47.4 +/- 1.4", "mask_jaccard": "0.8778", "pc_overlap": "0.0000", "source": "MAT-5 targeted rerun"},
        {"dataset": "Spambase", "lambda": "0.01", "full_acc": "0.9262 +/- 0.0045", "full_f1": "0.9055 +/- 0.0054", "selected_features": "26.8 +/- 2.3", "mask_jaccard": "0.7707", "pc_overlap": "0.0000", "source": "MAT-5 targeted rerun"},
        {"dataset": "Spambase", "lambda": "0.1", "full_acc": "0.9058 +/- 0.0064", "full_f1": "0.8776 +/- 0.0077", "selected_features": "9.2 +/- 0.4", "mask_jaccard": "0.9000", "pc_overlap": "0.0000", "source": "MAT-5 targeted rerun"},
    ]
    src = root / "artifacts/gisette_full_completion/mat5_threshold_sparsity_seed_rerun/lambda_seed_summary.csv"
    for row in read_csv(src):
        if row["dataset"] != "Gisette":
            continue
        rows.append(
            {
                "dataset": "Gisette",
                "lambda": row["lambda_sparsity"],
                "full_acc": mean_std(row, "full_acc_mean", "full_acc_std"),
                "full_f1": mean_std(row, "full_f1_mean", "full_f1_std"),
                "selected_features": mean_std(row, "full_selected_mean", "full_selected_std", decimals=1),
                "mask_jaccard": f4(row["full_mask_jaccard_mean"]),
                "pc_overlap": f4(row["primary_complementary_overlap_mean"]),
                "source": str(src),
            }
        )
    return rows


def build_checkpoint_table() -> list[dict[str, str]]:
    return [
        {"dataset": "Breast Cancer", "full_f1": "0.9630", "primary_f1": "0.9585", "delta": "+0.0044", "effective_overlap": "0.0", "source": "MAT-6 checkpoint attribution"},
        {"dataset": "Vowel", "full_f1": "0.9662", "primary_f1": "0.9651", "delta": "+0.0011", "effective_overlap": "0.0", "source": "MAT-6 checkpoint attribution"},
        {"dataset": "Spambase", "full_f1": "0.9136", "primary_f1": "0.9059", "delta": "+0.0076", "effective_overlap": "0.0", "source": "MAT-6 checkpoint attribution"},
        {"dataset": "Gisette", "full_f1": "0.9758", "primary_f1": "0.9755", "delta": "+0.0003", "effective_overlap": "0.0", "source": "MAT-6 checkpoint attribution"},
    ]


def build_sequential_table(root: Path) -> list[dict[str, str]]:
    rows = [
        {"dataset": "Breast Cancer", "sequential_f1": "0.9373", "joint_no_freeze_f1": "0.9171", "stage1_unfreeze_f1": "0.9288", "effective_overlap": "0.0", "claim_boundary": "numeric schedule comparison"},
        {"dataset": "Vowel", "sequential_f1": "0.9708", "joint_no_freeze_f1": "0.9347", "stage1_unfreeze_f1": "0.9696", "effective_overlap": "0.0", "claim_boundary": "numeric schedule comparison"},
        {"dataset": "Spambase", "sequential_f1": "0.9191", "joint_no_freeze_f1": "0.8045", "stage1_unfreeze_f1": "0.9056", "effective_overlap": "0.0", "claim_boundary": "numeric schedule comparison"},
    ]
    src = root / "artifacts/gisette_full_completion/mat7_sequential_gisette/comparison_table.csv"
    data = {r["variant"]: r for r in read_csv(src)}
    seq = data["sequential_freeze"]
    joint = data["joint_no_freeze"]
    unfreeze = data["stage1_unfreeze_refinement"]
    rows.append(
        {
            "dataset": "Gisette",
            "sequential_f1": f4(seq["final_val_f1_mean"]),
            "joint_no_freeze_f1": f4(joint["final_val_f1_mean"]),
            "stage1_unfreeze_f1": f4(unfreeze["final_val_f1_mean"]),
            "effective_overlap": f4(seq["mask_overlap_mean"]),
            "claim_boundary": "numeric fixed-fold rerun; mixed result, so no universal sequential-superiority claim",
        }
    )
    return rows


def build_architecture_table(root: Path) -> list[dict[str, str]]:
    sources = [
        root / "artifacts/legacy_preprocessing_bcwvowel/mat6_architecture_bcwd/attribution_summary.csv",
        root / "artifacts/legacy_preprocessing_bcwvowel/mat6_architecture_vowel/attribution_summary.csv",
        root / "artifacts/gisette_full_completion/mat6_architecture_spambase/attribution_summary.csv",
        root / "artifacts/gisette_full_completion/mat6_architecture_gisette/attribution_summary.csv",
    ]
    rows: list[dict[str, str]] = []
    for src in sources:
        for row in read_csv(src):
            rows.append(
                {
                    "dataset": dataset_name(row["dataset"]),
                    "variant": row["variant"],
                    "variant_label": row["variant_label"],
                    "full_acc": mean_std(row, "full_acc_mean", "full_acc_std"),
                    "full_f1": mean_std(row, "full_f1_mean", "full_f1_std"),
                    "selected_features": f2(row["selected_feature_count_mean"]),
                    "mask_overlap_rate": f4(row.get("mask_overlap_rate_mean")),
                    "primary_f1": f4(row.get("primary_f1_mean")),
                    "full_minus_primary_f1": f4(row.get("full_minus_primary_f1")),
                    "source": str(src),
                }
            )
    rows.sort(key=lambda r: (r["dataset"], VARIANT_ORDER.get(r["variant"], 99)))
    return rows


def build_correlated_table() -> list[dict[str, str]]:
    return [
        {"dataset": "Breast Cancer", "acc_delta": "0.0000", "selected_delta": "+3.2", "raw_jaccard": "0.8003", "collapsed_jaccard": "0.8207", "role_assignment_summary": "same 64%, single 24%, neither 12%", "scope": "full dataset"},
        {"dataset": "Vowel", "acc_delta": "+0.0091", "selected_delta": "+8.2", "raw_jaccard": "0.9102", "collapsed_jaccard": "0.8930", "role_assignment_summary": "same 100%", "scope": "full dataset"},
        {"dataset": "Spambase", "acc_delta": "+0.0017", "selected_delta": "0.0", "raw_jaccard": "0.7412", "collapsed_jaccard": "0.7970", "role_assignment_summary": "same 28%, single 44%, neither 8%, split 20%", "scope": "full dataset"},
        {"dataset": "Gisette", "acc_delta": "0.0000", "selected_delta": "+5.2", "raw_jaccard": "0.8745", "collapsed_jaccard": "0.8732", "role_assignment_summary": "same 76%, single 4%, neither 20%", "scope": "representative top-100 subset"},
    ]


def build_firing_table(root: Path) -> list[dict[str, str]]:
    rows = [
        {"dataset": "Breast Cancer", "mode": "product", "acc": "0.9657", "f1": "0.9501", "base_near_zero": "0.1354", "comp_near_zero": "0.0000", "comp_max_weight": "0.1482", "comp_effective_rules": "10.16", "source": "MAT-9 product-vs-HTSK summary"},
        {"dataset": "Breast Cancer", "mode": "HTSK", "acc": "0.9557", "f1": "0.9353", "base_near_zero": "0.0000", "comp_near_zero": "0.0000", "comp_max_weight": "0.1032", "comp_effective_rules": "10.94", "source": "MAT-9 product-vs-HTSK summary"},
        {"dataset": "Spambase", "mode": "product", "acc": "0.9270", "f1": "0.9056", "base_near_zero": "0.0710", "comp_near_zero": "0.1218", "comp_max_weight": "0.4449", "comp_effective_rules": "3.84", "source": "MAT-9 product-vs-HTSK summary"},
        {"dataset": "Spambase", "mode": "HTSK", "acc": "0.9287", "f1": "0.9078", "base_near_zero": "0.0000", "comp_near_zero": "0.0000", "comp_max_weight": "0.1514", "comp_effective_rules": "10.15", "source": "MAT-9 product-vs-HTSK summary"},
    ]
    src = root / "artifacts/gisette_full_completion/htsk_firing_vowel_gisette/reviewer_table.csv"
    for row in read_csv(src):
        rows.append(
            {
                "dataset": dataset_name(row["dataset"]),
                "mode": "HTSK" if row["mode"] == "htsk" else row["mode"],
                "acc": row["acc"],
                "f1": row["f1"],
                "base_near_zero": row["base_near_zero"],
                "comp_near_zero": row["resid_near_zero"],
                "comp_max_weight": row["resid_max_w"],
                "comp_effective_rules": row["resid_eff_rules"],
                "source": str(src),
            }
        )
    rows.sort(key=lambda r: (r["dataset"], r["mode"]))
    return rows


def build_strong_baselines_table() -> list[dict[str, str]]:
    return [
        {"dataset": "Breast Cancer", "input_dim": "80", "RF": "0.9467", "HGB": "0.9377", "XGBoost": "0.9295", "LightGBM": "0.9381", "CatBoost": "0.9459", "EBM": "0.9588"},
        {"dataset": "Vowel", "input_dim": "29", "RF": "0.9755", "HGB": "0.9143", "XGBoost": "0.9196", "LightGBM": "0.9390", "CatBoost": "0.9031", "EBM": "0.9027"},
        {"dataset": "Spambase", "input_dim": "57", "RF": "0.9401", "HGB": "0.9374", "XGBoost": "0.9375", "LightGBM": "0.9419", "CatBoost": "0.9347", "EBM": "0.9345"},
        {"dataset": "Gisette", "input_dim": "5000", "RF": "0.9700", "HGB": "0.9733", "XGBoost": "0.9774", "LightGBM": "0.9792", "CatBoost": "0.9732", "EBM": "0.9740"},
    ]


def build_local_deletion_table() -> list[dict[str, str]]:
    return [
        {"dataset": "Breast Cancer", "grs_intrinsic": "-0.0089", "shap": "0.1630", "lime": "0.0687", "interpretation": "SHAP strongest under local perturbation"},
        {"dataset": "Vowel", "grs_intrinsic": "0.7333", "shap": "0.8338", "lime": "0.6611", "interpretation": "all methods respond, SHAP highest"},
        {"dataset": "Spambase", "grs_intrinsic": "0.0693", "shap": "0.1582", "lime": "0.0065", "interpretation": "SHAP strongest, LIME weak"},
        {"dataset": "Gisette", "grs_intrinsic": "0.00065", "shap": "0.00065", "lime": "-0.00001", "interpretation": "deletion effect is negligible"},
    ]


def build_intrinsic_table() -> list[dict[str, str]]:
    return [
        {"dataset": "Breast Cancer", "mask_stability": "1.0000", "primary_coverage": "1.0000", "complementary_coverage": "1.0000", "top3_rule_agreement": "0.9928"},
        {"dataset": "Vowel", "mask_stability": "1.0000", "primary_coverage": "1.0000", "complementary_coverage": "1.0000", "top3_rule_agreement": "0.9101"},
        {"dataset": "Spambase", "mask_stability": "0.8490", "primary_coverage": "0.6571", "complementary_coverage": "1.0000", "top3_rule_agreement": "0.9763"},
        {"dataset": "Gisette", "mask_stability": "0.6011", "primary_coverage": "1.0000", "complementary_coverage": "1.0000", "top3_rule_agreement": "0.9998"},
    ]


def build_paired_stats_table() -> list[dict[str, str]]:
    return [
        {"dataset": "Breast Cancer", "comparator": "ANFIS", "acc_delta": "+0.0100", "acc_p": "0.0249", "f1_delta": "+0.0144", "f1_p": "0.0301"},
        {"dataset": "Vowel", "comparator": "GA-ANFIS", "acc_delta": "+0.1909", "acc_p": "0.0019", "f1_delta": "+0.1944", "f1_p": "0.0020"},
        {"dataset": "Vowel", "comparator": "PSO-ANFIS", "acc_delta": "+0.1455", "acc_p": "0.0032", "f1_delta": "+0.1466", "f1_p": "0.0036"},
        {"dataset": "Vowel", "comparator": "H-ANFIS(Avg)", "acc_delta": "+0.0273", "acc_p": "0.0197", "f1_delta": "+0.0271", "f1_p": "0.0202"},
        {"dataset": "Vowel", "comparator": "SVM", "acc_delta": "+0.1141", "acc_p": "<0.001", "f1_delta": "+0.1152", "f1_p": "<0.001"},
        {"dataset": "Spambase", "comparator": "GRS-ANFIS(base)", "acc_delta": "+0.0061", "acc_p": "0.0175", "f1_delta": "+0.0076", "f1_p": "0.0216"},
        {"dataset": "Spambase", "comparator": "GA-ANFIS", "acc_delta": "+0.0154", "acc_p": "0.0020", "f1_delta": "+0.0229", "f1_p": "<0.001"},
        {"dataset": "Spambase", "comparator": "PSO-ANFIS", "acc_delta": "+0.0352", "acc_p": "0.0011", "f1_delta": "+0.0520", "f1_p": "<0.001"},
        {"dataset": "Spambase", "comparator": "H-ANFIS(Avg)", "acc_delta": "+0.0130", "acc_p": "0.0026", "f1_delta": "+0.0188", "f1_p": "0.0014"},
        {"dataset": "Spambase", "comparator": "SVM", "acc_delta": "+0.2204", "acc_p": "<0.001", "f1_delta": "+0.3624", "f1_p": "<0.001"},
        {"dataset": "Gisette", "comparator": "ANFIS", "acc_delta": "+0.0263", "acc_p": "<0.001", "f1_delta": "+0.0268", "f1_p": "<0.001"},
        {"dataset": "Gisette", "comparator": "GA-ANFIS", "acc_delta": "+0.0470", "acc_p": "<0.001", "f1_delta": "+0.0469", "f1_p": "<0.001"},
        {"dataset": "Gisette", "comparator": "PSO-ANFIS", "acc_delta": "+0.0527", "acc_p": "<0.001", "f1_delta": "+0.0525", "f1_p": "<0.001"},
        {"dataset": "Gisette", "comparator": "H-ANFIS(Avg)", "acc_delta": "+0.0035", "acc_p": "0.0146", "f1_delta": "+0.0036", "f1_p": "0.0118"},
    ]


def build_coverage_table() -> list[dict[str, str]]:
    return [
        {"evidence_block": "Threshold tau sweep", "bcwd": "numeric", "vowel": "numeric", "spambase": "numeric", "gisette": "numeric", "claim_boundary": "4-dataset sensitivity evidence"},
        {"evidence_block": "Sparsity lambda/seed retraining", "bcwd": "numeric", "vowel": "numeric", "spambase": "numeric", "gisette": "numeric", "claim_boundary": "Gisette row is fixed fold 1, seeds 0-4; avoid claiming 5-fold lambda sweep"},
        {"evidence_block": "Full vs Primary checkpoint attribution", "bcwd": "numeric", "vowel": "numeric", "spambase": "numeric", "gisette": "numeric", "claim_boundary": "4-dataset Full-Primary delta and zero hard-route overlap"},
        {"evidence_block": "Sequential vs joint schedule", "bcwd": "numeric", "vowel": "numeric", "spambase": "numeric", "gisette": "numeric", "claim_boundary": "Gisette is numeric but mixed; no universal sequential-superiority claim"},
        {"evidence_block": "Architecture training-tier ablation", "bcwd": "numeric", "vowel": "numeric", "spambase": "numeric", "gisette": "numeric", "claim_boundary": "Use for attribution; preserve cases where dense/no-disjoint is higher"},
        {"evidence_block": "Correlated-copy stress", "bcwd": "numeric", "vowel": "numeric", "spambase": "numeric", "gisette": "numeric top-100 subset", "claim_boundary": "Group-level explanation under high collinearity"},
        {"evidence_block": "Product-vs-HTSK firing comparison", "bcwd": "numeric", "vowel": "numeric", "spambase": "numeric", "gisette": "numeric", "claim_boundary": "HTSK as numerical scale-control, not sole performance cause"},
        {"evidence_block": "Strong tabular baselines", "bcwd": "numeric", "vowel": "numeric", "spambase": "numeric", "gisette": "numeric", "claim_boundary": "Do not claim universal dominance over all tabular baselines"},
        {"evidence_block": "SHAP/LIME local deletion", "bcwd": "numeric", "vowel": "numeric", "spambase": "numeric", "gisette": "numeric", "claim_boundary": "GRS explanations are complementary, not universally more faithful locally"},
        {"evidence_block": "Intrinsic rule-path metrics", "bcwd": "numeric", "vowel": "numeric", "spambase": "numeric", "gisette": "numeric", "claim_boundary": "Model-native rule-path evidence"},
        {"evidence_block": "Paired statistics", "bcwd": "numeric", "vowel": "numeric", "spambase": "numeric", "gisette": "numeric", "claim_boundary": "Current revision-rerun fold evidence only"},
    ]


def html_table(title: str, rows: list[dict[str, str]], columns: list[str]) -> str:
    out = [f"<h2>{html.escape(title)}</h2>", '<table class="small">']
    out.append("<tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in columns) + "</tr>")
    for row in rows:
        out.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in columns) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def write_html_report(path: Path, table_specs: list[tuple[str, list[dict[str, str]], list[str]]]) -> None:
    body = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        "<title>Full Completion Additional Experiment Tables</title>",
        "<style>",
        'body { font-family: "Times New Roman", serif; font-size: 10.5pt; line-height: 1.35; color: #111; }',
        "h1 { font-size: 18pt; text-align: center; margin: 18pt 0 12pt; }",
        "h2 { font-size: 13pt; margin: 16pt 0 8pt; }",
        "p { margin: 5pt 0; }",
        "table { border-collapse: collapse; width: 100%; margin: 7pt 0 12pt; }",
        "th, td { border: 1px solid #333; padding: 4pt; vertical-align: top; }",
        "th { background: #f2f2f2; font-weight: bold; }",
        ".small { font-size: 8.6pt; }",
        ".note { color: #444; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Full Completion Additional Experiment Tables</h1>",
        "<p><b>Purpose.</b> This table package replaces the prior Gisette placeholder coverage rows with numeric rerun results where the completion artifacts exist. It keeps mixed results visible instead of turning every row into a positive narrative.</p>",
        "<p class=\"note\"><b>Key caveats.</b> Gisette lambda/seed rows are fixed-fold seed reruns, the Gisette sequential row is a fixed-fold schedule comparison with a mixed outcome, and the Gisette correlated-feature row remains a representative top-100 subset stress test.</p>",
    ]
    for title, rows, columns in table_specs:
        body.append(html_table(title, rows, columns))
    body.extend(["</body>", "</html>"])
    path.write_text("\n".join(body), encoding="utf-8")


def convert_docs(html_path: Path, out_dir: Path) -> None:
    if not shutil.which("libreoffice"):
        raise RuntimeError("LibreOffice is not available; cannot convert to DOCX/PDF.")
    env = {
        "HOME": "/tmp",
        "XDG_CONFIG_HOME": "/tmp",
        "XDG_CACHE_HOME": "/tmp",
    }
    subprocess.run(
        [
            "libreoffice",
            "--headless",
            "--infilter=HTML (StarWriter)",
            "--convert-to",
            "docx:MS Word 2007 XML",
            "--outdir",
            str(out_dir),
            str(html_path),
        ],
        check=True,
        env=env,
    )
    docx_path = out_dir / f"{html_path.stem}.docx"
    subprocess.run(
        [
            "libreoffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(out_dir),
            str(docx_path),
        ],
        check=True,
        env=env,
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve() if args.root is not None else infer_root()
    out_dir = (root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, tuple[list[dict[str, str]], list[str]]] = {
        "coverage_matrix": (
            build_coverage_table(),
            ["evidence_block", "bcwd", "vowel", "spambase", "gisette", "claim_boundary"],
        ),
        "threshold_sensitivity": (
            build_threshold_table(root),
            ["dataset", "tau_0.3_f1", "tau_0.4_f1", "tau_0.5_f1", "tau_0.6_f1", "tau_0.7_f1", "selected_at_0.5", "mask_jaccard_at_0.5", "source"],
        ),
        "sparsity_lambda_seed": (
            build_lambda_table(root),
            ["dataset", "lambda", "full_acc", "full_f1", "selected_features", "mask_jaccard", "pc_overlap", "source"],
        ),
        "full_vs_primary_checkpoint": (
            build_checkpoint_table(),
            ["dataset", "full_f1", "primary_f1", "delta", "effective_overlap", "source"],
        ),
        "sequential_vs_joint": (
            build_sequential_table(root),
            ["dataset", "sequential_f1", "joint_no_freeze_f1", "stage1_unfreeze_f1", "effective_overlap", "claim_boundary"],
        ),
        "architecture_ablation": (
            build_architecture_table(root),
            ["dataset", "variant", "variant_label", "full_acc", "full_f1", "selected_features", "mask_overlap_rate", "primary_f1", "full_minus_primary_f1", "source"],
        ),
        "correlated_feature_stress": (
            build_correlated_table(),
            ["dataset", "acc_delta", "selected_delta", "raw_jaccard", "collapsed_jaccard", "role_assignment_summary", "scope"],
        ),
        "product_vs_htsk_firing": (
            build_firing_table(root),
            ["dataset", "mode", "acc", "f1", "base_near_zero", "comp_near_zero", "comp_max_weight", "comp_effective_rules", "source"],
        ),
        "strong_tabular_baselines": (
            build_strong_baselines_table(),
            ["dataset", "input_dim", "RF", "HGB", "XGBoost", "LightGBM", "CatBoost", "EBM"],
        ),
        "local_deletion_faithfulness": (
            build_local_deletion_table(),
            ["dataset", "grs_intrinsic", "shap", "lime", "interpretation"],
        ),
        "intrinsic_rule_metrics": (
            build_intrinsic_table(),
            ["dataset", "mask_stability", "primary_coverage", "complementary_coverage", "top3_rule_agreement"],
        ),
        "paired_statistics": (
            build_paired_stats_table(),
            ["dataset", "comparator", "acc_delta", "acc_p", "f1_delta", "f1_p"],
        ),
    }

    for name, (rows, columns) in tables.items():
        write_csv(out_dir / f"{name}.csv", rows, columns)

    html_specs = [
        (name.replace("_", " ").title(), rows, columns)
        for name, (rows, columns) in tables.items()
    ]
    html_path = out_dir / "full_completion_experiment_tables.html"
    write_html_report(html_path, html_specs)

    if args.convert_docs:
        convert_docs(html_path, out_dir)

    print(f"Wrote {len(tables)} CSV tables and {html_path}")


if __name__ == "__main__":
    main()
