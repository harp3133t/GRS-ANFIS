#!/usr/bin/env python3
"""Compute Nauck/HFSi interpretability from saved fold checkpoints.

This script does not train models. It loads the fold-level torch artifacts
created by Notebook 01, reconstructs each ANFIS-family model, computes the
Nauck-based interpretability index on the saved model-input space, and writes
fold-level plus summary CSVs for manuscript table generation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import load_bcwd_data, load_gisette_data, load_spambase_data, load_vowel_data  # noqa: E402
from interpretability import (  # noqa: E402
    complexity_profile_model,
    nauck_index_grs_hierarchical,
    nauck_index_parallel_hier_tsk,
    nauck_index_tsk,
)
from model import GRS_ANFIS, ParallelHierarchicalTSKANFIS, TSKANFIS  # noqa: E402

DEFAULT_ARTIFACT_ROOT = (
    ROOT
    / "output"
    / "reproducible_paper_tables"
    / "model_artifacts"
    / "01_main_neuro_fuzzy_and_svm_experiments"
    / "cv_weights"
)
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "reproducible_paper_tables"
SEED = 42
N_FOLDS = 5

DATASET_LOADERS = {
    "Breast_Cancer_Wisconsin__Original_": load_bcwd_data,
    "Breast_Cancer_Wisconsin__Original___no_mi": load_bcwd_data,
    "Vowel": load_vowel_data,
    "Vowel__no_mi": load_vowel_data,
    "Spambase": load_spambase_data,
    "Spambase__no_mi": load_spambase_data,
    "Gisette": load_gisette_data,
    "Gisette__no_mi": load_gisette_data,
}
_DATA_CACHE: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
_SPLIT_CACHE: dict[tuple[str, int], np.ndarray] = {}

MODEL_FILES = {
    "ANFIS": "anfis.pt",
    "GA-ANFIS": "ga_anfis.pt",
    "PSO-ANFIS": "pso_anfis.pt",
    "H-ANFIS": "h_anfis.pt",
    "GRS-ANFIS": "grs_anfis.pt",
}

GRS_REPORT_MODES = {
    "GRS-ANFIS(primary)": "primary",
    "GRS-ANFIS Complementary-only diagnostic": "complementary",
    "GRS-ANFIS(full)": "full",
}


def finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def finite_std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def payload_feature_matrix(payload: dict) -> np.ndarray:
    dataset_key = str(payload.get("dataset_key") or payload.get("dataset") or "")
    feature_names = [str(f) for f in payload.get("feature_names") or []]
    if dataset_key and feature_names:
        try:
            X_df, y = load_dataset_for_payload(dataset_key)
            train_idx = train_indices_for_fold(dataset_key, y, int(payload.get("fold") or 1))
            aligned = align_payload_features(X_df, feature_names)
            X_train = np.asarray(aligned.iloc[train_idx], dtype=float)
            mean = np.asarray(payload.get("scaler_mean"), dtype=float).reshape(1, -1)
            scale = np.asarray(payload.get("scaler_scale"), dtype=float).reshape(1, -1)
            if mean.shape[1] == X_train.shape[1] and scale.shape[1] == X_train.shape[1]:
                scale = np.where(scale == 0.0, 1.0, scale)
                return ((X_train - mean) / scale).astype(np.float32)
        except Exception as exc:
            print(
                f"[warn] falling back to synthetic domains for {payload.get('model_name')} "
                f"{dataset_key} fold {payload.get('fold')}: {exc}",
                file=sys.stderr,
            )

    n_features = int(payload["n_features"])
    mean = payload.get("scaler_mean")
    scale = payload.get("scaler_scale")
    if mean is not None and scale is not None:
        mean_np = np.asarray(mean, dtype=float).reshape(1, -1)
        scale_np = np.asarray(scale, dtype=float).reshape(1, -1)
        if mean_np.shape[1] == n_features and scale_np.shape[1] == n_features:
            # The models were trained on standardized features.  A compact
            # synthetic support set spanning +/- 1 scaled unit per dimension is
            # enough for Nauck coverage because the MF centers/sigmas are
            # already in scaled model-input coordinates.
            return np.vstack(
                [
                    np.zeros((1, n_features), dtype=np.float32),
                    np.ones((1, n_features), dtype=np.float32),
                    -np.ones((1, n_features), dtype=np.float32),
                ]
            )
    return np.vstack(
        [
            np.zeros((1, n_features), dtype=np.float32),
            np.ones((1, n_features), dtype=np.float32),
            -np.ones((1, n_features), dtype=np.float32),
        ]
    )


def normalize_dataset_key(dataset_key: str) -> str:
    return str(dataset_key).replace("__no_mi", "")


def load_dataset_for_payload(dataset_key: str) -> tuple[pd.DataFrame, np.ndarray]:
    normalized = normalize_dataset_key(dataset_key)
    if normalized in _DATA_CACHE:
        return _DATA_CACHE[normalized]
    loader = DATASET_LOADERS.get(normalized) or DATASET_LOADERS.get(dataset_key)
    if loader is None:
        raise KeyError(f"no dataset loader for payload dataset_key={dataset_key}")
    X_df, y, _feature_names = loader()
    X_df = X_df.copy()
    X_df.columns = [str(c) for c in X_df.columns]
    y = np.asarray(y)
    _DATA_CACHE[normalized] = (X_df, y)
    return X_df, y


def train_indices_for_fold(dataset_key: str, y: np.ndarray, fold: int) -> np.ndarray:
    normalized = normalize_dataset_key(dataset_key)
    cache_key = (normalized, int(fold))
    if cache_key in _SPLIT_CACHE:
        return _SPLIT_CACHE[cache_key]
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold_no, (train_idx, _val_idx) in enumerate(splitter.split(np.zeros(len(y)), y), 1):
        _SPLIT_CACHE[(normalized, fold_no)] = train_idx
    if cache_key not in _SPLIT_CACHE:
        raise KeyError(f"fold {fold} not available for {dataset_key}")
    return _SPLIT_CACHE[cache_key]


def align_payload_features(X_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    names = [str(f) for f in feature_names]
    if names and names[0] not in X_df.columns and all(f"x{name}" in X_df.columns for name in names[:20]):
        names = [f"x{name}" for name in names]
    missing = [name for name in names if name not in X_df.columns]
    if missing:
        raise KeyError(f"missing {len(missing)} payload feature columns; first={missing[:5]}")
    return X_df.loc[:, names]


def load_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"{path} did not contain a dict payload")
    required = {"state_dict", "n_features", "n_outputs", "params", "model_name"}
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"{path} missing payload keys: {missing}")
    return payload


def build_tsk(payload: dict) -> TSKANFIS:
    params = payload.get("params") or {}
    model = TSKANFIS(
        n_inputs=int(payload["n_features"]),
        n_rules=int(params.get("n_rules", 0)),
        n_outputs=int(payload["n_outputs"]),
        mfs_per_input=int(params.get("mfs_per_input", 3)),
        rule_init_mode=str(params.get("rule_init_mode", "legacy")),
        rule_seed=int(params.get("rule_seed", 0)),
        firing_mode=str(params.get("firing_mode", "htsk")),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def build_h_anfis(payload: dict) -> ParallelHierarchicalTSKANFIS:
    params = payload.get("params") or {}
    state = payload["state_dict"]
    group_a = state.get("group_a_index_tensor")
    group_b = state.get("group_b_index_tensor")
    if group_a is None or group_b is None:
        raise KeyError("H-ANFIS payload state_dict missing group index tensors")
    model = ParallelHierarchicalTSKANFIS(
        n_inputs=int(payload["n_features"]),
        n_outputs=int(payload["n_outputs"]),
        group_a_idx=group_a.detach().cpu().numpy().astype(int).tolist(),
        group_b_idx=group_b.detach().cpu().numpy().astype(int).tolist(),
        branch_rules=int(params.get("branch_rules", params.get("n_rules", 12))),
        top_rules=int(params.get("top_rules", max(2, int(params.get("branch_rules", 12)) // 2))),
        fusion=str(params.get("fusion", "avg")),
        mfs_per_input=int(params.get("mfs_per_input", 3)),
        rule_init_mode=str(params.get("rule_init_mode", "legacy")),
        rule_seed=int(params.get("rule_seed", 0)),
        firing_mode=str(params.get("firing_mode", "htsk")),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def build_grs(payload: dict) -> GRS_ANFIS:
    params = payload.get("params") or {}
    model = GRS_ANFIS(
        n_features=int(payload["n_features"]),
        n_outputs=int(payload["n_outputs"]),
        complementary_rules=int(params.get("complementary_rules", 8)),
        primary_rules=int(params.get("primary_rules", 4)),
        mf_per_feature=int(params.get("mf_per_feature", 2)),
        device=torch.device("cpu"),
        complementary_gate_mode=str(params.get("complementary_gate_mode", "complement")),
        rule_init_mode=str(params.get("rule_init_mode", "balanced")),
        rule_seed=int(params.get("rule_seed", 0)),
        firing_mode=str(params.get("firing_mode", "htsk")),
        use_input_norm=bool(params.get("use_input_norm", False)),
        enable_complementary_branch=bool(params.get("enable_complementary_branch", True)),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def rows_for_payload(path: Path, grid_size: int) -> list[dict]:
    payload = load_payload(path)
    dataset_key = str(payload.get("dataset_key") or payload.get("dataset") or path.parents[1].name)
    dataset_display = (
        dataset_key.replace("__no_mi", "")
        .replace("Breast_Cancer_Wisconsin__Original_", "Breast Cancer")
        .replace("_", " ")
        .strip()
    )
    fold = int(payload.get("fold") or path.parent.name.replace("fold_", ""))
    model_name = str(payload["model_name"])
    X_probe = payload_feature_matrix(payload)
    n_classes = int(payload.get("n_outputs", 1) or 1)

    if model_name in {"ANFIS", "GA-ANFIS", "PSO-ANFIS"}:
        model = build_tsk(payload)
        info = nauck_index_tsk(model, X_probe, grid_size=grid_size, n_classes=n_classes)
        complexity = complexity_profile_model(model)
        return [
            {
                "dataset_key": dataset_key,
                "display_dataset": dataset_display,
                "fold": fold,
                "model": model_name,
                "index_type": "Nauck",
                "nauck_based_index": info.get("index", float("nan")),
                "comp": info.get("comp", float("nan")),
                "cov": info.get("cov", float("nan")),
                "part": info.get("part", float("nan")),
                "n_rules": info.get("n_rules", float("nan")),
                "n_features_for_index": info.get("n_features", float("nan")),
                **complexity,
                "checkpoint_path": str(path),
            }
        ]

    if model_name == "H-ANFIS":
        model = build_h_anfis(payload)
        info = nauck_index_parallel_hier_tsk(model, X_probe, grid_size=grid_size, n_classes=n_classes)
        overall = info.get("overall") or {}
        complexity = complexity_profile_model(model)
        return [
            {
                "dataset_key": dataset_key,
                "display_dataset": dataset_display,
                "fold": fold,
                "model": model_name,
                "index_type": "HFSi",
                "nauck_based_index": overall.get("index", float("nan")),
                "comp": overall.get("comp", float("nan")),
                "cov": overall.get("cov", float("nan")),
                "part": overall.get("part", float("nan")),
                "n_rules": overall.get("n_rules", float("nan")),
                "n_features_for_index": int(payload["n_features"]),
                **complexity,
                "checkpoint_path": str(path),
            }
        ]

    if model_name == "GRS-ANFIS":
        model = build_grs(payload)
        info = nauck_index_grs_hierarchical(model, X_probe, grid_size=grid_size, n_classes=n_classes)
        out = []
        for report_model, variant in GRS_REPORT_MODES.items():
            if variant == "primary":
                branch = info.get("primary") or {}
                index_type = "HFSi-primary-branch"
            elif variant == "complementary":
                branch = info.get("complementary") or {}
                index_type = "HFSi-complementary-branch"
            else:
                branch = info.get("overall") or {}
                index_type = "HFSi"
            complexity = complexity_profile_model(model, grs_variant="primary" if variant == "primary" else "full")
            out.append(
                {
                    "dataset_key": dataset_key,
                    "display_dataset": dataset_display,
                    "fold": fold,
                    "model": report_model,
                    "index_type": index_type,
                    "nauck_based_index": branch.get("index", float("nan")),
                    "comp": branch.get("comp", float("nan")),
                    "cov": branch.get("cov", float("nan")),
                    "part": branch.get("part", float("nan")),
                    "n_rules": branch.get("n_rules", float("nan")),
                    "n_features_for_index": branch.get("n_features", int(payload["n_features"])),
                    **complexity,
                    "checkpoint_path": str(path),
                }
            )
        return out

    return []


def iter_checkpoint_paths(artifact_root: Path, datasets: list[str] | None) -> Iterable[Path]:
    dataset_dirs = sorted(p for p in artifact_root.glob("*__no_mi") if p.is_dir())
    if datasets:
        wanted = {d.lower() for d in datasets}
        dataset_dirs = [
            p
            for p in dataset_dirs
            if p.name.lower() in wanted
            or p.name.replace("__no_mi", "").lower() in wanted
            or p.name.replace("__no_mi", "").replace("__", "_").lower() in wanted
        ]
    for dataset_dir in dataset_dirs:
        for fold_dir in sorted(dataset_dir.glob("fold_*")):
            for filename in MODEL_FILES.values():
                path = fold_dir / filename
                if path.is_file():
                    yield path


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["display_dataset", "model", "index_type"]
    numeric_cols = [
        "nauck_based_index",
        "comp",
        "cov",
        "part",
        "n_rules",
        "n_features_for_index",
        "chfs",
        "c_rb",
        "c_s",
        "n_layers",
        "n_modules",
        "n_rules_total",
        "n_rules_active",
        "avg_antecedent_len",
    ]
    for keys, group in folds.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys))
        row["n_folds"] = int(group["fold"].nunique())
        for col in numeric_cols:
            if col in group:
                row[f"{col}_mean"] = finite_mean(group[col])
                row[f"{col}_std"] = finite_std(group[col])
        row["source"] = "computed from saved model checkpoints"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--grid-size", type=int, default=201)
    args = parser.parse_args()

    rows = []
    for path in iter_checkpoint_paths(args.artifact_root, args.datasets):
        rows.extend(rows_for_payload(path, grid_size=args.grid_size))
    if not rows:
        raise SystemExit(f"No saved torch checkpoints found under {args.artifact_root}")

    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)
    folds = pd.DataFrame(rows).sort_values(["display_dataset", "fold", "model"])
    summary = summarize(folds).sort_values(["display_dataset", "model"])
    folds_path = out_root / "08_saved_model_interpretability_folds.csv"
    summary_path = out_root / "08_saved_model_interpretability_summary.csv"
    folds.to_csv(folds_path, index=False)
    summary.to_csv(summary_path, index=False)
    meta = {
        "artifact_root": str(args.artifact_root),
        "output_root": str(args.output_root),
        "grid_size": int(args.grid_size),
        "n_rows": int(len(folds)),
        "n_summary_rows": int(len(summary)),
        "note": "Nauck/HFSi metrics computed by loading saved fold checkpoints; no model training is performed.",
    }
    (out_root / "08_saved_model_interpretability_run_config.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {folds_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
