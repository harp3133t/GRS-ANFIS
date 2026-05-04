#!/usr/bin/env python3
"""
Ablation #2: GRS-ANFIS complementary boundary-sample flip analysis.

Goal:
- Compare GRS-ANFIS(primary_only) vs GRS-ANFIS(full)
- Count per-sample flips on boundary samples (Q1 confidence bin):
  wrong->right, right->wrong, net_gain
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from data import (
    coerce_numeric_frame,
    drop_nan_targets,
    encode_inputs_for_anfis,
    load_bcwd_data,
    load_gisette_data,
    load_spambase_data,
    load_vowel_data,
)
from model import GRS_ANFIS


DATASET_ORDER: List[str] = [
    "Breast_Cancer_Wisconsin_(Original)",
    "Vowel",
    "Spambase",
    "Gisette",
]

DATASET_KEYS: Dict[str, str] = {
    "Breast_Cancer_Wisconsin_(Original)": "Breast_Cancer_Wisconsin__Original_",
    "Vowel": "Vowel",
    "Spambase": "Spambase",
    "Gisette": "Gisette",
}

TARGET_NAME_PRIORITY: List[str] = ["Class", "class", "target", "label", "y"]

SPAMBASE_FEATURES: List[str] = [
    "word_freq_make",
    "word_freq_address",
    "word_freq_all",
    "word_freq_3d",
    "word_freq_our",
    "word_freq_over",
    "word_freq_remove",
    "word_freq_internet",
    "word_freq_order",
    "word_freq_mail",
    "word_freq_receive",
    "word_freq_will",
    "word_freq_people",
    "word_freq_report",
    "word_freq_addresses",
    "word_freq_free",
    "word_freq_business",
    "word_freq_email",
    "word_freq_you",
    "word_freq_credit",
    "word_freq_your",
    "word_freq_font",
    "word_freq_000",
    "word_freq_money",
    "word_freq_hp",
    "word_freq_hpl",
    "word_freq_george",
    "word_freq_650",
    "word_freq_lab",
    "word_freq_labs",
    "word_freq_telnet",
    "word_freq_857",
    "word_freq_data",
    "word_freq_415",
    "word_freq_85",
    "word_freq_technology",
    "word_freq_1999",
    "word_freq_parts",
    "word_freq_pm",
    "word_freq_direct",
    "word_freq_cs",
    "word_freq_meeting",
    "word_freq_original",
    "word_freq_project",
    "word_freq_re",
    "word_freq_edu",
    "word_freq_table",
    "word_freq_conference",
    "char_freq_;",
    "char_freq_(",
    "char_freq_[",
    "char_freq_!",
    "char_freq_$",
    "char_freq_#",
    "capital_run_length_average",
    "capital_run_length_longest",
    "capital_run_length_total",
]

BCWD_RAW_FEATURES: List[str] = [
    "Clump_thickness",
    "Uniformity_of_cell_size",
    "Uniformity_of_cell_shape",
    "Marginal_adhesion",
    "Single_epithelial_cell_size",
    "Bare_nuclei",
    "Bland_chromatin",
    "Normal_nucleoli",
    "Mitoses",
]

BCWD_WITH_ID_COLUMNS: List[str] = ["Sample_code_number"] + BCWD_RAW_FEATURES + ["Class"]

SUMMARY_CHECK_TARGETS: List[str] = ["Vowel", "Gisette"]


@dataclass
class DatasetBundle:
    name: str
    dataset_key: str
    X_df: pd.DataFrame
    y: np.ndarray
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation #2: GRS-ANFIS complementary boundary flip-count analysis."
    )
    parser.add_argument("--mode", type=str, default="no_mi", choices=["no_mi"])
    parser.add_argument("--weight-root", type=str, default="hyper_parameter/cv_weights")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=4)
    parser.add_argument("--boundary-bin", type=int, default=1)
    parser.add_argument(
        "--out-dir", type=str, default="output/ablation2_complementary_boundary"
    )
    parser.add_argument(
        "--summary-check-path",
        type=str,
        default=None,
    )
    return parser.parse_args()


def resolve_path(path_str: str, project_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def normalize_task_kind(task_kind: Any, n_outputs: int) -> str:
    t = str(task_kind).strip().lower()
    if t in {"binary", "multiclass"}:
        return t
    return "binary" if int(n_outputs) == 1 else "multiclass"


def _find_existing_file(candidates: Sequence[Path]) -> Optional[Path]:
    for path in candidates:
        if path.is_file():
            return path
    return None


def _read_table_candidates(path: Path) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    read_calls = [
        lambda p: pd.read_csv(p),
        lambda p: pd.read_csv(p, header=None),
        lambda p: pd.read_csv(p, header=None, sep=r"\s+|,", engine="python"),
    ]
    for read_fn in read_calls:
        try:
            df = read_fn(path)
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                frames.append(df)
        except Exception:
            continue
    return frames


def _coerce_target_binary(y_series: pd.Series, dataset_name: str) -> np.ndarray:
    if dataset_name == "Breast_Cancer_Wisconsin_(Original)":
        mapped = _map_bcwd_target(y_series)
        if mapped is None:
            raise ValueError("BCWD target mapping failed.")
        return mapped

    y_num = pd.to_numeric(y_series, errors="coerce")
    if y_num.notna().all():
        unique = sorted(set(y_num.astype(int).tolist()))
        if set(unique).issubset({0, 1}):
            return y_num.astype(int).to_numpy()

    y_s = y_series.astype(str).str.strip().str.lower()
    mapped = y_s.map({"0": 0, "1": 1, "false": 0, "true": 1, "no": 0, "yes": 1})
    if mapped.notna().all():
        return mapped.astype(int).to_numpy()

    non_na = y_series.dropna().astype(str).str.strip()
    uniques = sorted(non_na.unique().tolist())
    if len(uniques) != 2:
        raise ValueError(f"Cannot map target to binary classes. unique={uniques}")
    fallback_map = {uniques[0]: 0, uniques[1]: 1}
    return y_series.astype(str).str.strip().map(fallback_map).astype(int).to_numpy()


def _map_bcwd_target(y_series: pd.Series) -> Optional[np.ndarray]:
    y_num = pd.to_numeric(y_series, errors="coerce")
    if y_num.notna().all():
        unique = sorted(set(y_num.astype(int).tolist()))
        if set(unique).issubset({0, 1}):
            return y_num.astype(int).to_numpy()
        if set(unique).issubset({2, 4}):
            return (y_num.astype(int) == 4).astype(int).to_numpy()

    y_s = y_series.astype(str).str.strip().str.lower()
    known_map = {
        "2": 0,
        "4": 1,
        "benign": 0,
        "malignant": 1,
        "b": 0,
        "m": 1,
        "0": 0,
        "1": 1,
    }
    mapped = y_s.map(known_map)
    if mapped.notna().all():
        return mapped.astype(int).to_numpy()

    non_na = y_s.dropna()
    uniq = sorted(non_na.unique().tolist())
    if len(uniq) == 2:
        fallback_map = {uniq[0]: 0, uniq[1]: 1}
        return y_s.map(fallback_map).astype(int).to_numpy()
    return None


def _pick_target_column(df: pd.DataFrame) -> Any:
    for name in TARGET_NAME_PRIORITY:
        if name in df.columns:
            return name

    lowered = {str(col).strip().lower(): col for col in df.columns}
    for name in TARGET_NAME_PRIORITY:
        key = name.strip().lower()
        if key in lowered:
            return lowered[key]

    return df.columns[-1]


def _load_spambase_local(data_root: Path) -> DatasetBundle:
    candidates = [
        data_root / "spambase.csv",
        data_root / "spambase.data",
    ]
    src_path = _find_existing_file(candidates)
    if src_path is None:
        raise FileNotFoundError(
            "Spambase local file not found. Expected one of: "
            + ", ".join(str(p) for p in candidates)
        )

    frames = _read_table_candidates(src_path)
    if not frames:
        raise ValueError(f"Unable to parse Spambase file: {src_path}")

    best_df: Optional[pd.DataFrame] = None
    for df in frames:
        work = df.copy().dropna(axis=1, how="all")
        if "Unnamed: 0" in work.columns and work.shape[1] == 59:
            work = work.drop(columns=["Unnamed: 0"])
        if work.shape[1] == 58:
            best_df = work
            break
    if best_df is None:
        raise ValueError(
            f"Spambase file must have 58 columns (57 features + 1 target). got={[df.shape[1] for df in frames]}"
        )

    target_col = _pick_target_column(best_df)
    y_raw = best_df[target_col]
    X_df = best_df.drop(columns=[target_col])

    if X_df.shape[1] != 57:
        raise ValueError(f"Spambase feature count must be 57. got={X_df.shape[1]}")

    X_df = X_df.copy()
    X_df.columns = SPAMBASE_FEATURES
    X_df = X_df.apply(pd.to_numeric, errors="coerce")

    y = _coerce_target_binary(pd.Series(y_raw), dataset_name="Spambase")
    X_df, y = drop_nan_targets(X_df, y)
    y = np.asarray(y).astype(int)

    return DatasetBundle(
        name="Spambase",
        dataset_key=DATASET_KEYS["Spambase"],
        X_df=X_df,
        y=y,
        source=f"local:{src_path}",
    )


def _prepare_bcwd_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    work = df.copy().dropna(axis=1, how="all")
    if "Unnamed: 0" in work.columns:
        work = work.drop(columns=["Unnamed: 0"])

    n_cols = work.shape[1]
    if n_cols not in {10, 11}:
        raise ValueError(f"BCWD raw column count must be 10 or 11. got={n_cols}")

    lower_cols = {str(c).strip().lower(): c for c in work.columns}
    has_named_target = any(name.lower() in lower_cols for name in TARGET_NAME_PRIORITY)
    unnamed_like = all(str(c).isdigit() for c in work.columns)

    if unnamed_like or (not has_named_target):
        if n_cols == 11:
            work.columns = BCWD_WITH_ID_COLUMNS
        elif n_cols == 10:
            work.columns = BCWD_RAW_FEATURES + ["Class"]

    target_col = _pick_target_column(work)
    y_raw = work[target_col]
    X_raw = work.drop(columns=[target_col])

    id_col_candidates = {"sample_code_number", "id", "sample_code", "sample"}
    if X_raw.shape[1] == 10:
        first_name = str(X_raw.columns[0]).strip().lower()
        if first_name in id_col_candidates or X_raw.shape[1] == 10:
            X_raw = X_raw.iloc[:, 1:]

    if X_raw.shape[1] != 9:
        raise ValueError(f"BCWD feature count must be 9 after ID removal. got={X_raw.shape[1]}")

    X_raw = X_raw.copy()
    X_raw.columns = BCWD_RAW_FEATURES
    X_raw = X_raw.replace("?", np.nan).replace(" ?", np.nan)

    # Match the public BCWD loader: ordinal score columns become 80 drop-first one-hot inputs.
    for col in X_raw.columns:
        num = pd.to_numeric(X_raw[col], errors="coerce")
        X_raw[col] = num.astype(float)

    X_prepared, _ = encode_inputs_for_anfis(
        X_raw,
        treat_int_as_categorical=True,
        include_missing_as_category=False,
        drop_first=True,
    )
    X_prepared = coerce_numeric_frame(X_prepared)

    y = _coerce_target_binary(pd.Series(y_raw), dataset_name="Breast_Cancer_Wisconsin_(Original)")
    X_prepared, y = drop_nan_targets(X_prepared, y)
    y = np.asarray(y).astype(int)
    return X_prepared, y


def _load_bcwd_local(data_root: Path) -> DatasetBundle:
    candidates = [
        data_root / "bcwd.csv",
        data_root / "breast_cancer_wisconsin_original.csv",
        data_root / "breast-cancer-wisconsin.data",
    ]
    src_path = _find_existing_file(candidates)
    if src_path is None:
        raise FileNotFoundError(
            "BCWD local file not found. Expected one of: "
            + ", ".join(str(p) for p in candidates)
        )

    frames = _read_table_candidates(src_path)
    if not frames:
        raise ValueError(f"Unable to parse BCWD file: {src_path}")

    errors: List[str] = []
    for idx, frame in enumerate(frames, start=1):
        try:
            X_df, y = _prepare_bcwd_frame(frame)
            return DatasetBundle(
                name="Breast_Cancer_Wisconsin_(Original)",
                dataset_key=DATASET_KEYS["Breast_Cancer_Wisconsin_(Original)"],
                X_df=X_df,
                y=y,
                source=f"local:{src_path}:candidate_{idx}",
            )
        except Exception as exc:
            errors.append(f"candidate_{idx}: {type(exc).__name__}: {exc}")

    raise ValueError("BCWD local parse failed. " + " | ".join(errors))


def _load_dataset_bundle_maincode(dataset_name: str) -> DatasetBundle:
    # Match maincode_with_ph.ipynb loading order:
    # load_*_data -> coerce_numeric_frame -> drop_nan_targets
    if dataset_name == "Vowel":
        X_df, y, _ = load_vowel_data()
        loader_name = "load_vowel_data"
    elif dataset_name == "Spambase":
        X_df, y, _ = load_spambase_data()
        loader_name = "load_spambase_data"
    elif dataset_name == "Gisette":
        X_df, y, _ = load_gisette_data()
        loader_name = "load_gisette_data"
    elif dataset_name == "Breast_Cancer_Wisconsin_(Original)":
        X_df, y, _ = load_bcwd_data()
        loader_name = "load_bcwd_data"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    X_df = coerce_numeric_frame(X_df)
    X_df, y = drop_nan_targets(X_df, y)
    X_df = X_df.copy()

    # Historical GRS payloads use x0..xN feature names for Gisette.
    if dataset_name == "Gisette" and all(
        isinstance(col, (int, np.integer)) for col in X_df.columns
    ):
        X_df.columns = [f"x{i}" for i in range(X_df.shape[1])]

    return DatasetBundle(
        name=dataset_name,
        dataset_key=DATASET_KEYS[dataset_name],
        X_df=X_df,
        y=np.asarray(y).astype(int),
        source=f"maincode_loader:{loader_name}",
    )


def _load_dataset_bundle(dataset_name: str, data_root: Path) -> DatasetBundle:
    try:
        return _load_dataset_bundle_maincode(dataset_name)
    except Exception as maincode_exc:
        # Keep local-file fallback only for BCWD/Spambase when maincode loader is unavailable.
        if dataset_name == "Breast_Cancer_Wisconsin_(Original)":
            try:
                bundle = _load_bcwd_local(data_root)
                bundle.source = (
                    f"{bundle.source}|fallback_after_maincode:{type(maincode_exc).__name__}"
                )
                return bundle
            except Exception as local_exc:
                raise RuntimeError(
                    f"BCWD load failed. maincode={type(maincode_exc).__name__}: {maincode_exc} | "
                    f"local={type(local_exc).__name__}: {local_exc}"
                ) from local_exc

        if dataset_name == "Spambase":
            try:
                bundle = _load_spambase_local(data_root)
                bundle.source = (
                    f"{bundle.source}|fallback_after_maincode:{type(maincode_exc).__name__}"
                )
                return bundle
            except Exception as local_exc:
                raise RuntimeError(
                    f"Spambase load failed. maincode={type(maincode_exc).__name__}: {maincode_exc} | "
                    f"local={type(local_exc).__name__}: {local_exc}"
                ) from local_exc

        raise RuntimeError(
            f"{dataset_name} load failed. maincode={type(maincode_exc).__name__}: {maincode_exc}"
        ) from maincode_exc


def _dataset_key_for_mode(primary_key: str, mode: str) -> str:
    if mode == "no_mi":
        return primary_key
    return f"{primary_key}__{mode}"


def _candidate_dataset_keys_for_mode(primary_key: str, mode: str) -> List[str]:
    candidates: List[str] = []
    if mode == "no_mi":
        candidates.extend([f"{primary_key}__{mode}", primary_key])
    else:
        candidates.extend([f"{primary_key}__{mode}", primary_key])

    deduped: List[str] = []
    for key in candidates:
        if key not in deduped:
            deduped.append(key)
    return deduped


def _resolve_payload_path(
    weight_root: Path, primary_key: str, mode: str, fold_idx: int
) -> Tuple[str, Path]:
    attempted: List[str] = []
    for dataset_key in _candidate_dataset_keys_for_mode(primary_key, mode):
        payload_path = weight_root / dataset_key / f"fold_{int(fold_idx):02d}" / "grs_anfis.pt"
        attempted.append(str(payload_path))
        if payload_path.is_file():
            return dataset_key, payload_path

    raise FileNotFoundError("payload not found. attempted=" + " | ".join(attempted))


def _load_payload_strict(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"payload not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"payload must be dict. got={type(payload).__name__}")

    required = ["state_dict", "n_features", "n_outputs", "feature_names", "params"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"payload missing keys: {missing}")
    return payload


def _align_features_strict(X_df: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    if not feature_names:
        raise ValueError("feature_names is empty in payload.")
    normalized_feature_names = [str(f) for f in feature_names]
    if (
        normalized_feature_names
        and normalized_feature_names[0] not in X_df.columns
        and all(f"x{name}" in X_df.columns for name in normalized_feature_names[: min(20, len(normalized_feature_names))])
    ):
        normalized_feature_names = [f"x{name}" for name in normalized_feature_names]
    missing = [f for f in normalized_feature_names if f not in X_df.columns]
    if missing:
        preview = ", ".join(str(f) for f in missing[:10])
        raise ValueError(f"missing {len(missing)} payload features. first={preview}")
    return X_df.loc[:, normalized_feature_names]


def _apply_payload_scaler_strict(X_df: pd.DataFrame, payload: Dict[str, Any]) -> np.ndarray:
    mean = payload.get("scaler_mean")
    scale = payload.get("scaler_scale")
    if mean is None or scale is None:
        raise ValueError("payload scaler_mean/scaler_scale missing.")

    X_np = np.asarray(X_df, dtype=float)
    mean_np = np.asarray(mean, dtype=float)
    scale_np = np.asarray(scale, dtype=float)

    if mean_np.shape[0] != X_np.shape[1] or scale_np.shape[0] != X_np.shape[1]:
        raise ValueError(
            f"scaler length mismatch. n_features={X_np.shape[1]} "
            f"mean={mean_np.shape[0]} scale={scale_np.shape[0]}"
        )
    scale_np = np.where(scale_np == 0.0, 1.0, scale_np)
    return (X_np - mean_np) / scale_np


def _build_grs_model(payload: Dict[str, Any], device: torch.device) -> GRS_ANFIS:
    params = payload.get("params") or {}
    model = GRS_ANFIS(
        n_features=int(payload["n_features"]),
        n_outputs=int(payload["n_outputs"]),
        complementary_rules=int(params.get("complementary_rules", 8)),
        primary_rules=int(params.get("primary_rules", 4)),
        mf_per_feature=int(params.get("mf_per_feature", 2)),
        device=device,
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    if hasattr(model, "set_phase"):
        model.set_phase("complementary_complement")
    return model


def _predict_and_confidence(
    model: GRS_ANFIS, X_np: np.ndarray, task_kind: str, mode: str, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(model, "set_mode"):
        model.set_mode(mode)
    x_t = torch.tensor(X_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(x_t)

    if task_kind == "binary":
        if logits.dim() == 1:
            logits = logits.unsqueeze(1)
        probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        confidence = np.maximum(probs, 1.0 - probs)
        return preds, confidence

    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    preds = np.argmax(probs, axis=1).astype(int)
    confidence = np.max(probs, axis=1)
    return preds, confidence


def _assign_rank_bins(confidence: np.ndarray, n_bins: int) -> np.ndarray:
    if n_bins < 2:
        raise ValueError(f"n_bins must be >=2. got={n_bins}")
    conf = np.asarray(confidence, dtype=float)
    if conf.ndim != 1 or conf.size == 0:
        raise ValueError(f"confidence must be 1D non-empty. shape={conf.shape}")

    order = np.argsort(conf, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, conf.size + 1)
    bins = np.ceil((ranks.astype(float) / float(conf.size)) * float(n_bins)).astype(int)
    bins = np.clip(bins, 1, n_bins)
    return bins


def _compute_f1(y_true: np.ndarray, y_pred: np.ndarray, task_kind: str) -> float:
    avg = "weighted" if task_kind == "multiclass" else "binary"
    return float(f1_score(y_true, y_pred, average=avg, zero_division=0))


def _aggregate_q1_by_dataset(
    fold_df: pd.DataFrame, failed_folds: Sequence[Dict[str, Any]], n_folds: int, boundary_bin: int
) -> pd.DataFrame:
    ok_q1 = fold_df[
        (fold_df["status"] == "ok") & (fold_df["bin_id"] == int(boundary_bin))
    ].copy()

    fail_map: Dict[str, set] = {}
    for rec in failed_folds:
        ds = str(rec.get("dataset"))
        fold = rec.get("fold")
        if isinstance(fold, int):
            fail_map.setdefault(ds, set()).add(int(fold))

    rows: List[Dict[str, Any]] = []
    for dataset_name in DATASET_ORDER:
        subset = ok_q1[ok_q1["dataset"] == dataset_name]
        n_samples = int(subset["n_samples"].sum()) if len(subset) > 0 else 0
        w2r = int(subset["wrong_to_right"].sum()) if len(subset) > 0 else 0
        r2w = int(subset["right_to_wrong"].sum()) if len(subset) > 0 else 0
        sr = int(subset["stable_right"].sum()) if len(subset) > 0 else 0
        sw = int(subset["stable_wrong"].sum()) if len(subset) > 0 else 0
        net = w2r - r2w

        primary_correct = sr + r2w
        primary_wrong = w2r + sw
        full_correct = sr + w2r
        net_rate = (net / n_samples) if n_samples > 0 else np.nan
        primary_acc = (primary_correct / n_samples) if n_samples > 0 else np.nan
        full_acc = (full_correct / n_samples) if n_samples > 0 else np.nan
        help_rate = (w2r / n_samples) if n_samples > 0 else np.nan
        harm_rate = (r2w / n_samples) if n_samples > 0 else np.nan
        error_recovery_rate = (w2r / primary_wrong) if primary_wrong > 0 else np.nan

        n_success_folds = int(subset["fold"].nunique()) if len(subset) > 0 else 0
        failed = len(fail_map.get(dataset_name, set()))

        rows.append(
            {
                "dataset": dataset_name,
                "boundary_bin": int(boundary_bin),
                "n_success_folds": n_success_folds,
                "n_failed_folds": int(failed),
                "n_expected_folds": int(n_folds),
                "q1_n_samples": n_samples,
                "wrong_to_right": w2r,
                "right_to_wrong": r2w,
                "stable_right": sr,
                "stable_wrong": sw,
                "net_gain": net,
                "net_rate": net_rate,
                "primary_acc_q1": primary_acc,
                "full_acc_q1": full_acc,
                "help_rate_q1": help_rate,
                "harm_rate_q1": harm_rate,
                "error_recovery_rate_q1": error_recovery_rate,
                "q1_ratio_mean": float(subset["bin_ratio"].mean()) if len(subset) > 0 else np.nan,
                "q1_ratio_std": float(subset["bin_ratio"].std(ddof=0)) if len(subset) > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def _aggregate_q1_overall(q1_by_dataset: pd.DataFrame) -> pd.DataFrame:
    n_samples = int(q1_by_dataset["q1_n_samples"].sum())
    w2r = int(q1_by_dataset["wrong_to_right"].sum())
    r2w = int(q1_by_dataset["right_to_wrong"].sum())
    sr = int(q1_by_dataset["stable_right"].sum())
    sw = int(q1_by_dataset["stable_wrong"].sum())
    net = w2r - r2w

    primary_correct = sr + r2w
    primary_wrong = w2r + sw
    full_correct = sr + w2r

    row = {
        "scope": "overall",
        "q1_n_samples": n_samples,
        "wrong_to_right": w2r,
        "right_to_wrong": r2w,
        "stable_right": sr,
        "stable_wrong": sw,
        "net_gain": net,
        "net_rate": (net / n_samples) if n_samples > 0 else np.nan,
        "primary_acc_q1": (primary_correct / n_samples) if n_samples > 0 else np.nan,
        "full_acc_q1": (full_correct / n_samples) if n_samples > 0 else np.nan,
        "help_rate_q1": (w2r / n_samples) if n_samples > 0 else np.nan,
        "harm_rate_q1": (r2w / n_samples) if n_samples > 0 else np.nan,
        "error_recovery_rate_q1": (w2r / primary_wrong) if primary_wrong > 0 else np.nan,
        "datasets_with_success": int((q1_by_dataset["q1_n_samples"] > 0).sum()),
    }
    return pd.DataFrame([row])


def _repro_check(
    fold_metrics_df: pd.DataFrame, summary_csv_path: Path
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "summary_path": str(summary_csv_path),
        "available": bool(summary_csv_path.is_file()),
        "status": "skipped",
        "datasets": {},
    }
    if not summary_csv_path.is_file():
        result["reason"] = "summary_csv_not_found"
        return result

    summary_df = pd.read_csv(summary_csv_path)
    tol = 1e-8
    available_summary_datasets = set(
        summary_df.loc[
            (summary_df["mode"] == "no_mi")
            & (summary_df["model"].isin(["GRS-ANFIS(primary)", "GRS-ANFIS(full)"])),
            "dataset",
        ]
        .astype(str)
        .tolist()
    )
    target_datasets = [
        dataset for dataset in SUMMARY_CHECK_TARGETS if dataset in available_summary_datasets
    ]
    if not target_datasets:
        result["reason"] = "no_target_datasets_found_in_summary"
        result["available_datasets_in_summary"] = sorted(available_summary_datasets)
        return result

    all_pass = True
    checked_any = False
    result["checked_datasets"] = list(target_datasets)
    for dataset in target_datasets:
        obs = fold_metrics_df[
            (fold_metrics_df["dataset"] == dataset) & (fold_metrics_df["status"] == "ok")
        ]
        if len(obs) == 0:
            result["datasets"][dataset] = {"status": "skipped", "reason": "no_success_folds"}
            all_pass = False
            continue

        exp_primary = summary_df[
            (summary_df["dataset"] == dataset)
            & (summary_df["mode"] == "no_mi")
            & (summary_df["model"] == "GRS-ANFIS(primary)")
        ]
        exp_full = summary_df[
            (summary_df["dataset"] == dataset)
            & (summary_df["mode"] == "no_mi")
            & (summary_df["model"] == "GRS-ANFIS(full)")
        ]
        if len(exp_primary) == 0 or len(exp_full) == 0:
            result["datasets"][dataset] = {
                "status": "skipped",
                "reason": "expected_rows_not_found_in_summary",
            }
            all_pass = False
            continue

        checked_any = True
        exp_primary_row = exp_primary.iloc[0]
        exp_full_row = exp_full.iloc[0]

        obs_primary_acc = float(obs["primary_acc"].mean())
        obs_primary_f1 = float(obs["primary_f1"].mean())
        obs_full_acc = float(obs["full_acc"].mean())
        obs_full_f1 = float(obs["full_f1"].mean())

        cmp_payload = {
            "obs_primary_acc": obs_primary_acc,
            "obs_primary_f1": obs_primary_f1,
            "obs_full_acc": obs_full_acc,
            "obs_full_f1": obs_full_f1,
            "exp_primary_acc": float(exp_primary_row["acc_mean"]),
            "exp_primary_f1": float(exp_primary_row["f1_mean"]),
            "exp_full_acc": float(exp_full_row["acc_mean"]),
            "exp_full_f1": float(exp_full_row["f1_mean"]),
        }
        diffs = {
            "primary_acc_abs_diff": abs(cmp_payload["obs_primary_acc"] - cmp_payload["exp_primary_acc"]),
            "primary_f1_abs_diff": abs(cmp_payload["obs_primary_f1"] - cmp_payload["exp_primary_f1"]),
            "full_acc_abs_diff": abs(cmp_payload["obs_full_acc"] - cmp_payload["exp_full_acc"]),
            "full_f1_abs_diff": abs(cmp_payload["obs_full_f1"] - cmp_payload["exp_full_f1"]),
        }

        is_match = all(v <= tol for v in diffs.values())
        if not is_match:
            all_pass = False

        result["datasets"][dataset] = {
            "status": "pass" if is_match else "fail",
            "tolerance": tol,
            **cmp_payload,
            **diffs,
        }

    if checked_any and all_pass:
        result["status"] = "pass"
    elif checked_any:
        result["status"] = "fail"
    else:
        result["status"] = "skipped"
    return result


def _ensure_runtime_cache_paths(data_root: Path, project_root: Path) -> None:
    openml_cache = data_root / "openml_cache"
    local_cache = project_root / ".cache"
    openml_cache.mkdir(parents=True, exist_ok=True)
    local_cache.mkdir(parents=True, exist_ok=True)
    os.environ["OPENML_DATA_HOME"] = str(openml_cache)
    os.environ["XDG_CACHE_HOME"] = str(local_cache)


def _log(verbose: bool, message: str = "") -> None:
    if verbose:
        print(message)


def run_ablation(
    *,
    mode: str = "no_mi",
    weight_root: str = "hyper_parameter/cv_weights",
    data_root: str = "data",
    n_folds: int = 5,
    seed: int = 42,
    n_bins: int = 4,
    boundary_bin: int = 1,
    out_dir: str = "output/ablation2_complementary_boundary",
    summary_check_path: Optional[str] = None,
    project_root: Optional[Path] = None,
    device: Optional[Any] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the boundary-sample Q1 ablation and return in-memory results for notebooks."""
    project_root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parent

    weight_root_path = resolve_path(weight_root, project_root)
    data_root_path = resolve_path(data_root, project_root)
    out_dir_path = resolve_path(out_dir, project_root)
    summary_check_path_obj = (
        resolve_path(summary_check_path, project_root) if summary_check_path else None
    )

    if boundary_bin < 1 or boundary_bin > n_bins:
        raise ValueError(
            f"boundary_bin must satisfy 1 <= boundary_bin <= n_bins. "
            f"got boundary_bin={boundary_bin}, n_bins={n_bins}"
        )
    if n_folds < 2:
        raise ValueError(f"n_folds must be >=2. got={n_folds}")

    _ensure_runtime_cache_paths(data_root_path, project_root)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    if device is None:
        runtime_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        runtime_device = torch.device(device)

    _log(verbose, f"[Config] mode={mode}, n_folds={n_folds}, seed={seed}")
    _log(verbose, f"[Config] n_bins={n_bins}, boundary_bin={boundary_bin}")
    _log(verbose, f"[Path] weight_root={weight_root_path}")
    _log(verbose, f"[Path] data_root={data_root_path}")
    _log(verbose, f"[Path] out_dir={out_dir_path}")
    _log(verbose, f"[Device] {runtime_device}")

    fold_rows: List[Dict[str, Any]] = []
    failed_folds: List[Dict[str, Any]] = []
    fold_model_metrics: List[Dict[str, Any]] = []
    dataset_sources: Dict[str, str] = {}

    for dataset_name in DATASET_ORDER:
        _log(verbose, f"\n[Dataset] {dataset_name}")
        try:
            bundle = _load_dataset_bundle(dataset_name, data_root_path)
            dataset_sources[dataset_name] = bundle.source
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            _log(verbose, f"  -> load failed: {err}")
            for fold_idx in range(1, n_folds + 1):
                failed_rec = {
                    "dataset": dataset_name,
                    "dataset_key": DATASET_KEYS[dataset_name],
                    "fold": int(fold_idx),
                    "stage": "dataset_load",
                    "error": err,
                }
                failed_folds.append(failed_rec)
                fold_rows.append(
                    {
                        "dataset": dataset_name,
                        "dataset_key": DATASET_KEYS[dataset_name],
                        "mode": mode,
                        "fold": int(fold_idx),
                        "task_kind": np.nan,
                        "status": "failed",
                        "stage": "dataset_load",
                        "error": err,
                        "n_val_samples": np.nan,
                        "bin_id": np.nan,
                        "is_boundary": np.nan,
                        "bin_ratio": np.nan,
                        "n_samples": np.nan,
                        "wrong_to_right": np.nan,
                        "right_to_wrong": np.nan,
                        "stable_right": np.nan,
                        "stable_wrong": np.nan,
                        "net_gain": np.nan,
                        "net_rate": np.nan,
                        "primary_acc_bin": np.nan,
                        "full_acc_bin": np.nan,
                        "error_recovery_rate_bin": np.nan,
                        "boundary_conf_quantile": np.nan,
                    }
                )
            continue

        X_df = bundle.X_df.copy()
        X_df.columns = [str(c) for c in X_df.columns]
        y = np.asarray(bundle.y).astype(int)
        if len(np.unique(y)) < 2:
            err = "target has fewer than 2 classes."
            _log(verbose, f"  -> invalid target: {err}")
            for fold_idx in range(1, n_folds + 1):
                failed_folds.append(
                    {
                        "dataset": dataset_name,
                        "dataset_key": bundle.dataset_key,
                        "fold": int(fold_idx),
                        "stage": "target_validation",
                        "error": err,
                    }
                )
            continue

        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = list(kf.split(X_df, y))
        _log(
            verbose,
            f"  -> source={bundle.source}, n_samples={len(y)}, n_features={X_df.shape[1]}, n_splits={len(split_iter)}",
        )

        dataset_key_by_mode = _dataset_key_for_mode(bundle.dataset_key, mode)
        for fold_idx, (_train_idx, val_idx) in enumerate(split_iter, start=1):
            try:
                dataset_key_by_mode, payload_path = _resolve_payload_path(
                    weight_root=weight_root_path,
                    primary_key=bundle.dataset_key,
                    mode=mode,
                    fold_idx=fold_idx,
                )
                payload = _load_payload_strict(payload_path)
                n_outputs = int(payload["n_outputs"])
                task_kind = normalize_task_kind(payload.get("task_kind", "binary"), n_outputs)
                feature_names = list(payload["feature_names"])

                if int(payload["n_features"]) != len(feature_names):
                    raise ValueError(
                        f"payload n_features mismatch: n_features={payload['n_features']} "
                        f"feature_names_len={len(feature_names)}"
                    )

                X_val_df = X_df.iloc[val_idx].copy()
                y_val = y[val_idx]

                X_val_aligned = _align_features_strict(X_val_df, feature_names)
                X_val_scaled = _apply_payload_scaler_strict(X_val_aligned, payload)

                model = _build_grs_model(payload, device=runtime_device)

                primary_pred, primary_conf = _predict_and_confidence(
                    model=model,
                    X_np=X_val_scaled,
                    task_kind=task_kind,
                    mode="primary_only",
                    device=runtime_device,
                )
                full_pred, _full_conf = _predict_and_confidence(
                    model=model,
                    X_np=X_val_scaled,
                    task_kind=task_kind,
                    mode="full",
                    device=runtime_device,
                )

                primary_correct = primary_pred == y_val
                full_correct = full_pred == y_val

                primary_acc_fold = float(accuracy_score(y_val, primary_pred))
                full_acc_fold = float(accuracy_score(y_val, full_pred))
                primary_f1_fold = _compute_f1(y_val, primary_pred, task_kind=task_kind)
                full_f1_fold = _compute_f1(y_val, full_pred, task_kind=task_kind)

                fold_model_metrics.append(
                    {
                        "dataset": dataset_name,
                        "fold": int(fold_idx),
                        "status": "ok",
                        "task_kind": task_kind,
                        "primary_acc": primary_acc_fold,
                        "primary_f1": primary_f1_fold,
                        "full_acc": full_acc_fold,
                        "full_f1": full_f1_fold,
                    }
                )

                bins = _assign_rank_bins(primary_conf, n_bins=n_bins)
                n_val = int(len(y_val))
                boundary_quantile = float(np.quantile(primary_conf, float(boundary_bin) / float(n_bins)))

                bin_mass_total = 0
                for bin_id in range(1, n_bins + 1):
                    idx = bins == bin_id
                    n_bin = int(np.sum(idx))
                    bin_mass_total += n_bin

                    w2r = int(np.sum((~primary_correct) & full_correct & idx))
                    r2w = int(np.sum(primary_correct & (~full_correct) & idx))
                    stable_right = int(np.sum(primary_correct & full_correct & idx))
                    stable_wrong = int(np.sum((~primary_correct) & (~full_correct) & idx))

                    if (w2r + r2w + stable_right + stable_wrong) != n_bin:
                        raise ValueError(
                            f"mass conservation failed: fold={fold_idx}, bin={bin_id}, "
                            f"sum={w2r + r2w + stable_right + stable_wrong}, n_bin={n_bin}"
                        )

                    net = int(w2r - r2w)
                    primary_wrong_bin = int(w2r + stable_wrong)
                    bin_ratio = (n_bin / n_val) if n_val > 0 else np.nan
                    net_rate = (net / n_bin) if n_bin > 0 else np.nan
                    primary_acc_bin = (
                        float(np.sum(primary_correct & idx)) / float(n_bin) if n_bin > 0 else np.nan
                    )
                    full_acc_bin = (
                        float(np.sum(full_correct & idx)) / float(n_bin) if n_bin > 0 else np.nan
                    )
                    error_recovery_rate_bin = (
                        float(w2r) / float(primary_wrong_bin) if primary_wrong_bin > 0 else np.nan
                    )

                    fold_rows.append(
                        {
                            "dataset": dataset_name,
                            "dataset_key": dataset_key_by_mode,
                            "mode": mode,
                            "fold": int(fold_idx),
                            "task_kind": task_kind,
                            "status": "ok",
                            "stage": "evaluation",
                            "error": "",
                            "n_val_samples": n_val,
                            "bin_id": int(bin_id),
                            "is_boundary": bool(bin_id == boundary_bin),
                            "bin_ratio": bin_ratio,
                            "n_samples": int(n_bin),
                            "wrong_to_right": int(w2r),
                            "right_to_wrong": int(r2w),
                            "stable_right": int(stable_right),
                            "stable_wrong": int(stable_wrong),
                            "net_gain": int(net),
                            "net_rate": net_rate,
                            "primary_acc_bin": primary_acc_bin,
                            "full_acc_bin": full_acc_bin,
                            "error_recovery_rate_bin": error_recovery_rate_bin,
                            "boundary_conf_quantile": boundary_quantile,
                        }
                    )

                if bin_mass_total != n_val:
                    raise ValueError(
                        f"bin coverage failed: fold={fold_idx}, bin_mass_total={bin_mass_total}, n_val={n_val}"
                    )

                q1_count = int(np.sum(bins == boundary_bin))
                q1_ratio = q1_count / float(n_val) if n_val > 0 else np.nan
                _log(
                    verbose,
                    f"  fold {fold_idx:02d} ok | n_val={n_val} | primary_acc={primary_acc_fold:.4f} "
                    f"| full_acc={full_acc_fold:.4f} | q1_count={q1_count} ({q1_ratio:.3f})",
                )

            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                _log(verbose, f"  fold {fold_idx:02d} failed | {err}")
                failed_folds.append(
                    {
                        "dataset": dataset_name,
                        "dataset_key": dataset_key_by_mode,
                        "fold": int(fold_idx),
                        "stage": "fold_eval",
                        "payload_path": str(payload_path),
                        "error": err,
                    }
                )
                fold_rows.append(
                    {
                        "dataset": dataset_name,
                        "dataset_key": dataset_key_by_mode,
                        "mode": mode,
                        "fold": int(fold_idx),
                        "task_kind": np.nan,
                        "status": "failed",
                        "stage": "fold_eval",
                        "error": err,
                        "n_val_samples": np.nan,
                        "bin_id": np.nan,
                        "is_boundary": np.nan,
                        "bin_ratio": np.nan,
                        "n_samples": np.nan,
                        "wrong_to_right": np.nan,
                        "right_to_wrong": np.nan,
                        "stable_right": np.nan,
                        "stable_wrong": np.nan,
                        "net_gain": np.nan,
                        "net_rate": np.nan,
                        "primary_acc_bin": np.nan,
                        "full_acc_bin": np.nan,
                        "error_recovery_rate_bin": np.nan,
                        "boundary_conf_quantile": np.nan,
                    }
                )
                fold_model_metrics.append(
                    {
                        "dataset": dataset_name,
                        "fold": int(fold_idx),
                        "status": "failed",
                        "task_kind": np.nan,
                        "primary_acc": np.nan,
                        "primary_f1": np.nan,
                        "full_acc": np.nan,
                        "full_f1": np.nan,
                    }
                )

    fold_df = pd.DataFrame(fold_rows)
    fold_metrics_df = pd.DataFrame(fold_model_metrics)

    q1_by_dataset_df = _aggregate_q1_by_dataset(
        fold_df=fold_df,
        failed_folds=failed_folds,
        n_folds=n_folds,
        boundary_bin=boundary_bin,
    )
    q1_overall_df = _aggregate_q1_overall(q1_by_dataset_df)

    fold_csv = out_dir_path / "fold_bin_counts.csv"
    q1_dataset_csv = out_dir_path / "q1_summary_by_dataset.csv"
    q1_overall_csv = out_dir_path / "q1_summary_overall.csv"
    run_config_json = out_dir_path / "run_config.json"

    fold_df.to_csv(fold_csv, index=False)
    q1_by_dataset_df.to_csv(q1_dataset_csv, index=False)
    q1_overall_df.to_csv(q1_overall_csv, index=False)

    if summary_check_path_obj is None:
        repro = {
            "summary_path": None,
            "available": False,
            "status": "skipped",
            "reason": "summary_check_disabled",
            "datasets": {},
        }
    else:
        repro = _repro_check(
            fold_metrics_df=fold_metrics_df,
            summary_csv_path=summary_check_path_obj,
        )

    run_config = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "dataset_order": DATASET_ORDER,
        "dataset_keys": DATASET_KEYS,
        "dataset_sources": dataset_sources,
        "n_folds": int(n_folds),
        "seed": int(seed),
        "n_bins": int(n_bins),
        "boundary_bin": int(boundary_bin),
        "confidence_definition": {
            "binary": "max(sigmoid(logit), 1-sigmoid(logit))",
            "multiclass": "max(softmax(logits))",
        },
        "binning_definition": "rank-based equal-frequency bins",
        "weight_root": str(weight_root_path),
        "data_root": str(data_root_path),
        "out_dir": str(out_dir_path),
        "outputs": {
            "fold_bin_counts_csv": str(fold_csv),
            "q1_summary_by_dataset_csv": str(q1_dataset_csv),
            "q1_summary_overall_csv": str(q1_overall_csv),
        },
        "summary_check_path": str(summary_check_path_obj) if summary_check_path_obj is not None else None,
        "repro_check": repro,
        "failed_folds": failed_folds,
        "failed_fold_count": int(len(failed_folds)),
    }
    run_config_json.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    _log(verbose, "\n[Done]")
    _log(verbose, f"  fold_bin_counts:      {fold_csv}")
    _log(verbose, f"  q1_summary_by_dataset:{q1_dataset_csv}")
    _log(verbose, f"  q1_summary_overall:   {q1_overall_csv}")
    _log(verbose, f"  run_config:           {run_config_json}")
    _log(verbose, f"  failed_folds:         {len(failed_folds)}")

    if len(q1_by_dataset_df) > 0:
        _log(verbose, "\n[Q1 summary by dataset]")
        _log(
            verbose,
            q1_by_dataset_df[
                [
                    "dataset",
                    "q1_n_samples",
                    "wrong_to_right",
                    "right_to_wrong",
                    "net_gain",
                    "net_rate",
                    "error_recovery_rate_q1",
                ]
            ].to_string(index=False),
        )

    if len(q1_overall_df) > 0:
        _log(verbose, "\n[Q1 summary overall]")
        _log(verbose, q1_overall_df.to_string(index=False))

    return {
        "project_root": project_root,
        "device": runtime_device,
        "dataset_sources": dataset_sources,
        "failed_folds": failed_folds,
        "fold_df": fold_df,
        "fold_metrics_df": fold_metrics_df,
        "q1_by_dataset_df": q1_by_dataset_df,
        "q1_overall_df": q1_overall_df,
        "repro_check": repro,
        "run_config": run_config,
        "output_paths": {
            "fold_bin_counts_csv": fold_csv,
            "q1_summary_by_dataset_csv": q1_dataset_csv,
            "q1_summary_overall_csv": q1_overall_csv,
            "run_config_json": run_config_json,
        },
    }


def main() -> None:
    args = parse_args()
    run_ablation(
        mode=args.mode,
        weight_root=args.weight_root,
        data_root=args.data_root,
        n_folds=args.n_folds,
        seed=args.seed,
        n_bins=args.n_bins,
        boundary_bin=args.boundary_bin,
        out_dir=args.out_dir,
        summary_check_path=args.summary_check_path,
        project_root=Path(__file__).resolve().parent,
        verbose=True,
    )


if __name__ == "__main__":
    main()
