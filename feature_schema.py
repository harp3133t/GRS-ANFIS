from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TargetSummary:
    task: str  # 'binary' | 'multiclass' | 'regression'
    n_samples: int
    y_name: str
    dtype: str
    n_classes: Optional[int] = None
    class_counts: Optional[Dict[Any, int]] = None
    class_ratios: Optional[Dict[Any, float]] = None
    stats: Optional[Dict[str, float]] = None


def _is_ucirepo_dataset(obj: Any) -> bool:
    return hasattr(obj, "data") and hasattr(getattr(obj, "data"), "features") and hasattr(
        getattr(obj, "data"), "targets"
    )


def _extract_ucirepo_xy(
    dataset: Any,
) -> Tuple[pd.DataFrame, Any, Optional[List[str]]]:
    data = dataset.data
    X = data.features
    y = data.targets

    feature_names = None
    try:
        feature_names = list(data["headers"])
    except Exception:
        try:
            feature_names = list(data.get("headers"))  # type: ignore[attr-defined]
        except Exception:
            feature_names = None

    if feature_names is None and isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=feature_names)

    return X, y, feature_names


def _to_series(y: Any) -> pd.Series:
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 1:
            return y.iloc[:, 0]
        raise ValueError(f"y has multiple columns: {list(y.columns)}")
    if isinstance(y, pd.Series):
        return y
    arr = np.asarray(y)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"y must be 1D (or Nx1). Got shape {arr.shape}.")
    return pd.Series(arr)


def infer_feature_type(
    series: pd.Series,
    *,
    max_categorical_unique: int = 20,
    treat_int_as_categorical: bool = False,
) -> str:
    if pd.api.types.is_bool_dtype(series.dtype):
        return "categorical"
    if pd.api.types.is_categorical_dtype(series.dtype):
        return "categorical"
    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
        return "categorical"

    if pd.api.types.is_numeric_dtype(series.dtype):
        if treat_int_as_categorical:
            nunique = series.nunique(dropna=True)
            values = pd.to_numeric(series.dropna(), errors="coerce")
            values_np = values.to_numpy(dtype=float)
            integer_like = (
                values_np.size > 0
                and np.isfinite(values_np).all()
                and np.all(np.isclose(values_np, np.round(values_np)))
            )
            if integer_like and nunique <= max_categorical_unique:
                return "categorical"
        return "numerical"

    return "categorical"


def build_feature_summary(
    X: Union[pd.DataFrame, np.ndarray, Any],
    y: Any = None,
    *,
    feature_names: Optional[List[str]] = None,
    max_categorical_unique: int = 20,
    treat_int_as_categorical: bool = False,
) -> Tuple[pd.DataFrame, TargetSummary]:
    if y is None and _is_ucirepo_dataset(X):
        X, y, inferred_names = _extract_ucirepo_xy(X)
        if feature_names is None:
            feature_names = inferred_names

    if isinstance(X, np.ndarray):
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X.copy()

    y_s = _to_series(y)
    y_name = y_s.name if y_s.name is not None else "y"

    rows = []
    for col in X_df.columns:
        s = X_df[col]
        inferred = infer_feature_type(
            s,
            max_categorical_unique=max_categorical_unique,
            treat_int_as_categorical=treat_int_as_categorical,
        )
        nunique = int(s.nunique(dropna=True))
        cat_count = nunique if inferred == "categorical" else 0
        rows.append(
            {
                "column": str(col),
                "inferred_type": inferred,
                "pandas_dtype": str(s.dtype),
                "n_categories": cat_count,
                "n_missing": int(s.isna().sum()),
            }
        )

    feature_df = pd.DataFrame(rows)

    # Target summary
    y_arr = y_s.to_numpy()
    n_samples = int(len(y_s))
    dtype = str(y_s.dtype)

    unique = pd.unique(y_s.dropna())
    n_unique = len(unique)

    if pd.api.types.is_numeric_dtype(y_s.dtype) and n_unique > 50:
        task = "regression"
        stats = {
            "min": float(np.nanmin(y_arr.astype(float))),
            "max": float(np.nanmax(y_arr.astype(float))),
            "mean": float(np.nanmean(y_arr.astype(float))),
            "std": float(np.nanstd(y_arr.astype(float))),
        }
        target_summary = TargetSummary(
            task=task,
            n_samples=n_samples,
            y_name=y_name,
            dtype=dtype,
            stats=stats,
        )
    else:
        task = "binary" if n_unique == 2 else "multiclass"
        vc = y_s.value_counts(dropna=False)
        counts = {k: int(v) for k, v in vc.to_dict().items()}
        ratios = {k: float(v / n_samples) for k, v in counts.items()}
        target_summary = TargetSummary(
            task=task,
            n_samples=n_samples,
            y_name=y_name,
            dtype=dtype,
            n_classes=int(n_unique),
            class_counts=counts,
            class_ratios=ratios,
        )

    return feature_df, target_summary


def print_feature_report(
    X: Union[pd.DataFrame, np.ndarray, Any],
    y: Any = None,
    *,
    feature_names: Optional[List[str]] = None,
    max_categorical_unique: int = 20,
    treat_int_as_categorical: bool = False,
    max_rows: int = 200,
) -> Tuple[pd.DataFrame, TargetSummary]:
    feature_df, target_summary = build_feature_summary(
        X,
        y,
        feature_names=feature_names,
        max_categorical_unique=max_categorical_unique,
        treat_int_as_categorical=treat_int_as_categorical,
    )

    n_samples = target_summary.n_samples
    n_features = int(feature_df.shape[0])

    print("=== Input Feature Summary ===")
    print(f"n_samples: {n_samples}")
    print(f"n_features: {n_features}")

    print("\n[columns]")
    if len(feature_df) > max_rows:
        print(feature_df.head(max_rows).to_string(index=False))
        print(f"... ({len(feature_df) - max_rows} more rows)")
    else:
        print(feature_df.to_string(index=False))

    print("\n=== Target Summary ===")
    print(f"y_name: {target_summary.y_name}")
    print(f"dtype: {target_summary.dtype}")
    print(f"task: {target_summary.task}")

    if target_summary.task == "regression":
        assert target_summary.stats is not None
        print("stats:")
        for k, v in target_summary.stats.items():
            print(f"  {k}: {v}")
    else:
        print(f"n_classes: {target_summary.n_classes}")
        assert target_summary.class_counts is not None
        assert target_summary.class_ratios is not None
        counts_df = pd.DataFrame(
            {
                "class": list(target_summary.class_counts.keys()),
                "count": list(target_summary.class_counts.values()),
                "ratio": [target_summary.class_ratios[k] for k in target_summary.class_counts.keys()],
            }
        )
        print(counts_df.to_string(index=False))

    return feature_df, target_summary


def one_hot_encode_for_anfis(
    X: Union[pd.DataFrame, Any],
    *,
    min_categories_for_ohe: int = 3,
    max_categorical_unique: int = 20,
    treat_int_as_categorical: bool = False,
    include_missing_as_category: bool = True,
    drop_first: bool = False,
    dtype: str = "float32",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """ANFIS 입력을 위해 categorical을 숫자화합니다.

    기본 정책은 "진짜 categorical만 확장"입니다.
    따라서 integer-like ordinal / discrete numerical columns는
    명시적으로 허용하지 않는 한 one-hot 대상이 아닙니다.

    - categorical && n_categories >= min_categories_for_ohe: one-hot
    - categorical && n_categories == 2: 0/1로 매핑
    - bool: 0/1
    - drop_first=True: one-hot columns에서 기준 카테고리 1개 제거

    Returns:
        X_out: all-numeric DataFrame
        info: 변환 메타데이터
    """

    if _is_ucirepo_dataset(X):
        X, _, _ = _extract_ucirepo_xy(X)
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame (or a ucimlrepo dataset object).")

    X_df = X.copy()
    info: Dict[str, Any] = {
        "one_hot_columns": [],
        "binary_mappings": {},
        "categorical_columns": [],
    }

    categorical_cols = []
    binary_cols = []
    ohe_cols = []

    for col in X_df.columns:
        inferred = infer_feature_type(
            X_df[col],
            max_categorical_unique=max_categorical_unique,
            treat_int_as_categorical=treat_int_as_categorical,
        )
        if inferred != "categorical":
            continue

        s = X_df[col]
        if include_missing_as_category:
            s = s.astype("object")
            s = s.where(~s.isna(), other="__MISSING__")
            X_df[col] = s

        nunique = int(X_df[col].nunique(dropna=False))
        categorical_cols.append(col)

        if pd.api.types.is_bool_dtype(X_df[col].dtype):
            binary_cols.append(col)
        elif nunique >= min_categories_for_ohe:
            ohe_cols.append(col)
        elif nunique == 2:
            binary_cols.append(col)
        else:
            # single category -> map to 0
            binary_cols.append(col)

    info["categorical_columns"] = [str(c) for c in categorical_cols]

    # Binary mapping
    for col in binary_cols:
        s = X_df[col]
        if pd.api.types.is_bool_dtype(s.dtype):
            X_df[col] = s.astype(np.int64)
            info["binary_mappings"][str(col)] = {"True": 1, "False": 0}
            continue

        cats = list(pd.unique(s))
        if len(cats) == 1:
            mapping = {cats[0]: 0}
        else:
            mapping = {cats[0]: 0, cats[1]: 1}
        X_df[col] = s.map(mapping).astype(np.int64)
        info["binary_mappings"][str(col)] = {str(k): int(v) for k, v in mapping.items()}

    # One-hot
    if ohe_cols:
        X_df = pd.get_dummies(
            X_df,
            columns=ohe_cols,
            prefix=[str(c) for c in ohe_cols],
            prefix_sep="=",
            dtype=dtype,
            drop_first=drop_first,
        )
        info["one_hot_columns"] = [str(c) for c in ohe_cols]

    # Ensure numeric
    for col in X_df.columns:
        if not pd.api.types.is_numeric_dtype(X_df[col].dtype):
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

    # Cast dtype
    X_df = X_df.astype(dtype)

    return X_df, info
