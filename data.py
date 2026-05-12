import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from ucimlrepo import fetch_ucirepo

from feature_schema import one_hot_encode_for_anfis, print_feature_report

# ucirepo
# 15 : breast-cancer-wisconsin (diagnostic)
# 110 : yeast
# 94 : Spambase
# 52 : Ionosphere
# 54 : ISOLET
# 332 : Online News Popularity
# 186 : Wine Quality
# 320 : dataset.metadata.name

# openml
# 44127 : phoneme
# 37 : pima-diabetes
# 307 : vowel
# 44096 : credit-g


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

LOCAL_OPENML_SNAPSHOTS = {
    307: ("vowel_openml_307.csv", "vowel"),
}

LOCAL_UCI_SNAPSHOTS = {
    94: ("spambase_uci_94.csv", "Spambase"),
    15: ("bcwd_uci_15.csv", "Breast Cancer Wisconsin (Original)"),
}

ANFIS_PREPROCESS_KWARGS = {
    "min_categories_for_ohe": 3,
    "max_categorical_unique": 20,
    "treat_int_as_categorical": False,
    "include_missing_as_category": True,
    "dtype": "float32",
}

BCWD_DATASET_NAMES = {
    "Breast Cancer Wisconsin (Original)",
    "Breast_Cancer_Wisconsin_(Original)",
}

DATASET_PREPROCESS_OVERRIDES = {
    name: {
        "treat_int_as_categorical": True,
        "include_missing_as_category": False,
        "drop_first": True,
    }
    for name in BCWD_DATASET_NAMES
}


def _local_snapshot_path(filename):
    if not filename:
        return None
    path = DATA_DIR / filename
    return path if path.exists() else None


def _load_snapshot_dataset(
    snapshot_path,
    dataset_name,
    *,
    one_hot=False,
    task="auto",
    max_classes=50,
    report=False,
):
    df = pd.read_csv(snapshot_path)
    if "target" not in df.columns:
        raise ValueError(f"Snapshot is missing required 'target' column: {snapshot_path}")

    X_df = df.drop(columns=["target"])
    y_raw = df["target"]
    feature_names = list(X_df.columns)
    if report:
        print_feature_report(X_df, y_raw, feature_names=feature_names)
    return _finalize_dataset(
        dataset_name,
        X_df,
        y_raw,
        feature_names=feature_names,
        one_hot=one_hot,
        task=task,
        max_classes=max_classes,
    )


def _to_numpy_y(y_raw):
    if isinstance(y_raw, pd.DataFrame):
        if y_raw.shape[1] == 1:
            return y_raw.iloc[:, 0].to_numpy()
        return y_raw.to_numpy()
    if isinstance(y_raw, pd.Series):
        return y_raw.to_numpy()
    return np.asarray(y_raw)


def encode_inputs_for_anfis(X_df, **overrides):
    encode_kwargs = dict(ANFIS_PREPROCESS_KWARGS)
    encode_kwargs.update(overrides)
    return one_hot_encode_for_anfis(X_df, **encode_kwargs)

def _squeeze_y(y_arr):
    if y_arr.ndim == 2 and y_arr.shape[1] == 1:
        return y_arr.reshape(-1)
    return y_arr

def prepare_targets(y_raw, task="auto", max_classes=50):
    y_arr = _squeeze_y(_to_numpy_y(y_raw))

    if y_arr.ndim != 1:
        return y_arr, None, "multitarget", y_arr.shape[1]

    if task == "auto":
        if np.issubdtype(y_arr.dtype, np.number) and len(np.unique(y_arr)) > max_classes:
            task = "regression"
        else:
            task = "classification"

    if task == "regression":
        y_out = y_arr.astype(float)
        return y_out, None, "regression", 1

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y_arr)
    n_classes = len(encoder.classes_)
    task_kind = "binary" if n_classes == 2 else "multiclass"
    n_outputs = 1 if task_kind == "binary" else n_classes
    return y_enc, encoder, task_kind, n_outputs



def _ensure_dataframe(X, feature_names=None):
    if isinstance(X, pd.DataFrame):
        return X.copy(), list(X.columns)

    X_arr = np.asarray(X)
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X_arr.shape[1])]
    return pd.DataFrame(X_arr, columns=feature_names), list(feature_names)


def _finalize_dataset(
    name,
    X_df,
    y_raw,
    feature_names=None,
    *,
    one_hot=False,
    task="auto",
    max_classes=50,
):
    X_df, feature_names = _ensure_dataframe(X_df, feature_names)

    if one_hot:
        preprocess_overrides = DATASET_PREPROCESS_OVERRIDES.get(str(name), {})
        X_df, _ = encode_inputs_for_anfis(X_df, **preprocess_overrides)
        feature_names = list(X_df.columns)

    y, encoder, task_kind, n_outputs = prepare_targets(
        y_raw, task=task, max_classes=max_classes
    )

    info = {
        "name": name,
        "task_kind": task_kind,
        "n_inputs": int(X_df.shape[1]),
        "n_outputs": int(n_outputs),
        "one_hot": bool(one_hot),
    }
    return X_df, y, encoder, feature_names, info


def load_ucirepo_dataset(
    dataset_id,
    *,
    one_hot=False,
    task="auto",
    max_classes=50,
    report=False):
    snapshot = LOCAL_UCI_SNAPSHOTS.get(dataset_id)
    if snapshot is not None:
        snapshot_path = _local_snapshot_path(snapshot[0])
        if snapshot_path is not None:
            return _load_snapshot_dataset(
                snapshot_path,
                snapshot[1],
                one_hot=one_hot,
                task=task,
                max_classes=max_classes,
                report=report,
            )

    dataset = fetch_ucirepo(id=dataset_id)
    if report:
        print_feature_report(dataset)

    X_df = dataset.data.features
    feature_names = None
    try:
        feature_names = list(dataset.data.get("headers"))
    except Exception:
        try:
            feature_names = list(dataset.data["headers"])
        except Exception:
            feature_names = None

    name = dataset.metadata.get("name", f"ucirepo_{dataset_id}")
    return _finalize_dataset(
        name,
        X_df,
        dataset.data.targets,
        feature_names=feature_names,
        one_hot=one_hot,
        task=task,
        max_classes=max_classes,
    )


def load_openml_dataset(
    data_id,
    *,
    one_hot=False,
    task="auto",
    max_classes=50,
    report=False,
):
    snapshot = LOCAL_OPENML_SNAPSHOTS.get(data_id)
    if snapshot is not None:
        snapshot_path = _local_snapshot_path(snapshot[0])
        if snapshot_path is not None:
            return _load_snapshot_dataset(
                snapshot_path,
                snapshot[1],
                one_hot=one_hot,
                task=task,
                max_classes=max_classes,
                report=report,
            )

    dataset = fetch_openml(data_id=data_id, as_frame=True)
    X_df = dataset.data
    y_raw = dataset.target
    name = dataset.details.get("name", f"openml_{data_id}")
    if report:
        print_feature_report(X_df, y_raw, feature_names=list(X_df.columns))
    return _finalize_dataset(
        name,
        X_df,
        y_raw,
        feature_names=list(X_df.columns),
        one_hot=one_hot,
        task=task,
        max_classes=max_classes,
    )



def load_vowel_data():
    """Load Vowel with the legacy 29-dimensional encoded representation.

    The local OpenML snapshot stores the 10 acoustic features, speaker, sex, and
    target, but it omits the original Deterding train/test indicator.  The legacy
    experiments used that indicator as an additional categorical column, with the
    standard 528/462 Train/Test split preserved before cross-validation.  We
    reconstruct it here so all rebuttal reruns use the same 29-dimensional input:
    10 numeric features + 15 speaker dummies + 2 sex dummies + 2 split dummies.
    """
    snapshot_path = _local_snapshot_path("vowel_openml_307.csv")
    if snapshot_path is None:
        raise FileNotFoundError("Missing local Vowel snapshot: data/vowel_openml_307.csv")

    df = pd.read_csv(snapshot_path)
    if "target" not in df.columns:
        raise ValueError(f"Snapshot is missing required 'target' column: {snapshot_path}")

    X_df = df.drop(columns=["target"]).copy()
    y_raw = df["target"]
    if "Train_or_Test" not in X_df.columns:
        split = np.where(np.arange(len(X_df)) < 528, "Train", "Test")
        X_df.insert(0, "Train_or_Test", split)

    categorical_cols = ["Train_or_Test", "Speaker_Number", "Sex"]
    X_encoded = pd.get_dummies(
        X_df,
        columns=categorical_cols,
        prefix=categorical_cols,
        prefix_sep="_",
        drop_first=False,
        dummy_na=False,
        dtype="float32",
    )
    X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")

    y, encoder, task_kind, n_outputs = prepare_targets(
        y_raw, task="classification", max_classes=50
    )
    feature_names = list(X_encoded.columns)
    if X_encoded.shape[1] != 29:
        raise ValueError(f"Expected legacy Vowel dimension 29, got {X_encoded.shape[1]}")
    return X_encoded, y, feature_names

def load_spambase_data():
    X_df, y, encoder, feature_names, info = load_ucirepo_dataset(
        dataset_id=94,
        one_hot=True,
        task="classification",
        report=False,
    )
    return X_df, y, feature_names

def load_bcwd_data():
    X_df, y, encoder, feature_names, info = load_ucirepo_dataset(
        dataset_id=15,
        one_hot=True,
        task="classification",
        report=False,
    )
    return X_df, y, feature_names

def load_gisette_data():
    """Loads full labeled Gisette training data."""
    train_data_path = DATA_DIR / "gisette_train.data"
    train_labels_path = DATA_DIR / "gisette_train.labels"
    missing = [path for path in (train_data_path, train_labels_path) if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in missing)
        raise FileNotFoundError(
            "Gisette snapshots are not tracked in git. "
            f"Place the required files under {DATA_DIR} before loading: {missing_text}"
        )
    
    print(f"Loading {train_data_path}...")
    X = pd.read_csv(train_data_path, sep=r'\s+', header=None)
    
    print(f"Loading {train_labels_path}...")
    y = pd.read_csv(train_labels_path, header=None).values.flatten()
    y = np.where(y == -1, 0, 1)
    feature_name = [f"x{i}" for i in range(X.shape[1])]
    X.columns = feature_name

    X_df, y, encoder, feature_names, info = _finalize_dataset(
        "Gisette",
        X,
        y,
        feature_names=feature_name,
        one_hot=True,
        task="classification",
        max_classes=50,
    )
    return X_df, y, feature_names

def coerce_numeric_frame(X_df):
    if not hasattr(X_df, "select_dtypes"):
        return X_df
    cat_cols = X_df.select_dtypes(include=["object", "category", "bool"]).columns
    if len(cat_cols) > 0:
        X_df = pd.get_dummies(X_df, columns=cat_cols, dummy_na=True)
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.fillna(0)
    return X_df


def drop_nan_targets(X_df, y):
    if hasattr(y, "isna"):
        mask = ~y.isna()
    else:
        y_arr = np.asarray(y)
        if np.issubdtype(y_arr.dtype, np.number):
            mask = ~np.isnan(y_arr)
        else:
            mask = np.ones(len(y_arr), dtype=bool)

    if hasattr(X_df, "iloc"):
        X_df = X_df.loc[mask]
    else:
        X_df = X_df[mask]
    return X_df, np.asarray(y)[mask]

def split_train_test(X_df, y, task_kind, test_size, seed):
    n_total = len(y)
    if n_total == 0:
        raise ValueError("Empty dataset; cannot split.")
    indices = np.arange(n_total)
    try:
        if task_kind in {"binary", "multiclass"}:
            train_idx, test_idx = train_test_split(
                indices,
                test_size=test_size,
                random_state=seed,
                stratify=y,
            )
        else:
            train_idx, test_idx = train_test_split(
                indices,
                test_size=test_size,
                random_state=seed,
                shuffle=True,
            )
    except ValueError:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )
    return np.asarray(train_idx), np.asarray(test_idx)

def build_cv_splits(X_df, y, task_kind, n_splits, seed):
    if task_kind in {"binary", "multiclass"}:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(X_df, y))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(X_df, y))


def compute_f1(y_true, y_pred):
    average = "binary" if len(np.unique(y_true)) == 2 else "weighted"
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def evaluate_torch_classification(model, loader, criterion, device, task_kind):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0

    is_bce_logits = isinstance(criterion, nn.BCEWithLogitsLoss)

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)

            if task_kind == "binary":
                if y_batch.dim() == 1:
                    y_true = y_batch.float().unsqueeze(1)
                else:
                    y_true = y_batch.float()
                
                loss = criterion(logits, y_true)
                total_loss += loss.item() * x_batch.size(0)

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long().squeeze(1)
                targets = y_true.long().squeeze(1)
            else:
                loss = criterion(logits, y_batch.long())
                total_loss += loss.item() * x_batch.size(0)

                preds = torch.argmax(logits, dim=1)
                targets = y_batch.long()

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    acc = accuracy_score(y_true, y_pred)
    f1 = compute_f1(y_true, y_pred)
    return avg_loss, acc, f1


def evaluate_torch_regression(model, loader, device):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(x_batch)
            preds.append(outputs.cpu().numpy())
            targets.append(y_batch.cpu().numpy())

    y_pred = np.concatenate(preds, axis=0).reshape(-1)
    y_true = np.concatenate(targets, axis=0).reshape(-1)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2


def _scale_regression_targets(y_train, y_eval=None):
    y_train_arr = np.asarray(y_train, dtype=float).reshape(-1, 1)
    y_scaler = StandardScaler().fit(y_train_arr)
    y_train_scaled = y_scaler.transform(y_train_arr).reshape(-1)
    if y_eval is None:
        return y_train_scaled, None, y_scaler
    y_eval_arr = np.asarray(y_eval, dtype=float).reshape(-1, 1)
    y_eval_scaled = y_scaler.transform(y_eval_arr).reshape(-1)
    return y_train_scaled, y_eval_scaled, y_scaler
