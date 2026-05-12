import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json

from grs_config import GRS_PARAM_DEFAULTS, build_grs_experiment_presets, normalize_grs_params

PROJECT_ROOT = Path(__file__).resolve().parent


def set_deterministic(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_seed(seed):
    set_deterministic(seed)


def build_loader(X, y, batch_size, device, task_kind):
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    if task_kind == "regression":
        y_t = torch.tensor(y, dtype=torch.float32)
        if y_t.ndim == 1:
            y_t = y_t.unsqueeze(1)
        y_t = y_t.to(device)
    elif task_kind == "multiclass":
        y_t = torch.tensor(y, dtype=torch.long).to(device)
    else:
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def normalize_key(name):
    return str(name).strip().replace(" ", "_").replace("/", "_").replace("\\", "_").lower()


def load_params(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.is_file():
        return json.loads(path.read_text())
    return {}


def pick_params(source, dataset_key, dataset_name):
    if not source:
        return None
    if dataset_key in source:
        return source[dataset_key]
    if dataset_name in source:
        return source[dataset_name]
    nk = normalize_key(dataset_key)
    nn = normalize_key(dataset_name)
    for key, value in source.items():
        if isinstance(value, dict) and normalize_key(key) in {nk, nn}:
            return value
    return source.get("default")

def load_grs_params(dataset_key, dataset_name):
    source = load_params("hyper_parameter/best_GRS-ANFIS_HP.json")
    picked = pick_params(source, dataset_key, dataset_name) or {}
    merged = dict(GRS_PARAM_DEFAULTS)
    merged.update(picked)
    return normalize_grs_params(merged)


def load_anfis_params(dataset_key, dataset_name):
    defaults = {
        "lr": 0.1,
        "n_rules": 30,
        "epochs": 100,
        "mfs_per_input": 3,
        "selected_features": [],
        "rule_init_mode": "legacy",
        "rule_seed": 0,
        "firing_mode": "htsk",
    }
    source = load_params("hyper_parameter/paper_ANFIS_HP.json")
    if not source:
        source = load_params("hyper_parameter/best_ANFIS_HP.json")
    picked = pick_params(source, dataset_key, dataset_name) or {}
    merged = dict(defaults)
    merged.update(picked)
    return merged


def load_ga_params(dataset_key, dataset_name):
    defaults = {
        "lr": 0.01,
        "n_rules": 20,
        "epochs": 50,
        "selected_features": [],
    }
    source = load_params("hyper_parameter/best_GA-ANFIS_HP.json")
    picked = pick_params(source, dataset_key, dataset_name) or {}
    merged = dict(defaults)
    merged.update(picked)
    return merged



def load_pso_params(dataset_key, dataset_name):
    defaults = {
        "lr": 0.01,
        "n_rules": 20,
        "epochs": 50,
        "selected_features": [],
    }
    source = load_params("hyper_parameter/best_PSO-ANFIS_HP.json")
    picked = pick_params(source, dataset_key, dataset_name) or {}
    merged = dict(defaults)
    merged.update(picked)
    return merged


def load_h_params(dataset_key, dataset_name):
    defaults = {
        "lr": 0.05,
        "epochs": 80,
        "branch_rules": 16,
        "top_rules": 8,
        "mfs_per_input": 2,
        "weight_decay": 1e-5,
        "fusion": "avg",
        "seed_offset": 101,
    }
    source = load_params("hyper_parameter/best_H-ANFIS_HP.json")
    picked = pick_params(source, dataset_key, dataset_name) or {}
    merged = dict(defaults)
    merged.update(picked)
    return merged


def load_common_experiment_settings():
    return load_params("hyper_parameter/common_experiment_settings.json")


def load_svm_params():
    settings = load_common_experiment_settings()
    defaults = {"C": 1.0, "gamma": "scale", "kernel": "rbf", "probability": True, "random_state": 42}
    merged = dict(defaults)
    merged.update(settings.get("svm", {}))
    return merged


def load_tabular_baseline_candidates():
    source = load_params("hyper_parameter/tabular_baseline_candidates.json")
    if source:
        return source
    return {
        "RandomForest": [
            {"n_estimators": 300, "max_features": "sqrt"},
            {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 2, "class_weight": "balanced"},
        ],
        "HGB": [
            {"max_iter": 150, "learning_rate": 0.1, "max_leaf_nodes": 31, "l2_regularization": 0.0},
            {"max_iter": 250, "learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 0.1},
        ],
        "XGBoost": [
            {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4, "subsample": 0.9, "colsample_bytree": 0.9},
            {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 3, "subsample": 0.9, "colsample_bytree": 0.8},
        ],
        "LightGBM": [
            {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9},
            {"n_estimators": 500, "num_leaves": 15, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.8},
        ],
        "CatBoost": [
            {"iterations": 300, "depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
            {"iterations": 500, "depth": 3, "learning_rate": 0.03, "l2_leaf_reg": 5.0},
        ],
        "EBM": [
            {"early_stopping_rounds": 50, "interactions": 0, "learning_rate": 0.02, "max_bins": 128, "max_rounds": 2000, "outer_bags": 4},
            {"early_stopping_rounds": 50, "interactions": 0, "learning_rate": 0.03, "max_bins": 64, "max_rounds": 1000, "outer_bags": 4},
        ],
    }
