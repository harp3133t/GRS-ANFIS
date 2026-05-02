import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json

from gh_config import GH_PARAM_DEFAULTS, build_gh_experiment_presets, normalize_gh_params


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

def load_gh_params(dataset_key, dataset_name):
    source = load_params("./hyper_parameter/best_GH-ANFIS_HP.json")
    picked = pick_params(source, dataset_key, dataset_name) or {}
    merged = dict(GH_PARAM_DEFAULTS)
    merged.update(picked)
    return normalize_gh_params(merged)


def load_ga_params(dataset_key, dataset_name):
    defaults = {
        "lr": 0.01,
        "n_rules": 20,
        "epochs": 50,
        "selected_features": [],
    }
    source = load_params("./hyper_parameter/best_GA-ANFIS_HP.json")
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
    source = load_params("./hyper_parameter/best_PSO-ANFIS_HP.json")
    picked = pick_params(source, dataset_key, dataset_name) or {}
    merged = dict(defaults)
    merged.update(picked)
    return merged
