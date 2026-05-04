import numpy as np
import torch
import torch.nn.functional as F


def _to_numpy(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def feature_domains_from_data(X, pad: float = 1.0):
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] == 0:
        return []
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    spans = maxs - mins
    tiny = spans < 1e-6
    if np.any(tiny):
        mins = mins.copy()
        maxs = maxs.copy()
        mins[tiny] -= pad
        maxs[tiny] += pad
    return list(zip(mins.tolist(), maxs.tolist()))


def _hat_h(h: np.ndarray, p: int) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    out = np.empty_like(h, dtype=float)
    mask = (h > 0.0) & (h < 1.0)
    out[mask] = h[mask]
    if p <= 1:
        out[~mask] = 0.0
    else:
        out[~mask] = (p - h[~mask]) / (p - 1)
    return out


def nauck_comp(antecedent_counts, n_classes: int = 1) -> float:
    total = float(np.sum(antecedent_counts)) if antecedent_counts else 0.0
    if total <= 0.0:
        return 0.0
    m = max(1, int(n_classes))
    return float(m) / total


def nauck_cov_gaussian(
    centers: np.ndarray,
    sigmas: np.ndarray,
    domains,
    grid_size: int = 201,
) -> float:
    centers = np.asarray(centers, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    if centers.size == 0:
        return 0.0
    if centers.ndim != 2 or sigmas.ndim != 2:
        return 0.0

    cov_i = []
    for i in range(centers.shape[0]):
        p = int(centers.shape[1])
        if p <= 0:
            cov_i.append(0.0)
            continue
        xmin, xmax = domains[i]
        if xmin == xmax:
            xmin -= 1.0
            xmax += 1.0
        x = np.linspace(xmin, xmax, int(grid_size))
        c = centers[i]
        s = np.clip(sigmas[i], 1e-6, None)
        diff = (x[:, None] - c[None, :]) / s[None, :]
        mu = np.exp(-0.5 * diff**2)
        h = np.clip(mu, 0.0, 1.0).sum(axis=1)
        hat = _hat_h(h, p)
        cov_i.append(float(np.mean(hat)))
    return float(np.mean(cov_i)) if cov_i else 0.0


def nauck_part(mfs_per_var) -> float:
    if not mfs_per_var:
        return 0.0
    part_i = [0.0 if int(p) <= 1 else 1.0 / (int(p) - 1) for p in mfs_per_var]
    return float(np.mean(part_i)) if part_i else 0.0


def nauck_index_gaussian(
    centers: np.ndarray,
    sigmas: np.ndarray,
    domains,
    antecedent_counts,
    grid_size: int = 201,
    n_classes: int = 1,
):
    if centers is None or sigmas is None or not domains:
        return 0.0, {"comp": 0.0, "cov": 0.0, "part": 0.0}
    comp = nauck_comp(antecedent_counts, n_classes=n_classes)
    cov = nauck_cov_gaussian(centers, sigmas, domains, grid_size=grid_size)
    p = int(centers.shape[1]) if centers.ndim == 2 else 0
    part = nauck_part([p] * int(centers.shape[0]))
    return comp * cov * part, {"comp": comp, "cov": cov, "part": part}


def _hard_mask_from_logits(logits, threshold: float):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    return (probs >= float(threshold)).astype(np.float32)


def _grs_masks(model, gate_threshold: float):
    if getattr(model, "primary_mask_frozen", False) and getattr(model, "primary_mask_hard", None) is not None:
        primary_mask = _to_numpy(model.primary_mask_hard)
    else:
        primary_mask = _hard_mask_from_logits(model.primary_mask_logits, gate_threshold)

    if getattr(model, "complementary_mask_frozen", False) and getattr(model, "complementary_mask_hard", None) is not None:
        complementary_mask = _to_numpy(model.complementary_mask_hard)
    else:
        complementary_mask = _hard_mask_from_logits(model.complementary_mask_logits, gate_threshold)

    primary_mask = np.asarray(primary_mask, dtype=np.float32)
    complementary_mask = np.asarray(complementary_mask, dtype=np.float32)

    if getattr(model, "complementary_use_complement", True):
        complementary_mask = complementary_mask * (1.0 - primary_mask)

    return primary_mask, complementary_mask


def _grs_branch_params(model, branch: str):
    eps = float(getattr(model, "eps", 1e-6))
    if branch == "primary":
        centers = _to_numpy(getattr(model, "s_mf_centers", None))
        log_sigmas = getattr(model, "s_mf_log_sigmas", None)
        sigmas = _to_numpy(F.softplus(log_sigmas) + eps) if log_sigmas is not None else None
        n_rules = int(getattr(model, "primary_rules", 0))
    elif branch == "complementary":
        centers = _to_numpy(getattr(model, "p_mf_centers", None))
        log_sigmas = getattr(model, "p_mf_log_sigmas", None)
        sigmas = _to_numpy(F.softplus(log_sigmas) + eps) if log_sigmas is not None else None
        n_rules = int(getattr(model, "complementary_rules", 0))
    else:
        raise ValueError(f"Unknown GRS branch: {branch}")
    return centers, sigmas, n_rules


def _nauck_branch(centers, sigmas, domains, mask, n_rules, grid_size: int = 201, n_classes: int = 1):
    if centers is None or sigmas is None or n_rules <= 0:
        return {"index": 0.0, "comp": 0.0, "cov": 0.0, "part": 0.0, "n_rules": n_rules, "n_features": 0}

    centers = np.asarray(centers, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    if centers.ndim != 2 or sigmas.ndim != 2:
        return {"index": 0.0, "comp": 0.0, "cov": 0.0, "part": 0.0, "n_rules": n_rules, "n_features": 0}

    if mask is not None:
        idx = np.where(np.asarray(mask, dtype=bool))[0]
    else:
        idx = np.arange(centers.shape[0])

    if idx.size == 0:
        return {"index": 0.0, "comp": 0.0, "cov": 0.0, "part": 0.0, "n_rules": n_rules, "n_features": 0}

    centers_sel = centers[idx]
    sigmas_sel = sigmas[idx]
    domains_sel = [domains[i] for i in idx]
    n_features = int(idx.size)

    antecedent_counts = [n_features] * int(n_rules)
    index, parts = nauck_index_gaussian(
        centers_sel,
        sigmas_sel,
        domains_sel,
        antecedent_counts,
        grid_size=grid_size,
        n_classes=n_classes,
    )
    return {
        "index": float(index),
        "comp": float(parts["comp"]),
        "cov": float(parts["cov"]),
        "part": float(parts["part"]),
        "n_rules": int(n_rules),
        "n_features": n_features,
    }


def nauck_index_grs(model, X, gate_threshold: float = 0.5, grid_size: int = 201, n_classes: int = None):
    domains = feature_domains_from_data(X)
    if not domains:
        return {"primary": None, "complementary": None, "overall": None}

    if n_classes is None:
        n_classes = int(getattr(model, "n_outputs", 1) or 1)

    primary_mask, complementary_mask = _grs_masks(model, gate_threshold)

    primary_centers, primary_sigmas, primary_rules = _grs_branch_params(model, "primary")
    complementary_centers, complementary_sigmas, complementary_rules = _grs_branch_params(model, "complementary")

    primary_info = _nauck_branch(
        primary_centers,
        primary_sigmas,
        domains,
        primary_mask,
        primary_rules,
        grid_size=grid_size,
        n_classes=n_classes,
    )
    complementary_info = _nauck_branch(
        complementary_centers,
        complementary_sigmas,
        domains,
        complementary_mask,
        complementary_rules,
        grid_size=grid_size,
        n_classes=n_classes,
    )

    total_rules = int(primary_rules) + int(complementary_rules)
    if total_rules > 0:
        overall_index = (
            primary_info["index"] * float(primary_rules) + complementary_info["index"] * float(complementary_rules)
        ) / float(total_rules)
    else:
        overall_index = 0.0

    return {
        "primary": primary_info,
        "complementary": complementary_info,
        "overall": {"index": float(overall_index), "n_rules": total_rules},
    }


def nauck_index_tsk(model, X, grid_size: int = 201, n_classes: int = None):
    domains = feature_domains_from_data(X)
    if not domains:
        return {"index": 0.0, "comp": 0.0, "cov": 0.0, "part": 0.0, "n_rules": 0, "n_features": 0}

    if n_classes is None:
        n_classes = int(getattr(model, "n_outputs", 1) or 1)

    centers = _to_numpy(model.centers)
    sigmas = _to_numpy(torch.exp(model.log_sigmas) + float(getattr(model, "eps", 1e-6)))
    n_rules = int(getattr(model, "n_rules", 0))
    n_features = int(getattr(model, "n_inputs", centers.shape[0] if centers is not None else 0))

    antecedent_counts = [n_features] * int(n_rules) if n_rules > 0 else []
    index, parts = nauck_index_gaussian(
        centers,
        sigmas,
        domains,
        antecedent_counts,
        grid_size=grid_size,
        n_classes=n_classes,
    )
    return {
        "index": float(index),
        "comp": float(parts["comp"]),
        "cov": float(parts["cov"]),
        "part": float(parts["part"]),
        "n_rules": int(n_rules),
        "n_features": int(n_features),
    }


def nauck_index_parallel_hier_tsk(model, X, grid_size: int = 201, n_classes: int = None):
    X_np = np.asarray(X)
    if X_np.ndim != 2 or X_np.shape[0] == 0:
        return {"branch_a": None, "branch_b": None, "top": None, "overall": None, "hfsi": None}

    if n_classes is None:
        n_classes = int(getattr(model, "n_outputs", 1) or 1)

    idx_a = np.asarray(getattr(model, "group_a_idx", []), dtype=int).reshape(-1)
    idx_b = np.asarray(getattr(model, "group_b_idx", []), dtype=int).reshape(-1)
    if idx_a.size == 0 or idx_b.size == 0:
        return {"branch_a": None, "branch_b": None, "top": None, "overall": None, "hfsi": None}

    idx_a = idx_a[(idx_a >= 0) & (idx_a < X_np.shape[1])]
    idx_b = idx_b[(idx_b >= 0) & (idx_b < X_np.shape[1])]
    if idx_a.size == 0 or idx_b.size == 0:
        return {"branch_a": None, "branch_b": None, "top": None, "overall": None, "hfsi": None}

    branch_a_model = getattr(model, "branch_a", None)
    branch_b_model = getattr(model, "branch_b", None)
    if branch_a_model is None or branch_b_model is None:
        return {"branch_a": None, "branch_b": None, "top": None, "overall": None, "hfsi": None}

    branch_a_info = nauck_index_tsk(
        branch_a_model,
        X_np[:, idx_a],
        grid_size=grid_size,
        n_classes=n_classes,
    )
    branch_b_info = nauck_index_tsk(
        branch_b_model,
        X_np[:, idx_b],
        grid_size=grid_size,
        n_classes=n_classes,
    )

    idx_layers = [[branch_a_info.get("index", np.nan), branch_b_info.get("index", np.nan)]]
    comp_layers = [[branch_a_info.get("comp", np.nan), branch_b_info.get("comp", np.nan)]]
    cov_layers = [[branch_a_info.get("cov", np.nan), branch_b_info.get("cov", np.nan)]]
    part_layers = [[branch_a_info.get("part", np.nan), branch_b_info.get("part", np.nan)]]

    top_info = None
    top_model = getattr(model, "top_anfis", None)
    if top_model is not None and hasattr(model, "forward_branches"):
        try:
            device = next(model.parameters()).device
            with torch.no_grad():
                x_t = torch.as_tensor(X_np, dtype=torch.float32, device=device)
                y_a, y_b = model.forward_branches(x_t)
                top_input = torch.cat([y_a, y_b], dim=1).detach().cpu().numpy()
            top_info = nauck_index_tsk(
                top_model,
                top_input,
                grid_size=grid_size,
                n_classes=n_classes,
            )
            idx_layers.append([top_info.get("index", np.nan)])
            comp_layers.append([top_info.get("comp", np.nan)])
            cov_layers.append([top_info.get("cov", np.nan)])
            part_layers.append([top_info.get("part", np.nan)])
        except Exception:
            top_info = None

    idx_hfsi = hfsi_aggregate(idx_layers)
    comp_hfsi = hfsi_aggregate(comp_layers)
    cov_hfsi = hfsi_aggregate(cov_layers)
    part_hfsi = hfsi_aggregate(part_layers)

    n_rules = int(branch_a_info.get("n_rules", 0) or 0) + int(branch_b_info.get("n_rules", 0) or 0)
    if top_info is not None:
        n_rules += int(top_info.get("n_rules", 0) or 0)

    overall = {
        "index": float(idx_hfsi["index"]),
        "comp": float(comp_hfsi["index"]),
        "cov": float(cov_hfsi["index"]),
        "part": float(part_hfsi["index"]),
        "n_rules": int(n_rules),
        "n_layers": int(idx_hfsi["n_layers"]),
        "layer_weights": idx_hfsi["layer_weights"],
    }

    return {
        "branch_a": branch_a_info,
        "branch_b": branch_b_info,
        "top": top_info,
        "overall": overall,
        "hfsi": idx_hfsi,
    }


def _window_domains_from_data(X, kernel_size: int, stride: int, inp_channel: int = 1):
    X = np.asarray(X)
    if X.ndim == 2:
        Xc = X[:, None, :]
    elif X.ndim == 3:
        Xc = X
    else:
        return []

    _, C, L = Xc.shape
    if inp_channel != C:
        inp_channel = C

    if L < kernel_size or kernel_size <= 0:
        return []

    n_windows = 1 + (L - kernel_size) // max(1, stride)
    domains_per_window = []

    for w in range(n_windows):
        start = w * stride
        end = start + kernel_size
        window = Xc[:, :inp_channel, start:end]
        mins = np.min(window, axis=0)
        maxs = np.max(window, axis=0)
        domains = []
        for c in range(inp_channel):
            for k in range(kernel_size):
                domains.append((float(mins[c, k]), float(maxs[c, k])))
        domains_per_window.append(domains)

    return domains_per_window


def _nauck_tanfis(tanfis, domains, grid_size: int = 201, rule_mask=None, n_classes: int = 1):
    centers = _to_numpy(getattr(tanfis, "center", None))
    sigmas = _to_numpy(getattr(tanfis, "sigma", None))
    n_rules = int(getattr(tanfis, "rules_num", 0))
    n_features = int(getattr(tanfis, "input_dim", centers.shape[0] if centers is not None else 0))

    if centers is None or sigmas is None or not domains or n_rules <= 0:
        return {"index": 0.0, "comp": 0.0, "cov": 0.0, "part": 0.0, "n_rules": n_rules, "n_features": n_features}

    if len(domains) != centers.shape[0]:
        domains = domains[: centers.shape[0]]

    if rule_mask is not None:
        mask_np = _to_numpy(rule_mask)
        active_rules = int(np.sum(mask_np > 0)) if mask_np is not None else n_rules
        if active_rules > 0:
            n_rules = active_rules

    antecedent_counts = [n_features] * int(n_rules)
    index, parts = nauck_index_gaussian(
        centers,
        sigmas,
        domains,
        antecedent_counts,
        grid_size=grid_size,
        n_classes=n_classes,
    )
    return {
        "index": float(index),
        "comp": float(parts["comp"]),
        "cov": float(parts["cov"]),
        "part": float(parts["part"]),
        "n_rules": int(n_rules),
        "n_features": int(n_features),
    }


def _aggregate_nauck(modules):
    total_rules = sum(int(m.get("n_rules", 0) or 0) for m in modules)
    total_features = sum(int(m.get("n_features", 0) or 0) for m in modules)
    if total_rules <= 0:
        return {"index": 0.0, "comp": 0.0, "cov": 0.0, "part": 0.0, "n_rules": 0, "n_features": total_features}

    def _wavg(key):
        return float(
            sum(float(m.get(key, 0.0)) * float(m.get("n_rules", 0) or 0) for m in modules)
            / float(total_rules)
        )

    return {
        "index": _wavg("index"),
        "comp": _wavg("comp"),
        "cov": _wavg("cov"),
        "part": _wavg("part"),
        "n_rules": int(total_rules),
        "n_features": int(total_features),
    }


def _acnn_intermediate_features(model, X):
    device = getattr(model, "device", torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(X, dtype=torch.float32, device=device)
        if (getattr(model, "in_channel", 1) == 1) or (x.dim() < 2):
            x = x.unsqueeze(1)
        x_conv = model.tAnfisConv1(x, model.rule_masks[0])
        if getattr(model, "bnorm_en", False):
            x_conv = model.b_norm1(x_conv)
        x_conv = model.act(x_conv)
        if getattr(model, "module_attn_en", False):
            v_in = model.module_attn(x)
            x_tan = x_conv * v_in
        else:
            x_tan = x_conv
        return x_tan.detach().cpu().numpy()


def nauck_index_acnn(model, X, grid_size: int = 201):
    """
    AH-ANFIS/ACNN용 Nauck index 계산
    - conv1 모듈별 및 fc 모듈별 지표 + 전체 평균 제공
    """
    X_np = np.asarray(X)
    if X_np.ndim < 2 or X_np.shape[0] == 0:
        return {"conv": None, "fc": None, "overall": None}

    n_classes = None
    fc = getattr(model, "tAnfisFC", None)
    if fc is not None:
        n_classes = getattr(fc, "out_dim", None)
    if n_classes is None:
        n_classes = 1

    conv_modules = []
    conv = getattr(model, "tAnfisConv1", None)
    if conv is not None:
        conv_domains = _window_domains_from_data(
            X_np,
            kernel_size=int(getattr(conv, "filter_size", 1)),
            stride=int(getattr(conv, "stride", 1)),
            inp_channel=int(getattr(conv, "inp_channel", 1)),
        )
        for i, tanfis in enumerate(getattr(conv, "tAnfis", [])):
            domains = conv_domains[i] if i < len(conv_domains) else feature_domains_from_data(X_np)
            rule_mask = None
            if hasattr(model, "rule_masks") and len(model.rule_masks) > 0:
                rule_mask = model.rule_masks[0][i] if i < len(model.rule_masks[0]) else None
            info = _nauck_tanfis(
                tanfis, domains, grid_size=grid_size, rule_mask=rule_mask, n_classes=n_classes
            )
            conv_modules.append(info)

    fc_modules = []
    if fc is not None:
        x_h1 = _acnn_intermediate_features(model, X_np)
        fc_domains = _window_domains_from_data(
            x_h1,
            kernel_size=int(getattr(fc, "filter_size", 1)),
            stride=int(getattr(fc, "stride", 1)),
            inp_channel=int(getattr(fc, "inp_channel", 1)),
        )
        for i, tanfis in enumerate(getattr(fc, "tAnfis", [])):
            domains = fc_domains[i] if i < len(fc_domains) else feature_domains_from_data(x_h1.reshape(x_h1.shape[0], -1))
            rule_mask = None
            if hasattr(model, "rule_masks") and len(model.rule_masks) > 1:
                rule_mask = model.rule_masks[1][i] if i < len(model.rule_masks[1]) else None
            info = _nauck_tanfis(
                tanfis, domains, grid_size=grid_size, rule_mask=rule_mask, n_classes=n_classes
            )
            fc_modules.append(info)

    conv_summary = _aggregate_nauck(conv_modules) if conv_modules else None
    fc_summary = _aggregate_nauck(fc_modules) if fc_modules else None

    overall_modules = []
    if conv_modules:
        overall_modules.extend(conv_modules)
    if fc_modules:
        overall_modules.extend(fc_modules)
    overall = _aggregate_nauck(overall_modules) if overall_modules else None

    return {
        "conv": {"modules": conv_modules, "summary": conv_summary} if conv_modules else None,
        "fc": {"modules": fc_modules, "summary": fc_summary} if fc_modules else None,
        "overall": overall,
    }


def hmean_index_acnn(nauck_info, layer_weights=None):
    """
    Hmean interpretability for ACNN (Hierarchical ANFIS) using module-level indices.

    Hmean = sum_j l_j * (1/s_j) * sum_k E_jk
    Here E_jk is the module-level interpretability index (e.g., Nauck index).
    """
    if not nauck_info:
        return float("nan")

    layers = []
    conv = nauck_info.get("conv")
    if conv and conv.get("modules"):
        layers.append([m.get("index", 0.0) for m in conv["modules"]])

    fc = nauck_info.get("fc")
    if fc and fc.get("modules"):
        layers.append([m.get("index", 0.0) for m in fc["modules"]])

    if not layers:
        return float("nan")

    q = len(layers)
    if layer_weights is None:
        layer_weights = [1.0 / q] * q
    if len(layer_weights) != q:
        raise ValueError("layer_weights length must match number of layers")

    hmean = 0.0
    for w, vals in zip(layer_weights, layers):
        if not vals:
            continue
        hmean += float(w) * (float(np.mean(vals)))
    return float(hmean)


def hfsi_layer_weights(n_layers: int):
    """
    Hierarchical layer weights from Eq. (7):
      l_i = 2 (n - i + 1) / (n (n + 1)), i=1..n
    """
    n = int(n_layers)
    if n <= 0:
        return []
    denom = float(n * (n + 1))
    return [float(2 * (n - i) / denom) for i in range(n)]


def hfsi_aggregate(layer_values, layer_weights=None):
    """
    Aggregate component interpretability values with HFSi Eq. (6):
      HFSi = sum_i l_i * (1/m_i) * sum_j E_ij
    """
    cleaned_layers = []
    for values in layer_values or []:
        arr = np.asarray(values, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            cleaned_layers.append(arr)

    if not cleaned_layers:
        return {
            "index": float("nan"),
            "n_layers": 0,
            "layer_weights": [],
            "layer_means": [],
            "layer_sizes": [],
        }

    n_layers = len(cleaned_layers)
    if layer_weights is None:
        weights = hfsi_layer_weights(n_layers)
    else:
        weights = [float(w) for w in layer_weights]
        if len(weights) != n_layers:
            raise ValueError("layer_weights length must match number of non-empty layers")

    layer_means = [float(np.mean(arr)) for arr in cleaned_layers]
    layer_sizes = [int(arr.size) for arr in cleaned_layers]
    index = float(sum(w * m for w, m in zip(weights, layer_means)))

    return {
        "index": index,
        "n_layers": int(n_layers),
        "layer_weights": weights,
        "layer_means": layer_means,
        "layer_sizes": layer_sizes,
    }


def nauck_index_grs_hierarchical(model, X, gate_threshold: float = 0.5, grid_size: int = 201, n_classes: int = None):
    """
    GRS-ANFIS hierarchical Nauck (user-selected: 1-layer parallel).
    - Layer 1 has two components: primary and complementary.
    - Eq. (6) with n=1, m1=2 => mean(primary, complementary).
    """
    grs_info = nauck_index_grs(
        model,
        X,
        gate_threshold=gate_threshold,
        grid_size=grid_size,
        n_classes=n_classes,
    )
    primary = grs_info.get("primary") or {}
    complementary = grs_info.get("complementary") or {}

    idx_hfsi = hfsi_aggregate([[primary.get("index", np.nan), complementary.get("index", np.nan)]])
    comp_hfsi = hfsi_aggregate([[primary.get("comp", np.nan), complementary.get("comp", np.nan)]])
    cov_hfsi = hfsi_aggregate([[primary.get("cov", np.nan), complementary.get("cov", np.nan)]])
    part_hfsi = hfsi_aggregate([[primary.get("part", np.nan), complementary.get("part", np.nan)]])

    n_rules = int((primary.get("n_rules", 0) or 0) + (complementary.get("n_rules", 0) or 0))
    overall = {
        "index": float(idx_hfsi["index"]),
        "comp": float(comp_hfsi["index"]),
        "cov": float(cov_hfsi["index"]),
        "part": float(part_hfsi["index"]),
        "n_rules": n_rules,
        "n_layers": int(idx_hfsi["n_layers"]),
        "layer_weights": idx_hfsi["layer_weights"],
    }

    return {
        "primary": primary,
        "complementary": complementary,
        "overall": overall,
        "hfsi": idx_hfsi,
    }


def nauck_index_acnn_hierarchical(model, X, grid_size: int = 201):
    """
    AH-ANFIS hierarchical Nauck with two layers:
    - Layer 1: conv modules
    - Layer 2: fc modules
    Aggregation follows Eq. (6) and Eq. (7).
    """
    acnn_info = nauck_index_acnn(model, X, grid_size=grid_size)
    conv_modules = ((acnn_info.get("conv") or {}).get("modules") or [])
    fc_modules = ((acnn_info.get("fc") or {}).get("modules") or [])

    idx_layers = []
    comp_layers = []
    cov_layers = []
    part_layers = []
    if conv_modules:
        idx_layers.append([m.get("index", np.nan) for m in conv_modules])
        comp_layers.append([m.get("comp", np.nan) for m in conv_modules])
        cov_layers.append([m.get("cov", np.nan) for m in conv_modules])
        part_layers.append([m.get("part", np.nan) for m in conv_modules])
    if fc_modules:
        idx_layers.append([m.get("index", np.nan) for m in fc_modules])
        comp_layers.append([m.get("comp", np.nan) for m in fc_modules])
        cov_layers.append([m.get("cov", np.nan) for m in fc_modules])
        part_layers.append([m.get("part", np.nan) for m in fc_modules])

    idx_hfsi = hfsi_aggregate(idx_layers)
    comp_hfsi = hfsi_aggregate(comp_layers)
    cov_hfsi = hfsi_aggregate(cov_layers)
    part_hfsi = hfsi_aggregate(part_layers)

    total_rules = 0
    for item in conv_modules:
        total_rules += int(item.get("n_rules", 0) or 0)
    for item in fc_modules:
        total_rules += int(item.get("n_rules", 0) or 0)

    overall = {
        "index": float(idx_hfsi["index"]),
        "comp": float(comp_hfsi["index"]),
        "cov": float(cov_hfsi["index"]),
        "part": float(part_hfsi["index"]),
        "n_rules": int(total_rules),
        "n_layers": int(idx_hfsi["n_layers"]),
        "layer_weights": idx_hfsi["layer_weights"],
    }

    return {
        "conv": acnn_info.get("conv"),
        "fc": acnn_info.get("fc"),
        "overall": overall,
        "hfsi": idx_hfsi,
    }


def _active_rule_count_from_mask(rule_mask, fallback_rules: int) -> int:
    if rule_mask is None:
        return int(fallback_rules)
    mask_np = _to_numpy(rule_mask)
    if mask_np is None:
        return int(fallback_rules)
    active = int(np.sum(np.asarray(mask_np) > 0))
    return active if active > 0 else int(fallback_rules)


def _complexity_summary(
    *,
    c_rb: float,
    c_s: float,
    n_layers: int,
    n_modules: int,
    n_rules_total: int,
    n_rules_active: int,
    antecedents_total: float,
    antecedents_active: float,
) -> dict:
    if n_rules_active > 0:
        avg_ante_len = float(antecedents_active) / float(n_rules_active)
    elif n_rules_total > 0:
        avg_ante_len = float(antecedents_total) / float(n_rules_total)
    else:
        avg_ante_len = float("nan")

    if np.isfinite(c_rb) and np.isfinite(c_s):
        chfs = float(np.mean([c_rb, c_s]))
    else:
        chfs = float("nan")

    return {
        # CHFS framework in the cited paper: CHFS = CRB ⊕ CS, with mean operator.
        "chfs": float(chfs),
        "c_rb": float(c_rb),
        "c_s": float(c_s),
        "n_layers": int(n_layers),
        "n_modules": int(n_modules),
        "n_rules_total": int(n_rules_total),
        "n_rules_active": int(n_rules_active),
        "avg_antecedent_len": float(avg_ante_len),
        "antecedents_total": float(antecedents_total),
        "antecedents_active": float(antecedents_active),
    }


def complexity_profile_tsk(model):
    n_rules = int(getattr(model, "n_rules", 0) or 0)
    n_inputs = int(getattr(model, "n_inputs", 0) or 0)
    n_modules = 1 if n_rules > 0 else 0
    n_layers = 1 if n_modules > 0 else 0
    antecedents_total = float(n_rules * n_inputs)
    antecedents_active = antecedents_total
    c_rb = antecedents_active
    c_s = float(n_modules + n_layers)
    return _complexity_summary(
        c_rb=c_rb,
        c_s=c_s,
        n_layers=n_layers,
        n_modules=n_modules,
        n_rules_total=n_rules,
        n_rules_active=n_rules,
        antecedents_total=antecedents_total,
        antecedents_active=antecedents_active,
    )


def complexity_profile_parallel_hier_tsk(model):
    branch_a = getattr(model, "branch_a", None)
    branch_b = getattr(model, "branch_b", None)
    if branch_a is None or branch_b is None:
        return complexity_profile_tsk(model)

    n_rules_a = int(getattr(branch_a, "n_rules", 0) or 0)
    n_inputs_a = int(getattr(branch_a, "n_inputs", 0) or 0)
    n_rules_b = int(getattr(branch_b, "n_rules", 0) or 0)
    n_inputs_b = int(getattr(branch_b, "n_inputs", 0) or 0)

    top = getattr(model, "top_anfis", None)
    if top is not None:
        n_rules_top = int(getattr(top, "n_rules", 0) or 0)
        n_inputs_top = int(getattr(top, "n_inputs", 0) or 0)
        n_modules = 3
        n_layers = 2
    else:
        n_rules_top = 0
        n_inputs_top = 0
        n_modules = 2
        n_layers = 1

    n_rules_total = int(n_rules_a + n_rules_b + n_rules_top)
    n_rules_active = n_rules_total
    antecedents_total = float(
        n_rules_a * n_inputs_a
        + n_rules_b * n_inputs_b
        + n_rules_top * n_inputs_top
    )
    antecedents_active = antecedents_total

    c_rb = antecedents_active
    c_s = float(n_modules + n_layers)
    return _complexity_summary(
        c_rb=c_rb,
        c_s=c_s,
        n_layers=n_layers,
        n_modules=n_modules,
        n_rules_total=n_rules_total,
        n_rules_active=n_rules_active,
        antecedents_total=antecedents_total,
        antecedents_active=antecedents_active,
    )


def complexity_profile_grs(model, variant: str = "full"):
    if variant not in {"full", "primary"}:
        raise ValueError("variant must be one of {'full', 'primary'}")

    primary_rules = int(getattr(model, "primary_rules", 0) or 0)
    complementary_rules = int(getattr(model, "complementary_rules", 0) or 0)
    n_features = int(getattr(model, "n_features", 0) or 0)

    primary_thr = float(getattr(model, "primary_mask_threshold", 0.5))
    complementary_thr = float(getattr(model, "complementary_mask_threshold", 0.5))

    if getattr(model, "primary_mask_frozen", False) and getattr(model, "primary_mask_hard", None) is not None:
        primary_mask = _to_numpy(model.primary_mask_hard)
    else:
        primary_mask = (_to_numpy(torch.sigmoid(model.primary_mask_logits)) >= primary_thr).astype(np.float32)

    if getattr(model, "complementary_mask_frozen", False) and getattr(model, "complementary_mask_hard", None) is not None:
        complementary_mask = _to_numpy(model.complementary_mask_hard)
    else:
        complementary_mask = (_to_numpy(torch.sigmoid(model.complementary_mask_logits)) >= complementary_thr).astype(np.float32)

    if primary_mask is None:
        primary_mask = np.ones(n_features, dtype=np.float32)
    if complementary_mask is None:
        complementary_mask = np.ones(n_features, dtype=np.float32)

    primary_mask = np.asarray(primary_mask, dtype=np.float32).reshape(-1)
    complementary_mask = np.asarray(complementary_mask, dtype=np.float32).reshape(-1)
    if getattr(model, "complementary_use_complement", True):
        complementary_mask = complementary_mask * (1.0 - primary_mask)

    primary_feats = int(np.sum(primary_mask > 0))
    complementary_feats = int(np.sum(complementary_mask > 0))

    if variant == "primary":
        n_modules = 1 if primary_rules > 0 else 0
        n_layers = 1 if n_modules > 0 else 0
        n_rules_total = int(primary_rules)
        n_rules_active = int(primary_rules)
        antecedents_total = float(primary_rules * primary_feats)
        antecedents_active = antecedents_total
    else:
        n_modules = int((1 if primary_rules > 0 else 0) + (1 if complementary_rules > 0 else 0))
        n_layers = 1 if n_modules > 0 else 0
        n_rules_total = int(primary_rules + complementary_rules)
        n_rules_active = int(primary_rules + complementary_rules)
        antecedents_total = float(primary_rules * primary_feats + complementary_rules * complementary_feats)
        antecedents_active = antecedents_total

    c_rb = antecedents_active
    c_s = float(n_modules + n_layers)
    return _complexity_summary(
        c_rb=c_rb,
        c_s=c_s,
        n_layers=n_layers,
        n_modules=n_modules,
        n_rules_total=n_rules_total,
        n_rules_active=n_rules_active,
        antecedents_total=antecedents_total,
        antecedents_active=antecedents_active,
    )


def complexity_profile_acnn(model):
    modules = []

    conv = getattr(model, "tAnfisConv1", None)
    conv_list = list(getattr(conv, "tAnfis", []) or []) if conv is not None else []
    for i, tanfis in enumerate(conv_list):
        n_rules_total = int(getattr(tanfis, "rules_num", 0) or 0)
        n_inputs = int(getattr(tanfis, "input_dim", 0) or 0)
        n_rules_active = n_rules_total
        if hasattr(model, "rule_masks") and len(model.rule_masks) > 0:
            rm = model.rule_masks[0][i] if i < len(model.rule_masks[0]) else None
            n_rules_active = _active_rule_count_from_mask(rm, n_rules_total)
        modules.append((n_rules_total, n_rules_active, n_inputs, "conv"))

    fc = getattr(model, "tAnfisFC", None)
    fc_list = list(getattr(fc, "tAnfis", []) or []) if fc is not None else []
    for i, tanfis in enumerate(fc_list):
        n_rules_total = int(getattr(tanfis, "rules_num", 0) or 0)
        n_inputs = int(getattr(tanfis, "input_dim", 0) or 0)
        n_rules_active = n_rules_total
        if hasattr(model, "rule_masks") and len(model.rule_masks) > 1:
            rm = model.rule_masks[1][i] if i < len(model.rule_masks[1]) else None
            n_rules_active = _active_rule_count_from_mask(rm, n_rules_total)
        modules.append((n_rules_total, n_rules_active, n_inputs, "fc"))

    n_conv_modules = len(conv_list)
    n_fc_modules = len(fc_list)
    n_layers = int((1 if n_conv_modules > 0 else 0) + (1 if n_fc_modules > 0 else 0))
    n_modules = len(modules)
    n_rules_total = int(sum(m[0] for m in modules))
    n_rules_active = int(sum(m[1] for m in modules))
    antecedents_total = float(sum(m[0] * m[2] for m in modules))
    antecedents_active = float(sum(m[1] * m[2] for m in modules))

    c_rb = antecedents_active
    c_s = float(n_modules + n_layers)
    return _complexity_summary(
        c_rb=c_rb,
        c_s=c_s,
        n_layers=n_layers,
        n_modules=n_modules,
        n_rules_total=n_rules_total,
        n_rules_active=n_rules_active,
        antecedents_total=antecedents_total,
        antecedents_active=antecedents_active,
    )


def complexity_profile_model(model, grs_variant: str = "full"):
    if model is None:
        return {
            "chfs": float("nan"),
            "c_rb": float("nan"),
            "c_s": float("nan"),
            "n_layers": float("nan"),
            "n_modules": float("nan"),
            "n_rules_total": float("nan"),
            "n_rules_active": float("nan"),
            "avg_antecedent_len": float("nan"),
            "antecedents_total": float("nan"),
            "antecedents_active": float("nan"),
        }

    if getattr(model, "parallel_hierarchical_anfis", False):
        return complexity_profile_parallel_hier_tsk(model)
    if hasattr(model, "tAnfisConv1") and hasattr(model, "tAnfisFC"):
        return complexity_profile_acnn(model)
    if hasattr(model, "primary_rules") and hasattr(model, "complementary_rules"):
        return complexity_profile_grs(model, variant=grs_variant)
    if hasattr(model, "n_rules") and hasattr(model, "n_inputs"):
        return complexity_profile_tsk(model)

    return {
        "chfs": float("nan"),
        "c_rb": float("nan"),
        "c_s": float("nan"),
        "n_layers": float("nan"),
        "n_modules": float("nan"),
        "n_rules_total": float("nan"),
        "n_rules_active": float("nan"),
        "avg_antecedent_len": float("nan"),
        "antecedents_total": float("nan"),
        "antecedents_active": float("nan"),
    }
