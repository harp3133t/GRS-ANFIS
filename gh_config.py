from collections import OrderedDict
import copy


GH_PARAM_DEFAULTS = {
    "lr_base": 0.01,
    "lr_residual": 0.01,
    "residual_rules": 16,
    "base_rules": 8,
    "mf_per_feature": 2,
    "epochs_stage1": 20,
    "epochs_stage2": 20,
    "lambda_resid_s2": 0.05,
    "lambda_base_s1": 1e-4,
    "weight_decay": 1e-4,
    "base_hard_epochs": 0,
    "residual_hard_epochs": 0,
    "base_mask_threshold": None,
    "residual_mask_threshold": None,
    "lambda_base_hard": 0.0,
    "lambda_resid_hard": 0.0,
    # GH core / ablation switches
    "rule_init_mode": "balanced",
    "rule_seed": 0,
    "firing_mode": "htsk",
    "residual_gate_mode": "complement",
    "use_input_norm": False,
    "enable_residual_branch": True,
    "random_role_assignment": False,
}


def normalize_gh_params(params=None):
    merged = copy.deepcopy(GH_PARAM_DEFAULTS)
    incoming = copy.deepcopy(params or {})
    merged.update(incoming)

    if "residual_use_complement" in incoming and "residual_gate_mode" not in incoming:
        merged["residual_gate_mode"] = (
            "complement" if bool(incoming["residual_use_complement"]) else "independent"
        )

    if "use_htsk_firing" in incoming and "firing_mode" not in incoming:
        merged["firing_mode"] = "htsk" if bool(incoming["use_htsk_firing"]) else "prod"

    merged["rule_init_mode"] = str(merged["rule_init_mode"]).strip().lower()
    merged["firing_mode"] = str(merged["firing_mode"]).strip().lower()
    merged["residual_gate_mode"] = str(merged["residual_gate_mode"]).strip().lower()
    merged["rule_seed"] = int(merged["rule_seed"])
    merged["use_input_norm"] = bool(merged["use_input_norm"])
    merged["enable_residual_branch"] = bool(merged["enable_residual_branch"])
    merged["random_role_assignment"] = bool(merged["random_role_assignment"])

    if merged["rule_init_mode"] not in {"balanced", "legacy"}:
        raise ValueError("rule_init_mode must be one of {'balanced', 'legacy'}")
    if merged["firing_mode"] not in {"prod", "htsk"}:
        raise ValueError("firing_mode must be one of {'prod', 'htsk'}")
    if merged["residual_gate_mode"] not in {"complement", "independent"}:
        raise ValueError(
            "residual_gate_mode must be one of {'complement', 'independent'}"
        )

    if not merged["enable_residual_branch"]:
        merged["epochs_stage2"] = 0
        merged["residual_hard_epochs"] = 0
        merged["lambda_resid_s2"] = 0.0
        merged["lambda_resid_hard"] = 0.0

    return merged


def build_gh_experiment_presets(base_params=None):
    """
    Presets for fair GH ablations.
    - gh_core: gate + structural split only with product firing
    - gh_core_htsk: gh_core + HTSK firing
    - gh_no_complement: independent residual gate instead of complement split
    - gh_legacy_reproduction: closest to the recent GH-ANFIS_exp behavior
    - gh_base_only: ablation without residual branch
    """
    base = normalize_gh_params(base_params)
    presets = OrderedDict()

    def _add(name, **updates):
        cfg = copy.deepcopy(base)
        cfg.update(updates)
        presets[name] = normalize_gh_params(cfg)

    _add(
        "gh_core",
        rule_init_mode="balanced",
        firing_mode="prod",
        residual_gate_mode="complement",
        enable_residual_branch=True,
    )
    _add(
        "gh_core_htsk",
        rule_init_mode="balanced",
        firing_mode="htsk",
        residual_gate_mode="complement",
        enable_residual_branch=True,
    )
    _add(
        "gh_no_complement",
        rule_init_mode="balanced",
        firing_mode="htsk",
        residual_gate_mode="independent",
        enable_residual_branch=True,
    )
    _add(
        "gh_legacy_reproduction",
        rule_init_mode="legacy",
        firing_mode="htsk",
        residual_gate_mode="complement",
        enable_residual_branch=True,
    )
    _add(
        "gh_base_only",
        rule_init_mode="balanced",
        firing_mode="htsk",
        residual_gate_mode="independent",
        enable_residual_branch=False,
    )

    return presets
