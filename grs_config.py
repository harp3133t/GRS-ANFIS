from collections import OrderedDict
import copy


GRS_PARAM_DEFAULTS = {
    "lr_primary": 0.01,
    "lr_complementary": 0.01,
    "complementary_rules": 16,
    "primary_rules": 8,
    "mf_per_feature": 2,
    "epochs_stage1": 20,
    "epochs_stage2": 20,
    "lambda_complementary_s2": 0.05,
    "lambda_primary_s1": 1e-4,
    "weight_decay": 1e-4,
    "primary_hard_epochs": 0,
    "complementary_hard_epochs": 0,
    "primary_mask_threshold": None,
    "complementary_mask_threshold": None,
    "lambda_primary_hard": 0.0,
    "lambda_complementary_hard": 0.0,
    # GRS core / ablation switches
    "rule_init_mode": "balanced",
    "rule_seed": 0,
    "firing_mode": "htsk",
    "complementary_gate_mode": "complement",
    "use_input_norm": False,
    "enable_complementary_branch": True,
    "random_role_assignment": False,
}


def normalize_grs_params(params=None):
    merged = copy.deepcopy(GRS_PARAM_DEFAULTS)
    incoming = copy.deepcopy(params or {})
    merged.update(incoming)

    if "complementary_use_complement" in incoming and "complementary_gate_mode" not in incoming:
        merged["complementary_gate_mode"] = (
            "complement" if bool(incoming["complementary_use_complement"]) else "independent"
        )

    if "use_htsk_firing" in incoming and "firing_mode" not in incoming:
        merged["firing_mode"] = "htsk" if bool(incoming["use_htsk_firing"]) else "prod"

    merged["rule_init_mode"] = str(merged["rule_init_mode"]).strip().lower()
    merged["firing_mode"] = str(merged["firing_mode"]).strip().lower()
    merged["complementary_gate_mode"] = str(merged["complementary_gate_mode"]).strip().lower()
    merged["rule_seed"] = int(merged["rule_seed"])
    merged["use_input_norm"] = bool(merged["use_input_norm"])
    merged["enable_complementary_branch"] = bool(merged["enable_complementary_branch"])
    merged["random_role_assignment"] = bool(merged["random_role_assignment"])

    if merged["rule_init_mode"] not in {"balanced", "legacy"}:
        raise ValueError("rule_init_mode must be one of {'balanced', 'legacy'}")
    if merged["firing_mode"] not in {"prod", "htsk"}:
        raise ValueError("firing_mode must be one of {'prod', 'htsk'}")
    if merged["complementary_gate_mode"] not in {"complement", "independent"}:
        raise ValueError(
            "complementary_gate_mode must be one of {'complement', 'independent'}"
        )

    if not merged["enable_complementary_branch"]:
        merged["epochs_stage2"] = 0
        merged["complementary_hard_epochs"] = 0
        merged["lambda_complementary_s2"] = 0.0
        merged["lambda_complementary_hard"] = 0.0

    return merged


def build_grs_experiment_presets(primary_params=None):
    """
    Presets for fair GRS ablations.
    - grs_core: gate + structural split only with product firing
    - grs_core_htsk: grs_core + HTSK firing
    - grs_no_complement: independent complementary gate instead of complement split
    - grs_legacy_reproduction: closest to the recent GRS-ANFIS behavior
    - grs_primary_only: ablation without complementary branch
    """
    primary = normalize_grs_params(primary_params)
    presets = OrderedDict()

    def _add(name, **updates):
        cfg = copy.deepcopy(primary)
        cfg.update(updates)
        presets[name] = normalize_grs_params(cfg)

    _add(
        "grs_core",
        rule_init_mode="balanced",
        firing_mode="prod",
        complementary_gate_mode="complement",
        enable_complementary_branch=True,
    )
    _add(
        "grs_core_htsk",
        rule_init_mode="balanced",
        firing_mode="htsk",
        complementary_gate_mode="complement",
        enable_complementary_branch=True,
    )
    _add(
        "grs_no_complement",
        rule_init_mode="balanced",
        firing_mode="htsk",
        complementary_gate_mode="independent",
        enable_complementary_branch=True,
    )
    _add(
        "grs_legacy_reproduction",
        rule_init_mode="legacy",
        firing_mode="htsk",
        complementary_gate_mode="complement",
        enable_complementary_branch=True,
    )
    _add(
        "grs_primary_only",
        rule_init_mode="balanced",
        firing_mode="htsk",
        complementary_gate_mode="independent",
        enable_complementary_branch=False,
    )

    return presets
