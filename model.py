# 모델 구조(GH/TSK-ANFIS) 정의 모음
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def _estimate_rule_grid_size(num_features, num_mfs, limit=None):
    total = 1
    for _ in range(int(max(0, num_features))):
        total *= int(max(1, num_mfs))
        if limit is not None and total > limit:
            return None
    return total


def _build_rule_mf_indices_legacy(num_rules, num_features, num_mfs, device):
    """Legacy rule enumeration with base-M counting over feature slots."""
    num_rules = int(num_rules)
    num_features = int(num_features)
    num_mfs = int(num_mfs)

    if num_rules <= 0:
        return torch.zeros(0, max(num_features, 0), dtype=torch.long, device=device)
    if num_features <= 0 or num_mfs <= 1:
        return torch.zeros(num_rules, max(num_features, 0), dtype=torch.long, device=device)

    indices = torch.zeros(num_rules, num_features, dtype=torch.long, device=device)
    for r in range(num_rules):
        code = r
        for d in range(num_features):
            indices[r, d] = code % num_mfs
            code //= num_mfs
    return indices


def _build_rule_mf_indices_balanced(num_rules, num_features, num_mfs, device, seed=0):
    """
    Approximate balanced rulebook.
    - Small spaces: sample unique rules from the full Cartesian grid.
    - Large spaces: build column-balanced assignments and repair duplicates.
    """
    num_rules = int(num_rules)
    num_features = int(num_features)
    num_mfs = int(num_mfs)

    if num_rules <= 0:
        return torch.zeros(0, max(num_features, 0), dtype=torch.long, device=device)
    if num_features <= 0 or num_mfs <= 1:
        return torch.zeros(num_rules, max(num_features, 0), dtype=torch.long, device=device)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    total_grid = _estimate_rule_grid_size(num_features, num_mfs, limit=50000)
    if total_grid is not None:
        axes = [torch.arange(num_mfs, dtype=torch.long) for _ in range(num_features)]
        grid = torch.cartesian_prod(*axes)
        if num_rules <= total_grid:
            perm = torch.randperm(total_grid, generator=generator)[:num_rules]
            return grid.index_select(0, perm).to(device)

        repeats = math.ceil(num_rules / total_grid)
        sampled = []
        for _ in range(repeats):
            perm = torch.randperm(total_grid, generator=generator)
            sampled.append(grid.index_select(0, perm))
        return torch.cat(sampled, dim=0)[:num_rules].to(device)

    indices = torch.empty(num_rules, num_features, dtype=torch.long)
    repeats = math.ceil(num_rules / num_mfs)
    base_pattern = torch.arange(num_mfs, dtype=torch.long).repeat(repeats)[:num_rules]

    for d in range(num_features):
        perm = torch.randperm(num_rules, generator=generator)
        shift = int(torch.randint(0, num_mfs, (1,), generator=generator).item())
        indices[:, d] = (base_pattern.index_select(0, perm) + shift) % num_mfs

    for _ in range(32):
        seen = set()
        duplicate_rows = []
        for idx, row in enumerate(indices.tolist()):
            key = tuple(row)
            if key in seen:
                duplicate_rows.append(idx)
            else:
                seen.add(key)

        if not duplicate_rows:
            return indices.to(device)

        for idx in duplicate_rows:
            indices[idx] = torch.randint(
                num_mfs,
                (num_features,),
                generator=generator,
                dtype=torch.long,
            )

    return indices.to(device)


def _build_rule_mf_indices(num_rules, num_features, num_mfs, device, mode="legacy", seed=0):
    """Build fixed TSK-style MF assignments per rule/feature."""
    mode = str(mode).strip().lower()
    if mode in {"legacy", "biased", "base_m"}:
        return _build_rule_mf_indices_legacy(num_rules, num_features, num_mfs, device)
    if mode in {"balanced", "balanced_random", "random"}:
        return _build_rule_mf_indices_balanced(
            num_rules,
            num_features,
            num_mfs,
            device,
            seed=seed,
        )
    raise ValueError(f"Unsupported rule_init_mode: {mode}")


def _indices_to_compat_logits(rule_mf_indices, num_mfs):
    """
    Keep analysis code compatible by exposing fixed MF indices as near-one-hot logits.
    """
    num_mfs = int(max(1, num_mfs))
    r, d = rule_mf_indices.shape
    if d == 0:
        return torch.empty(r, d, num_mfs, device=rule_mf_indices.device)

    logits = torch.full(
        (r, d, num_mfs),
        -10.0,
        device=rule_mf_indices.device,
        dtype=torch.float32,
    )
    logits.scatter_(2, rule_mf_indices.unsqueeze(-1), 10.0)
    return logits


def _selector_logits_to_indices(selector_logits):
    if selector_logits is None:
        return None
    if selector_logits.ndim != 3:
        raise ValueError(
            f"Expected selector logits with shape (R, D, M), got {tuple(selector_logits.shape)}"
        )
    return selector_logits.argmax(dim=-1).to(dtype=torch.long)


def _aggregate_rule_firing(selected_mf_activations, eps, firing_mode="prod", effective_dim=None):
    """Aggregate selected MF activations into normalized rule weights."""
    if selected_mf_activations.ndim != 3:
        raise ValueError(
            "selected_mf_activations must have shape (B, R, D), "
            f"got {tuple(selected_mf_activations.shape)}"
        )

    bsz, n_rules, n_dims = selected_mf_activations.shape
    if n_dims == 0:
        return torch.full(
            (bsz, n_rules),
            1.0 / max(n_rules, 1),
            device=selected_mf_activations.device,
            dtype=selected_mf_activations.dtype,
        )

    firing_mode = str(firing_mode).strip().lower()
    if firing_mode == "prod":
        firing = selected_mf_activations.prod(dim=2)
    elif firing_mode == "htsk":
        log_activations = torch.log(selected_mf_activations + eps)
        sum_log_activations = log_activations.sum(dim=2)
        if effective_dim is None:
            effective_dim = float(n_dims)
        if not torch.is_tensor(effective_dim):
            effective_dim = torch.tensor(
                effective_dim,
                device=selected_mf_activations.device,
                dtype=selected_mf_activations.dtype,
            )
        effective_dim = torch.clamp(effective_dim, min=1.0)
        firing = torch.exp(sum_log_activations / effective_dim)
    else:
        raise ValueError(f"Unsupported firing_mode: {firing_mode}")

    return firing / (firing.sum(dim=1, keepdim=True) + eps)


def _hard_mask_from_probs(mask_probs, threshold):
    threshold = float(threshold)
    return (mask_probs >= threshold).to(dtype=mask_probs.dtype)



class GH_ANFIS(nn.Module):
    """
    - 각 변수마다 M개의 Gaussian MF (center, sigma)
    - base / residual 두 개의 규칙 집합
    - base(primary):
        * soft stage: 후건부만 soft gate로 feature weight를 조절
        * hard stage: 전건부/후건부 모두 gate의 hard mask를 사용
    - residual(complementary):
        * base hard complement를 우선 사용
        * soft stage: 후건부만 soft gate를 사용
        * hard stage: 전건부/후건부 모두 hard mask를 사용
    - phase에 따라 branch 관계를 다르게 설정:
        * 'base'               : base만 사용, residual은 off
        * 'residual_complement': residual은 base complement로만 사용
        * 'joint'              : base + residual 모두 사용 (residual은 complement)
    - mode에 따라 출력 결합 방식 결정:
        * 'full'          : y = y_base + y_residual
        * 'base_only'     : y = y_base
        * 'residual_only' : y = y_residual
    """

    def __init__(
        self,
        n_features,
        n_outputs,
        residual_rules=8,
        base_rules=4,
        mf_per_feature=2,
        device=None,
        eps=1e-6,
        residual_gate_mode="complement",
        rule_init_mode="balanced",
        rule_seed=0,
        firing_mode="htsk",
        use_input_norm=False,
        enable_residual_branch=True,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.n_features = n_features
        self.n_outputs = n_outputs

        self.residual_rules = residual_rules
        self.base_rules = base_rules
        self.mf_per_feature = mf_per_feature
        self.eps = eps
        self.rule_init_mode = str(rule_init_mode).strip().lower()
        self.rule_seed = int(rule_seed)
        self.firing_mode = str(firing_mode).strip().lower()
        self.use_input_norm = bool(use_input_norm)
        self.enable_residual_branch = bool(enable_residual_branch)
        self.residual_gate_mode = str(residual_gate_mode).strip().lower()
        if self.residual_gate_mode not in {"complement", "independent"}:
            raise ValueError(
                "residual_gate_mode must be one of {'complement', 'independent'}"
            )

        # 출력 결합 모드
        self.mode = "full"  # 'full' | 'base_only' | 'residual_only'

        # 학습 단계 / 게이트 관계 모드
        self.phase = "joint"  # 'joint' | 'base' | 'residual_complement'

        # 입력 정규화 (원하면 forward에서 self.norm(x)로 사용)
        self.norm = nn.BatchNorm1d(n_features)

        # 전역(base) 마스크: soft 확률 → threshold로 hard 고정
        self.base_mask_logits = nn.Parameter(torch.zeros(n_features, device=self.device))
        self.register_buffer(
            "base_mask_hard", torch.zeros(n_features, device=self.device)
        )
        self.base_mask_threshold = 0.5
        self.base_mask_frozen = False

        # 전역(residual) 마스크: soft 확률 → threshold로 hard 고정
        self.residual_mask_logits = nn.Parameter(torch.zeros(n_features, device=self.device))
        self.register_buffer(
            "residual_mask_hard", torch.zeros(n_features, device=self.device)
        )
        self.residual_mask_threshold = 0.5
        self.residual_mask_frozen = False
        self.residual_use_complement = self.residual_gate_mode == "complement"

        # MF / Rule / Consequent 초기화
        self.residual_input_dim = self.n_features
        self._init_residual(self.residual_input_dim, self.mf_per_feature)
        self._init_base(self.n_features, self.mf_per_feature)


        self.is_structured = True
        self.config = {
            "rule_init_mode": self.rule_init_mode,
            "rule_seed": self.rule_seed,
            "firing_mode": self.firing_mode,
            "residual_gate_mode": self.residual_gate_mode,
            "use_input_norm": self.use_input_norm,
            "enable_residual_branch": self.enable_residual_branch,
        }

    # ───────── residual 초기화 ─────────
    def _init_residual(self, D, M):
        if M > 1:
            min_val, max_val = -1.5, 1.5
            centers = torch.linspace(min_val, max_val, M, device=self.device)
            self.p_mf_centers = nn.Parameter(centers.unsqueeze(0).repeat(D, 1))  # (D, M)

            sigma = (max_val - min_val) / (M - 1)
            initial_log_sigma = torch.log(
                torch.exp(torch.tensor(sigma, device=self.device)) - 1 + self.eps
            )
            self.p_mf_log_sigmas = nn.Parameter(
                torch.full((D, M), initial_log_sigma, device=self.device)
            )
        else:
            self.p_mf_centers = nn.Parameter(torch.zeros(D, M, device=self.device))
            self.p_mf_log_sigmas = nn.Parameter(torch.zeros(D, M, device=self.device))

        self.register_buffer(
            "p_rule_mf_indices",
            _build_rule_mf_indices(
                self.residual_rules,
                D,
                M,
                self.device,
                mode=self.rule_init_mode,
                seed=self.rule_seed + 1009,
            ),
            persistent=False,
        )
        self.register_buffer(
            "p_rule_mf_selector_logits",
            _indices_to_compat_logits(self.p_rule_mf_indices, M),
        )

        self.residual_consequents = nn.ModuleList(
            [nn.Linear(D + 1, 1) for _ in range(self.residual_rules * self.n_outputs)]
        )
        if self.device.type == "cuda":
            for layer in self.residual_consequents:
                layer.to(self.device)

    # ───────── base 초기화 ─────────
    def _init_base(self, D, M):
        if D > 0:
            if M > 1:
                min_val, max_val = -1.5, 1.5
                centers = torch.linspace(min_val, max_val, M, device=self.device)
                self.s_mf_centers = nn.Parameter(centers.unsqueeze(0).repeat(D, 1))  # (D, M)

                sigma = (max_val - min_val) / (M - 1)
                initial_log_sigma = torch.log(
                    torch.exp(torch.tensor(sigma, device=self.device)) - 1 + self.eps
                )
                self.s_mf_log_sigmas = nn.Parameter(
                    torch.full((D, M), initial_log_sigma, device=self.device)
                )
            else:
                self.s_mf_centers = nn.Parameter(torch.zeros(D, M, device=self.device))
                self.s_mf_log_sigmas = nn.Parameter(torch.zeros(D, M, device=self.device))

            self.register_buffer(
                "s_rule_mf_indices",
                _build_rule_mf_indices(
                    self.base_rules,
                    D,
                    M,
                    self.device,
                    mode=self.rule_init_mode,
                    seed=self.rule_seed + 17,
                ),
                persistent=False,
            )
            self.register_buffer(
                "s_rule_mf_selector_logits",
                _indices_to_compat_logits(self.s_rule_mf_indices, M),
            )

            self.base_consequents = nn.ModuleList(
                [nn.Linear(D + 1, 1) for _ in range(self.base_rules * self.n_outputs)]
            )
            if self.device.type == "cuda":
                for layer in self.base_consequents:
                    layer.to(self.device)
        else:
            self.register_parameter("s_mf_centers", None)
            self.register_parameter("s_mf_log_sigmas", None)
            self.register_buffer(
                "s_rule_mf_indices",
                torch.zeros(self.base_rules, 0, dtype=torch.long, device=self.device),
                persistent=False,
            )
            self.register_buffer(
                "s_rule_mf_selector_logits",
                torch.empty(
                    self.base_rules,
                    0,
                    max(1, int(M)),
                    dtype=torch.float32,
                    device=self.device,
                ),
                persistent=False,
            )
            self.base_consequents = None

    def _resolve_gate_state(
        self,
        mask_logits,
        mask_hard_buffer,
        mask_frozen,
        threshold,
        batch_size,
        use_soft_eval=False,
    ):
        mask_probs = torch.sigmoid(mask_logits)
        if mask_frozen:
            mask_hard = mask_hard_buffer
        else:
            mask_hard = _hard_mask_from_probs(mask_probs, threshold)

        mask_probs_b = mask_probs.unsqueeze(0).expand(batch_size, -1)
        mask_hard_b = mask_hard.unsqueeze(0).expand(batch_size, -1)

        if mask_frozen:
            antecedent_mask = mask_hard_b
            consequent_gate = mask_hard_b
        elif self.training or use_soft_eval:
            antecedent_mask = None
            consequent_gate = mask_probs_b
        else:
            antecedent_mask = mask_hard_b
            consequent_gate = mask_hard_b

        return mask_probs_b, mask_hard_b, antecedent_mask, consequent_gate

    def _calculate_membership(
        self,
        x_sel,
        mf_centers_per_feature,
        mf_log_sigmas_per_feature,
        rule_mf_indices,
        antecedent_mask=None,
    ):
        mf_sigmas_per_feature = F.softplus(mf_log_sigmas_per_feature) + self.eps
        B, D = x_sel.shape

        # 1. Gaussian MF 계산 (B, D, M)
        x_expanded = x_sel.unsqueeze(2)
        dist_sq_per_mf = ((x_expanded - mf_centers_per_feature) / mf_sigmas_per_feature).pow(2)
        mf_activations = torch.exp(-0.5 * dist_sq_per_mf)

        # 2. Hard antecedent mask 적용
        # mask=0인 feature는 don't-care로 간주하여 membership을 1.0으로 둔다.
        if antecedent_mask is not None:
            antecedent_mask = antecedent_mask.unsqueeze(-1)  # (B, D, 1)
            mf_activations = antecedent_mask * mf_activations + (1.0 - antecedent_mask) * 1.0

        # 3. 룰별 고정 MF 인덱스로 직접 선택 (B, R, D)
        if D == 0:
            n_rules = int(rule_mf_indices.shape[0])
            return torch.full(
                (B, n_rules),
                1.0 / max(n_rules, 1),
                device=x_sel.device,
                dtype=x_sel.dtype,
            )

        if rule_mf_indices.device != x_sel.device:
            rule_mf_indices = rule_mf_indices.to(x_sel.device)
        rule_mf_indices = rule_mf_indices.long()

        batch_idx = torch.arange(B, device=x_sel.device).view(B, 1, 1)
        feature_idx = torch.arange(D, device=x_sel.device).view(1, 1, D)
        selected_idx = rule_mf_indices.unsqueeze(0).expand(B, -1, -1)
        selected_mf_activations = mf_activations[batch_idx, feature_idx, selected_idx]

        effective_dim = None
        if antecedent_mask is not None:
            effective_dim = antecedent_mask.squeeze(-1).sum(dim=1, keepdim=True)
        return _aggregate_rule_firing(
            selected_mf_activations,
            eps=self.eps,
            firing_mode=self.firing_mode,
            effective_dim=effective_dim,
        )

    # ───────── 후건부 출력 ─────────
    def _module_consequent(self, x_sel, consequents, n_rules, gate=None):
        """
        x_sel: (B, D)
        gate: (B, D) or None
        consequents: ModuleList of Linear(D+1, 1)
        gate는 feature 값이 아니라 후건부 feature weight에만 곱해진다.
        """
        B, D = x_sel.shape
        O = self.n_outputs

        out_list = []
        for o in range(O):
            per_rule_out = []
            for r in range(n_rules):
                layer = consequents[o * n_rules + r]
                feat_weight = layer.weight[:, :D]
                bias_weight = layer.weight[:, D:]

                if D > 0:
                    if gate is not None:
                        gated_feat_weight = feat_weight * gate
                    else:
                        gated_feat_weight = feat_weight.expand(B, -1)
                    feat_out = torch.sum(x_sel * gated_feat_weight, dim=1)
                else:
                    feat_out = torch.zeros(B, device=x_sel.device, dtype=x_sel.dtype)

                bias_out = bias_weight.squeeze(1).expand(B)
                if layer.bias is not None:
                    bias_out = bias_out + layer.bias.view(1).expand(B)

                per_rule_out.append(feat_out + bias_out)
            out_list.append(torch.stack(per_rule_out, dim=1))  # (B, R)

        # (B, R, O)
        return torch.stack(out_list, dim=0).permute(1, 2, 0)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for attr_name, compat_name in (
            ("s_rule_mf_indices", "s_rule_mf_selector_logits"),
            ("p_rule_mf_indices", "p_rule_mf_selector_logits"),
        ):
            key = prefix + compat_name
            if key not in state_dict or not hasattr(self, attr_name):
                continue
            selector_logits = state_dict[key]
            rule_mf_indices = _selector_logits_to_indices(selector_logits)
            getattr(self, attr_name).copy_(rule_mf_indices.to(getattr(self, attr_name).device))
            compat_logits = _indices_to_compat_logits(
                getattr(self, attr_name),
                selector_logits.shape[-1],
            )
            state_dict[key] = compat_logits.to(
                device=selector_logits.device,
                dtype=selector_logits.dtype,
            )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    # ───────── 모드 & freeze 유틸 ─────────
    def set_mode(self, mode: str):
        """
        mode: 'full' | 'base_only' | 'residual_only'
        - 'full'          : y = s_out + p_out
        - 'base_only'     : y = s_out
        - 'residual_only' : y = p_out
        """
        assert mode in ["full", "base_only", "residual_only"]
        self.mode = mode

    def set_phase(self, phase: str):
        """
        phase: 'joint' | 'base' | 'residual_complement'
        - 'base'               : base만 사용, residual off
        - 'residual_complement': residual은 base hard complement만 사용
        - 'joint'              : base + residual 모두 사용 (residual은 complement)
        """
        assert phase in ["joint", "base", "residual_complement"]
        self.phase = phase

    def set_residual_use_complement(self, enabled: bool):
        self.residual_use_complement = bool(enabled)
        self.residual_gate_mode = "complement" if self.residual_use_complement else "independent"

    def freeze_residual(self):
        """residual branch 파라미터 gradients 막기"""
        self.p_mf_centers.requires_grad_(False)
        self.p_mf_log_sigmas.requires_grad_(False)
        for layer in self.residual_consequents:
            for p in layer.parameters():
                p.requires_grad_(False)

    def unfreeze_residual(self):
        """residual branch 파라미터 gradients 다시 허용"""
        self.p_mf_centers.requires_grad_(True)
        self.p_mf_log_sigmas.requires_grad_(True)
        for layer in self.residual_consequents:
            for p in layer.parameters():
                p.requires_grad_(True)

    def freeze_base(self):
        """base branch 파라미터 gradients 막기"""
        if self.base_consequents is None:
            return
        self.s_mf_centers.requires_grad_(False)
        self.s_mf_log_sigmas.requires_grad_(False)
        for layer in self.base_consequents:
            for p in layer.parameters():
                p.requires_grad_(False)

    def unfreeze_base(self):
        """base branch 파라미터 gradients 다시 허용"""
        if self.base_consequents is None:
            return
        self.s_mf_centers.requires_grad_(True)
        self.s_mf_log_sigmas.requires_grad_(True)
        for layer in self.base_consequents:
            for p in layer.parameters():
                p.requires_grad_(True)

    def get_base_mask_probs(self):
        return torch.sigmoid(self.base_mask_logits)

    def freeze_base_mask(self, threshold: float = None):
        if threshold is None:
            threshold = self.base_mask_threshold
        with torch.no_grad():
            base_soft = torch.sigmoid(self.base_mask_logits)
            hard = _hard_mask_from_probs(base_soft, threshold)
            self.base_mask_hard.copy_(hard)
        self.base_mask_frozen = True
        self.base_mask_logits.requires_grad_(False)

    def unfreeze_base_mask(self):
        self.base_mask_frozen = False
        self.base_mask_logits.requires_grad_(True)

    def get_residual_mask_probs(self):
        return torch.sigmoid(self.residual_mask_logits)

    def freeze_residual_mask(self, threshold: float = None):
        if threshold is None:
            threshold = self.residual_mask_threshold
        with torch.no_grad():
            residual_soft = torch.sigmoid(self.residual_mask_logits)
            hard = _hard_mask_from_probs(residual_soft, threshold)
            self.residual_mask_hard.copy_(hard)
        self.residual_mask_frozen = True
        self.residual_mask_logits.requires_grad_(False)

    def unfreeze_residual_mask(self):
        self.residual_mask_frozen = False
        self.residual_mask_logits.requires_grad_(True)

    def freeze_base_routing(self, threshold: float = None):
        self.freeze_base_mask(threshold=threshold)

    def unfreeze_base_routing(self):
        self.unfreeze_base_mask()

    def freeze_residual_routing(self):
        self.freeze_residual_mask(threshold=self.residual_mask_threshold)
        if hasattr(self, "residual_routing_net") and self.residual_routing_net is not None:
            for p in self.residual_routing_net.parameters():
                p.requires_grad_(False)

    def unfreeze_residual_routing(self):
        self.unfreeze_residual_mask()
        if hasattr(self, "residual_routing_net") and self.residual_routing_net is not None:
            for p in self.residual_routing_net.parameters():
                p.requires_grad_(True)

    # 기존 이름 호환
    def freeze_routing(self):
        self.freeze_base_routing()
        self.freeze_residual_routing()

    def unfreeze_routing(self):
        self.unfreeze_base_routing()
        self.unfreeze_residual_routing()

    # ───────── Forward ─────────
    def forward(self, x, return_activations: bool = False, use_soft_eval: bool = False):
        """
        x: (B, D)
        return:
            y: (B, O)
            + optionally dict of activations if return_activations=True

        use_soft_eval:
            평가 단계에서 후건부 gate만 soft(sigmoid)로 볼지 여부 (해석용 옵션)
        """
        x = x.to(self.device)

        x_n = self.norm(x) if self.use_input_norm else x
        B, D = x_n.shape
        # print(B, D)

        # ── Base(primary) gate
        # soft stage: consequent only
        # hard stage: antecedent + consequent both hard masked
        base_soft_b, base_hard_b, antecedent_base_mask, gate_base = self._resolve_gate_state(
            self.base_mask_logits,
            self.base_mask_hard,
            self.base_mask_frozen,
            self.base_mask_threshold,
            B,
            use_soft_eval=use_soft_eval,
        )

        # ── Residual(complementary) gate ────────────────────────────────
        residual_soft_b, residual_hard_b, residual_self_antecedent_mask, gate_residual_raw = self._resolve_gate_state(
            self.residual_mask_logits,
            self.residual_mask_hard,
            self.residual_mask_frozen,
            self.residual_mask_threshold,
            B,
            use_soft_eval=use_soft_eval,
        )

        # ── base branch ───────────────────────────
        if self.base_consequents is not None:
            s_w = self._calculate_membership(
                x_n,
                self.s_mf_centers,
                self.s_mf_log_sigmas,
                self.s_rule_mf_indices,
                antecedent_mask=antecedent_base_mask,
            )  # (B, R_s)

            s_rule_out = self._module_consequent(
                x_n, self.base_consequents, self.base_rules, gate=gate_base
            )  # (B, R_s, O)
            s_out = (s_w.unsqueeze(-1) * s_rule_out).sum(dim=1)  # (B, O)
        else:
            s_w = None
            s_out = torch.zeros(B, self.n_outputs, device=self.device)

        # ── residual branch ─────────────────────────────
        if (not self.enable_residual_branch) or self.phase == "base":
            residual_available_mask = torch.zeros(B, D, device=self.device)
            antecedent_residual_mask = torch.zeros(B, D, device=self.device)
            gate_residual = torch.zeros(B, D, device=self.device)
            p_w = torch.zeros(B, self.residual_rules, device=self.device)
            p_out = torch.zeros(B, self.n_outputs, device=self.device)
        else:
            if self.residual_gate_mode == "complement":
                residual_available_mask = (1.0 - base_hard_b).detach()
                gate_residual = residual_available_mask * gate_residual_raw
                if residual_self_antecedent_mask is None:
                    antecedent_residual_mask = residual_available_mask
                else:
                    antecedent_residual_mask = residual_available_mask * residual_self_antecedent_mask
            elif self.residual_gate_mode == "independent":
                residual_available_mask = torch.ones(B, D, device=self.device)
                antecedent_residual_mask = residual_self_antecedent_mask
                gate_residual = gate_residual_raw
            else:
                raise ValueError(f"Unsupported residual_gate_mode: {self.residual_gate_mode}")

            # residual input = original features only (removed s_out)
            residual_input = x_n

            p_w = self._calculate_membership(
                residual_input,
                self.p_mf_centers,
                self.p_mf_log_sigmas,
                self.p_rule_mf_indices,
                antecedent_mask=antecedent_residual_mask,
            )  # (B, R_p)

            p_rule_out = self._module_consequent(
                residual_input,
                self.residual_consequents,
                self.residual_rules,
                gate=gate_residual,
            )  # (B, R_p, O)
            p_out = (p_w.unsqueeze(-1) * p_rule_out).sum(dim=1)  # (B, O)

        # ── 최종 출력 결합 (mode에 따라) ─────────────────
        if self.mode == "base_only":
            y = s_out
        elif self.mode == "residual_only":
            y = p_out
        else:  # "full"
            y = s_out + p_out

        if return_activations:
            activations = {
                "gate_base": gate_base,                            # consequent gate (B, D)
                "gate_base_raw": base_soft_b,                      # sigmoid gate probs (B, D)
                "gate_base_hard": base_hard_b,                     # antecedent hard mask (B, D)
                "antecedent_mask_base": antecedent_base_mask,      # (B, D) or None in soft stage
                "gate_residual_raw": gate_residual_raw,            # residual sigmoid gate probs (B, D)
                "gate_residual_hard": residual_hard_b,             # residual self hard mask (B, D)
                "gate_residual": gate_residual,                    # effective consequent gate (B, D)
                "antecedent_mask_residual": antecedent_residual_mask,  # (B, D) or None in soft stage
                "residual_available_mask": residual_available_mask, # base complement mask (B, D)
                "residual_rule_weights": p_w,                      # (B, R_p)
                "base_rule_weights": s_w,                          # (B, R_s) or None
                "residual_output": p_out,                          # (B, O)
                "base_output": s_out,                              # (B, O)
            }
            return y, activations

        return y

class TSKANFIS(nn.Module):
    """
    공유 MF 기반 TSK-ANFIS
    - 입력별 Gaussian MF를 공유하고, 룰별 antecedent는 고정 TSK-style MF index를 사용
    - 룰 발화도 = 입력별로 지정된 MF 활성도를 prod 또는 HTSK 방식으로 집계
    - 정규화된 발화도로 TSK 선형 후건부를 가중합
    """

    def __init__(
        self,
        n_inputs: int,
        n_rules: int,
        n_outputs: int,
        mfs_per_input: int = 3,
        eps: float = 1e-6,
        rule_init_mode: str = "legacy",
        rule_seed: int = 0,
        firing_mode: str = "htsk",
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_rules = n_rules
        self.n_outputs = n_outputs
        self.mfs_per_input = mfs_per_input
        self.eps = eps
        self.rule_init_mode = str(rule_init_mode).strip().lower()
        self.rule_seed = int(rule_seed)
        self.firing_mode = str(firing_mode).strip().lower()

        # 1) 전건부(Antecedent) 파라미터: 입력별 공유 (D, M)
        self.centers = nn.Parameter(torch.randn(n_inputs, mfs_per_input) * 0.1)
        self.log_sigmas = nn.Parameter(torch.zeros(n_inputs, mfs_per_input))

        # 2) 룰별 고정 MF 선택 인덱스 (R, D)
        self.register_buffer(
            "rule_mf_indices",
            _build_rule_mf_indices(
                n_rules,
                n_inputs,
                mfs_per_input,
                self.centers.device,
                mode=self.rule_init_mode,
                seed=self.rule_seed,
            ),
            persistent=False,
        )
        self.register_buffer(
            "rule_mf_selector_logits",
            _indices_to_compat_logits(self.rule_mf_indices, mfs_per_input),
        )

        # 3) 후건부(Consequent) 파라미터: (R, O, D+1)
        self.consequents = nn.Parameter(
            torch.randn(n_rules, n_outputs, n_inputs + 1) * 0.1
        )

    @staticmethod
    def _gaussian_mf(
        x: torch.Tensor, centers: torch.Tensor, sigmas: torch.Tensor
    ) -> torch.Tensor:
        x_ = x.unsqueeze(-1)                # (B, D, 1)
        c_ = centers.unsqueeze(0)           # (1, D, M)
        s_ = sigmas.unsqueeze(0).clamp_min(1e-6)
        return torch.exp(-0.5 * ((x_ - c_) / s_) ** 2)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        key = prefix + "rule_mf_selector_logits"
        if key in state_dict:
            selector_logits = state_dict[key]
            rule_mf_indices = _selector_logits_to_indices(selector_logits)
            self.rule_mf_indices.copy_(rule_mf_indices.to(self.rule_mf_indices.device))
            compat_logits = _indices_to_compat_logits(
                self.rule_mf_indices,
                selector_logits.shape[-1],
            )
            state_dict[key] = compat_logits.to(
                device=selector_logits.device,
                dtype=selector_logits.dtype,
            )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor, return_activations: bool = False):
        """
        x: (B, D)
        return:
            y: (B, O)
        """
        B, D = x.shape
        assert D == self.n_inputs, "입력 차원이 모델 설정과 다릅니다."

        sigmas = torch.exp(self.log_sigmas) + self.eps  # (D, M)
        mu = self._gaussian_mf(x, self.centers, sigmas)  # (B, D, M)

        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1)
        feature_idx = torch.arange(D, device=x.device).view(1, 1, D)
        selected_idx = self.rule_mf_indices.unsqueeze(0).expand(B, -1, -1)  # (B, R, D)
        weighted = mu[batch_idx, feature_idx, selected_idx]  # (B, R, D)

        if self.firing_mode == "prod":
            firing = weighted.prod(dim=2)
        elif self.firing_mode == "htsk":
            firing = torch.exp(torch.log(weighted + self.eps).sum(dim=2) / max(float(D), 1.0))
        else:
            raise ValueError(f"Unsupported firing_mode: {self.firing_mode}")
        norm = firing / (firing.sum(dim=1, keepdim=True) + self.eps)

        ones = torch.ones(B, 1, device=x.device, dtype=x.dtype)
        x_aug = torch.cat([x, ones], dim=1)  # (B, D+1)

        rule_out = torch.einsum("bd,rod->bro", x_aug, self.consequents)  # (B, R, O)
        y = torch.einsum("br,bro->bo", norm, rule_out)  # (B, O)

        if return_activations:
            activations = {
                "rule_weights": norm,   # (B, R)
                "rule_firing": firing,  # (B, R)
            }
            return y, activations

        return y


class ParallelHierarchicalTSKANFIS(nn.Module):
    """
    MI로 선택된 입력을 2개 그룹으로 나눈 뒤 병렬 TSK-ANFIS를 통과시키는 계층형 구조.
    - fusion='avg'    : 두 브랜치 출력을 학습 가능한 가중 평균으로 결합(1-layer parallel)
    - fusion='stacked': 두 브랜치 출력을 상위 TSK-ANFIS로 결합(2-layer hierarchical)
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        group_a_idx,
        group_b_idx,
        branch_rules: int = 12,
        top_rules: int = 6,
        fusion: str = "avg",
        mfs_per_input: int = 3,
        eps: float = 1e-6,
        rule_init_mode: str = "legacy",
        rule_seed: int = 0,
        firing_mode: str = "htsk",
    ):
        super().__init__()
        if int(n_inputs) <= 1:
            raise ValueError("ParallelHierarchicalTSKANFIS requires at least 2 input features.")

        self.n_inputs = int(n_inputs)
        self.n_outputs = int(n_outputs)
        self.branch_rules = int(branch_rules)
        self.top_rules = int(top_rules)
        self.fusion = str(fusion).strip().lower()
        if self.fusion not in {"avg", "stacked"}:
            raise ValueError("fusion must be one of {'avg', 'stacked'}")

        self.group_a_idx, self.group_b_idx = self._normalize_groups(
            self.n_inputs, group_a_idx, group_b_idx
        )
        self.register_buffer(
            "group_a_index_tensor",
            torch.tensor(self.group_a_idx, dtype=torch.long),
        )
        self.register_buffer(
            "group_b_index_tensor",
            torch.tensor(self.group_b_idx, dtype=torch.long),
        )

        self.branch_a = TSKANFIS(
            n_inputs=len(self.group_a_idx),
            n_rules=self.branch_rules,
            n_outputs=self.n_outputs,
            mfs_per_input=mfs_per_input,
            eps=eps,
            rule_init_mode=rule_init_mode,
            rule_seed=int(rule_seed) + 101,
            firing_mode=firing_mode,
        )
        self.branch_b = TSKANFIS(
            n_inputs=len(self.group_b_idx),
            n_rules=self.branch_rules,
            n_outputs=self.n_outputs,
            mfs_per_input=mfs_per_input,
            eps=eps,
            rule_init_mode=rule_init_mode,
            rule_seed=int(rule_seed) + 503,
            firing_mode=firing_mode,
        )

        self.top_anfis = None
        self.fusion_logits = None
        if self.fusion == "avg":
            self.fusion_logits = nn.Parameter(torch.zeros(self.n_outputs))
        else:
            self.top_anfis = TSKANFIS(
                n_inputs=self.n_outputs * 2,
                n_rules=self.top_rules,
                n_outputs=self.n_outputs,
                mfs_per_input=mfs_per_input,
                eps=eps,
                rule_init_mode=rule_init_mode,
                rule_seed=int(rule_seed) + 907,
                firing_mode=firing_mode,
            )

        # interpretability/complexity 루틴에서 모델 타입 식별용
        self.parallel_hierarchical_anfis = True

    @staticmethod
    def _normalize_groups(n_inputs: int, group_a_idx, group_b_idx):
        def _clean(indices):
            out = []
            seen = set()
            for item in list(indices or []):
                idx = int(item)
                if 0 <= idx < n_inputs and idx not in seen:
                    seen.add(idx)
                    out.append(idx)
            return out

        group_a = _clean(group_a_idx)
        used = set(group_a)
        group_b = []
        for item in list(group_b_idx or []):
            idx = int(item)
            if 0 <= idx < n_inputs and idx not in used:
                used.add(idx)
                group_b.append(idx)

        # 빠진 특성은 더 작은 그룹에 채워넣어 모든 입력을 사용.
        for idx in range(n_inputs):
            if idx in used:
                continue
            if len(group_a) <= len(group_b):
                group_a.append(idx)
            else:
                group_b.append(idx)
            used.add(idx)

        if len(group_a) == 0 and len(group_b) > 0:
            group_a.append(group_b.pop())
        if len(group_b) == 0 and len(group_a) > 0:
            group_b.append(group_a.pop())

        group_a = sorted(group_a)
        group_b = sorted(group_b)
        if not group_a or not group_b:
            raise ValueError("Feature groups must both be non-empty.")
        return group_a, group_b

    def split_inputs(self, x: torch.Tensor):
        if x.dim() != 2 or x.size(1) != self.n_inputs:
            raise ValueError(
                f"Expected input shape (B, {self.n_inputs}), got {tuple(x.shape)}"
            )
        xa = torch.index_select(x, dim=1, index=self.group_a_index_tensor)
        xb = torch.index_select(x, dim=1, index=self.group_b_index_tensor)
        return xa, xb

    def forward_branches(self, x: torch.Tensor):
        xa, xb = self.split_inputs(x)
        ya = self.branch_a(xa)
        yb = self.branch_b(xb)
        return ya, yb

    def forward(self, x: torch.Tensor, return_activations: bool = False):
        y_a, y_b = self.forward_branches(x)

        if self.fusion == "avg":
            w = torch.sigmoid(self.fusion_logits).view(1, -1)
            y = w * y_a + (1.0 - w) * y_b
            top_input = torch.cat([y_a, y_b], dim=1)
        else:
            top_input = torch.cat([y_a, y_b], dim=1)
            y = self.top_anfis(top_input)

        if return_activations:
            activations = {
                "branch_a": y_a,
                "branch_b": y_b,
                "top_input": top_input,
            }
            if self.fusion_logits is not None:
                activations["fusion_weights"] = torch.sigmoid(self.fusion_logits)
            return y, activations

        return y


# ─────────────────────────────────────────────────────────────
# AHANFIS (Attentive Hybrid ANFIS) 계열
# - 기존 code_AHANFIS/model_ahanfis_gaus_gattn_mattn_drinit_200309_0907.py 기반
# - GH-ANFIS 실험에서 import 가능하도록 model.py에 포함
# ─────────────────────────────────────────────────────────────

class mf:
    def gaussmf(x, mean, sigma):
        """
        Gaussian fuzzy membership function.
        """
        if "torch" in str(type(x)):
            return torch.exp(-((x - mean) ** 2.0) / (2 * sigma ** 2.0))
        return np.exp(-((x - mean) ** 2.0) / (2 * sigma ** 2.0))

    def sigmmf(x, c, a):
        """
        Sigmoid fuzzy membership function.
        """
        if "torch" in str(type(x)):
            return 1 / (1 + torch.exp(-a * (x - c)))
        return 1 / (1 + np.exp(-a * (x - c)))


class ModuleAttention(nn.Module):
    def __init__(self, input_size, module_num, reduction_ratio=2):
        super(ModuleAttention, self).__init__()
        self.input_size = input_size
        self.module_num = module_num
        self.fc = nn.Sequential(
            nn.Linear(input_size, input_size * reduction_ratio, bias=True),
            nn.Sigmoid(),
            nn.Linear(input_size * reduction_ratio, module_num, bias=True),
            nn.Sigmoid(),
        )
        self.fc._modules["0"].bias.data.uniform_(-1, 1)
        self.fc._modules["2"].bias.data.uniform_(-1, 1)
        v_np = np.random.rand(module_num)
        stdv = 1.0 / math.sqrt(v_np.shape[0])
        v_np = np.random.uniform(-stdv, stdv, size=module_num)
        self.v = nn.Parameter(torch.from_numpy(v_np))
        self.gumbel_seed = nn.Parameter(torch.Tensor(1), requires_grad=False)

    def sample_gumbel(self, shape, device, eps=1e-20):
        if self.training:
            self.gumbel_seed.data = torch.LongTensor([np.random.randint(10000)]).to(device)
        np.random.seed(int(self.gumbel_seed.item()))
        rd = np.random.rand(*shape)
        U = torch.from_numpy(rd).float().to(device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size(), logits.device)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        return y_hard

    def forward(self, x, gb_temp=0):
        xc = self.fc(x)
        if gb_temp > 0:
            xc = self.gumbel_softmax(xc, gb_temp)
        else:
            softmax_dim = 2 if xc.dim() == 3 else 1
            xc = F.softmax(xc, dim=softmax_dim)
        return xc


class RuleAttention(nn.Module):
    def __init__(self, rule_num, params, reduction_ratio=2):
        super(RuleAttention, self).__init__()
        self.seed = params["SEED"]
        self.device = params["device"]
        self.rule_num = rule_num
        self.fc = nn.Sequential(
            nn.Linear(rule_num, rule_num * reduction_ratio, bias=True),
            nn.Sigmoid(),
            nn.Linear(rule_num * reduction_ratio, rule_num, bias=True),
            nn.Sigmoid(),
        )
        stdv = 1.0 / math.sqrt(self.fc._modules["0"].weight.size(0))
        self.fc._modules["0"].bias.data.uniform_(-stdv, stdv)
        stdv = 1.0 / math.sqrt(self.fc._modules["2"].weight.size(0))
        self.fc._modules["2"].bias.data.uniform_(-stdv, stdv)

        self.v = nn.Parameter(torch.from_numpy(np.random.rand(rule_num)).float())
        self.gumbel_seed = nn.Parameter(torch.Tensor(1), requires_grad=False)
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def sample_gumbel(self, shape, eps=1e-20):
        if self.training:
            self.gumbel_seed.data = torch.LongTensor([np.random.randint(10000)]).to(
                self.device
            )
        np.random.seed(int(self.gumbel_seed.item()))
        rd = np.random.rand(shape[0], shape[1])
        U = torch.from_numpy(rd).float()
        return -Variable(torch.log(-torch.log(U + eps) + eps).to(self.device))

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        return y_hard

    def forward(self, x, gb_temp=0):
        xc = self.fc(x)
        if gb_temp > 0:
            xc = self.gumbel_softmax(xc, gb_temp)
        else:
            xc = F.softmax(xc, dim=1)
        v = self.v.repeat(x.size()[0], 1)
        xc = v * xc
        return x * xc


class TorchAnfis(nn.Module):
    def __init__(
        self,
        params,
        X=None,
        rules_num=200,
        input_dim=2,
        output_dim=1,
        mf_num=3,
        mf_func="sigmmf",
        attn=False,
        input_range=(-10, 10),
    ):
        super(TorchAnfis, self).__init__()
        self.device = params["device"]
        esp = 1e-16
        self.input_dim = input_dim
        self.ante_num = self.input_dim
        self.eps = esp
        self.rules_num = rules_num
        self.mf_func = mf_func
        self.mf_num = mf_num
        self.seed = params["SEED"]
        self.temp = params["G_TEMP"]
        self.attn_en = attn
        self.rule_select = params["RULE_SELECT"]
        mf_list = []
        min_val, max_val = input_range

        univ_of_dis = max_val - min_val
        mf_step = univ_of_dis / (self.mf_num - 1)
        center_min = torch.FloatTensor(self.mf_num).zero_()
        center_max = torch.FloatTensor(self.mf_num).zero_()

        for i in range(self.mf_num):
            if i == 0:
                center_min[i] = min_val
                center_max[i] = min_val + mf_step * i + mf_step / 2
            elif i == self.mf_num - 1:
                center_min[i] = min_val + mf_step * i - mf_step / 2
                center_max[i] = max_val
            else:
                center_min[i] = min_val + mf_step * i - mf_step / 2
                center_max[i] = min_val + mf_step * i + mf_step / 2

        self.register_buffer("center_min", center_min)
        self.register_buffer("center_max", center_max)

        for i in range(self.input_dim):
            mfi = []

            if self.mf_func == "sigmmf":
                sig = 0.5 + np.random.rand(1)[0] / 10
            elif self.mf_func == "gaussmf":
                sig = univ_of_dis / (2 * mf_num) + np.random.rand(1)[0] / 20 * univ_of_dis

            for j in range(mf_num):
                mf_dict = {}
                if j == 0:
                    mf_dict["mean"] = min_val + (np.random.rand(1)[0] / 5) * univ_of_dis
                elif j == mf_num - 1:
                    mf_dict["mean"] = max_val - (np.random.rand(1)[0] / 5) * univ_of_dis
                else:
                    mf_dict["mean"] = min_val + mf_step * j + (np.random.randn(1)[0] / 20) * univ_of_dis
                mf_dict["sigma"] = sig
                mfi.append(mf_dict)
            mf_list.append(mfi)
        self.MFlist = mf_list

        self.output_dim = output_dim
        rule_set = self.rule_init(params, X)

        self.rules_set = torch.nn.Parameter(
            torch.FloatTensor(np.eye(self.mf_num)[np.array(rule_set)]),
            requires_grad=False,
        )
        self.rules_num = len(self.rules_set)

        sigma = np.array(
            [[self.MFlist[ant][mf]["sigma"] for mf in range(self.mf_num)] for ant in range(self.ante_num)]
        )
        center = np.array(
            [[self.MFlist[ant][mf]["mean"] for mf in range(self.mf_num)] for ant in range(self.ante_num)]
        )

        self.sigma = torch.nn.Parameter(torch.FloatTensor(sigma), requires_grad=True).to(self.device)
        self.center = torch.nn.Parameter(torch.FloatTensor(center), requires_grad=True).to(self.device)
        self.par_f = torch.nn.Parameter(
            torch.from_numpy(np.random.uniform(size=(self.input_dim, self.rules_num * self.output_dim)))
            .float()
            .to(self.device),
            requires_grad=True,
        )
        self.par_r = torch.nn.Parameter(
            torch.from_numpy(np.random.uniform(size=(1, self.rules_num * self.output_dim)))
            .float()
            .to(self.device),
            requires_grad=True,
        )

        self.dropout = torch.nn.Dropout(params["DR"])
        if self.attn_en:
            self.gattn = RuleAttention(self.rules_num, params)

    def rule_init(self, params, data=None):
        rule_set = []
        rule_set_str = []
        if data is None:
            for _ in range(self.rules_num):
                rule = np.random.randint(self.mf_num, size=self.input_dim)
                str_rule = np.array_str(rule)
                if str_rule not in rule_set_str:
                    rule_set_str.append(str_rule)
                    rule_set.append(rule)
        else:
            for i in range(len(data)):
                evalMF = self.evaluateMF(data[i, :])
                rule = np.argmax(evalMF, axis=1)
                str_rule = np.array_str(rule)
                if str_rule not in rule_set_str:
                    rule_set_str.append(str_rule)
                    rule_set.append(rule)
        return rule_set

    def evaluateMF(self, rowInput):
        MFlist = self.MFlist
        if len(rowInput) != len(MFlist):
            print("Number of variables does not match number of rule sets")
        result = []
        for i in range(len(rowInput)):
            rs = []
            for k in range(len(MFlist[i])):
                mf_params = MFlist[i][k]
                rs.append(getattr(mf, self.mf_func)(rowInput[i], **mf_params))
            result.append(rs)
        return result

    def forward(self, X, r_mask=None):
        if self.training:
            for i in range(self.mf_num):
                self.center.data[:, i].clamp_(self.center_min[i], self.center_max[i])
            for i in range(self.mf_num):
                self.sigma.data[:, i].clamp_(0.01, 1.0)

        n_sample = X.size()[0]

        x2 = copy.copy(X.unsqueeze(2).repeat(1, 1, self.mf_num))
        layerOne = getattr(mf, self.mf_func)(x2, self.center, self.sigma)
        mf2rule = layerOne.unsqueeze(1).repeat(1, self.rules_num, 1, 1)
        rules_set2 = self.rules_set.unsqueeze(0).repeat(n_sample, 1, 1, 1)
        x2 = torch.min(torch.sum(mf2rule * rules_set2, dim=3), dim=2)[0]
        wSum = torch.norm(x2, p=1, dim=1, keepdim=True)
        wSum = wSum + self.eps
        x2 = x2 / wSum
        if self.rule_select:
            x2 = x2 * r_mask
        if self.attn_en:
            x2 = self.gattn(x2, self.temp)

        f = torch.mm(X, self.par_f) + self.par_r.repeat(n_sample, 1)
        z = x2.repeat(1, self.output_dim) * f

        z = torch.reshape(z, (z.size()[0], self.output_dim, self.rules_num))
        output = torch.sum(z, dim=2)

        return output


class AnfisConv(nn.Module):
    def __init__(
        self,
        params,
        x=None,
        inp_channel=1,
        out_channel=1,
        kernel_size=5,
        stride=1,
        input_range=(-10, 10),
        rules_num=300,
        mf_func="sigmmf",
        attn=False,
        mf_num=2,
        output_dim=1,
        aggregate_positions=False,
    ):
        super(AnfisConv, self).__init__()
        self.device = params["device"]
        self.input_dim = inp_channel * kernel_size
        self.filter_size = kernel_size
        self.inp_channel = inp_channel
        self.out_dim = output_dim
        self.stride = stride
        self.aggregate_positions = bool(aggregate_positions)

        if x is not None:
            b_size = x.shape[0]
            conv_idx = []
            R = self.filter_size
            for r in np.uint16(np.arange(R / 2, x.shape[1] - R / 2 + 1, self.stride)):
                conv_idx.append([r - np.uint16(np.floor(R / 2)), r + np.uint16(np.ceil(R / 2))])
            conv_img2 = np.zeros([b_size, len(conv_idx), R])

            for i in range(len(conv_idx)):
                idx = conv_idx[i]
                conv_img2[:, i, :] = x[:, idx[0] : idx[1]]

            # Build one ANFIS module per valid window position.
            n_positions = len(conv_idx)
            self.tAnfis = nn.ModuleList(
                [
                    TorchAnfis(
                        params,
                        X=conv_img2[:, i, :],
                        rules_num=rules_num,
                        input_dim=self.input_dim,
                        output_dim=self.out_dim,
                        mf_num=mf_num,
                        mf_func=mf_func,
                        input_range=input_range,
                        attn=attn,
                    )
                    for i in range(n_positions)
                ]
            )
        else:
            self.tAnfis = nn.ModuleList(
                [
                    TorchAnfis(
                        params,
                        rules_num=rules_num,
                        input_dim=self.input_dim,
                        output_dim=self.out_dim,
                        mf_num=mf_num,
                        mf_func=mf_func,
                        input_range=input_range,
                        attn=attn,
                    )
                    for _ in range(out_channel)
                ]
            )

    def forward(self, x, rule_mask=None):
        b_size = x.size()[0]
        conv_idx = []
        R = self.filter_size
        for r in np.uint16(np.arange(R / 2, x.size()[2] - R / 2 + 1, self.stride)):
            conv_idx.append([r - np.uint16(np.floor(R / 2)), r + np.uint16(np.ceil(R / 2))])
        conv_img2 = torch.zeros([b_size, len(conv_idx), self.inp_channel, R]).to(self.device)

        for i in range(len(conv_idx)):
            idx = conv_idx[i]
            conv_img2[:, i, :, :] = x[:, :, idx[0] : idx[1]]

        conv_img2 = conv_img2.view(
            conv_img2.shape[0], conv_img2.shape[1], conv_img2.shape[2] * conv_img2.shape[3]
        ).contiguous()
        n_positions = conv_img2.shape[1]
        n_modules = len(self.tAnfis)
        if n_positions != n_modules:
            raise RuntimeError(
                f"AnfisConv window/module mismatch: windows={n_positions}, modules={n_modules}. "
                "This would drop positions; initialize modules with matching window count."
            )
        x_out = torch.zeros(
            conv_img2.shape[0], self.out_dim, n_modules, dtype=torch.float
        ).to(self.device)

        for i in range(n_modules):
            if rule_mask is not None:
                x_out[:, :, i] = self.tAnfis[i](conv_img2[:, i, :], rule_mask[i])
            else:
                x_out[:, :, i] = self.tAnfis[i](conv_img2[:, i, :])

        if self.aggregate_positions:
            x_out = torch.mean(x_out, dim=2)
        return x_out


class ACNN(torch.nn.Module):
    def __init__(self, params, X=None):
        super(ACNN, self).__init__()
        import importlib

        utils = importlib.import_module(params["utils"])
        self.inp_size = params["IN_SIZE"]
        output_dim = params["N_CLASS"]
        self.dr = params["DR"]
        self.module_attn_en = params["MODULE_ATTN_EN"]
        self.bnorm_en = False
        self.device = params["device"]
        utils.set_seed(params["SEED"])

        pars = params["ACONV"]["CONV1"]
        self.in_channel = pars["INP"]
        self.add_module(
            "tAnfisConv1",
            AnfisConv(
                params,
                x=X,
                inp_channel=pars["INP"],
                out_channel=pars["OUT"],
                kernel_size=pars["SIZE"],
                stride=pars["STRIDE"],
                input_range=pars["RANGE"],
                rules_num=pars["RULE_NUM"],
                mf_func=pars["MF_FUNC"],
                attn=pars["ATTN"],
                mf_num=pars["MF_NUM"],
            ),
        )
        if pars["BNORM"]:
            self.bnorm_en = True
            self.b_norm1 = torch.nn.BatchNorm1d(params["ACONV"]["FC"]["OUT"], momentum=0.5)
        self.actfn = pars["ACT_FN"]
        self.act = getattr(nn, self.actfn)()

        if self.module_attn_en:
            self.module_attn = ModuleAttention(
                input_size=self.inp_size, module_num=len(self.tAnfisConv1.tAnfis)
            )

        pars = params["ACONV"]["FC"]
        conv_len = len(self.tAnfisConv1.tAnfis)
        fc_kernel = int(pars["SIZE"])
        fc_stride = int(pars.get("STRIDE", 1))
        if conv_len >= fc_kernel:
            fc_out_channels = 1 + (conv_len - fc_kernel) // max(1, fc_stride)
        else:
            fc_out_channels = 1
        self.add_module(
            "tAnfisFC",
            AnfisConv(
                params,
                x=None,
                inp_channel=pars["INP"],
                out_channel=fc_out_channels,
                kernel_size=pars["SIZE"],
                stride=fc_stride,
                input_range=pars["RANGE"],
                rules_num=pars["RULE_NUM"],
                mf_func=pars["MF_FUNC"],
                mf_num=pars["MF_NUM"],
                output_dim=pars["OUT_DIM"],
                attn=pars["ATTN"],
                aggregate_positions=True,
            ),
        )

        self.rule_masks = []
        self.rule_masks.append(
            [
                torch.ones(self.tAnfisConv1.tAnfis[i].rules_num, device=self.device)
                for i in range(len(self.tAnfisConv1.tAnfis))
            ]
        )
        self.rule_masks.append(
            [
                torch.ones(self.tAnfisFC.tAnfis[i].rules_num, device=self.device)
                for i in range(len(self.tAnfisFC.tAnfis))
            ]
        )

    def forward(self, x):
        if (self.in_channel == 1) or (len(x.size()) < 2):
            x = torch.unsqueeze(x, 1)
        x_conv = self.tAnfisConv1(x, self.rule_masks[0])
        if self.bnorm_en:
            x_conv = self.b_norm1(x_conv)
        x_conv = self.act(x_conv)
        if self.module_attn_en:
            v_in = self.module_attn(x)
            x_tan = x_conv * v_in
        else:
            x_tan = x_conv

        x_h1 = x_tan
        x_out = self.tAnfisFC(x_h1, self.rule_masks[1])
        return x_out


class AnfisConvOld(nn.Module):
    """
    Legacy AH-ANFIS conv block kept for reproducible old experiments.
    Matches pre-update behavior where module count follows `out_channel`.
    """

    def __init__(
        self,
        params,
        x=None,
        inp_channel=1,
        out_channel=1,
        kernel_size=5,
        stride=1,
        input_range=(-10, 10),
        rules_num=300,
        mf_func="sigmmf",
        attn=False,
        mf_num=2,
        output_dim=1,
    ):
        super(AnfisConvOld, self).__init__()
        self.device = params["device"]
        self.input_dim = inp_channel * kernel_size
        self.filter_size = kernel_size
        self.inp_channel = inp_channel
        self.out_dim = output_dim
        self.stride = stride

        if x is not None:
            b_size = x.shape[0]
            conv_idx = []
            R = self.filter_size
            for r in np.uint16(np.arange(R / 2, x.shape[1] - R / 2 + 1, self.stride)):
                conv_idx.append([r - np.uint16(np.floor(R / 2)), r + np.uint16(np.ceil(R / 2))])
            conv_img2 = np.zeros([b_size, len(conv_idx), R])

            for i in range(len(conv_idx)):
                idx = conv_idx[i]
                conv_img2[:, i, :] = x[:, idx[0] : idx[1]]

            self.tAnfis = nn.ModuleList(
                [
                    TorchAnfis(
                        params,
                        X=conv_img2[:, i, :],
                        rules_num=rules_num,
                        input_dim=self.input_dim,
                        output_dim=self.out_dim,
                        mf_num=mf_num,
                        mf_func=mf_func,
                        input_range=input_range,
                        attn=attn,
                    )
                    for i in range(out_channel)
                ]
            )
        else:
            self.tAnfis = nn.ModuleList(
                [
                    TorchAnfis(
                        params,
                        rules_num=rules_num,
                        input_dim=self.input_dim,
                        output_dim=self.out_dim,
                        mf_num=mf_num,
                        mf_func=mf_func,
                        input_range=input_range,
                        attn=attn,
                    )
                    for _ in range(out_channel)
                ]
            )

    def forward(self, x, rule_mask=None):
        b_size = x.size()[0]
        conv_idx = []
        R = self.filter_size
        for r in np.uint16(np.arange(R / 2, x.size()[2] - R / 2 + 1, self.stride)):
            conv_idx.append([r - np.uint16(np.floor(R / 2)), r + np.uint16(np.ceil(R / 2))])
        conv_img2 = torch.zeros([b_size, len(conv_idx), self.inp_channel, R]).to(self.device)

        for i in range(len(conv_idx)):
            idx = conv_idx[i]
            conv_img2[:, i, :, :] = x[:, :, idx[0] : idx[1]]

        conv_img2 = conv_img2.view(
            conv_img2.shape[0], conv_img2.shape[1], conv_img2.shape[2] * conv_img2.shape[3]
        ).contiguous()
        x_out = torch.zeros(
            conv_img2.shape[0], self.out_dim, len(self.tAnfis), dtype=torch.float
        ).to(self.device)

        for i in range(len(self.tAnfis)):
            if rule_mask is not None:
                x_out[:, :, i] = self.tAnfis[i](conv_img2[:, i, :], rule_mask[i])
            else:
                x_out[:, :, i] = self.tAnfis[i](conv_img2[:, i, :])

        if self.out_dim > 1:
            x_out = x_out.view(b_size, self.out_dim).contiguous()
        return x_out


class ACNNOld(torch.nn.Module):
    """
    Legacy AH-ANFIS model (old) for side-by-side experiments.
    """

    def __init__(self, params, X=None):
        super(ACNNOld, self).__init__()
        import importlib

        utils = importlib.import_module(params["utils"])
        self.inp_size = params["IN_SIZE"]
        output_dim = params["N_CLASS"]
        self.dr = params["DR"]
        self.module_attn_en = params["MODULE_ATTN_EN"]
        self.bnorm_en = False
        self.device = params["device"]
        utils.set_seed(params["SEED"])

        pars = params["ACONV"]["CONV1"]
        self.in_channel = pars["INP"]
        self.add_module(
            "tAnfisConv1",
            AnfisConvOld(
                params,
                x=X,
                inp_channel=pars["INP"],
                out_channel=pars["OUT"],
                kernel_size=pars["SIZE"],
                stride=pars["STRIDE"],
                input_range=pars["RANGE"],
                rules_num=pars["RULE_NUM"],
                mf_func=pars["MF_FUNC"],
                attn=pars["ATTN"],
                mf_num=pars["MF_NUM"],
            ),
        )
        if pars["BNORM"]:
            self.bnorm_en = True
            self.b_norm1 = torch.nn.BatchNorm1d(params["ACONV"]["FC"]["OUT"], momentum=0.5)
        self.actfn = pars["ACT_FN"]

        pars = params["ACONV"]["FC"]
        self.add_module(
            "tAnfisFC",
            AnfisConvOld(
                params,
                inp_channel=pars["INP"],
                out_channel=pars["OUT"],
                kernel_size=pars["SIZE"],
                input_range=pars["RANGE"],
                rules_num=pars["RULE_NUM"],
                mf_func=pars["MF_FUNC"],
                mf_num=pars["MF_NUM"],
                output_dim=pars["OUT_DIM"],
                attn=pars["ATTN"],
            ),
        )

        self.act = getattr(nn, self.actfn)()

        if self.module_attn_en:
            self.module_attn = ModuleAttention(
                input_size=self.inp_size, module_num=params["ACONV"]["CONV1"]["OUT"]
            )

        self.rule_masks = []
        self.rule_masks.append(
            [
                torch.ones(self.tAnfisConv1.tAnfis[i].rules_num, device=self.device)
                for i in range(len(self.tAnfisConv1.tAnfis))
            ]
        )
        self.rule_masks.append(
            [
                torch.ones(self.tAnfisFC.tAnfis[i].rules_num, device=self.device)
                for i in range(len(self.tAnfisFC.tAnfis))
            ]
        )

    def forward(self, x):
        if (self.in_channel == 1) or (len(x.size()) < 2):
            x = torch.unsqueeze(x, 1)
        x_conv = self.tAnfisConv1(x, self.rule_masks[0])
        if self.bnorm_en:
            x_conv = self.b_norm1(x_conv)
        x_conv = self.act(x_conv)
        if self.module_attn_en:
            v_in = self.module_attn(x)
            x_tan = x_conv * v_in
        else:
            x_tan = x_conv

        x_h1 = x_tan
        x_out = self.tAnfisFC(x_h1, self.rule_masks[1])
        return x_out
