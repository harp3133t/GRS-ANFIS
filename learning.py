import copy
import numpy as np

from grs_config import normalize_grs_params
from model import ACNN, ACNNOld, GRS_ANFIS, ParallelHierarchicalTSKANFIS, TSKANFIS
from torch import nn, optim
import torch
import matplotlib.pyplot as plt

def make_optimizer(model, lr=1e-3, weight_decay=1e-5):
    return optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )


def _effective_complementary_l1(model, activations, device):
    gate_complementary = activations.get("gate_complementary", None)
    if gate_complementary is not None:
        return torch.mean(torch.abs(gate_complementary))
    if hasattr(model, "get_complementary_mask_probs") and not getattr(model, "complementary_mask_frozen", False):
        return torch.mean(model.get_complementary_mask_probs())
    return torch.tensor(0.0, device=device)


def train_one_epoch_dual_cls(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    phase="primary",
    lambda_primary=0.0,
    lambda_complementary=0.0,
    lambda_bimodal=0.0,
    feature_names=None,
    binary_columns=None,
    task="auto",
    eps=1e-12,
):
    model.train()

    if hasattr(model, "set_phase"):
        model.set_phase(phase)

    running_loss = 0.0
    running_pred = 0.0
    running_l1_primary = 0.0
    running_l1_complementary = 0.0
    running_overlap = 0.0

    target_sigma = torch.log(torch.exp(torch.tensor(0.01)) - 1 + 1e-6).to(device)
    zero_val = torch.tensor(-1.0, device=device)
    one_val = torch.tensor(1.0, device=device)

    binary_indices = None
    if feature_names is not None and binary_columns is not None:
        binary_indices = [feature_names.index(col) for col in binary_columns if col in feature_names]

    is_bce_logits = isinstance(criterion, nn.BCEWithLogitsLoss)
    is_ce_logits = isinstance(criterion, nn.CrossEntropyLoss)
    is_nll = isinstance(criterion, nn.NLLLoss)

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits, activations = model(inputs, return_activations=True)

        if task not in {"auto", "binary", "multiclass"}:
            raise ValueError(f"task must be one of ['auto','binary','multiclass'], got {task}")

        if task == "auto":
            if logits.dim() == 2 and logits.size(1) == 1:
                task_kind = "binary"
            elif logits.dim() == 2 and logits.size(1) > 1:
                task_kind = "multiclass"
            else:
                raise ValueError(f"Cannot infer task from logits shape: {tuple(logits.shape)}")
        else:
            task_kind = task

        if task_kind == "binary":
            if targets.dim() == 1:
                y = targets.float().unsqueeze(1)
            else:
                y = targets.float()

            if logits.dim() == 1:
                z = logits.unsqueeze(1)
            else:
                z = logits

            if is_bce_logits:
                pred_loss = criterion(z, y)
            else:
                probs = torch.sigmoid(z)
                pred_loss = criterion(probs, y)

        else:
            if logits.dim() != 2 or logits.size(1) < 2:
                raise ValueError(f"Multiclass expects logits shape (B,C), got {tuple(logits.shape)}")

            C = logits.size(1)

            if targets.dim() == 2 and targets.size(1) == C:
                y_onehot = targets.float()
                y_index = torch.argmax(targets, dim=1).long()
            elif targets.dim() == 2 and targets.size(1) == 1:
                y_index = targets.squeeze(1).long()
                y_onehot = None
            else:
                y_index = targets.long()
                y_onehot = None

            if is_ce_logits:
                pred_loss = criterion(logits, y_index)
            else:
                probs = torch.softmax(logits, dim=1)

                if is_nll:
                    pred_loss = criterion(torch.log(probs + eps), y_index)
                elif y_onehot is not None:
                    pred_loss = criterion(probs, y_onehot)
                else:
                    pred_loss = criterion(probs, y_index)

        gate_primary = activations.get("gate_primary", None)
        gate_complementary = activations.get("gate_complementary", None)

        if hasattr(model, "get_primary_mask_probs") and not getattr(model, "primary_mask_frozen", False):
            primary_probs = model.get_primary_mask_probs()
            l1_primary = torch.mean(primary_probs)
            bimodal = torch.mean(primary_probs * (1.0 - primary_probs))
        else:
            l1_primary = (
                torch.mean(torch.abs(gate_primary))
                if gate_primary is not None
                else torch.tensor(0.0, device=device)
            )
            bimodal = (
                torch.mean(gate_primary * (1.0 - gate_primary))
                if gate_primary is not None
                else torch.tensor(0.0, device=device)
            )

        l1_complementary = _effective_complementary_l1(model, activations, device)

        if (gate_primary is not None) and (gate_complementary is not None):
            overlap_loss = torch.mean(gate_primary * gate_complementary)
        else:
            overlap_loss = torch.tensor(0.0, device=device)

        total_loss = (
            pred_loss
            + lambda_primary * l1_primary
            + lambda_complementary * l1_complementary
            + lambda_bimodal * bimodal
        )

        total_loss.backward()
        optimizer.step()

        if binary_indices:
            with torch.no_grad():
                if hasattr(model, "p_mf_centers"):
                    model.p_mf_centers.data[binary_indices, 0] = zero_val
                    if model.p_mf_centers.shape[1] >= 2:
                        model.p_mf_centers.data[binary_indices, 1] = one_val
                    if hasattr(model, "p_mf_log_sigmas"):
                        model.p_mf_log_sigmas.data[binary_indices, :] = target_sigma

                if hasattr(model, "s_mf_centers") and model.s_mf_centers is not None:
                    model.s_mf_centers.data[binary_indices, 0] = zero_val
                    if model.s_mf_centers.shape[1] >= 2:
                        model.s_mf_centers.data[binary_indices, 1] = one_val
                    if hasattr(model, "s_mf_log_sigmas"):
                        model.s_mf_log_sigmas.data[binary_indices, :] = target_sigma

        batch_size = inputs.size(0)
        running_loss += total_loss.item() * batch_size
        running_pred += pred_loss.item() * batch_size
        running_l1_primary += l1_primary.item() * batch_size
        running_l1_complementary += l1_complementary.item() * batch_size
        running_overlap += overlap_loss.item() * batch_size

    n_data = len(dataloader.dataset)
    return (
        running_loss / n_data,
        running_pred / n_data,
        running_l1_primary / n_data,
        running_l1_complementary / n_data,
        running_overlap / n_data,
    )


def train_grs_anfis(
    train_loader,
    test_loader,
    n_features,
    n_outputs,
    device,
    feature_names,
    binary_columns,
    hparams,
    task_kind,
):
    # model = GRS_AANFIS(
    #     n_features=n_features,
    #     n_outputs=n_outputs,
    #     primary_rules=hparams["primary_rules"],
    #     complementary_rules=hparams["complementary_rules"],
    #     mf_per_feature=hparams["mf_per_feature"],
    #     attn_hidden=64,
    #     attn_tau=1.0,
    #     attn_topk=3,      # 해석을 위해 top-3 룰만 남김 (선택)
    #     attn_hard=False,  # 일단 soft로 시작 추천
    #     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # ).to(device)
    model = GRS_ANFIS(
        n_features=n_features,
        n_outputs=n_outputs,
        complementary_rules=hparams["complementary_rules"],
        primary_rules=hparams["primary_rules"],
        mf_per_feature=hparams["mf_per_feature"],
        device=device,
    ).to(device)

    if task_kind == "multiclass":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCELoss()

    num_epochs_stage1 = hparams.get("primary_epochs", hparams.get("epochs_stage1", 20))
    lambda_primary_s1 = hparams.get("lambda_primary_s1", 1.0)
    lambda_complementary_s1 = 0.0
    lambda_bimodal_s1 = hparams.get("lambda_bimodal_s1", 0.0)

    model.set_phase("primary")
    model.set_mode("primary_only")
    model.unfreeze_primary()
    model.unfreeze_primary_routing()
    model.freeze_complementary()
    model.freeze_complementary_routing()

    optimizer = make_optimizer(
        model,
        lr=hparams["lr_primary"],
        weight_decay=hparams["weight_decay"],
    )

    for epoch in range(1, num_epochs_stage1 + 1):
        train_total_loss, train_pred_loss, l1_b, l1_r, l1_overlap = train_one_epoch_dual_cls(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            phase="primary",
            lambda_primary=lambda_primary_s1,
            lambda_complementary=lambda_complementary_s1,
            lambda_bimodal=lambda_bimodal_s1,
            feature_names=list(feature_names),
            binary_columns=binary_columns,
        )
        val_loss, val_acc, val_f1 = evaluate_torch_classification(
            model, test_loader, criterion, device, task_kind
        )
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[GRS-ANFIS Stage1 {epoch:03d}] "
                f"Total Loss: {train_total_loss:.4f} "
                f"(BCE: {train_pred_loss:.4f} | "
                f"L1(Primary): {l1_b:.2f} | L1(Complementary): {l1_r:.2f} | "
                f"L1(Overlap): {l1_overlap:.2f}) "
                f"Val BCE: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
            )

    num_epochs_stage1_hard = hparams.get("primary_hard_epochs", 0)
    if num_epochs_stage1_hard > 0:
        hard_threshold = hparams.get("primary_mask_threshold", None)
        model.freeze_primary_routing(threshold=hard_threshold)
        model.set_phase("primary")
        model.set_mode("primary_only")
        model.unfreeze_primary()
        model.freeze_complementary()
        model.freeze_complementary_routing()

        optimizer = make_optimizer(
            model,
            lr=hparams.get("lr_primary_hard", hparams["lr_primary"]),
            weight_decay=hparams["weight_decay"],
        )

        lambda_primary_hard = hparams.get("lambda_primary_hard", 0.0)

        for epoch in range(1, num_epochs_stage1_hard + 1):
            train_total_loss, train_pred_loss, l1_b, l1_r, l1_overlap = train_one_epoch_dual_cls(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                phase="primary",
                lambda_primary=lambda_primary_hard,
                lambda_complementary=lambda_complementary_s1,
                lambda_bimodal=0.0,
                feature_names=list(feature_names),
                binary_columns=binary_columns,
            )
            val_loss, val_acc, val_f1 = evaluate_torch_classification(
                model, test_loader, criterion, device, task_kind
            )
            if epoch % 10 == 0 or epoch == 1 or epoch == num_epochs_stage1_hard:
                print(
                    f"[GRS-ANFIS Stage1-Hard {epoch:03d}] "
                    f"Total Loss: {train_total_loss:.4f} "
                    f"(BCE: {train_pred_loss:.4f} | "
                    f"L1(Primary): {l1_b:.2f} | L1(Complementary): {l1_r:.2f} | "
                    f"L1(Overlap): {l1_overlap:.2f}) "
                    f"Val BCE: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
                )

    num_epochs_stage2 = hparams.get("complementary_epochs", hparams.get("epochs_stage2", 20))
    lambda_primary_s2 = 0.0
    lambda_complementary_s2 = hparams.get("lambda_complementary_s2", 0.1)

    model.set_phase("complementary_complement")
    model.set_mode("full")
    model.freeze_primary()
    model.freeze_primary_routing()
    model.unfreeze_complementary()
    model.unfreeze_complementary_routing()

    optimizer = make_optimizer(
        model,
        lr=hparams["lr_complementary"],
        weight_decay=hparams["weight_decay"],
    )

    for epoch in range(1, num_epochs_stage2 + 1):
        train_total_loss, train_pred_loss, l1_b, l1_r, l1_overlap = train_one_epoch_dual_cls(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            phase="complementary_complement",
            lambda_primary=lambda_primary_s2,
            lambda_complementary=lambda_complementary_s2,
            feature_names=list(feature_names),
            binary_columns=binary_columns,
        )
        val_loss, val_acc, val_f1 = evaluate_torch_classification(
            model, test_loader, criterion, device, task_kind
        )
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[GRS-ANFIS Stage2 {epoch:03d}] "
                f"Total Loss: {train_total_loss:.4f} "
                f"(BCE: {train_pred_loss:.4f} | "
                f"L1(Primary): {l1_b:.2f} | L1(Complementary): {l1_r:.2f} | "
                f"L1(Overlap): {l1_overlap:.2f}) "
                f"Val BCE: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
            )

    num_epochs_stage2_hard = hparams.get("complementary_hard_epochs", 0)
    if num_epochs_stage2_hard > 0:
        hard_threshold = hparams.get("complementary_mask_threshold", None)
        if hard_threshold is not None:
            model.complementary_mask_threshold = hard_threshold
        model.freeze_complementary_routing()
        model.set_phase("complementary_complement")
        model.set_mode("full")
        model.freeze_primary()
        model.freeze_primary_routing()
        model.unfreeze_complementary()

        optimizer = make_optimizer(
            model,
            lr=hparams.get("lr_complementary_hard", hparams["lr_complementary"]),
            weight_decay=hparams["weight_decay"],
        )

        lambda_complementary_hard = hparams.get("lambda_complementary_hard", 0.0)

        for epoch in range(1, num_epochs_stage2_hard + 1):
            train_total_loss, train_pred_loss, l1_b, l1_r, l1_overlap = train_one_epoch_dual_cls(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                phase="complementary_complement",
                lambda_primary=lambda_primary_s2,
                lambda_complementary=lambda_complementary_hard,
                feature_names=list(feature_names),
                binary_columns=binary_columns,
            )
            val_loss, val_acc, val_f1 = evaluate_torch_classification(
                model, test_loader, criterion, device, task_kind
            )
            if epoch % 10 == 0 or epoch == 1 or epoch == num_epochs_stage2_hard:
                print(
                    f"[GRS-ANFIS Stage2-Hard {epoch:03d}] "
                    f"Total Loss: {train_total_loss:.4f} "
                    f"(BCE: {train_pred_loss:.4f} | "
                    f"L1(Primary): {l1_b:.2f} | L1(Complementary): {l1_r:.2f} | "
                    f"L1(Overlap): {l1_overlap:.2f}) "
                    f"Val BCE: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
                )

    metrics = {}

    model.set_phase("primary")
    model.set_mode("primary_only")
    _, test_acc, test_f1 = evaluate_torch_classification(
        model, test_loader, criterion, device, task_kind, use_soft_eval=False
    )
    metrics["primary_hard"] = (test_acc, test_f1)

    was_frozen = model.primary_mask_frozen
    if was_frozen:
        model.primary_mask_frozen = False
    _, test_acc, test_f1 = evaluate_torch_classification(
        model, test_loader, criterion, device, task_kind, use_soft_eval=True
    )
    metrics["primary_soft"] = (test_acc, test_f1)
    if was_frozen:
        model.primary_mask_frozen = True

    model.set_phase("complementary_complement")
    model.set_mode("complementary_only")
    _, test_acc, test_f1 = evaluate_torch_classification(
        model, test_loader, criterion, device, task_kind
    )
    metrics["complementary"] = (test_acc, test_f1)

    model.set_phase("complementary_complement")
    model.set_mode("full")
    _, test_acc, test_f1 = evaluate_torch_classification(
        model, test_loader, criterion, device, task_kind
    )
    metrics["full"] = (test_acc, test_f1)

    was_primary_frozen = model.primary_mask_frozen
    was_complementary_frozen = model.complementary_mask_frozen
    if was_primary_frozen:
        model.primary_mask_frozen = False
    if was_complementary_frozen:
        model.complementary_mask_frozen = False
    _, test_acc, test_f1 = evaluate_torch_classification(
        model, test_loader, criterion, device, task_kind, use_soft_eval=True
    )
    metrics["full_soft"] = (test_acc, test_f1)
    if was_primary_frozen:
        model.primary_mask_frozen = True
    if was_complementary_frozen:
        model.complementary_mask_frozen = True

    top_k = hparams.get("primary_top_k", 10)
    if feature_names is not None:
        primary_probs = model.get_primary_mask_probs().detach().cpu()
        top_k = max(1, min(int(top_k), primary_probs.numel()))
        top_vals, top_idx = torch.topk(primary_probs, k=top_k)
        print(f"GRS-ANFIS primary mask top-{top_k} (probabilities):")
        for idx, val in zip(top_idx.tolist(), top_vals.tolist()):
            name = feature_names[idx] if idx < len(feature_names) else f"x{idx}"
            print(f"  {name}: {val:.4f}")

    gate_stats = evaluate_gate_overlap(model, test_loader, device)

    return model, metrics, gate_stats


def train_tsk_anfis(
    train_loader,
    test_loader,
    n_features,
    n_outputs,
    device,
    n_rules=10,
    epochs=30,
    lr=1e-3,
    task_kind="binary",
):
    model = TSKANFIS(n_inputs=n_features, n_rules=n_rules, n_outputs=n_outputs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if task_kind == "multiclass":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device).float()
            targets = targets.to(device)

            optimizer.zero_grad()

            logits = model(inputs)
            if task_kind == "multiclass":
                loss = criterion(logits, targets.long())
            else:
                outputs = torch.sigmoid(logits)
                loss = criterion(outputs, targets.float())

            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = running_loss / max(1, n_samples)
            label = "CE" if task_kind == "multiclass" else "BCE"
            print(f"[TSK-ANFIS {epoch:03d}] Train {label}: {avg_loss:.4f}")

    _, test_acc, test_f1 = evaluate_torch_classification(
        model, test_loader, criterion, device, task_kind
    )
    return model, test_acc, test_f1


def train_sklearn_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return compute_metrics(y_test, y_pred)



def fit_grs_anfis(
    params,
    train_loader,
    n_features,
    n_outputs,
    task_kind,
    feature_names,
    device,
    verbose=False,
    complementary_use_complement=True,
    random_role_assignment=False,
):
    params = dict(params or {})
    if "complementary_gate_mode" not in params:
        params["complementary_gate_mode"] = "complement" if complementary_use_complement else "independent"
    if "random_role_assignment" not in params:
        params["random_role_assignment"] = random_role_assignment
    params = normalize_grs_params(params)

    model = GRS_ANFIS(
        n_features=n_features,
        n_outputs=n_outputs,
        complementary_rules=params["complementary_rules"],
        primary_rules=params["primary_rules"],
        mf_per_feature=params["mf_per_feature"],
        device=device,
        complementary_gate_mode=params["complementary_gate_mode"],
        rule_init_mode=params["rule_init_mode"],
        rule_seed=params["rule_seed"],
        firing_mode=params["firing_mode"],
        use_input_norm=params["use_input_norm"],
        enable_complementary_branch=params["enable_complementary_branch"],
    ).to(device)

    if task_kind == "multiclass":
        criterion = nn.CrossEntropyLoss()
    elif task_kind == "binary":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    binary_columns = []
    random_role_assignment = bool(params.get("random_role_assignment", False))

    if random_role_assignment:
        with torch.no_grad():
            primary_rand = torch.rand_like(model.primary_mask_logits)
            primary_hard = (primary_rand > 0.5).float()
            if primary_hard.sum() == 0:
                idx = torch.randint(0, primary_hard.numel(), (1,), device=primary_hard.device)
                primary_hard[idx] = 1.0
            if primary_hard.sum() == primary_hard.numel():
                idx = torch.randint(0, primary_hard.numel(), (1,), device=primary_hard.device)
                primary_hard[idx] = 0.0

            model.primary_mask_hard.copy_(primary_hard)
            model.primary_mask_frozen = True
            model.primary_mask_logits.requires_grad_(False)

            if params["complementary_gate_mode"] == "complement":
                complementary_hard = torch.ones_like(model.complementary_mask_logits)
            else:
                complementary_hard = (1.0 - primary_hard).clone()
            model.complementary_mask_hard.copy_(complementary_hard)
            model.complementary_mask_frozen = True
            model.complementary_mask_logits.requires_grad_(False)

    model.set_phase("primary")
    model.set_mode("primary_only")
    model.unfreeze_primary()
    model.unfreeze_primary_routing()
    model.freeze_complementary()
    model.freeze_complementary_routing()

    optimizer = make_optimizer(
        model,
        lr=params["lr_primary"],
        weight_decay=params["weight_decay"],
    )

    history = {"stage1": [], "primary_hard": [], "stage2": [], "complementary_hard": []}

    for _ in range(params["epochs_stage1"]):
        if task_kind in {"binary", "multiclass"}:
            loss_tuple = train_one_epoch_dual_cls(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                phase="primary",
                lambda_primary=params["lambda_primary_s1"],
                lambda_complementary=0.0,
                feature_names=list(feature_names),
                binary_columns=binary_columns,
                task=task_kind,
            )
            history["stage1"].append(loss_tuple[0])
        else:
            loss = train_one_epoch_dual_reg(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                phase="primary",
                lambda_primary=params["lambda_primary_s1"],
                lambda_complementary=0.0,
                feature_names=list(feature_names),
                binary_columns=binary_columns,
            )
            history["stage1"].append(loss)

    primary_hard_epochs = int(params.get("primary_hard_epochs", 0) or 0)
    if primary_hard_epochs > 0:
        hard_threshold = params.get("primary_mask_threshold", None)
        model.freeze_primary_routing(threshold=hard_threshold)
        model.set_phase("primary")
        model.set_mode("primary_only")
        model.unfreeze_primary()
        model.freeze_complementary()
        model.freeze_complementary_routing()

        optimizer = make_optimizer(
            model,
            lr=params.get("lr_primary_hard", params["lr_primary"]),
            weight_decay=params["weight_decay"],
        )
        lambda_primary_hard = params.get("lambda_primary_hard", 0.0)

        for _ in range(primary_hard_epochs):
            if task_kind in {"binary", "multiclass"}:
                loss_tuple = train_one_epoch_dual_cls(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                    phase="primary",
                    lambda_primary=lambda_primary_hard,
                    lambda_complementary=0.0,
                    lambda_bimodal=0.0,
                    feature_names=list(feature_names),
                    binary_columns=binary_columns,
                    task=task_kind,
                )
                history["primary_hard"].append(loss_tuple[0])
            else:
                loss = train_one_epoch_dual_reg(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                    phase="primary",
                    lambda_primary=lambda_primary_hard,
                    lambda_complementary=0.0,
                    lambda_bimodal=0.0,
                    feature_names=list(feature_names),
                    binary_columns=binary_columns,
                )
                history["primary_hard"].append(loss)

    if params["enable_complementary_branch"] and int(params["epochs_stage2"]) > 0:
        model.set_phase("complementary_complement")
        model.set_mode("full")
        model.freeze_primary()
        model.freeze_primary_routing()
        model.unfreeze_complementary()
        model.unfreeze_complementary_routing()

        optimizer = make_optimizer(
            model,
            lr=params["lr_complementary"],
            weight_decay=params["weight_decay"],
        )

        for _ in range(params["epochs_stage2"]):
            if task_kind in {"binary", "multiclass"}:
                loss_tuple = train_one_epoch_dual_cls(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                    phase="complementary_complement",
                    lambda_primary=0.0,
                    lambda_complementary=params["lambda_complementary_s2"],
                    feature_names=list(feature_names),
                    binary_columns=binary_columns,
                    task=task_kind,
                )
                history["stage2"].append(loss_tuple[0])
            else:
                loss = train_one_epoch_dual_reg(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                    phase="complementary_complement",
                    lambda_primary=0.0,
                    lambda_complementary=params["lambda_complementary_s2"],
                    feature_names=list(feature_names),
                    binary_columns=binary_columns,
                )
                history["stage2"].append(loss)

        complementary_hard_epochs = int(params.get("complementary_hard_epochs", 0) or 0)
        if complementary_hard_epochs > 0:
            hard_threshold = params.get("complementary_mask_threshold", None)
            if hard_threshold is not None:
                model.complementary_mask_threshold = hard_threshold

            model.freeze_complementary_routing()
            model.set_phase("complementary_complement")
            model.set_mode("full")
            model.freeze_primary()
            model.freeze_primary_routing()
            model.unfreeze_complementary()

            optimizer = make_optimizer(
                model,
                lr=params.get("lr_complementary_hard", params["lr_complementary"]),
                weight_decay=params["weight_decay"],
            )
            lambda_complementary_hard = params.get("lambda_complementary_hard", 0.0)

            for _ in range(complementary_hard_epochs):
                if task_kind in {"binary", "multiclass"}:
                    loss_tuple = train_one_epoch_dual_cls(
                        model,
                        train_loader,
                        criterion,
                        optimizer,
                        device,
                        phase="complementary_complement",
                        lambda_primary=0.0,
                        lambda_complementary=lambda_complementary_hard,
                        lambda_bimodal=0.0,
                        feature_names=list(feature_names),
                        binary_columns=binary_columns,
                        task=task_kind,
                    )
                    history["complementary_hard"].append(loss_tuple[0])
                else:
                    loss = train_one_epoch_dual_reg(
                        model,
                        train_loader,
                        criterion,
                        optimizer,
                        device,
                        phase="complementary_complement",
                        lambda_primary=0.0,
                        lambda_complementary=lambda_complementary_hard,
                        lambda_bimodal=0.0,
                        feature_names=list(feature_names),
                        binary_columns=binary_columns,
                    )
                    history["complementary_hard"].append(loss)
    else:
        model.set_phase("primary")
        model.set_mode("primary_only")

    if verbose:
        plt.figure(figsize=(10, 5))
        all_losses = []
        labels = []
        for stage, losses in history.items():
            if losses:
                all_losses.extend(losses)
                labels.extend([stage] * len(losses))
        
        plt.plot(all_losses, marker='.')
        plt.title("GRS-ANFIS Training Loss (All Stages)")
        plt.xlabel("Total Epochs")
        plt.ylabel("Loss")
        
        # Draw vertical lines for stage splits
        curr = 0
        for stage, losses in history.items():
            if losses:
                curr += len(losses)
                plt.axvline(x=curr-0.5, color='gray', linestyle='--')
                plt.text(curr - len(losses)/2, max(all_losses)*0.9, stage, rotation=45, ha='center')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return model, criterion


def fit_tsk_anfis(
    params,
    train_loader,
    n_features,
    n_outputs,
    task_kind,
    device,
):
    mfs_per_input = int(params.get("mfs_per_input", 3))
    model = TSKANFIS(
        n_inputs=n_features,
        n_rules=params["n_rules"],
        n_outputs=n_outputs,
        mfs_per_input=mfs_per_input,
        rule_init_mode=str(params.get("rule_init_mode", "legacy")),
        rule_seed=int(params.get("rule_seed", 0)),
        firing_mode=str(params.get("firing_mode", "htsk")),
    ).to(device)

    if task_kind == "multiclass":
        criterion = nn.CrossEntropyLoss()
    elif task_kind == "binary":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=params["lr"])

    for _ in range(params["epochs"]):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device).float()
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)

            if task_kind == "multiclass":
                loss = criterion(logits, targets.long())
            elif task_kind == "binary":
                loss = criterion(logits, targets.float())
            else:
                if targets.dim() == 1:
                    y = targets.float().unsqueeze(1)
                else:
                    y = targets.float()
                loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

    return model, criterion


def _make_random_feature_groups(n_features, split_seed):
    n_features = int(n_features)
    if n_features < 2:
        raise ValueError("Parallel hierarchical ANFIS requires at least 2 features.")

    rng = np.random.default_rng(int(split_seed))
    order = np.arange(n_features, dtype=int)
    rng.shuffle(order)

    cut = int(max(1, n_features // 2))
    if cut >= n_features:
        cut = n_features - 1

    group_a = np.sort(order[:cut]).tolist()
    group_b = np.sort(order[cut:]).tolist()
    return group_a, group_b


def fit_parallel_hier_anfis(
    params,
    train_loader,
    n_features,
    n_outputs,
    task_kind,
    device,
    fusion="avg",
    split_seed=0,
    feature_names=None,
    verbose=False,
):
    fusion = str(fusion).strip().lower()
    if fusion != "avg":
        raise ValueError("H-ANFIS supports only fusion='avg'.")

    group_a, group_b = _make_random_feature_groups(n_features, split_seed)
    branch_rules = int(params.get("branch_rules", params.get("n_rules", 12)))
    top_rules = int(params.get("top_rules", max(2, branch_rules // 2)))
    mfs_per_input = int(params.get("mfs_per_input", 3))

    model = ParallelHierarchicalTSKANFIS(
        n_inputs=int(n_features),
        n_outputs=int(n_outputs),
        group_a_idx=group_a,
        group_b_idx=group_b,
        branch_rules=branch_rules,
        top_rules=top_rules,
        fusion=fusion,
        mfs_per_input=mfs_per_input,
        rule_init_mode=str(params.get("rule_init_mode", "legacy")),
        rule_seed=int(params.get("rule_seed", split_seed)),
        firing_mode=str(params.get("firing_mode", "htsk")),
    ).to(device)

    if task_kind == "multiclass":
        criterion = nn.CrossEntropyLoss()
    elif task_kind == "binary":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(params.get("lr", 0.01)),
        weight_decay=float(params.get("weight_decay", 0.0)),
    )
    epochs = int(params.get("epochs", 50))

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device).float()
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)

            if task_kind == "multiclass":
                loss = criterion(logits, targets.long())
            elif task_kind == "binary":
                loss = criterion(logits, targets.float())
            else:
                if targets.dim() == 1:
                    y = targets.float().unsqueeze(1)
                else:
                    y = targets.float()
                loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            running += float(loss.item()) * batch_size
            seen += batch_size

        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            avg = running / max(1, seen)
            print(f"[H-ANFIS {epoch:03d}] Loss={avg:.4f}")

    meta = {
        "group_a_idx": list(group_a),
        "group_b_idx": list(group_b),
        "fusion": str(fusion),
    }
    if feature_names is not None:
        names = list(feature_names)
        meta["group_a_features"] = [names[i] for i in group_a if i < len(names)]
        meta["group_b_features"] = [names[i] for i in group_b if i < len(names)]

    return model, criterion, meta


def _collect_acnn_init_data(train_loader, max_samples=2048):
    if max_samples is None or max_samples <= 0:
        max_samples = 2048
    chunks = []
    n_seen = 0
    for inputs, _ in train_loader:
        if torch.is_tensor(inputs):
            x_np = inputs.detach().cpu().numpy()
        else:
            x_np = np.asarray(inputs)
        chunks.append(x_np)
        n_seen += x_np.shape[0]
        if n_seen >= max_samples:
            break
    if not chunks:
        return None
    X = np.concatenate(chunks, axis=0)
    return X[:max_samples]


def _ensure_acnn_params(params, n_features, n_outputs, device):
    cfg = copy.deepcopy(params) if params is not None else {}

    cfg.setdefault("device", device)
    cfg.setdefault("SEED", 0)
    cfg.setdefault("G_TEMP", 1.0)
    cfg.setdefault("RULE_SELECT", False)
    cfg.setdefault("DR", 0.0)
    # Always sync runtime input width (e.g., after MI/RFE feature selection).
    cfg["IN_SIZE"] = int(n_features)
    cfg.setdefault("MODULE_ATTN_EN", True)
    cfg.setdefault("utils", "utils")
    cfg.setdefault("ACNN_VARIANT", "new")

    lr_cfg = cfg.get("LR", {})
    if not lr_cfg:
        lr_cfg = {
            "ant": cfg.get("lr_ant", 1e-5),
            "cons": cfg.get("lr_cons", 1e-5),
            "all": cfg.get("lr_all", 1e-4),
        }
    cfg["LR"] = {
        "ant": lr_cfg.get("ant", 1e-5),
        "cons": lr_cfg.get("cons", 1e-5),
        "all": lr_cfg.get("all", 1e-4),
    }
    cfg.setdefault("WD", cfg.get("weight_decay", 0.0))

    aconv = cfg.setdefault("ACONV", {})
    conv1 = aconv.setdefault("CONV1", {})
    conv1.setdefault("INP", 1)
    conv1.setdefault("OUT", int(min(12, max(1, n_features))))
    conv1.setdefault("SIZE", int(min(7, max(1, n_features))))
    conv1.setdefault("RANGE", (-1, 1))
    conv1.setdefault("STRIDE", 2 if conv1["SIZE"] > 2 else 1)
    conv1.setdefault("RULE_NUM", 200)
    conv1.setdefault("MF_FUNC", "gaussmf")
    conv1.setdefault("ATTN", True)
    conv1.setdefault("MF_NUM", 2)
    conv1.setdefault("ACT_FN", "Tanh")
    conv1.setdefault("BNORM", True)

    # Clamp conv config to current feature width for robust forward indexing.
    conv1["SIZE"] = int(max(1, min(int(conv1["SIZE"]), int(n_features))))
    conv1["STRIDE"] = int(max(1, int(conv1["STRIDE"])))
    if int(n_features) >= conv1["SIZE"]:
        max_windows = 1 + (int(n_features) - conv1["SIZE"]) // conv1["STRIDE"]
    else:
        max_windows = 1
    conv1["OUT"] = int(max(1, min(int(conv1["OUT"]), int(max_windows))))

    fc = aconv.setdefault("FC", {})
    fc.setdefault("INP", 1)
    fc.setdefault("OUT", 1)
    fc.setdefault("SIZE", conv1["OUT"])
    if fc["SIZE"] > conv1["OUT"]:
        fc["SIZE"] = conv1["OUT"]
    fc.setdefault("STRIDE", 1)
    fc.setdefault("RANGE", (-1, 1))
    fc.setdefault("RULE_NUM", 250)
    fc.setdefault("MF_FUNC", "gaussmf")
    fc.setdefault("ATTN", False)
    fc.setdefault("MF_NUM", 2)
    fc.setdefault("OUT_DIM", int(n_outputs))

    cfg["N_CLASS"] = int(fc["OUT_DIM"])
    return cfg


def fit_acnn(
    params,
    train_loader,
    n_features,
    n_outputs,
    task_kind,
    device,
    init_samples=2048,
    verbose=False,
):
    cfg = _ensure_acnn_params(params, n_features, n_outputs, device)

    X_init = _collect_acnn_init_data(train_loader, max_samples=init_samples)
    variant = str(cfg.get("ACNN_VARIANT", "new")).strip().lower()
    if variant in {"new", "current", "default"}:
        model_cls = ACNN
    elif variant in {"old", "legacy"}:
        model_cls = ACNNOld
    else:
        raise ValueError(
            f"Unsupported ACNN_VARIANT='{cfg.get('ACNN_VARIANT')}'. "
            "Use one of: new, old."
        )
    model = model_cls(cfg, X=X_init).to(device)

    out_dim = cfg["ACONV"]["FC"].get("OUT_DIM", n_outputs)
    if task_kind == "binary" and out_dim == 1:
        criterion = nn.BCEWithLogitsLoss()
        task_kind_eff = "binary"
    elif task_kind in {"binary", "multiclass"}:
        criterion = nn.CrossEntropyLoss()
        task_kind_eff = "multiclass"
    else:
        criterion = nn.MSELoss()
        task_kind_eff = "regression"

    ant_pars = []
    cons_pars = []
    other_pars = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ("sigma" in name) or ("center" in name):
            ant_pars.append(param)
        elif ("par_f" in name) or ("par_r" in name):
            cons_pars.append(param)
        else:
            other_pars.append(param)

    lr_ant = cfg["LR"]["ant"]
    lr_cons = cfg["LR"]["cons"]
    lr_all = cfg["LR"]["all"]
    wd = cfg.get("WD", 0.0)

    param_groups = []
    if ant_pars:
        param_groups.append({"params": ant_pars, "lr": lr_ant, "weight_decay": wd})
    if cons_pars:
        param_groups.append({"params": cons_pars, "lr": lr_cons, "weight_decay": wd})
    if other_pars:
        param_groups.append({"params": other_pars, "lr": lr_all, "weight_decay": wd})

    optimizer = optim.Adam(param_groups, lr=lr_all, weight_decay=wd)
    scheduler = None
    if cfg.get("SCH", False):
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.get("MST", []), gamma=cfg.get("GAMMA", 0.1)
        )

    epochs = int(cfg.get("epochs", cfg.get("EPOCHS", 30)))
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device).float()
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            if logits.dim() == 3 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)

            if task_kind_eff == "binary":
                if targets.dim() == 1:
                    y = targets.float().unsqueeze(1)
                else:
                    y = targets.float()
                loss = criterion(logits, y)
            elif task_kind_eff == "multiclass":
                y_idx = targets
                if y_idx.dim() > 1:
                    y_idx = y_idx.squeeze(1)
                loss = criterion(logits, y_idx.long())
            else:
                y = targets.float()
                if y.dim() == 1:
                    y = y.unsqueeze(1)
                loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

        if scheduler is not None:
            scheduler.step()

        if verbose and (epoch % 10 == 0 or epoch == 1):
            avg_loss = running_loss / max(1, n_samples)
            print(f"[ACNN {epoch:03d}] Train loss: {avg_loss:.4f}")

    return model, criterion
