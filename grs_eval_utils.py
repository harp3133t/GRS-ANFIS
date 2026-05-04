from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import inspect
import numpy as np

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    average = "binary" if len(np.unique(y_true)) == 2 else "weighted"
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    return acc, f1


def _supports_use_soft_eval(model):
    try:
        sig = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind == param.VAR_KEYWORD:
            return True
        if param.name == "use_soft_eval":
            return True
    return False


# GRS-ANFIS Evaluation Helper Functions
"""
Helper functions to evaluate GRS-ANFIS with Primary/Complementary/Combined metrics
"""

def evaluate_grs_anfis_detailed_classification(model, loader, criterion, device, task_kind):
    """
    Evaluate GRS-ANFIS separately for Primary, Complementary, and Combined
    Returns: dict with keys 'primary', 'complementary', 'combined', each containing (acc, f1)
    """

    results = {}
    
    # Evaluate each mode
    for mode_name in ['primary_only', 'complementary_only', 'full']:
        model.set_mode(mode_name)
        model.eval()
        
        all_preds = []
        all_targets = []
        is_bce_logits = isinstance(criterion, torch.nn.BCEWithLogitsLoss)
        
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
                    probs = torch.sigmoid(logits)
                    preds = (probs >= 0.5).long().squeeze(1)
                    targets = y_true.long().squeeze(1)
                else:
                    preds = torch.argmax(logits, dim=1)
                    targets = y_batch.long()
                
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_targets, axis=0)
        acc = accuracy_score(y_true, y_pred)
        
        # Compute F1
        if task_kind == "multiclass":
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        else:
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        # Store results
        key = 'combined' if mode_name == 'full' else mode_name.replace('_only', '')
        results[key] = (acc, f1)
    
    # Reset to full mode
    model.set_mode('full')
    
    return results


def evaluate_grs_anfis_detailed_regression(model, loader, device):
    """
    Evaluate GRS-ANFIS separately for Primary, Complementary, and Combined
    Returns: dict with keys 'primary', 'complementary', 'combined', each containing (mse, r2)
    """
    from sklearn.metrics import mean_squared_error, r2_score
    import torch
    import numpy as np
    
    results = {}
    
    # Evaluate each mode
    for mode_name in ['primary_only', 'complementary_only', 'full']:
        model.set_mode(mode_name)
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
        
        # Store results
        key = 'combined' if mode_name == 'full' else mode_name.replace('_only', '')
        results[key] = (mse, r2)
    
    # Reset to full mode
    model.set_mode('full')
    
    return results


@torch.no_grad()
def evaluate_torch_classification(
    model, loader, criterion, device, task_kind, use_soft_eval=False
):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_preds = []
    all_targets = []

    is_bce_logits = isinstance(criterion, nn.BCEWithLogitsLoss)

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        if use_soft_eval and _supports_use_soft_eval(model):
            logits = model(x_batch, use_soft_eval=True)
        else:
            logits = model(x_batch)

        if task_kind == "binary":
            if y_batch.dim() == 1:
                y_true = y_batch.float().unsqueeze(1)
            else:
                y_true = y_batch.float()

            probs = torch.sigmoid(logits)
            if is_bce_logits:
                loss = criterion(logits, y_true)
            else:
                loss = criterion(probs, y_true)

            preds = (probs >= 0.5).long().squeeze(1)
            targets = y_true.long().squeeze(1)
        else:
            loss = criterion(logits, y_batch.long())
            preds = torch.argmax(logits, dim=1)
            targets = y_batch.long()

        batch_size = x_batch.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / max(1, n_samples)
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    acc, f1 = compute_metrics(y_true, y_pred)
    return avg_loss, acc, f1
