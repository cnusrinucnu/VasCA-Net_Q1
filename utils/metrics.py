"""
Evaluation metrics matching Section 4.3 of the paper:
  Se (Sensitivity / TPR), Sp (Specificity), F1, ACC, AUC, FPR.
"""
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


@torch.no_grad()
def compute_confusion(pred_bin: torch.Tensor, target_bin: torch.Tensor):
    pred_bin = pred_bin.view(-1).bool()
    target_bin = target_bin.view(-1).bool()

    tp = (pred_bin & target_bin).sum().item()
    tn = ((~pred_bin) & (~target_bin)).sum().item()
    fp = (pred_bin & (~target_bin)).sum().item()
    fn = ((~pred_bin) & target_bin).sum().item()
    return tp, tn, fp, fn


@torch.no_grad()
def compute_metrics(
    probs: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    probs:  model output AFTER sigmoid, same shape as target, values in [0, 1]
    target: binary ground truth (0/1), same shape as probs
    """
    pred_bin = (probs >= threshold).float()
    tp, tn, fp, fn = compute_confusion(pred_bin, target)

    se = tp / (tp + fn + eps)          # Sensitivity / TPR / Recall
    sp = tn / (fp + tn + eps)          # Specificity
    precision = tp / (tp + fp + eps)
    f1 = 2 * tp / (2 * tp + fn + fp + eps)
    acc = (tp + tn) / (tp + tn + fn + fp + eps)
    fpr = fp / (fp + tn + eps)

    metrics = {
        "Se": se,
        "Sp": sp,
        "Precision": precision,
        "F1": f1,
        "ACC": acc,
        "FPR": fpr,
    }

    try:
        y_true = target.view(-1).cpu().numpy().astype(np.uint8)
        y_score = probs.view(-1).cpu().numpy()
        # AUC/PR-AUC are expensive over full images; subsample if huge.
        if y_true.size > 2_000_000:
            idx = np.random.choice(y_true.size, 2_000_000, replace=False)
            y_true, y_score = y_true[idx], y_score[idx]
        if y_true.min() != y_true.max():
            metrics["AUC"] = roc_auc_score(y_true, y_score)
            metrics["PR_AUC"] = average_precision_score(y_true, y_score)
    except Exception:
        pass

    return metrics


class MetricAccumulator:
    """Accumulates confusion-matrix counts across an epoch/dataset for
    globally-correct Se/Sp/F1/ACC (rather than averaging per-batch metrics)."""

    def __init__(self):
        self.tp = self.tn = self.fp = self.fn = 0
        self.probs_buffer = []
        self.targets_buffer = []

    def update(self, probs: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
        pred_bin = (probs >= threshold).float()
        tp, tn, fp, fn = compute_confusion(pred_bin, target)
        self.tp += tp
        self.tn += tn
        self.fp += fp
        self.fn += fn
        # Keep a light-weight subsample for AUC to avoid unbounded memory growth.
        flat_p = probs.view(-1).detach().cpu().numpy()
        flat_t = target.view(-1).detach().cpu().numpy().astype(np.uint8)
        if flat_p.size > 20000:
            idx = np.random.choice(flat_p.size, 20000, replace=False)
            flat_p, flat_t = flat_p[idx], flat_t[idx]
        self.probs_buffer.append(flat_p)
        self.targets_buffer.append(flat_t)

    def compute(self, eps: float = 1e-8) -> Dict[str, float]:
        tp, tn, fp, fn = self.tp, self.tn, self.fp, self.fn
        se = tp / (tp + fn + eps)
        sp = tn / (fp + tn + eps)
        precision = tp / (tp + fp + eps)
        f1 = 2 * tp / (2 * tp + fn + fp + eps)
        acc = (tp + tn) / (tp + tn + fn + fp + eps)
        fpr = fp / (fp + tn + eps)

        result = {"Se": se, "Sp": sp, "Precision": precision, "F1": f1, "ACC": acc, "FPR": fpr}

        try:
            y_score = np.concatenate(self.probs_buffer)
            y_true = np.concatenate(self.targets_buffer)
            if y_true.min() != y_true.max():
                result["AUC"] = roc_auc_score(y_true, y_score)
                result["PR_AUC"] = average_precision_score(y_true, y_score)
        except Exception:
            pass
        return result
