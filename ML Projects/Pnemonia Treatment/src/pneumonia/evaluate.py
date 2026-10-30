"""Evaluation metrics: accuracy, precision, recall, F1, AUC, AP, ROC, confusion matrix.

Metrics follow the paper (Tables 2-4, Section 4, Eq. 5-6):
- Accuracy, Precision, Recall, F1-Score per class.
- Validation Accuracy, Test Accuracy.
- AUC (Eq. 5) and Average Precision AP (Eq. 6).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)


@torch.no_grad()
def predict_probs(
    model: torch.nn.Module,
    loader,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_prob) for a dataloader."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    use_amp = device.type == "cuda"
    y_true_list = []
    y_prob_list = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
        if isinstance(outputs, dict):
            # TwoStagePipeline output -> use detection logits or pred
            logits = outputs.get("detection_logits", outputs.get("logits"))
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        probs = torch.softmax(logits.float(), dim=1)
        y_true_list.append(labels.cpu().numpy())
        y_prob_list.append(probs.cpu().numpy())
    y_true = np.concatenate(y_true_list)
    y_prob = np.vstack(y_prob_list)
    return y_true, y_prob


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[list[str]] = None,
) -> dict:
    """Compute classification metrics.

    Returns a dict with accuracy, per-class precision/recall/f1, macro
    averages, AUC (one-vs-rest), and Average Precision.
    """
    if class_names is None:
        class_names = [f"class_{i}" for i in range(y_prob.shape[1])]
    y_pred = y_prob.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": float(acc),
        "precision_per_class": [float(p) for p in precision],
        "recall_per_class": [float(r) for r in recall],
        "f1_per_class": [float(f) for f in f1],
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }

    # AUC and AP (one-vs-rest); only meaningful if >1 class present in y_true
    try:
        if len(np.unique(y_true)) > 1:
            y_true_bin = (y_true[:, None] == np.arange(y_prob.shape[1])).astype(int)
            metrics["auc_ovr"] = float(roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr"))
            metrics["ap_macro"] = float(average_precision_score(y_true_bin, y_prob, average="macro"))
    except ValueError:
        metrics["auc_ovr"] = None
        metrics["ap_macro"] = None

    return metrics


def evaluate_model(
    model: torch.nn.Module,
    loader,
    device: Optional[torch.device] = None,
    class_names: Optional[list[str]] = None,
) -> dict:
    """Evaluate a model on a dataloader and return metrics."""
    y_true, y_prob = predict_probs(model, loader, device=device)
    return compute_metrics(y_true, y_prob, class_names=class_names)


def get_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """Return per-class ROC curve data (fpr, tpr, thresholds) and AUC."""
    n_classes = y_prob.shape[1]
    y_true_bin = (y_true[:, None] == np.arange(n_classes)).astype(int)
    curves = {}
    aucs = {}
    for c in range(n_classes):
        try:
            fpr, tpr, thr = roc_curve(y_true_bin[:, c], y_prob[:, c])
            auc = roc_auc_score(y_true_bin[:, c], y_prob[:, c])
        except ValueError:
            fpr, tpr, thr = np.array([0.0]), np.array([0.0]), np.array([])
            auc = None
        curves[f"class_{c}"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thr.tolist()}
        aucs[f"class_{c}"] = float(auc) if auc is not None else None
    return {"curves": curves, "aucs": aucs}


def get_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """Return per-class Precision-Recall curve data and AP."""
    n_classes = y_prob.shape[1]
    y_true_bin = (y_true[:, None] == np.arange(n_classes)).astype(int)
    curves = {}
    aps = {}
    for c in range(n_classes):
        try:
            prec, rec, thr = precision_recall_curve(y_true_bin[:, c], y_prob[:, c])
            ap = average_precision_score(y_true_bin[:, c], y_prob[:, c])
        except ValueError:
            prec, rec, thr = np.array([1.0]), np.array([0.0]), np.array([])
            ap = None
        curves[f"class_{c}"] = {
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "thresholds": thr.tolist(),
        }
        aps[f"class_{c}"] = float(ap) if ap is not None else None
    return {"curves": curves, "aps": aps}
