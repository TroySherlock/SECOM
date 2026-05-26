"""Holdout / reporting metrics aligned with benchmark CV scorers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)


def compute_holdout_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict:
    """BER, TPR/TNR, optional ROC/PR AUC, and confusion matrix for binary labels."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    true_positive_rate = tp / (tp + fn) if (tp + fn) else 0.0
    true_negative_rate = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (true_positive_rate + true_negative_rate) / 2
    balanced_error_rate = 1 - balanced_accuracy

    metrics = {
        "true_positive_rate": float(true_positive_rate),
        "true_negative_rate": float(true_negative_rate),
        "balanced_accuracy": float(balanced_accuracy),
        "ber_percent": float(100 * balanced_error_rate),
        "true_positive_percent": float(100 * true_positive_rate),
        "true_negative_percent": float(100 * true_negative_rate),
        "confusion_matrix": cm.tolist(),
    }
    if y_score is not None:
        y_score = np.asarray(y_score)
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    return metrics
