"""Holdout / reporting metrics aligned with benchmark CV scorers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)


def predict_with_threshold(y_score: np.ndarray, threshold: float) -> np.ndarray:
    """Binary labels from positive-class scores and a probability threshold."""
    return (np.asarray(y_score) >= threshold).astype(int)


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


def _empty_bootstrap_ci() -> dict[str, None]:
    return {
        "pr_auc_median": None,
        "pr_auc_ci_low": None,
        "pr_auc_ci_high": None,
        "ber_percent_median": None,
        "ber_percent_ci_low": None,
        "ber_percent_ci_high": None,
    }


def stratified_bootstrap_holdout_metrics(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray | None = None,
    *,
    threshold: float | None = None,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> dict[str, float | None]:
    """Stratified bootstrap CIs for holdout PR AUC and BER (no model refit)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if y_pred is None:
        if threshold is None:
            raise ValueError("y_pred or threshold required for BER bootstrap")
        y_pred = predict_with_threshold(y_score, threshold)
    else:
        y_pred = np.asarray(y_pred).astype(int)

    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return _empty_bootstrap_ci()

    rng = rng or np.random.default_rng()
    alpha = (1.0 - float(ci_level)) / 2.0
    lo_pct = 100.0 * alpha
    hi_pct = 100.0 * (1.0 - alpha)

    pr_aucs: list[float] = []
    bers: list[float] = []
    for _ in range(int(n_bootstrap)):
        bp = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        bn = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([bp, bn])
        y_b = y_true[idx]
        score_b = y_score[idx]
        pred_b = y_pred[idx]
        pr_aucs.append(float(average_precision_score(y_b, score_b)))
        bers.append(float(compute_holdout_metrics(y_b, pred_b)["ber_percent"]))

    pr_arr = np.asarray(pr_aucs)
    ber_arr = np.asarray(bers)
    return {
        "pr_auc_median": float(np.median(pr_arr)),
        "pr_auc_ci_low": float(np.percentile(pr_arr, lo_pct)),
        "pr_auc_ci_high": float(np.percentile(pr_arr, hi_pct)),
        "ber_percent_median": float(np.median(ber_arr)),
        "ber_percent_ci_low": float(np.percentile(ber_arr, lo_pct)),
        "ber_percent_ci_high": float(np.percentile(ber_arr, hi_pct)),
    }
