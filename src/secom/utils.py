"""Shared helpers for model tuning and benchmark evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from secom.pipelines import BENCHMARK_MODEL_IDS, TUNED_BLOCKED_PARAMS_DIR, TUNED_PARAMS_DIR

METRIC_SPECS = [
    ("balanced_accuracy", "ber_percent", True),
    ("true_positive_rate", "true_positive_percent", True),
    ("true_negative_rate", "true_negative_percent", True),
    ("roc_auc", "roc_auc", False),
    ("pr_auc", "pr_auc", False),
]


def json_safe(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def tuned_params_path(model_id: str, base_dir: Path = TUNED_PARAMS_DIR) -> Path:
    return base_dir / f"{model_id}.json"


def tuned_blocked_params_path(model_id: str) -> Path:
    return tuned_params_path(model_id, TUNED_BLOCKED_PARAMS_DIR)


def load_tuned_params(model_id: str, base_dir: Path = TUNED_PARAMS_DIR) -> dict:
    path = tuned_params_path(model_id, base_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run tuning/{model_id}.ipynb or tuning/tune_all.ipynb."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_tuned_blocked_params(model_id: str) -> dict:
    return load_tuned_params(model_id, TUNED_BLOCKED_PARAMS_DIR)


def load_all_tuned_params(base_dir: Path = TUNED_PARAMS_DIR) -> dict[str, dict]:
    missing = [
        model_id
        for model_id in BENCHMARK_MODEL_IDS
        if not tuned_params_path(model_id, base_dir).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing tuned params for: "
            + ", ".join(missing)
            + f". Expected JSON files under {base_dir}/"
        )
    return {model_id: load_tuned_params(model_id, base_dir) for model_id in BENCHMARK_MODEL_IDS}


def load_all_tuned_blocked_params() -> dict[str, dict]:
    return load_all_tuned_params(TUNED_BLOCKED_PARAMS_DIR)


def fitted_base_classifier(pipeline) -> object:
    """Underlying sklearn classifier after fit (unwraps threshold + calibration)."""
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "estimator_"):
        classifier = classifier.estimator_
    calibrated = getattr(classifier, "calibrated_classifiers_", None)
    if calibrated:
        return calibrated[0].estimator
    return classifier


def score_row_from_cv_result(result: dict) -> dict:
    """Aggregate one cross_validate result dict to mean/std metric columns."""
    row = {}
    for scorer_name, out_col, as_percent in METRIC_SPECS:
        values = result[f"test_{scorer_name}"]
        if scorer_name == "balanced_accuracy":
            values = 1 - values
        if as_percent:
            values = 100 * values
        row[f"mean_{out_col}"] = float(np.mean(values))
        row[f"std_{out_col}"] = float(np.std(values, ddof=0))
    return row
