"""Load and shape pipeline benchmark results for the Models dashboard page."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.secom_pipelines import BENCHMARK_MODEL_IDS, BENCHMARK_RESULTS_PATH


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    display_name: str
    family: str
    classifier: str
    feature_path: str
    description: str
    tuning_notebook: str


MODEL_CATALOG: dict[str, ModelInfo] = {
    "mspc_lr": ModelInfo(
        model_id="mspc_lr",
        display_name="MSPC + Elastic Net",
        family="MSPC (PLS + Q + T²)",
        classifier="Logistic regression (elastic net)",
        feature_path="Median impute → cluster → PLS+Q → Hotelling T²",
        description=(
            "Latent MSPC features with a sparse linear classifier. Strong baseline when "
            "defect signal is spread across many correlated sensors."
        ),
        tuning_notebook="tuning/mspc_lr.ipynb",
    ),
    "mspc_rf": ModelInfo(
        model_id="mspc_rf",
        display_name="MSPC + Random Forest",
        family="MSPC (PLS + Q + T²)",
        classifier="Random forest",
        feature_path="Median impute → cluster → PLS+Q → Hotelling T²",
        description=(
            "Same MSPC feature stack with a shallow random forest for nonlinear decision boundaries."
        ),
        tuning_notebook="tuning/mspc_rf.ipynb",
    ),
    "xgb_mspc": ModelInfo(
        model_id="xgb_mspc",
        display_name="MSPC + XGBoost",
        family="MSPC (PLS + Q + T²)",
        classifier="XGBoost",
        feature_path="Median impute → cluster → PLS+Q → Hotelling T²",
        description=(
            "MSPC features with gradient boosting; handles class imbalance via scale_pos_weight."
        ),
        tuning_notebook="tuning/xgb_mspc.ipynb",
    ),
    "rf_k_lr": ModelInfo(
        model_id="rf_k_lr",
        display_name="RF top-k + Elastic Net",
        family="RF-K (top-k + T²)",
        classifier="Logistic regression (elastic net)",
        feature_path="Median impute → cluster → RF SelectFromModel → Hotelling T²",
        description=(
            "Sparse sensor subset from RF importance, then linear classification with T² monitoring."
        ),
        tuning_notebook="tuning/rf_k_lr.ipynb",
    ),
    "rf_k_rf": ModelInfo(
        model_id="rf_k_rf",
        display_name="RF top-k + Random Forest",
        family="RF-K (top-k + T²)",
        classifier="Random forest",
        feature_path="Median impute → cluster → RF SelectFromModel → Hotelling T²",
        description=(
            "Top-k sensor selection plus forest classifier; often strong holdout PR AUC in this benchmark."
        ),
        tuning_notebook="tuning/rf_k_rf.ipynb",
    ),
    "rf_k_knn": ModelInfo(
        model_id="rf_k_knn",
        display_name="RF top-k + k-NN",
        family="RF-K (top-k + T²)",
        classifier="k-nearest neighbors",
        feature_path="Median impute → cluster → RF SelectFromModel → Hotelling T²",
        description=(
            "Instance-based classifier on a compact RF-selected feature set."
        ),
        tuning_notebook="tuning/rf_k_knn.ipynb",
    ),
}


def load_benchmark_results(path: Path | str = BENCHMARK_RESULTS_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing benchmark results at `{path}`.\n"
            "Run: `python -m scripts.benchmark_models` or `benchmark_models.ipynb`."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def cv_leaderboard_df(payload: dict[str, Any]) -> pd.DataFrame:
    rows = payload.get("leaderboard") or []
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def holdout_df(payload: dict[str, Any]) -> pd.DataFrame:
    rows = payload.get("holdout") or []
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def merged_comparison_df(payload: dict[str, Any]) -> pd.DataFrame:
    cv = cv_leaderboard_df(payload)
    ho = holdout_df(payload)
    if cv.empty or ho.empty:
        return pd.DataFrame()

    cv_cols = {
        "pipeline": "pipeline",
        "mean_pr_auc": "pr_auc_cv",
        "std_pr_auc": "std_pr_auc_cv",
        "mean_roc_auc": "roc_auc_cv",
        "std_roc_auc": "std_roc_auc_cv",
        "mean_ber_percent": "ber_cv",
        "std_ber_percent": "std_ber_cv",
        "mean_true_positive_percent": "tpr_cv",
        "mean_true_negative_percent": "tnr_cv",
    }
    ho_cols = {
        "pipeline": "pipeline",
        "pr_auc": "pr_auc_holdout",
        "roc_auc": "roc_auc_holdout",
        "ber_percent": "ber_holdout",
        "true_positive_percent": "tpr_holdout",
        "true_negative_percent": "tnr_holdout",
    }
    cv_sub = cv[[c for c in cv_cols if c in cv.columns]].rename(columns=cv_cols)
    ho_sub = ho[[c for c in ho_cols if c in ho.columns]].rename(columns=ho_cols)
    merged = cv_sub.merge(ho_sub, on="pipeline", how="outer")
    merged["cv_rank"] = merged["pr_auc_cv"].rank(ascending=False, method="min")
    merged["holdout_rank"] = merged["pr_auc_holdout"].rank(ascending=False, method="min")
    return merged.sort_values("pr_auc_cv", ascending=False).reset_index(drop=True)


def model_catalog() -> dict[str, ModelInfo]:
    return MODEL_CATALOG


def model_info(model_id: str) -> ModelInfo:
    if model_id not in MODEL_CATALOG:
        raise KeyError(f"Unknown model_id: {model_id}")
    return MODEL_CATALOG[model_id]


def list_model_ids(payload: dict[str, Any] | None = None) -> list[str]:
    if payload and payload.get("model_ids"):
        return list(payload["model_ids"])
    return list(BENCHMARK_MODEL_IDS)
