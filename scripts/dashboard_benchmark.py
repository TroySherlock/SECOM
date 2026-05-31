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
    "linear_lr": ModelInfo(
        model_id="linear_lr",
        display_name="Linear (hub interactions)",
        family="Linear",
        classifier="Logistic regression (elastic net, saga)",
        feature_path=(
            "Median impute → cluster → RF top-k → Hotelling T² → "
            "hub×hub interactions → neighbor fail rate → isolation forest score → scale → elastic-net LR"
        ),
        description=(
            "Sparse linear path: RF-selected sensors, T², hub×hub products, "
            "KNN neighbor fail-rate meta feature, isolation-forest score, with elastic-net LR."
        ),
        tuning_notebook="tuning/linear_lr.ipynb",
    ),
    "topk_rf": ModelInfo(
        model_id="topk_rf",
        display_name="Hub features + Random Forest",
        family="Top-k + hubs",
        classifier="Random forest",
        feature_path=(
            "Median impute → cluster → RF top-k → Hotelling T² → "
            "hub×hub interactions → neighbor fail rate → isolation forest score → scale → RF"
        ),
        description=(
            "Same hub preprocess as linear_lr with a random forest classifier."
        ),
        tuning_notebook="tuning/topk_rf.ipynb",
    ),
    "topk_knn": ModelInfo(
        model_id="topk_knn",
        display_name="Hub features + k-NN",
        family="Top-k + hubs",
        classifier="k-nearest neighbors",
        feature_path=(
            "Median impute → cluster → RF top-k → Hotelling T² → "
            "hub×hub interactions → neighbor fail rate → isolation forest score → scale → k-NN"
        ),
        description="Shared hub preprocess with instance-based classification.",
        tuning_notebook="tuning/topk_knn.ipynb",
    ),
    "topk_xgb": ModelInfo(
        model_id="topk_xgb",
        display_name="Hub features + XGBoost",
        family="Top-k + hubs",
        classifier="XGBoost",
        feature_path=(
            "Median impute → cluster → RF top-k → Hotelling T² → "
            "hub×hub interactions → neighbor fail rate → isolation forest score → scale → XGBoost"
        ),
        description=(
            "Shared hub preprocess with gradient boosting (scale_pos_weight for imbalance)."
        ),
        tuning_notebook="tuning/topk_xgb.ipynb",
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
        "pr_auc_median": "pr_auc_holdout_median",
        "pr_auc_ci_low": "pr_auc_holdout_ci_low",
        "pr_auc_ci_high": "pr_auc_holdout_ci_high",
        "roc_auc": "roc_auc_holdout",
        "ber_percent": "ber_holdout",
        "ber_percent_median": "ber_holdout_median",
        "ber_percent_ci_low": "ber_holdout_ci_low",
        "ber_percent_ci_high": "ber_holdout_ci_high",
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
