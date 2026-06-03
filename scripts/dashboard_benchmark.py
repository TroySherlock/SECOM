"""Load and shape pipeline benchmark results for the Models dashboard page."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.secom_costs import (
    DEFAULT_PROFILE_ID,
    PROFILE_IDS,
    THRESHOLD_PROFILES,
    has_multi_profile_thresholds,
    normalize_profile_id,
    threshold_profile_config,
)
from scripts.secom_pipelines import (
    BENCHMARK_MODEL_IDS,
    BENCHMARK_RESULTS_PATH,
    TUNED_PARAMS_DIR,
)


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    display_name: str
    family: str
    classifier: str
    feature_path: str
    description: str
    tuning_notebook: str


_SHARED_FEATURE_PATH = (
    "Median impute → cluster → RF top-k → T² → hub pairs → scale"
)

MODEL_CATALOG: dict[str, ModelInfo] = {
    "linear_lr": ModelInfo(
        model_id="linear_lr",
        display_name="Linear LR",
        family="Shared preprocess",
        classifier="Logistic regression (elastic net, saga)",
        feature_path=f"{_SHARED_FEATURE_PATH} → elastic-net LR",
        description="Elastic-net logistic regression on the shared sensor path.",
        tuning_notebook="tuning/linear_lr.ipynb",
    ),
    "topk_rf": ModelInfo(
        model_id="topk_rf",
        display_name="Random Forest",
        family="Shared preprocess",
        classifier="Random forest",
        feature_path=f"{_SHARED_FEATURE_PATH} → RF",
        description="Random forest on the shared sensor path.",
        tuning_notebook="tuning/topk_rf.ipynb",
    ),
    "topk_knn": ModelInfo(
        model_id="topk_knn",
        display_name="k-NN",
        family="Shared preprocess",
        classifier="k-nearest neighbors",
        feature_path=f"{_SHARED_FEATURE_PATH} → k-NN",
        description="k-nearest neighbors on the shared sensor path.",
        tuning_notebook="tuning/topk_knn.ipynb",
    ),
    "topk_xgb": ModelInfo(
        model_id="topk_xgb",
        display_name="XGBoost",
        family="Shared preprocess",
        classifier="XGBoost",
        feature_path=f"{_SHARED_FEATURE_PATH} → XGBoost",
        description=(
            "Gradient boosting on the shared sensor path (scale_pos_weight for imbalance)."
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


def load_tuned_payload(
    model_id: str,
    base_dir: Path | str = TUNED_PARAMS_DIR,
) -> dict[str, Any]:
    path = Path(base_dir) / f"{model_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing tuned params: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_threshold_curves(model_id: str) -> pd.DataFrame:
    """Objective curves from tuned JSON (empty if not yet re-tuned)."""
    payload = load_tuned_payload(model_id)
    rows = payload.get("objective_curves") or []
    if not rows:
        legacy = payload.get("threshold_tuning") or {}
        grid = legacy.get("threshold_grid") or []
        per_ber = legacy.get("per_threshold_mean_ber") or []
        if per_ber:
            return pd.DataFrame(per_ber)
        if grid and legacy.get("mean_ber_percent") is not None:
            return pd.DataFrame()
    return pd.DataFrame(rows)


def holdout_by_profile_df(payload: dict[str, Any]) -> pd.DataFrame:
    """Long holdout table: pipeline, profile, fbeta, ber_percent, threshold."""
    ho = holdout_df(payload)
    if ho.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for _, row in ho.iterrows():
        pipeline = row["pipeline"]
        for profile_id in PROFILE_IDS:
            ber_col = f"{profile_id}_ber_percent"
            if ber_col not in row.index:
                continue
            entry: dict = {
                "pipeline": pipeline,
                "profile": profile_id,
                "display_name": THRESHOLD_PROFILES[profile_id].display_name,
                "threshold": row.get(f"{profile_id}_threshold"),
                "ber_percent": row.get(ber_col),
                "true_positive_percent": row.get(f"{profile_id}_true_positive_percent"),
                "true_negative_percent": row.get(f"{profile_id}_true_negative_percent"),
            }
            fbeta_col = f"{profile_id}_fbeta"
            if fbeta_col in row.index:
                entry["fbeta"] = row.get(fbeta_col)
            rows.append(entry)
    if not rows:
        return pd.DataFrame(
            columns=[
                "pipeline",
                "profile",
                "display_name",
                "threshold",
                "ber_percent",
                "true_positive_percent",
                "true_negative_percent",
                "fbeta",
            ]
        )
    return pd.DataFrame(rows)


def _parse_confusion_matrix(raw: object) -> list[list[int]] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, list):
        return raw
    return None


def holdout_confusion_by_profile(
    ho_df: pd.DataFrame,
    pipeline_id: str,
) -> dict[str, list[list[int]] | None]:
    """Holdout confusion matrices per F-beta profile for one pipeline."""
    out: dict[str, list[list[int]] | None] = {pid: None for pid in PROFILE_IDS}
    if ho_df.empty or "pipeline" not in ho_df.columns:
        return out
    rows = ho_df.loc[ho_df["pipeline"] == pipeline_id]
    if rows.empty:
        return out
    row = rows.iloc[0]
    for pid in PROFILE_IDS:
        cm_col = f"{pid}_confusion_matrix"
        if cm_col in row.index:
            out[pid] = _parse_confusion_matrix(row[cm_col])
    if out[DEFAULT_PROFILE_ID] is None and "confusion_matrix" in row.index:
        legacy = _parse_confusion_matrix(row["confusion_matrix"])
        if legacy is not None:
            out[DEFAULT_PROFILE_ID] = legacy
    return out


def _holdout_row_for_profile(
    ho_long: pd.DataFrame,
    model_id: str,
    profile_id: str,
) -> pd.Series | None:
    if ho_long.empty or "pipeline" not in ho_long.columns:
        return None
    ho_sub = ho_long.loc[
        (ho_long["pipeline"] == model_id) & (ho_long["profile"] == profile_id)
    ]
    return ho_sub.iloc[0] if not ho_sub.empty else None


def profile_threshold_summary_table(
    payload: dict[str, Any],
    tuned_dir: Path | str = TUNED_PARAMS_DIR,
) -> pd.DataFrame:
    """CV + holdout metrics for each model × threshold profile."""
    ho_long = holdout_by_profile_df(payload)
    rows: list[dict] = []
    for model_id in list_model_ids(payload):
        try:
            tuned = load_tuned_payload(model_id, tuned_dir)
        except FileNotFoundError:
            continue
        raw_profiles = tuned.get("threshold_profiles") or {}
        raw_keys = {str(k) for k in raw_profiles}
        profiles = {
            normalize_profile_id(k, raw_keys): v for k, v in raw_profiles.items()
        }
        for profile_id in PROFILE_IDS:
            cv = profiles.get(profile_id) or {}
            ho_row = _holdout_row_for_profile(ho_long, model_id, profile_id)
            row = {
                "pipeline": model_id,
                "profile": profile_id,
                "cv_threshold": cv.get("best_threshold"),
                "cv_mean_fbeta": cv.get("mean_fbeta"),
                "cv_mean_ber_percent": cv.get("mean_ber_percent"),
                "holdout_threshold": ho_row["threshold"] if ho_row is not None else None,
                "holdout_fbeta": ho_row.get("fbeta") if ho_row is not None else None,
                "holdout_ber_percent": ho_row["ber_percent"] if ho_row is not None else None,
                "holdout_tpr_percent": ho_row["true_positive_percent"]
                if ho_row is not None
                else None,
                "holdout_tnr_percent": ho_row["true_negative_percent"]
                if ho_row is not None
                else None,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def benchmark_has_multi_profile_thresholds(payload: dict[str, Any]) -> bool:
    for model_id in list_model_ids(payload):
        try:
            if has_multi_profile_thresholds(load_tuned_payload(model_id)):
                return True
        except FileNotFoundError:
            continue
    return False


def resolved_threshold_profile_config(
    payload: dict[str, Any],
) -> dict[str, float | str]:
    cfg = payload.get("threshold_profile_config")
    if isinstance(cfg, dict) and cfg:
        return dict(cfg)
    frozen = payload.get("frozen_config") or {}
    if isinstance(frozen, dict) and (
        frozen.get("f1_beta") is not None or frozen.get("f2_beta") is not None
    ):
        return {
            "f1_beta": frozen.get("f1_beta"),
            "f2_beta": frozen.get("f2_beta"),
            "f3_beta": frozen.get("f3_beta"),
            "default_profile": frozen.get("default_profile", "f2"),
        }
    return threshold_profile_config()
