"""Per-model pipeline specs, param grids, and GridSearchCV helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import FixedThresholdClassifier, GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline

from secom.progress import tqdm_joblib_context
from secom.costs import (
    DEFAULT_PROFILE_ID,
    PROFILE_IDS,
    THRESHOLD_PROFILES,
    fbeta_at_threshold,
)
from secom.metrics import compute_holdout_metrics, predict_with_threshold
from secom.pipelines import (
    CHAMPION_IMPUTATION_METHOD,
    CV_SCORING,
    GRID_SEARCH_VERBOSE,
    KNN_IMPUTE_NEIGHBORS,
    PRIMARY_TUNING_METRIC,
    TARGET_COL,
    THRESHOLD_GRID,
    TUNED_PARAMS_DIR,
    feature_columns,
    frozen_config,
    load_mart,
    make_repeated_stratified_cv,
    split_train_test,
)
from secom.utils import json_safe, tuned_params_path

from secom.pipelines import (
    C_GRID,
    CORRELATED_SELECTION_THRESHOLD,
    CORRELATED_SELECTION_THRESHOLD_GRID,
    KNN_CLASSIFIER_NEIGHBORS,
    KNN_NEIGHBORS_GRID,
    L1_RATIO_GRID,
    N_HUBS_DEFAULT,
    N_HUBS_GRID,
    RF_MAX_DEPTH,
    RF_MAX_DEPTH_GRID,
    RF_SELECT_TOP_K,
    RF_SELECT_TOP_K_GRID,
    XGB_LEARNING_RATE,
    XGB_LEARNING_RATE_GRID,
    XGB_MAX_DEPTH,
    XGB_MAX_DEPTH_GRID,
    elastic_net_lr,
    feature_pipeline,
    knn_classifier,
    linear_preprocess,
    random_forest_classifier,
    xgboost_classifier,
)

LINEAR_TOP_K_PARAM = "preprocess__sensor_branch__select_t2_hubs__top_k"
LINEAR_N_HUBS_PARAM = "preprocess__sensor_branch__select_t2_hubs__n_hubs"
LEGACY_TOP_K_PARAM = "preprocess__sensor_branch__select__max_features"
LEGACY_META_KNN_N_NEIGHBORS_PARAM = (
    "preprocess__sensor_branch__neighbor_fail_rate__n_neighbors"
)
LEGACY_META_KNN_IN_HUB_PARAM = (
    "preprocess__sensor_branch__select_t2_hubs__neighbor_n_neighbors"
)
SMART_CORR_THRESHOLD_PARAM = (
    "preprocess__sensor_branch__cluster__smart_corr__threshold"
)
CLASSIFIER_ESTIMATOR_PARAMS = (
    "C",
    "l1_ratio",
    "max_depth",
    "n_neighbors",
    "learning_rate",
)


def _hub_preprocess_grid() -> dict:
    return {
        LINEAR_TOP_K_PARAM: [int(k) for k in RF_SELECT_TOP_K_GRID],
        LINEAR_N_HUBS_PARAM: [int(k) for k in N_HUBS_GRID],
        SMART_CORR_THRESHOLD_PARAM: [float(t) for t in CORRELATED_SELECTION_THRESHOLD_GRID],
    }


def _hub_best_params(cv_summary: dict) -> dict:
    return {
        LINEAR_TOP_K_PARAM: int(cv_summary["best_top_k"]),
        LINEAR_N_HUBS_PARAM: int(cv_summary["best_n_hubs"]),
        SMART_CORR_THRESHOLD_PARAM: float(
            cv_summary.get("best_corr_threshold", CORRELATED_SELECTION_THRESHOLD)
        ),
    }


def _hub_preprocess_param_renames() -> dict[str, str]:
    return {
        f"param_{LINEAR_TOP_K_PARAM}": "top_k",
        f"param_{LINEAR_N_HUBS_PARAM}": "n_hubs",
        f"param_{SMART_CORR_THRESHOLD_PARAM}": "corr_threshold",
    }


def _hub_preprocess_groupby_cols() -> list[str]:
    return ["top_k", "n_hubs", "corr_threshold"]


def _hub_preprocess_best_defaults() -> dict[str, object]:
    return {
        "top_k": int(RF_SELECT_TOP_K),
        "n_hubs": int(N_HUBS_DEFAULT),
        "corr_threshold": float(CORRELATED_SELECTION_THRESHOLD),
    }


def _hub_feature_pipeline(classifier, *, top_k: int = RF_SELECT_TOP_K, n_hubs: int = N_HUBS_DEFAULT):
    return feature_pipeline(
        classifier,
        linear_preprocess(top_k=top_k, n_hubs=n_hubs),
    )


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    build_pipeline: Callable[[], Pipeline]
    make_param_grid: Callable[[], dict]
    param_renames: dict[str, str]
    groupby_cols: list[str]
    best_defaults: dict[str, object]
    build_grid_search_best_params: Callable[[dict], dict]


def _linear_lr_pipeline() -> Pipeline:
    return _hub_feature_pipeline(
        elastic_net_lr(
            C=float(C_GRID[0]),
            l1_ratio=float(L1_RATIO_GRID[0]),
        )
    )


def _linear_lr_grid() -> dict:
    return {
        **_hub_preprocess_grid(),
        "classifier__estimator__C": [float(c) for c in C_GRID],
        "classifier__estimator__l1_ratio": [float(r) for r in L1_RATIO_GRID],
    }


def _linear_lr_best_params(cv_summary: dict) -> dict:
    return {
        **_hub_best_params(cv_summary),
        "classifier__estimator__C": float(cv_summary["best_c"]),
        "classifier__estimator__l1_ratio": float(cv_summary["best_l1_ratio"]),
    }


def _topk_rf_pipeline() -> Pipeline:
    return _hub_feature_pipeline(random_forest_classifier())


def _topk_rf_grid() -> dict:
    return {
        **_hub_preprocess_grid(),
        "classifier__estimator__max_depth": [int(d) for d in RF_MAX_DEPTH_GRID],
    }


def _topk_rf_best_params(cv_summary: dict) -> dict:
    return {
        **_hub_best_params(cv_summary),
        "classifier__estimator__max_depth": int(cv_summary["best_max_depth"]),
    }


def _topk_knn_pipeline() -> Pipeline:
    return _hub_feature_pipeline(knn_classifier())


def _topk_knn_grid() -> dict:
    return {
        **_hub_preprocess_grid(),
        "classifier__estimator__n_neighbors": [int(k) for k in KNN_NEIGHBORS_GRID],
    }


def _topk_knn_best_params(cv_summary: dict) -> dict:
    return {
        **_hub_best_params(cv_summary),
        "classifier__estimator__n_neighbors": int(cv_summary["best_n_neighbors"]),
    }


def _topk_xgb_pipeline() -> Pipeline:
    return _hub_feature_pipeline(xgboost_classifier())


def _topk_xgb_grid() -> dict:
    return {
        **_hub_preprocess_grid(),
        "classifier__estimator__max_depth": [int(d) for d in XGB_MAX_DEPTH_GRID],
        "classifier__estimator__learning_rate": [
            float(x) for x in XGB_LEARNING_RATE_GRID
        ],
    }


def _topk_xgb_best_params(cv_summary: dict) -> dict:
    return {
        **_hub_best_params(cv_summary),
        "classifier__estimator__max_depth": int(cv_summary["best_max_depth"]),
        "classifier__estimator__learning_rate": float(
            cv_summary["best_learning_rate"]
        ),
    }


MODEL_SPECS: dict[str, ModelSpec] = {
    "linear_lr": ModelSpec(
        model_id="linear_lr",
        build_pipeline=_linear_lr_pipeline,
        make_param_grid=_linear_lr_grid,
        param_renames={
            **_hub_preprocess_param_renames(),
            "param_classifier__estimator__C": "c",
            "param_classifier__estimator__l1_ratio": "l1_ratio",
        },
        groupby_cols=[*_hub_preprocess_groupby_cols(), "c", "l1_ratio"],
        best_defaults={
            **_hub_preprocess_best_defaults(),
            "c": float(C_GRID[0]),
            "l1_ratio": float(L1_RATIO_GRID[0]),
        },
        build_grid_search_best_params=_linear_lr_best_params,
    ),
    "topk_rf": ModelSpec(
        model_id="topk_rf",
        build_pipeline=_topk_rf_pipeline,
        make_param_grid=_topk_rf_grid,
        param_renames={
            **_hub_preprocess_param_renames(),
            "param_classifier__estimator__max_depth": "max_depth",
        },
        groupby_cols=[*_hub_preprocess_groupby_cols(), "max_depth"],
        best_defaults={
            **_hub_preprocess_best_defaults(),
            "max_depth": int(RF_MAX_DEPTH),
        },
        build_grid_search_best_params=_topk_rf_best_params,
    ),
    "topk_knn": ModelSpec(
        model_id="topk_knn",
        build_pipeline=_topk_knn_pipeline,
        make_param_grid=_topk_knn_grid,
        param_renames={
            **_hub_preprocess_param_renames(),
            "param_classifier__estimator__n_neighbors": "n_neighbors",
        },
        groupby_cols=[*_hub_preprocess_groupby_cols(), "n_neighbors"],
        best_defaults={
            **_hub_preprocess_best_defaults(),
            "n_neighbors": int(KNN_CLASSIFIER_NEIGHBORS),
        },
        build_grid_search_best_params=_topk_knn_best_params,
    ),
    "topk_xgb": ModelSpec(
        model_id="topk_xgb",
        build_pipeline=_topk_xgb_pipeline,
        make_param_grid=_topk_xgb_grid,
        param_renames={
            **_hub_preprocess_param_renames(),
            "param_classifier__estimator__max_depth": "max_depth",
            "param_classifier__estimator__learning_rate": "learning_rate",
        },
        groupby_cols=[*_hub_preprocess_groupby_cols(), "max_depth", "learning_rate"],
        best_defaults={
            **_hub_preprocess_best_defaults(),
            "max_depth": int(XGB_MAX_DEPTH),
            "learning_rate": float(XGB_LEARNING_RATE),
        },
        build_grid_search_best_params=_topk_xgb_best_params,
    ),
}


def _resolved_classifier_threshold(tuned_payload: dict) -> float:
    raw = tuned_payload.get("classifier_threshold", 0.5)
    if isinstance(raw, str):
        if raw == "default_0.5":
            return 0.5
        return float(raw)
    return float(raw)


def _remap_legacy_classifier_estimator_params(raw: dict) -> None:
    """Map pre-calibration wrapper keys (classifier__max_depth) to nested paths."""
    for param in CLASSIFIER_ESTIMATOR_PARAMS:
        legacy = f"classifier__{param}"
        nested = f"classifier__estimator__{param}"
        if legacy in raw and nested not in raw:
            raw[nested] = raw.pop(legacy)


def resolve_grid_search_best_params(model_id: str, tuned_payload: dict) -> dict:
    """Map legacy tuned JSON keys to the current hub preprocess param names."""
    raw = dict(tuned_payload.get("grid_search_best_params") or {})
    cv_summary = tuned_payload.get("cv_summary") or {}

    if LEGACY_TOP_K_PARAM in raw:
        raw[LINEAR_TOP_K_PARAM] = int(raw.pop(LEGACY_TOP_K_PARAM))

    if LINEAR_TOP_K_PARAM not in raw and cv_summary.get("best_top_k") is not None:
        raw[LINEAR_TOP_K_PARAM] = int(cv_summary["best_top_k"])

    if LINEAR_N_HUBS_PARAM not in raw:
        if cv_summary.get("best_n_hubs") is not None:
            raw[LINEAR_N_HUBS_PARAM] = int(cv_summary["best_n_hubs"])
        else:
            raw[LINEAR_N_HUBS_PARAM] = int(N_HUBS_DEFAULT)

    for legacy_key in (LEGACY_META_KNN_N_NEIGHBORS_PARAM, LEGACY_META_KNN_IN_HUB_PARAM):
        raw.pop(legacy_key, None)

    if SMART_CORR_THRESHOLD_PARAM not in raw:
        if cv_summary.get("best_corr_threshold") is not None:
            raw[SMART_CORR_THRESHOLD_PARAM] = float(cv_summary["best_corr_threshold"])
        else:
            raw[SMART_CORR_THRESHOLD_PARAM] = float(CORRELATED_SELECTION_THRESHOLD)

    _remap_legacy_classifier_estimator_params(raw)

    spec = MODEL_SPECS[model_id]
    valid = spec.build_pipeline().get_params(deep=True)
    return {k: v for k, v in raw.items() if k in valid}


def build_tuned_pipeline(model_id: str, tuned_payload: dict) -> Pipeline:
    """Clone model pipeline, apply frozen params, and wrap classifier with tuned threshold."""
    spec = MODEL_SPECS[model_id]
    pipeline = clone(spec.build_pipeline())
    pipeline.set_params(**resolve_grid_search_best_params(model_id, tuned_payload))
    threshold = _resolved_classifier_threshold(tuned_payload)
    classifier = pipeline.named_steps["classifier"]
    pipeline.steps[-1] = (
        "classifier",
        FixedThresholdClassifier(classifier, threshold=threshold),
    )
    return pipeline


def grid_search_workload(
    param_grid: dict,
    cv=None,
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
) -> tuple[int, int, int]:
    cv = cv or make_repeated_stratified_cv()
    n_candidates = len(list(ParameterGrid(param_grid)))
    n_splits = cv.get_n_splits(X, y) if X is not None and y is not None else 5
    return n_candidates, n_splits, n_candidates * n_splits


def fit_with_progress(search: GridSearchCV, X: pd.DataFrame, y: pd.Series) -> GridSearchCV:
    _n_candidates, _n_splits, total_fits = grid_search_workload(
        search.param_grid, search.cv, X, y
    )
    with tqdm_joblib_context(total_fits, f"GridSearchCV {total_fits} fits"):
        search.fit(X, y)
    return search


def _fold_metric_series(
    selected_rows: pd.DataFrame,
    cv_results: pd.DataFrame,
    metric_name: str,
    *,
    as_percent: bool = False,
) -> pd.Series:
    suffix = f"_test_{metric_name}"
    split_cols = [
        col for col in cv_results.columns if col.startswith("split") and col.endswith(suffix)
    ]
    values = []
    for _, row in selected_rows.iterrows():
        for col in split_cols:
            score = 1 - row[col] if metric_name == "balanced_accuracy" else row[col]
            values.append(100 * score if as_percent else score)
    return pd.Series(values)


def summarize_cv_search(
    search: GridSearchCV,
    spec: ModelSpec,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    cv_results = pd.DataFrame(search.cv_results_)
    param_renames = spec.param_renames
    groupby_cols = spec.groupby_cols
    best_defaults = spec.best_defaults

    for param_col, friendly_col in param_renames.items():
        if param_col not in cv_results.columns:
            cv_results[param_col] = np.nan
    cv_results = cv_results.rename(columns=param_renames)

    for col in groupby_cols:
        if col not in cv_results.columns:
            cv_results[col] = np.nan

    cv_results["mean_balanced_accuracy"] = cv_results["mean_test_balanced_accuracy"]
    cv_results["mean_ber_percent"] = 100 * (1 - cv_results["mean_balanced_accuracy"])
    cv_results["std_ber_percent"] = 100 * cv_results["std_test_balanced_accuracy"].fillna(0)
    cv_results["mean_true_positive_percent"] = 100 * cv_results["mean_test_true_positive_rate"]
    cv_results["std_true_positive_percent"] = (
        100 * cv_results["std_test_true_positive_rate"].fillna(0)
    )
    cv_results["mean_true_negative_percent"] = 100 * cv_results["mean_test_true_negative_rate"]
    cv_results["std_true_negative_percent"] = (
        100 * cv_results["std_test_true_negative_rate"].fillna(0)
    )
    cv_results["mean_roc_auc"] = cv_results["mean_test_roc_auc"]
    cv_results["std_roc_auc"] = cv_results["std_test_roc_auc"].fillna(0)
    cv_results["mean_pr_auc"] = cv_results["mean_test_pr_auc"]
    cv_results["std_pr_auc"] = cv_results["std_test_pr_auc"].fillna(0)

    aggregated = (
        cv_results.groupby(groupby_cols, as_index=False, dropna=False)
        .agg(
            mean_ber_percent=("mean_ber_percent", "mean"),
            std_ber_percent=("std_ber_percent", "mean"),
            mean_balanced_accuracy=("mean_balanced_accuracy", "mean"),
            mean_true_positive_percent=("mean_true_positive_percent", "mean"),
            std_true_positive_percent=("std_true_positive_percent", "mean"),
            mean_true_negative_percent=("mean_true_negative_percent", "mean"),
            std_true_negative_percent=("std_true_negative_percent", "mean"),
            mean_roc_auc=("mean_roc_auc", "mean"),
            std_roc_auc=("std_roc_auc", "mean"),
            mean_pr_auc=("mean_pr_auc", "mean"),
            std_pr_auc=("std_pr_auc", "mean"),
        )
        .sort_values(
            ["mean_pr_auc", "std_pr_auc"] + groupby_cols,
            ascending=[False, False] + [True] * len(groupby_cols),
            kind="mergesort",
        )
    )

    best_row = aggregated.iloc[0]
    best_values = {
        key: best_row[key] if key in best_row.index and pd.notna(best_row[key]) else default
        for key, default in best_defaults.items()
    }

    selected_mask = pd.Series(True, index=cv_results.index)
    for col, value in best_values.items():
        if col in cv_results.columns:
            selected_mask &= cv_results[col] == value
    selected_rows = cv_results.loc[selected_mask]

    summary: dict = {
        "model_id": spec.model_id,
        "primary_selection_metric": PRIMARY_TUNING_METRIC,
        "imputation_method": CHAMPION_IMPUTATION_METHOD,
        "knn_impute_neighbors": int(KNN_IMPUTE_NEIGHBORS),
        "mean_ber_percent": float(best_row["mean_ber_percent"]),
        "std_ber_percent": float(best_row["std_ber_percent"]),
        "mean_balanced_accuracy": float(best_row["mean_balanced_accuracy"]),
        "mean_true_positive_percent": float(best_row["mean_true_positive_percent"]),
        "std_true_positive_percent": float(best_row["std_true_positive_percent"]),
        "mean_true_negative_percent": float(best_row["mean_true_negative_percent"]),
        "std_true_negative_percent": float(best_row["std_true_negative_percent"]),
        "mean_roc_auc": float(best_row["mean_roc_auc"]),
        "std_roc_auc": float(best_row["std_roc_auc"]),
        "mean_pr_auc": float(best_row["mean_pr_auc"]),
        "std_pr_auc": float(best_row["std_pr_auc"]),
    }
    for col, value in best_values.items():
        summary[f"best_{col}"] = type(value)(value)

    fold_results = pd.DataFrame(
        {
            **{col: best_values[col] for col in groupby_cols if col in best_values},
            "ber_percent": _fold_metric_series(
                selected_rows, cv_results, "balanced_accuracy", as_percent=True
            ),
            "true_positive_percent": _fold_metric_series(
                selected_rows, cv_results, "true_positive_rate", as_percent=True
            ),
            "true_negative_percent": _fold_metric_series(
                selected_rows, cv_results, "true_negative_rate", as_percent=True
            ),
            "roc_auc": _fold_metric_series(selected_rows, cv_results, "roc_auc"),
            "pr_auc": _fold_metric_series(selected_rows, cv_results, "pr_auc"),
        }
    )
    return summary, fold_results, aggregated


def _fold_metrics_at_threshold(
    fold_probas: list[np.ndarray],
    fold_y_val: list[pd.Series],
    threshold: float,
    *,
    beta: float,
) -> tuple[list[float], list[float], list[float]]:
    """Per-fold F-beta, BER %, and TPR at a single threshold."""
    fold_fbetas: list[float] = []
    fold_bers: list[float] = []
    fold_tprs: list[float] = []
    for proba, y_val in zip(fold_probas, fold_y_val, strict=True):
        pred = predict_with_threshold(proba, threshold)
        metrics = compute_holdout_metrics(y_val, pred)
        fold_fbetas.append(fbeta_at_threshold(y_val, pred, beta=beta))
        fold_bers.append(metrics["ber_percent"])
        fold_tprs.append(metrics["true_positive_percent"])
    return fold_fbetas, fold_bers, fold_tprs


def _profile_result_at_best(
    profile_id: str,
    best_threshold: float,
    fold_probas: list[np.ndarray],
    fold_y_val: list[pd.Series],
    *,
    beta: float,
) -> dict:
    fold_fbetas, fold_bers, fold_tprs = _fold_metrics_at_threshold(
        fold_probas, fold_y_val, best_threshold, beta=beta
    )
    fold_tnrs: list[float] = []
    fold_results: list[dict] = []
    for fold_idx, (proba, y_val) in enumerate(zip(fold_probas, fold_y_val, strict=True)):
        pred = predict_with_threshold(proba, best_threshold)
        metrics = compute_holdout_metrics(y_val, pred)
        fold_tnrs.append(metrics["true_negative_percent"])
        fold_results.append(
            {
                "fold": fold_idx + 1,
                "threshold": float(best_threshold),
                "fbeta": fbeta_at_threshold(y_val, pred, beta=beta),
                "ber_percent": metrics["ber_percent"],
                "true_positive_percent": metrics["true_positive_percent"],
                "true_negative_percent": metrics["true_negative_percent"],
            }
        )

    return {
        "profile_id": profile_id,
        "beta": float(beta),
        "best_threshold": float(best_threshold),
        "mean_fbeta": float(np.mean(fold_fbetas)),
        "std_fbeta": float(np.std(fold_fbetas, ddof=0)),
        "mean_ber_percent": float(np.mean(fold_bers)),
        "std_ber_percent": float(np.std(fold_bers, ddof=0)),
        "mean_true_positive_percent": float(np.mean(fold_tprs)),
        "mean_true_negative_percent": float(np.mean(fold_tnrs)),
        "fold_results_at_best_threshold": fold_results,
    }


def tune_classifier_threshold_profiles(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    cv_summary: dict,
    cv=None,
) -> dict:
    """Stage 2: sweep thresholds on CV validation probs; maximise mean F-beta per profile."""
    cv = cv or make_repeated_stratified_cv()
    best_params = spec.build_grid_search_best_params(cv_summary)
    base_pipeline = clone(spec.build_pipeline())
    base_pipeline.set_params(**best_params)

    threshold_grid = [float(t) for t in THRESHOLD_GRID]
    splits = list(cv.split(X, y))
    fold_probas: list[np.ndarray] = []
    fold_y_val: list[pd.Series] = []

    try:
        from tqdm.auto import tqdm

        split_iter = tqdm(splits, desc="Threshold CV folds")
    except ImportError:
        split_iter = splits

    for train_idx, val_idx in split_iter:
        fold_pipe = clone(base_pipeline)
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        fold_pipe.fit(X_tr, y_tr)
        fold_probas.append(fold_pipe.predict_proba(X_val)[:, 1])
        fold_y_val.append(y_val)

    mean_fbeta_by_profile: dict[str, dict[float, float]] = {
        pid: {} for pid in PROFILE_IDS
    }
    mean_ber_by_threshold: dict[float, float] = {}

    for threshold in threshold_grid:
        fold_bers, _, _ = _fold_metrics_at_threshold(
            fold_probas, fold_y_val, threshold, beta=1.0
        )
        mean_ber_by_threshold[threshold] = float(np.mean(fold_bers))
        for pid in PROFILE_IDS:
            beta = THRESHOLD_PROFILES[pid].beta
            fold_fbetas, _, _ = _fold_metrics_at_threshold(
                fold_probas, fold_y_val, threshold, beta=beta
            )
            mean_fbeta_by_profile[pid][threshold] = float(np.mean(fold_fbetas))

    best_thresholds = {
        pid: max(scores, key=scores.get)
        for pid, scores in mean_fbeta_by_profile.items()
    }

    profiles = {
        pid: _profile_result_at_best(
            pid,
            best_thresholds[pid],
            fold_probas,
            fold_y_val,
            beta=THRESHOLD_PROFILES[pid].beta,
        )
        for pid in PROFILE_IDS
    }

    curve_cols = {
        "threshold": threshold_grid,
        "mean_ber_percent": [mean_ber_by_threshold[t] for t in threshold_grid],
        **{
            f"mean_fbeta_{pid}": [
                mean_fbeta_by_profile[pid][t] for t in threshold_grid
            ]
            for pid in PROFILE_IDS
        },
    }
    objective_curves = pd.DataFrame(curve_cols)

    default_prof = profiles[DEFAULT_PROFILE_ID]
    neutral_col = f"mean_fbeta_{DEFAULT_PROFILE_ID}"
    per_threshold_mean_ber = (
        objective_curves[["threshold", "mean_ber_percent"]]
        .sort_values("mean_ber_percent", ascending=True, kind="mergesort")
    )
    return {
        "profiles": profiles,
        "threshold_grid": threshold_grid,
        "objective_curves": objective_curves,
        "best_threshold": default_prof["best_threshold"],
        "mean_fbeta": default_prof["mean_fbeta"],
        "std_fbeta": default_prof["std_fbeta"],
        "mean_ber_percent": default_prof["mean_ber_percent"],
        "std_ber_percent": default_prof["std_ber_percent"],
        "per_threshold_mean_ber": per_threshold_mean_ber,
        f"per_threshold_mean_fbeta_{DEFAULT_PROFILE_ID}": objective_curves[
            ["threshold", neutral_col]
        ].sort_values(neutral_col, ascending=False, kind="mergesort"),
        "fold_results_at_best_threshold": default_prof[
            "fold_results_at_best_threshold"
        ],
    }


def save_tuned_params(
    spec: ModelSpec,
    cv_summary: dict,
    fold_results: pd.DataFrame,
    aggregated: pd.DataFrame,
    *,
    threshold_result: dict | None = None,
    path: Path | None = None,
) -> dict:
    path = path or tuned_params_path(spec.model_id)
    best_params = spec.build_grid_search_best_params(cv_summary)
    summary_out = dict(cv_summary)
    profile_map = (
        threshold_result.get("profiles") if threshold_result is not None else None
    )
    default_profile = (
        profile_map.get(DEFAULT_PROFILE_ID) if profile_map else None
    )
    if default_profile is None and profile_map:
        default_profile = profile_map.get("ber") or profile_map.get("f3") or profile_map.get("f2")

    if default_profile is not None:
        summary_out["mean_fbeta_at_threshold"] = default_profile.get("mean_fbeta")
        summary_out["mean_ber_percent_at_threshold"] = default_profile["mean_ber_percent"]
        summary_out["std_ber_percent_at_threshold"] = default_profile["std_ber_percent"]
        summary_out["classifier_threshold"] = default_profile["best_threshold"]
        summary_out["mean_true_positive_percent_at_threshold"] = default_profile[
            "mean_true_positive_percent"
        ]
        summary_out["mean_true_negative_percent_at_threshold"] = default_profile[
            "mean_true_negative_percent"
        ]
    elif threshold_result is not None:
        summary_out["mean_fbeta_at_threshold"] = threshold_result.get("mean_fbeta")
        summary_out["mean_ber_percent_at_threshold"] = threshold_result.get(
            "mean_ber_percent"
        )
        summary_out["std_ber_percent_at_threshold"] = threshold_result.get(
            "std_ber_percent"
        )
        summary_out["classifier_threshold"] = threshold_result["best_threshold"]
        tpr_vals = [
            r["true_positive_percent"]
            for r in threshold_result["fold_results_at_best_threshold"]
        ]
        tnr_vals = [
            r["true_negative_percent"]
            for r in threshold_result["fold_results_at_best_threshold"]
        ]
        summary_out["mean_true_positive_percent_at_threshold"] = float(np.mean(tpr_vals))
        summary_out["mean_true_negative_percent_at_threshold"] = float(np.mean(tnr_vals))
    else:
        summary_out["classifier_threshold"] = 0.5

    best_threshold = float(
        default_profile["best_threshold"]
        if default_profile
        else (threshold_result["best_threshold"] if threshold_result else 0.5)
    )
    payload = {
        "model_id": spec.model_id,
        "classifier_threshold": best_threshold,
        "grid_search_best_params": best_params,
        "cv_summary": json_safe(summary_out),
        "cv_fold_results": fold_results.to_dict(orient="records"),
        "aggregated_top_configs": aggregated.head(10).to_dict(orient="records"),
        "frozen_config": frozen_config(),
        "tuned_at": datetime.now(timezone.utc).isoformat(),
    }
    if threshold_result is not None:
        fold_at_best = (
            default_profile["fold_results_at_best_threshold"]
            if default_profile
            else threshold_result["fold_results_at_best_threshold"]
        )
        per_thresh = threshold_result.get(
            f"per_threshold_mean_fbeta_{DEFAULT_PROFILE_ID}"
        )
        if per_thresh is None:
            per_thresh = threshold_result.get("per_threshold_mean_fbeta_f3")
        if per_thresh is None:
            per_thresh = threshold_result.get("per_threshold_mean_ber")
        payload["threshold_tuning"] = json_safe(
            {
                "metric": "fbeta",
                "default_profile": DEFAULT_PROFILE_ID,
                "best_threshold": best_threshold,
                "mean_fbeta": (
                    default_profile.get("mean_fbeta")
                    if default_profile
                    else threshold_result.get("mean_fbeta")
                ),
                "mean_ber_percent": (
                    default_profile["mean_ber_percent"]
                    if default_profile
                    else threshold_result.get("mean_ber_percent")
                ),
                "std_ber_percent": (
                    default_profile["std_ber_percent"]
                    if default_profile
                    else threshold_result.get("std_ber_percent")
                ),
                "threshold_grid": threshold_result["threshold_grid"],
                "per_threshold_top": (
                    per_thresh.head(10).to_dict(orient="records")
                    if hasattr(per_thresh, "head")
                    else per_thresh
                ),
                "fold_results_at_best_threshold": fold_at_best,
            }
        )
        payload["cv_fold_results_at_threshold"] = fold_at_best

        if profile_map:
            payload["threshold_profiles"] = json_safe(profile_map)
            curves = threshold_result.get("objective_curves")
            if curves is not None:
                payload["objective_curves"] = json_safe(
                    curves.to_dict(orient="records")
                )
    if "best_top_k" in cv_summary:
        payload["best_top_k"] = int(cv_summary["best_top_k"])
    if "best_n_hubs" in cv_summary:
        payload["best_n_hubs"] = int(cv_summary["best_n_hubs"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    return payload


def run_grid_search(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    verbose: int = GRID_SEARCH_VERBOSE,
) -> tuple[GridSearchCV, int, int, int]:
    param_grid = spec.make_param_grid()
    cv = make_repeated_stratified_cv()
    n_candidates, n_splits, total_fits = grid_search_workload(param_grid, cv, X, y)
    search = GridSearchCV(
        spec.build_pipeline(),
        param_grid=param_grid,
        cv=cv,
        scoring=CV_SCORING,
        refit=False,
        n_jobs=-1,
        verbose=verbose,
        error_score="raise",
    )
    return search, n_candidates, n_splits, total_fits
