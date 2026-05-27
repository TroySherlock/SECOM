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

from scripts.progress import tqdm_joblib_context
from scripts.secom_metrics import compute_holdout_metrics, predict_with_threshold
from scripts.secom_pipelines import (
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
from scripts.secom_utils import json_safe

from scripts.secom_pipelines import (
    C_GRID,
    KNN_CLASSIFIER_NEIGHBORS,
    KNN_NEIGHBORS_GRID,
    L1_RATIO_GRID,
    PLS_N_COMPONENTS,
    PLS_N_COMPONENTS_GRID,
    RF_MAX_DEPTH,
    RF_MAX_DEPTH_GRID,
    RF_SELECT_TOP_K,
    RF_SELECT_TOP_K_GRID,
    XGB_LEARNING_RATE,
    XGB_LEARNING_RATE_GRID,
    XGB_MAX_DEPTH,
    XGB_MAX_DEPTH_GRID,
    elastic_net,
    knn_classifier,
    mspc_preprocess,
    random_forest_classifier,
    rf_top_k_preprocess,
    secom_pipeline,
    xgboost_classifier,
)

PLS_N_PARAM = "preprocess__sensor_mspc__pls__n_components"
TOP_K_PARAM = "preprocess__sensor_mspc__select__max_features"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    build_pipeline: Callable[[], Pipeline]
    make_param_grid: Callable[[], dict]
    param_renames: dict[str, str]
    groupby_cols: list[str]
    best_defaults: dict[str, object]
    build_grid_search_best_params: Callable[[dict], dict]


def _mspc_lr_pipeline() -> Pipeline:
    return secom_pipeline(
        elastic_net(C=float(C_GRID[0]), l1_ratio=float(L1_RATIO_GRID[0])),
        n_components=int(PLS_N_COMPONENTS),
    )


def _mspc_lr_grid() -> dict:
    return {
        PLS_N_PARAM: [int(k) for k in PLS_N_COMPONENTS_GRID],
        "classifier__C": [float(c) for c in C_GRID],
        "classifier__l1_ratio": [float(r) for r in L1_RATIO_GRID],
    }


def _mspc_lr_best_params(cv_summary: dict) -> dict:
    return {
        PLS_N_PARAM: int(cv_summary["best_n_components"]),
        "classifier__C": float(cv_summary["best_c"]),
        "classifier__l1_ratio": float(cv_summary["best_l1_ratio"]),
    }


def _mspc_rf_pipeline() -> Pipeline:
    return secom_pipeline(
        random_forest_classifier(),
        n_components=int(PLS_N_COMPONENTS),
    )


def _mspc_rf_grid() -> dict:
    return {
        PLS_N_PARAM: [int(k) for k in PLS_N_COMPONENTS_GRID],
        "classifier__max_depth": [int(d) for d in RF_MAX_DEPTH_GRID],
    }


def _mspc_rf_best_params(cv_summary: dict) -> dict:
    return {
        PLS_N_PARAM: int(cv_summary["best_n_components"]),
        "classifier__max_depth": int(cv_summary["best_max_depth"]),
    }


def _xgb_mspc_pipeline() -> Pipeline:
    return secom_pipeline(
        xgboost_classifier(),
        n_components=int(PLS_N_COMPONENTS),
    )


def _xgb_mspc_grid() -> dict:
    return {
        PLS_N_PARAM: [int(k) for k in PLS_N_COMPONENTS_GRID],
        "classifier__max_depth": [int(d) for d in XGB_MAX_DEPTH_GRID],
        "classifier__learning_rate": [float(x) for x in XGB_LEARNING_RATE_GRID],
    }


def _xgb_mspc_best_params(cv_summary: dict) -> dict:
    return {
        PLS_N_PARAM: int(cv_summary["best_n_components"]),
        "classifier__max_depth": int(cv_summary["best_max_depth"]),
        "classifier__learning_rate": float(cv_summary["best_learning_rate"]),
    }


def _rf_k_lr_pipeline() -> Pipeline:
    return secom_pipeline(
        elastic_net(C=float(C_GRID[0]), l1_ratio=float(L1_RATIO_GRID[0])),
        preprocess=rf_top_k_preprocess(top_k=int(RF_SELECT_TOP_K)),
    )


def _rf_k_lr_grid() -> dict:
    return {
        TOP_K_PARAM: [int(k) for k in RF_SELECT_TOP_K_GRID],
        "classifier__C": [float(c) for c in C_GRID],
        "classifier__l1_ratio": [float(r) for r in L1_RATIO_GRID],
    }


def _rf_k_lr_best_params(cv_summary: dict) -> dict:
    return {
        TOP_K_PARAM: int(cv_summary["best_top_k"]),
        "classifier__C": float(cv_summary["best_c"]),
        "classifier__l1_ratio": float(cv_summary["best_l1_ratio"]),
    }


def _rf_k_rf_pipeline() -> Pipeline:
    return secom_pipeline(
        random_forest_classifier(),
        preprocess=rf_top_k_preprocess(top_k=int(RF_SELECT_TOP_K)),
    )


def _rf_k_rf_grid() -> dict:
    return {
        TOP_K_PARAM: [int(k) for k in RF_SELECT_TOP_K_GRID],
        "classifier__max_depth": [int(d) for d in RF_MAX_DEPTH_GRID],
    }


def _rf_k_rf_best_params(cv_summary: dict) -> dict:
    return {
        TOP_K_PARAM: int(cv_summary["best_top_k"]),
        "classifier__max_depth": int(cv_summary["best_max_depth"]),
    }


def _rf_k_knn_pipeline() -> Pipeline:
    return secom_pipeline(
        knn_classifier(),
        preprocess=rf_top_k_preprocess(top_k=int(RF_SELECT_TOP_K)),
    )


def _rf_k_knn_grid() -> dict:
    return {
        TOP_K_PARAM: [int(k) for k in RF_SELECT_TOP_K_GRID],
        "classifier__n_neighbors": [int(k) for k in KNN_NEIGHBORS_GRID],
    }


def _rf_k_knn_best_params(cv_summary: dict) -> dict:
    return {
        TOP_K_PARAM: int(cv_summary["best_top_k"]),
        "classifier__n_neighbors": int(cv_summary["best_n_neighbors"]),
    }


MODEL_SPECS: dict[str, ModelSpec] = {
    "mspc_lr": ModelSpec(
        model_id="mspc_lr",
        build_pipeline=_mspc_lr_pipeline,
        make_param_grid=_mspc_lr_grid,
        param_renames={
            f"param_{PLS_N_PARAM}": "n_components",
            "param_classifier__C": "c",
            "param_classifier__l1_ratio": "l1_ratio",
        },
        groupby_cols=["n_components", "c", "l1_ratio"],
        best_defaults={
            "n_components": int(PLS_N_COMPONENTS),
            "c": float(C_GRID[0]),
            "l1_ratio": float(L1_RATIO_GRID[0]),
        },
        build_grid_search_best_params=_mspc_lr_best_params,
    ),
    "mspc_rf": ModelSpec(
        model_id="mspc_rf",
        build_pipeline=_mspc_rf_pipeline,
        make_param_grid=_mspc_rf_grid,
        param_renames={
            f"param_{PLS_N_PARAM}": "n_components",
            "param_classifier__max_depth": "max_depth",
        },
        groupby_cols=["n_components", "max_depth"],
        best_defaults={
            "n_components": int(PLS_N_COMPONENTS),
            "max_depth": int(RF_MAX_DEPTH),
        },
        build_grid_search_best_params=_mspc_rf_best_params,
    ),
    "xgb_mspc": ModelSpec(
        model_id="xgb_mspc",
        build_pipeline=_xgb_mspc_pipeline,
        make_param_grid=_xgb_mspc_grid,
        param_renames={
            f"param_{PLS_N_PARAM}": "n_components",
            "param_classifier__max_depth": "max_depth",
            "param_classifier__learning_rate": "learning_rate",
        },
        groupby_cols=["n_components", "max_depth", "learning_rate"],
        best_defaults={
            "n_components": int(PLS_N_COMPONENTS),
            "max_depth": int(XGB_MAX_DEPTH),
            "learning_rate": float(XGB_LEARNING_RATE),
        },
        build_grid_search_best_params=_xgb_mspc_best_params,
    ),
    "rf_k_lr": ModelSpec(
        model_id="rf_k_lr",
        build_pipeline=_rf_k_lr_pipeline,
        make_param_grid=_rf_k_lr_grid,
        param_renames={
            f"param_{TOP_K_PARAM}": "top_k",
            "param_classifier__C": "c",
            "param_classifier__l1_ratio": "l1_ratio",
        },
        groupby_cols=["top_k", "c", "l1_ratio"],
        best_defaults={
            "top_k": int(RF_SELECT_TOP_K),
            "c": float(C_GRID[0]),
            "l1_ratio": float(L1_RATIO_GRID[0]),
        },
        build_grid_search_best_params=_rf_k_lr_best_params,
    ),
    "rf_k_rf": ModelSpec(
        model_id="rf_k_rf",
        build_pipeline=_rf_k_rf_pipeline,
        make_param_grid=_rf_k_rf_grid,
        param_renames={
            f"param_{TOP_K_PARAM}": "top_k",
            "param_classifier__max_depth": "max_depth",
        },
        groupby_cols=["top_k", "max_depth"],
        best_defaults={
            "top_k": int(RF_SELECT_TOP_K),
            "max_depth": int(RF_MAX_DEPTH),
        },
        build_grid_search_best_params=_rf_k_rf_best_params,
    ),
    "rf_k_knn": ModelSpec(
        model_id="rf_k_knn",
        build_pipeline=_rf_k_knn_pipeline,
        make_param_grid=_rf_k_knn_grid,
        param_renames={
            f"param_{TOP_K_PARAM}": "top_k",
            "param_classifier__n_neighbors": "n_neighbors",
        },
        groupby_cols=["top_k", "n_neighbors"],
        best_defaults={
            "top_k": int(RF_SELECT_TOP_K),
            "n_neighbors": int(KNN_CLASSIFIER_NEIGHBORS),
        },
        build_grid_search_best_params=_rf_k_knn_best_params,
    ),
}


def _resolved_classifier_threshold(tuned_payload: dict) -> float:
    raw = tuned_payload.get("classifier_threshold", 0.5)
    if isinstance(raw, str):
        if raw == "default_0.5":
            return 0.5
        return float(raw)
    return float(raw)


def build_tuned_pipeline(model_id: str, tuned_payload: dict) -> Pipeline:
    """Clone model pipeline, apply frozen params, and wrap classifier with tuned threshold."""
    spec = MODEL_SPECS[model_id]
    pipeline = clone(spec.build_pipeline())
    pipeline.set_params(**tuned_payload["grid_search_best_params"])
    threshold = _resolved_classifier_threshold(tuned_payload)
    classifier = pipeline.named_steps["classifier"]
    pipeline.steps[-1] = (
        "classifier",
        FixedThresholdClassifier(classifier, threshold=threshold),
    )
    return pipeline


def tuned_params_path(model_id: str, base_dir: Path = TUNED_PARAMS_DIR) -> Path:
    return base_dir / f"{model_id}.json"


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


def tune_classifier_threshold(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    cv_summary: dict,
    cv=None,
) -> dict:
    """Stage 2: sweep thresholds on CV validation probs; minimize mean BER."""
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

    mean_ber_by_threshold: dict[float, float] = {}
    for threshold in threshold_grid:
        fold_bers = []
        for proba, y_val in zip(fold_probas, fold_y_val, strict=True):
            pred = predict_with_threshold(proba, threshold)
            metrics = compute_holdout_metrics(y_val, pred)
            fold_bers.append(metrics["ber_percent"])
        mean_ber_by_threshold[threshold] = float(np.mean(fold_bers))

    best_threshold = min(mean_ber_by_threshold, key=mean_ber_by_threshold.get)
    best_fold_bers = []
    fold_results_at_best: list[dict] = []
    for fold_idx, (proba, y_val) in enumerate(zip(fold_probas, fold_y_val, strict=True)):
        pred = predict_with_threshold(proba, best_threshold)
        metrics = compute_holdout_metrics(y_val, pred)
        best_fold_bers.append(metrics["ber_percent"])
        fold_results_at_best.append(
            {
                "fold": fold_idx + 1,
                "threshold": best_threshold,
                "ber_percent": metrics["ber_percent"],
                "true_positive_percent": metrics["true_positive_percent"],
                "true_negative_percent": metrics["true_negative_percent"],
            }
        )

    per_threshold_mean_ber = (
        pd.DataFrame(
            [
                {"threshold": t, "mean_ber_percent": mean_ber_by_threshold[t]}
                for t in threshold_grid
            ]
        )
        .sort_values("mean_ber_percent", ascending=True, kind="mergesort")
        .reset_index(drop=True)
    )

    return {
        "best_threshold": float(best_threshold),
        "mean_ber_percent": float(np.mean(best_fold_bers)),
        "std_ber_percent": float(np.std(best_fold_bers, ddof=0)),
        "threshold_grid": threshold_grid,
        "per_threshold_mean_ber": per_threshold_mean_ber,
        "fold_results_at_best_threshold": fold_results_at_best,
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
    if threshold_result is not None:
        summary_out["mean_ber_percent_at_threshold"] = threshold_result["mean_ber_percent"]
        summary_out["std_ber_percent_at_threshold"] = threshold_result["std_ber_percent"]
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
        threshold_result["best_threshold"] if threshold_result else 0.5
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
        payload["threshold_tuning"] = json_safe(
            {
                "metric": "ber",
                "best_threshold": best_threshold,
                "mean_ber_percent": threshold_result["mean_ber_percent"],
                "std_ber_percent": threshold_result["std_ber_percent"],
                "threshold_grid": threshold_result["threshold_grid"],
                "per_threshold_mean_ber": threshold_result["per_threshold_mean_ber"]
                .head(10)
                .to_dict(orient="records"),
                "fold_results_at_best_threshold": threshold_result[
                    "fold_results_at_best_threshold"
                ],
            }
        )
        payload["cv_fold_results_at_threshold"] = threshold_result[
            "fold_results_at_best_threshold"
        ]
    if "best_n_components" in cv_summary:
        payload["best_n_components"] = int(cv_summary["best_n_components"])
    if "best_top_k" in cv_summary:
        payload["best_top_k"] = int(cv_summary["best_top_k"])
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
