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
    DECAY_LAMBDA_GRID,
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
    time_decay_weights,
)
from secom.utils import json_safe, tuned_params_path

from sklearn.metrics import average_precision_score


def _sample_weight_kwargs(timestamps, train_idx, decay_lambda: float) -> dict:
    """Fit kwargs for time-decay weighting; empty when timestamps is None.

    Passing no weights (timestamps=None) is required for models whose fit does
    not accept sample_weight (e.g. k-NN).
    """
    if timestamps is None:
        return {}
    w = time_decay_weights(timestamps.iloc[train_idx], decay_lambda)
    return {"classifier__sample_weight": w}

from secom.pipelines import (
    C_GRID,
    CORRELATED_SELECTION_THRESHOLD,
    CORRELATED_SELECTION_THRESHOLD_GRID,
    L1_RATIO_GRID,
    N_HUBS_DEFAULT,
    N_HUBS_GRID,
    PLS_N_COMPONENTS_DEFAULT,
    PLS_N_COMPONENTS_GRID,
    RF_MAX_DEPTH,
    RF_MAX_DEPTH_GRID,
    RF_SELECT_TOP_K,
    RF_SELECT_TOP_K_GRID,
    elastic_net_lr,
    feature_pipeline,
    interp_preprocess,
    random_forest_classifier,
)
from secom.extrap_pipelines import EXTRAP_MODEL_IDS
from secom.extrap_pipelines import is_bayesian as _is_bayesian_id

LINEAR_TOP_K_PARAM = "preprocess__sensor_branch__select_t2_hubs__top_k"
LINEAR_N_HUBS_PARAM = "preprocess__sensor_branch__select_t2_hubs__n_hubs"
SMART_CORR_THRESHOLD_PARAM = (
    "preprocess__sensor_branch__cluster__smart_corr__threshold"
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


PLS_N_COMPONENTS_PARAM = "preprocess__sensor_branch__pls__n_components"


def _hub_feature_pipeline(classifier, *, top_k: int = RF_SELECT_TOP_K, n_hubs: int = N_HUBS_DEFAULT):
    """Interpolation RF-selection front-end + EFA T²/Q gate features → classifier."""
    return feature_pipeline(
        classifier,
        interp_preprocess(front_end="rf", top_k=top_k, n_hubs=n_hubs, with_gate=True),
    )


def _pls_feature_pipeline(classifier, *, pls_n_components: int = PLS_N_COMPONENTS_DEFAULT):
    """Interpolation PLS reduction front-end + EFA T²/Q gate features → classifier."""
    return feature_pipeline(
        classifier,
        interp_preprocess(
            front_end="pls", pls_n_components=pls_n_components, with_gate=True
        ),
    )


def _pls_preprocess_grid() -> dict:
    return {
        PLS_N_COMPONENTS_PARAM: [int(k) for k in PLS_N_COMPONENTS_GRID],
        SMART_CORR_THRESHOLD_PARAM: [float(t) for t in CORRELATED_SELECTION_THRESHOLD_GRID],
    }


def _pls_best_params(cv_summary: dict) -> dict:
    return {
        PLS_N_COMPONENTS_PARAM: int(cv_summary["best_pls_n_components"]),
        SMART_CORR_THRESHOLD_PARAM: float(
            cv_summary.get("best_corr_threshold", CORRELATED_SELECTION_THRESHOLD)
        ),
    }


def _pls_preprocess_param_renames() -> dict[str, str]:
    return {
        f"param_{PLS_N_COMPONENTS_PARAM}": "pls_n_components",
        f"param_{SMART_CORR_THRESHOLD_PARAM}": "corr_threshold",
    }


def _pls_preprocess_groupby_cols() -> list[str]:
    return ["pls_n_components", "corr_threshold"]


def _pls_preprocess_best_defaults() -> dict[str, object]:
    return {
        "pls_n_components": int(PLS_N_COMPONENTS_DEFAULT),
        "corr_threshold": float(CORRELATED_SELECTION_THRESHOLD),
    }


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    build_pipeline: Callable[[], Pipeline]
    make_param_grid: Callable[[], dict]
    param_renames: dict[str, str]
    groupby_cols: list[str]
    best_defaults: dict[str, object]
    build_grid_search_best_params: Callable[[dict], dict]
    track: str = "interpolation"


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


def _pls_enet_pipeline() -> Pipeline:
    return _pls_feature_pipeline(
        elastic_net_lr(
            C=float(C_GRID[0]),
            l1_ratio=float(L1_RATIO_GRID[0]),
        )
    )


def _pls_enet_grid() -> dict:
    return {
        **_pls_preprocess_grid(),
        "classifier__estimator__C": [float(c) for c in C_GRID],
        "classifier__estimator__l1_ratio": [float(r) for r in L1_RATIO_GRID],
    }


def _pls_enet_best_params(cv_summary: dict) -> dict:
    return {
        **_pls_best_params(cv_summary),
        "classifier__estimator__C": float(cv_summary["best_c"]),
        "classifier__estimator__l1_ratio": float(cv_summary["best_l1_ratio"]),
    }


def _pls_rf_pipeline() -> Pipeline:
    return _pls_feature_pipeline(random_forest_classifier())


def _pls_rf_grid() -> dict:
    return {
        **_pls_preprocess_grid(),
        "classifier__estimator__max_depth": [int(d) for d in RF_MAX_DEPTH_GRID],
    }


def _pls_rf_best_params(cv_summary: dict) -> dict:
    return {
        **_pls_best_params(cv_summary),
        "classifier__estimator__max_depth": int(cv_summary["best_max_depth"]),
    }


_LINEAR_LR_RENAMES = {
    **_hub_preprocess_param_renames(),
    "param_classifier__estimator__C": "c",
    "param_classifier__estimator__l1_ratio": "l1_ratio",
}
_LINEAR_LR_GROUPBY = [*_hub_preprocess_groupby_cols(), "c", "l1_ratio"]
_LINEAR_LR_DEFAULTS = {
    **_hub_preprocess_best_defaults(),
    "c": float(C_GRID[0]),
    "l1_ratio": float(L1_RATIO_GRID[0]),
}

_TOPK_RF_RENAMES = {
    **_hub_preprocess_param_renames(),
    "param_classifier__estimator__max_depth": "max_depth",
}
_TOPK_RF_GROUPBY = [*_hub_preprocess_groupby_cols(), "max_depth"]
_TOPK_RF_DEFAULTS = {
    **_hub_preprocess_best_defaults(),
    "max_depth": int(RF_MAX_DEPTH),
}

_PLS_ENET_RENAMES = {
    **_pls_preprocess_param_renames(),
    "param_classifier__estimator__C": "c",
    "param_classifier__estimator__l1_ratio": "l1_ratio",
}
_PLS_ENET_GROUPBY = [*_pls_preprocess_groupby_cols(), "c", "l1_ratio"]
_PLS_ENET_DEFAULTS = {
    **_pls_preprocess_best_defaults(),
    "c": float(C_GRID[0]),
    "l1_ratio": float(L1_RATIO_GRID[0]),
}

_PLS_RF_RENAMES = {
    **_pls_preprocess_param_renames(),
    "param_classifier__estimator__max_depth": "max_depth",
}
_PLS_RF_GROUPBY = [*_pls_preprocess_groupby_cols(), "max_depth"]
_PLS_RF_DEFAULTS = {
    **_pls_preprocess_best_defaults(),
    "max_depth": int(RF_MAX_DEPTH),
}

MODEL_SPECS: dict[str, ModelSpec] = {
    "intrap_linear_lr": ModelSpec(
        model_id="intrap_linear_lr",
        build_pipeline=_linear_lr_pipeline,
        make_param_grid=_linear_lr_grid,
        param_renames=_LINEAR_LR_RENAMES,
        groupby_cols=_LINEAR_LR_GROUPBY,
        best_defaults=_LINEAR_LR_DEFAULTS,
        build_grid_search_best_params=_linear_lr_best_params,
        track="interpolation",
    ),
    "intrap_topk_rf": ModelSpec(
        model_id="intrap_topk_rf",
        build_pipeline=_topk_rf_pipeline,
        make_param_grid=_topk_rf_grid,
        param_renames=_TOPK_RF_RENAMES,
        groupby_cols=_TOPK_RF_GROUPBY,
        best_defaults=_TOPK_RF_DEFAULTS,
        build_grid_search_best_params=_topk_rf_best_params,
        track="interpolation",
    ),
    "intrap_pls_enet": ModelSpec(
        model_id="intrap_pls_enet",
        build_pipeline=_pls_enet_pipeline,
        make_param_grid=_pls_enet_grid,
        param_renames=_PLS_ENET_RENAMES,
        groupby_cols=_PLS_ENET_GROUPBY,
        best_defaults=_PLS_ENET_DEFAULTS,
        build_grid_search_best_params=_pls_enet_best_params,
        track="interpolation",
    ),
    "intrap_pls_rf": ModelSpec(
        model_id="intrap_pls_rf",
        build_pipeline=_pls_rf_pipeline,
        make_param_grid=_pls_rf_grid,
        param_renames=_PLS_RF_RENAMES,
        groupby_cols=_PLS_RF_GROUPBY,
        best_defaults=_PLS_RF_DEFAULTS,
        build_grid_search_best_params=_pls_rf_best_params,
        track="interpolation",
    ),
}

# Track-aware unified lookup: sklearn MODEL_SPECS (interpolation) are tuned via
# GridSearchCV; the extrapolation ids are Bayesian and tuned via bayes/harness.py.
TRACKS = ("interpolation", "extrapolation")
ALL_MODEL_IDS: tuple[str, ...] = tuple(MODEL_SPECS.keys()) + tuple(EXTRAP_MODEL_IDS)


def is_bayesian(model_id: str) -> bool:
    """True for extrapolation-track Bayesian models (no sklearn ModelSpec)."""
    return bool(_is_bayesian_id(model_id))


def model_ids_for_track(track: str | None = None) -> list[str]:
    """Model ids filtered by track across both registries; all ids when None."""
    if track == "extrapolation":
        return list(EXTRAP_MODEL_IDS)
    if track == "interpolation":
        return [mid for mid, spec in MODEL_SPECS.items() if spec.track == "interpolation"]
    return list(ALL_MODEL_IDS)


def _resolved_classifier_threshold(tuned_payload: dict) -> float:
    raw = tuned_payload.get("classifier_threshold", 0.5)
    if isinstance(raw, str):
        if raw == "default_0.5":
            return 0.5
        return float(raw)
    return float(raw)


def resolve_grid_search_best_params(model_id: str, tuned_payload: dict) -> dict:
    """Resolve tuned grid-search params for the current pipeline param names."""
    raw = dict(tuned_payload.get("grid_search_best_params") or {})
    cv_summary = tuned_payload.get("cv_summary") or {}

    if LINEAR_TOP_K_PARAM not in raw and cv_summary.get("best_top_k") is not None:
        raw[LINEAR_TOP_K_PARAM] = int(cv_summary["best_top_k"])

    if LINEAR_N_HUBS_PARAM not in raw:
        if cv_summary.get("best_n_hubs") is not None:
            raw[LINEAR_N_HUBS_PARAM] = int(cv_summary["best_n_hubs"])
        else:
            raw[LINEAR_N_HUBS_PARAM] = int(N_HUBS_DEFAULT)

    if SMART_CORR_THRESHOLD_PARAM not in raw:
        if cv_summary.get("best_corr_threshold") is not None:
            raw[SMART_CORR_THRESHOLD_PARAM] = float(cv_summary["best_corr_threshold"])
        else:
            raw[SMART_CORR_THRESHOLD_PARAM] = float(CORRELATED_SELECTION_THRESHOLD)

    spec = MODEL_SPECS[model_id]
    valid = spec.build_pipeline().get_params(deep=True)
    return {k: v for k, v in raw.items() if k in valid}


def build_tuned_pipeline(
    model_id: str,
    tuned_payload: dict,
) -> Pipeline:
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


def fit_pipeline_weighted(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: np.ndarray | None = None,
) -> tuple[Pipeline, float | None]:
    """Fit a tuned pipeline, routing ``sample_weight`` to the inner estimator.

    ``build_tuned_pipeline`` wraps the final classifier in
    ``FixedThresholdClassifier``, which cannot forward ``sample_weight`` to its
    estimator without sklearn metadata routing. When weights are present we fit
    the unwrapped estimator and return its tuned decision threshold so callers
    that need labels can reproduce ``predict`` via ``predict_with_threshold``
    (``>= threshold`` on the positive class). ``predict_proba`` is unaffected by
    unwrapping. When ``sample_weight`` is None the wrapped pipeline is fit in
    place (unchanged behaviour). Returns ``(fitted_pipeline, threshold_or_None)``.
    """
    final = pipeline.steps[-1][1]
    threshold = (
        float(final.threshold) if isinstance(final, FixedThresholdClassifier) else None
    )
    if sample_weight is None:
        pipeline.fit(X, y)
        return pipeline, threshold
    if isinstance(final, FixedThresholdClassifier):
        pipeline = clone(pipeline)
        pipeline.steps[-1] = ("classifier", clone(final.estimator))
    pipeline.fit(X, y, classifier__sample_weight=sample_weight)
    return pipeline, threshold


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
    beta: float | None,
    objective: str = "fbeta",
) -> dict:
    fbeta_beta = beta if beta is not None else 1.0
    fold_fbetas, fold_bers, fold_tprs = _fold_metrics_at_threshold(
        fold_probas, fold_y_val, best_threshold, beta=fbeta_beta
    )
    fold_tnrs: list[float] = []
    fold_results: list[dict] = []
    for fold_idx, (proba, y_val) in enumerate(zip(fold_probas, fold_y_val, strict=True)):
        pred = predict_with_threshold(proba, best_threshold)
        metrics = compute_holdout_metrics(y_val, pred)
        fold_tnrs.append(metrics["true_negative_percent"])
        fold_entry: dict = {
            "fold": fold_idx + 1,
            "threshold": float(best_threshold),
            "ber_percent": metrics["ber_percent"],
            "true_positive_percent": metrics["true_positive_percent"],
            "true_negative_percent": metrics["true_negative_percent"],
        }
        if objective == "fbeta" and beta is not None:
            fold_entry["fbeta"] = fbeta_at_threshold(y_val, pred, beta=beta)
        fold_results.append(fold_entry)

    result: dict = {
        "profile_id": profile_id,
        "objective": objective,
        "best_threshold": float(best_threshold),
        "mean_ber_percent": float(np.mean(fold_bers)),
        "std_ber_percent": float(np.std(fold_bers, ddof=0)),
        "mean_true_positive_percent": float(np.mean(fold_tprs)),
        "mean_true_negative_percent": float(np.mean(fold_tnrs)),
        "fold_results_at_best_threshold": fold_results,
    }
    if objective == "fbeta" and beta is not None:
        result["beta"] = float(beta)
        result["mean_fbeta"] = float(np.mean(fold_fbetas))
        result["std_fbeta"] = float(np.std(fold_fbetas, ddof=0))
    return result


def tune_time_decay_lambda(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    cv_summary: dict,
    cv,
    *,
    lambda_grid: list[float] | None = None,
) -> dict:
    """Select exponential time-decay lambda by mean PR-AUC over blocked CV folds.

    Fits the structurally-tuned pipeline per fold with time-decay sample weights
    and scores average_precision on each validation block. lambda is tuned only
    on CV (never the holdout); lambda=0 recovers the unweighted model.
    """
    lambda_grid = [float(x) for x in (lambda_grid or DECAY_LAMBDA_GRID)]
    best_params = spec.build_grid_search_best_params(cv_summary)
    base_pipeline = clone(spec.build_pipeline())
    base_pipeline.set_params(**best_params)

    splits = list(cv.split(X, y))
    per_lambda: dict[float, float] = {}
    for decay_lambda in lambda_grid:
        fold_scores: list[float] = []
        for train_idx, val_idx in splits:
            fold_pipe = clone(base_pipeline)
            fold_pipe.fit(
                X.iloc[train_idx],
                y.iloc[train_idx],
                **_sample_weight_kwargs(timestamps, train_idx, decay_lambda),
            )
            proba = fold_pipe.predict_proba(X.iloc[val_idx])[:, 1]
            fold_scores.append(
                float(average_precision_score(y.iloc[val_idx], proba))
            )
        per_lambda[decay_lambda] = float(np.mean(fold_scores)) if fold_scores else 0.0

    best_decay_lambda = max(per_lambda, key=per_lambda.get)
    return {
        "best_decay_lambda": float(best_decay_lambda),
        "best_mean_pr_auc": float(per_lambda[best_decay_lambda]),
        "per_lambda_pr_auc": {str(k): v for k, v in per_lambda.items()},
        "lambda_grid": lambda_grid,
    }


def tune_classifier_threshold_profiles(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    cv_summary: dict,
    cv=None,
    timestamps: pd.Series | None = None,
    decay_lambda: float = 0.0,
) -> dict:
    """Stage 2: sweep thresholds on CV validation probs; maximise mean F-beta per profile.

    When ``timestamps`` is provided (extrapolation, weight-capable models), each
    fold fit is time-decay weighted at ``decay_lambda``.
    """
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
        fold_pipe.fit(X_tr, y_tr, **_sample_weight_kwargs(timestamps, train_idx, decay_lambda))
        fold_probas.append(fold_pipe.predict_proba(X_val)[:, 1])
        fold_y_val.append(y_val)

    fbeta_profile_ids = [
        pid for pid in PROFILE_IDS if THRESHOLD_PROFILES[pid].objective == "fbeta"
    ]
    mean_fbeta_by_profile: dict[str, dict[float, float]] = {
        pid: {} for pid in fbeta_profile_ids
    }
    mean_ber_by_threshold: dict[float, float] = {}
    mean_tpr_by_threshold: dict[float, float] = {}

    for threshold in threshold_grid:
        _, fold_bers, fold_tprs = _fold_metrics_at_threshold(
            fold_probas, fold_y_val, threshold, beta=1.0
        )
        mean_ber_by_threshold[threshold] = float(np.mean(fold_bers))
        mean_tpr_by_threshold[threshold] = float(np.mean(fold_tprs))
        for pid in fbeta_profile_ids:
            beta = THRESHOLD_PROFILES[pid].beta
            fold_fbetas, _, _ = _fold_metrics_at_threshold(
                fold_probas, fold_y_val, threshold, beta=beta
            )
            mean_fbeta_by_profile[pid][threshold] = float(np.mean(fold_fbetas))

    best_thresholds: dict[str, float] = {
        pid: max(scores, key=scores.get)
        for pid, scores in mean_fbeta_by_profile.items()
    }
    ber_eligible = {
        thr: ber
        for thr, ber in mean_ber_by_threshold.items()
        if mean_tpr_by_threshold[thr] > 0.0
    }
    ber_pool = ber_eligible if ber_eligible else mean_ber_by_threshold
    best_thresholds["ber"] = min(ber_pool, key=ber_pool.get)

    profiles = {
        pid: _profile_result_at_best(
            pid,
            best_thresholds[pid],
            fold_probas,
            fold_y_val,
            beta=THRESHOLD_PROFILES[pid].beta,
            objective=THRESHOLD_PROFILES[pid].objective,
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
            for pid in fbeta_profile_ids
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
    cv_protocol: str = "repeated_stratified_5x2",
    decay_lambda: float = 0.0,
    decay_lambda_search: dict | None = None,
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
        default_profile = profile_map.get("ber") or profile_map.get("f4") or profile_map.get("f2")

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
        "cv_protocol": cv_protocol,
        "classifier_threshold": best_threshold,
        "grid_search_best_params": best_params,
        "decay_lambda": float(decay_lambda),
        "cv_summary": json_safe(summary_out),
        "cv_fold_results": fold_results.to_dict(orient="records"),
        "aggregated_top_configs": aggregated.head(10).to_dict(orient="records"),
        "frozen_config": frozen_config(),
        "tuned_at": datetime.now(timezone.utc).isoformat(),
    }
    if decay_lambda_search is not None:
        payload["decay_lambda_search"] = json_safe(decay_lambda_search)
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
            per_thresh = threshold_result.get("per_threshold_mean_fbeta_f4")
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
    cv=None,
    verbose: int = GRID_SEARCH_VERBOSE,
) -> tuple[GridSearchCV, int, int, int]:
    param_grid = spec.make_param_grid()
    cv = cv or make_repeated_stratified_cv()
    n_candidates, n_splits, total_fits = grid_search_workload(param_grid, cv, X, y)
    pipeline = spec.build_pipeline()
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=CV_SCORING,
        refit=False,
        n_jobs=-1,
        verbose=verbose,
        error_score="raise",
    )
    return search, n_candidates, n_splits, total_fits
