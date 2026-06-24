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
    BER_BAND_TOLERANCE,
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
    THRESHOLD_GRID,
    frozen_config,
    make_repeated_stratified_cv,
    time_decay_weights,
)
from secom.pipelines import (
    BAYES_C_GRID,
    BAYES_L1_RATIO_GRID,
    BAYES_N_HUBS,
    BAYES_PLS_COMPONENTS,
    BAYES_POS_WEIGHT_GRID,
    BAYES_TOP_K,
    C_GRID,
    CORRELATED_SELECTION_THRESHOLD,
    CORRELATED_SELECTION_THRESHOLD_GRID,
    L1_RATIO_GRID,
    MODEL_CELLS,
    MODEL_IDS,
    N_HUBS_DEFAULT,
    N_HUBS_GRID,
    PLS_N_COMPONENTS_DEFAULT,
    PLS_N_COMPONENTS_GRID,
    RF_MAX_DEPTH,
    RF_MAX_DEPTH_GRID,
    TOP_K_DEFAULT,
    TOP_K_GRID,
    build_model_pipeline,
    is_bayesian as _is_bayesian_id,
)
from secom.utils import json_safe, tuned_params_path

from sklearn.metrics import average_precision_score

# --- Unified pipeline param paths --------------------------------------------
SELECT_TOP_K_PARAM = "preprocess__sensor_branch__front_end__top_k"
SELECT_N_HUBS_PARAM = "preprocess__sensor_branch__front_end__n_hubs"
PLS_N_COMPONENTS_PARAM = "preprocess__sensor_branch__front_end__n_components"
SMART_CORR_THRESHOLD_PARAM = (
    "preprocess__sensor_branch__cluster__smart_corr__threshold"
)
CLF_C_PARAM = "classifier__estimator__C"
CLF_L1_PARAM = "classifier__estimator__l1_ratio"
CLF_MAX_DEPTH_PARAM = "classifier__estimator__max_depth"
CLF_POS_WEIGHT_PARAM = "classifier__estimator__pos_weight"


def _sample_weight_kwargs(timestamps, train_idx, decay_lambda: float) -> dict:
    """Fit kwargs for time-decay weighting; empty when timestamps is None."""
    if timestamps is None:
        return {}
    w = time_decay_weights(timestamps.iloc[train_idx], decay_lambda)
    return {"classifier__sample_weight": w}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    build_pipeline: Callable[[], Pipeline]
    make_param_grid: Callable[[], dict]
    param_renames: dict[str, str]
    groupby_cols: list[str]
    best_defaults: dict[str, object]
    build_grid_search_best_params: Callable[[dict], dict]
    n_jobs: int = -1
    error_score: object = "raise"


# --- Front-end grid fragments (selection vs PLS) -----------------------------
def _selection_front_end_fragment(*, bayes: bool) -> dict:
    """Grid / renames / defaults / best-params for HSIC & RF-selection front-ends."""
    if bayes:
        grid = {
            SELECT_TOP_K_PARAM: [int(BAYES_TOP_K)],
            SELECT_N_HUBS_PARAM: [int(BAYES_N_HUBS)],
            SMART_CORR_THRESHOLD_PARAM: [float(CORRELATED_SELECTION_THRESHOLD)],
        }
    else:
        grid = {
            SELECT_TOP_K_PARAM: [int(k) for k in TOP_K_GRID],
            SELECT_N_HUBS_PARAM: [int(k) for k in N_HUBS_GRID],
            SMART_CORR_THRESHOLD_PARAM: [
                float(t) for t in CORRELATED_SELECTION_THRESHOLD_GRID
            ],
        }
    return {
        "grid": grid,
        "renames": {
            f"param_{SELECT_TOP_K_PARAM}": "top_k",
            f"param_{SELECT_N_HUBS_PARAM}": "n_hubs",
            f"param_{SMART_CORR_THRESHOLD_PARAM}": "corr_threshold",
        },
        "groupby": ["top_k", "n_hubs", "corr_threshold"],
        "defaults": {
            "top_k": int(BAYES_TOP_K if bayes else TOP_K_DEFAULT),
            "n_hubs": int(BAYES_N_HUBS if bayes else N_HUBS_DEFAULT),
            "corr_threshold": float(CORRELATED_SELECTION_THRESHOLD),
        },
        "best": lambda cv: {
            SELECT_TOP_K_PARAM: int(cv["best_top_k"]),
            SELECT_N_HUBS_PARAM: int(cv["best_n_hubs"]),
            SMART_CORR_THRESHOLD_PARAM: float(
                cv.get("best_corr_threshold", CORRELATED_SELECTION_THRESHOLD)
            ),
        },
    }


def _pls_front_end_fragment(*, bayes: bool) -> dict:
    if bayes:
        grid = {
            PLS_N_COMPONENTS_PARAM: [int(BAYES_PLS_COMPONENTS)],
            SMART_CORR_THRESHOLD_PARAM: [float(CORRELATED_SELECTION_THRESHOLD)],
        }
    else:
        grid = {
            PLS_N_COMPONENTS_PARAM: [int(k) for k in PLS_N_COMPONENTS_GRID],
            SMART_CORR_THRESHOLD_PARAM: [
                float(t) for t in CORRELATED_SELECTION_THRESHOLD_GRID
            ],
        }
    return {
        "grid": grid,
        "renames": {
            f"param_{PLS_N_COMPONENTS_PARAM}": "pls_n_components",
            f"param_{SMART_CORR_THRESHOLD_PARAM}": "corr_threshold",
        },
        "groupby": ["pls_n_components", "corr_threshold"],
        "defaults": {
            "pls_n_components": int(BAYES_PLS_COMPONENTS if bayes else PLS_N_COMPONENTS_DEFAULT),
            "corr_threshold": float(CORRELATED_SELECTION_THRESHOLD),
        },
        "best": lambda cv: {
            PLS_N_COMPONENTS_PARAM: int(cv["best_pls_n_components"]),
            SMART_CORR_THRESHOLD_PARAM: float(
                cv.get("best_corr_threshold", CORRELATED_SELECTION_THRESHOLD)
            ),
        },
    }


def _front_end_fragment(front_end: str, *, bayes: bool) -> dict:
    if front_end == "pls":
        return _pls_front_end_fragment(bayes=bayes)
    return _selection_front_end_fragment(bayes=bayes)


# --- Classifier-head grid fragments ------------------------------------------
def _enet_head_fragment() -> dict:
    return {
        "grid": {
            CLF_C_PARAM: [float(c) for c in C_GRID],
            CLF_L1_PARAM: [float(r) for r in L1_RATIO_GRID],
        },
        "renames": {
            f"param_{CLF_C_PARAM}": "c",
            f"param_{CLF_L1_PARAM}": "l1_ratio",
        },
        "groupby": ["c", "l1_ratio"],
        "defaults": {"c": float(C_GRID[0]), "l1_ratio": float(L1_RATIO_GRID[0])},
        "best": lambda cv: {
            CLF_C_PARAM: float(cv["best_c"]),
            CLF_L1_PARAM: float(cv["best_l1_ratio"]),
        },
    }


def _rf_head_fragment() -> dict:
    return {
        "grid": {CLF_MAX_DEPTH_PARAM: [int(d) for d in RF_MAX_DEPTH_GRID]},
        "renames": {f"param_{CLF_MAX_DEPTH_PARAM}": "max_depth"},
        "groupby": ["max_depth"],
        "defaults": {"max_depth": int(RF_MAX_DEPTH)},
        "best": lambda cv: {CLF_MAX_DEPTH_PARAM: int(cv["best_max_depth"])},
    }


def _bayes_head_fragment() -> dict:
    return {
        "grid": {
            CLF_C_PARAM: [float(c) for c in BAYES_C_GRID],
            CLF_L1_PARAM: [float(r) for r in BAYES_L1_RATIO_GRID],
            CLF_POS_WEIGHT_PARAM: [float(w) for w in BAYES_POS_WEIGHT_GRID],
        },
        "renames": {
            f"param_{CLF_C_PARAM}": "c",
            f"param_{CLF_L1_PARAM}": "l1_ratio",
            f"param_{CLF_POS_WEIGHT_PARAM}": "pos_weight",
        },
        "groupby": ["c", "l1_ratio", "pos_weight"],
        "defaults": {
            "c": float(BAYES_C_GRID[0]),
            "l1_ratio": float(BAYES_L1_RATIO_GRID[0]),
            "pos_weight": float(BAYES_POS_WEIGHT_GRID[0]),
        },
        "best": lambda cv: {
            CLF_C_PARAM: float(cv["best_c"]),
            CLF_L1_PARAM: float(cv["best_l1_ratio"]),
            CLF_POS_WEIGHT_PARAM: float(cv["best_pos_weight"]),
        },
    }


_HEAD_FRAGMENTS = {
    "enet": _enet_head_fragment,
    "rf": _rf_head_fragment,
    "bayes": _bayes_head_fragment,
}


def _make_spec(model_id: str) -> ModelSpec:
    front_end, classifier_kind = MODEL_CELLS[model_id]
    bayes = classifier_kind == "bayes"
    fe = _front_end_fragment(front_end, bayes=bayes)
    head = _HEAD_FRAGMENTS[classifier_kind]()

    def build_pipeline() -> Pipeline:
        return build_model_pipeline(front_end, classifier_kind)

    def make_param_grid() -> dict:
        return {**fe["grid"], **head["grid"]}

    def best_params(cv_summary: dict) -> dict:
        return {**fe["best"](cv_summary), **head["best"](cv_summary)}

    return ModelSpec(
        model_id=model_id,
        build_pipeline=build_pipeline,
        make_param_grid=make_param_grid,
        param_renames={**fe["renames"], **head["renames"]},
        groupby_cols=[*fe["groupby"], *head["groupby"]],
        best_defaults={**fe["defaults"], **head["defaults"]},
        build_grid_search_best_params=best_params,
        n_jobs=1 if bayes else -1,
        # HSIC folds can go singular and Bayesian ADVI can blow up; nan keeps one
        # bad fold from aborting the whole search.
        error_score=(np.nan if (bayes or front_end == "hsic") else "raise"),
    )


MODEL_SPECS: dict[str, ModelSpec] = {mid: _make_spec(mid) for mid in MODEL_IDS}

# Both protocols run every cell; tuning loops (id, track).
TRACKS = ("interpolation", "extrapolation")
ALL_MODEL_IDS: tuple[str, ...] = tuple(MODEL_IDS)


def is_bayesian(model_id: str) -> bool:
    """True for the Bayesian-head cells (still plain sklearn pipelines now)."""
    return bool(_is_bayesian_id(model_id))


def model_ids_for_track(track: str | None = None) -> list[str]:
    """All 9 ids run on every protocol, so track only narrows the CLI label."""
    return list(MODEL_IDS)


def _resolved_classifier_threshold(tuned_payload: dict) -> float:
    raw = tuned_payload.get("classifier_threshold", 0.5)
    if isinstance(raw, str):
        if raw == "default_0.5":
            return 0.5
        return float(raw)
    return float(raw)


def resolve_grid_search_best_params(model_id: str, tuned_payload: dict) -> dict:
    """Tuned grid-search params filtered to the current pipeline param names.

    Falls back to rebuilding them from ``cv_summary`` (``best_*`` keys) when the
    payload omits the explicit param dict.
    """
    spec = MODEL_SPECS[model_id]
    raw = dict(tuned_payload.get("grid_search_best_params") or {})
    if not raw:
        cv_summary = tuned_payload.get("cv_summary") or {}
        try:
            raw = spec.build_grid_search_best_params(cv_summary)
        except KeyError:
            raw = {}
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

    mean_ber_by_threshold: dict[float, float] = {}
    mean_tpr_by_threshold: dict[float, float] = {}

    for threshold in threshold_grid:
        _, fold_bers, fold_tprs = _fold_metrics_at_threshold(
            fold_probas, fold_y_val, threshold, beta=1.0
        )
        mean_ber_by_threshold[threshold] = float(np.mean(fold_bers))
        mean_tpr_by_threshold[threshold] = float(np.mean(fold_tprs))

    # BER-tolerance band: minimise mean BER on the eligible (TPR>0) pool, then
    # take the high/low ends of the band within BER_BAND_TOLERANCE of that min.
    ber_eligible = {
        thr: ber
        for thr, ber in mean_ber_by_threshold.items()
        if mean_tpr_by_threshold[thr] > 0.0
    }
    ber_pool = ber_eligible if ber_eligible else mean_ber_by_threshold
    ber_min_thr = min(ber_pool, key=ber_pool.get)
    ber_min = ber_pool[ber_min_thr]
    band = [
        thr for thr, ber in ber_pool.items() if ber <= ber_min + BER_BAND_TOLERANCE
    ]
    if not band:
        band = [ber_min_thr]
    best_thresholds: dict[str, float] = {
        "conservative": float(max(band)),
        "ber": float(ber_min_thr),
        "aggressive": float(min(band)),
    }

    profiles = {
        pid: _profile_result_at_best(
            pid,
            best_thresholds[pid],
            fold_probas,
            fold_y_val,
            beta=None,
            objective="ber",
        )
        for pid in PROFILE_IDS
    }

    objective_curves = pd.DataFrame(
        {
            "threshold": threshold_grid,
            "mean_ber_percent": [mean_ber_by_threshold[t] for t in threshold_grid],
        }
    )

    default_prof = profiles[DEFAULT_PROFILE_ID]
    per_threshold_mean_ber = (
        objective_curves[["threshold", "mean_ber_percent"]]
        .sort_values("mean_ber_percent", ascending=True, kind="mergesort")
    )
    return {
        "profiles": profiles,
        "threshold_grid": threshold_grid,
        "objective_curves": objective_curves,
        "ber_band_tolerance": float(BER_BAND_TOLERANCE),
        "best_threshold": default_prof["best_threshold"],
        "mean_ber_percent": default_prof["mean_ber_percent"],
        "std_ber_percent": default_prof["std_ber_percent"],
        "per_threshold_mean_ber": per_threshold_mean_ber,
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
        default_profile = profile_map.get("ber")

    if default_profile is not None:
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
        per_thresh = threshold_result.get("per_threshold_mean_ber")
        payload["threshold_tuning"] = json_safe(
            {
                "metric": "ber_band",
                "default_profile": DEFAULT_PROFILE_ID,
                "ber_band_tolerance": threshold_result.get("ber_band_tolerance"),
                "best_threshold": best_threshold,
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
        n_jobs=spec.n_jobs,
        verbose=verbose,
        error_score=spec.error_score,
    )
    return search, n_candidates, n_splits, total_fits
