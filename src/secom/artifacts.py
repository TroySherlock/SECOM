"""Extract holdout-fit pipeline reporting artifacts for the dashboard."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from feature_engine.selection import SmartCorrelatedSelection
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from secom.hub_interactions import (
    HSICSelectHubBlock,
    LinearSelectT2HubBlock,
    extract_hub_interaction_info,
    sensor_value_columns,
)
from secom.pipelines import (
    BENCHMARK_MODEL_IDS,
    N_SENSORS,
    CORRELATED_SELECTION_CRITERION,
    CORRELATED_SELECTION_METHOD,
    CORRELATED_SELECTION_THRESHOLD,
    PIPELINE_ARTIFACTS_PATH,
    RANDOM_SEED,
    TEST_SIZE,
)
from secom.utils import json_safe

# Reference models for the reduction widget + shared cluster example. Both are
# RF-selection front-ends so the stage breakdown extracts cleanly.
REFERENCE_MODELS = {"linear": "rfsel_enet", "topk": "rfsel_rf"}


def _sensor_branch_pipeline(preprocess: ColumnTransformer) -> Pipeline:
    fitted = getattr(preprocess, "transformers_", None) or preprocess.transformers
    for name, trans, _ in fitted:
        if name == "sensor_branch":
            return trans
    raise KeyError("preprocess has no sensor_branch transformer")


def _sensor_branch_columns(sensor_pipe: Pipeline) -> list[str]:
    """Columns the fitted sensor branch consumed (raw, plus rolling-Z for extrap).

    Read from the imputer's ``feature_names_in_`` so it reflects whatever the
    branch's ``make_column_selector`` resolved at fit time, rather than assuming
    a raw-only sensor set.
    """
    return list(sensor_pipe.named_steps["impute"].feature_names_in_)


def _model_family(model_id: str) -> str:
    if model_id.endswith("_rf"):
        return "topk"
    if model_id.endswith("_enet") or model_id.endswith("_bayes"):
        return "linear"
    return "unknown"


def _cluster_drop_counts(cluster: Pipeline, X_imp: pd.DataFrame) -> dict[str, int]:
    """Drop counts from each fitted cluster sub-step."""
    key_map = {
        "drop_constant": "drop_constant",
        "drop_duplicates": "drop_duplicate",
        "smart_corr": "drop_correlated",
    }
    counts: dict[str, int] = {}
    X_cur = X_imp
    for step_name, step in cluster.named_steps.items():
        n_before = X_cur.shape[1]
        X_cur = step.transform(X_cur)
        key = key_map.get(step_name)
        if key:
            counts[key] = n_before - X_cur.shape[1]
    return counts


def _extract_rf_selection(select: SelectFromModel, cluster_cols: list[str]) -> dict[str, Any]:
    support = select.get_support()
    selected = [c for c, keep in zip(cluster_cols, support) if keep]
    estimator = select.estimator_
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        ranked: list[dict[str, float]] = []
    else:
        pairs = sorted(
            zip(cluster_cols, importances),
            key=lambda x: x[1],
            reverse=True,
        )
        ranked = [
            {"feature": str(f), "importance": float(imp)}
            for f, imp in pairs[:30]
        ]
    max_features = select.max_features
    top_k = int(max_features) if isinstance(max_features, int) else len(selected)
    return {
        "top_k": top_k,
        "selected_count": len(selected),
        "selected_features": [str(f) for f in selected],
        "importances": ranked,
    }


def _spearman_cluster_example(
    smart_corr: SmartCorrelatedSelection,
    X_pre_correlation: pd.DataFrame,
) -> dict[str, Any] | None:
    """Pick a correlated group (≥3 members) and compute Spearman ρ on training data."""
    sets = getattr(smart_corr, "correlated_feature_sets_", None)
    if not sets:
        return None

    chosen = None
    for group in sets:
        members = sorted(str(f) for f in group)
        available = [m for m in members if m in X_pre_correlation.columns]
        if len(available) >= 3:
            chosen = available[:6]
            break
    if chosen is None:
        return None

    sub = X_pre_correlation[chosen]
    corr = sub.corr(method="spearman").to_numpy()
    return {
        "representative": chosen[0],
        "members": chosen,
        "correlations": corr.tolist(),
    }


def _dataframe_before_smart_corr(cluster: Pipeline, X_imp: pd.DataFrame) -> pd.DataFrame:
    """Sensor matrix after constant/duplicate drops, before correlated selection."""
    X_cur = X_imp
    for step_name, step in cluster.named_steps.items():
        if step_name == "smart_corr":
            break
        X_cur = step.transform(X_cur)
    return X_cur


def extract_model_artifacts(
    pipeline: Pipeline,
    model_id: str,
    X_train: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize stages and family-specific details from a fitted pipeline."""
    preprocess: ColumnTransformer = pipeline.named_steps["preprocess"]
    scale = pipeline.named_steps["scale"]
    sensor_pipe = _sensor_branch_pipeline(preprocess)
    branch_cols = _sensor_branch_columns(sensor_pipe)

    # Raw physical sensors (mart) vs. all sensor-derived inputs the branch sees
    # (raw + rolling-Z for the extrapolation track).
    mart_sensors = len(sensor_value_columns(X_train.columns))
    auxiliary_features = len(X_train.columns) - len(branch_cols)
    stg_sensors = int(N_SENSORS)
    stages: dict[str, int] = {
        "stg_sensors": stg_sensors,
        "mart_sensors": mart_sensors,
        "dbt_dropped_sensors": max(0, stg_sensors - mart_sensors),
        "auxiliary_features": auxiliary_features,
    }

    X_sensors = X_train[branch_cols]
    impute = sensor_pipe.named_steps["impute"]
    X_imp = pd.DataFrame(
        impute.transform(X_sensors),
        columns=branch_cols,
        index=X_sensors.index,
    )
    stages["after_impute"] = len(X_imp.columns)

    cluster = sensor_pipe.named_steps["cluster"]
    X_clust = cluster.transform(X_imp)
    stages["after_cluster"] = len(X_clust.columns)
    stages.update(_cluster_drop_counts(cluster, X_imp))

    family = _model_family(model_id)
    rf_selection = None
    hub_interactions = None

    block = sensor_pipe.named_steps.get("front_end")
    cluster_cols = list(X_clust.columns)
    if isinstance(block, LinearSelectT2HubBlock):
        rf_selection = _extract_rf_selection(block.select_, cluster_cols)
        stages["after_selection"] = rf_selection["selected_count"]
        hub_interactions = extract_hub_interaction_info(block)
        n_interact = int(hub_interactions.get("n_interaction_features", 0))
        stages["after_hub_interactions"] = (
            int(rf_selection["selected_count"]) + 1 + n_interact
        )
    elif isinstance(block, HSICSelectHubBlock):
        selected = list(getattr(block, "selected_columns_", []))
        hubs = [str(h) for h in getattr(block, "hubs_", [])]
        n_interact = int(getattr(block, "n_interaction_features_", 0))
        stages["after_selection"] = len(selected)
        stages["after_hub_interactions"] = len(selected) + 1 + n_interact
        hub_interactions = {
            "top_k_requested": int(block.top_k),
            "n_hubs_requested": int(block.n_hubs),
            "n_hubs_selected": len(hubs),
            "n_interaction_features": n_interact,
            "hub_sensors": hubs,
            "selected_features": [str(c) for c in selected],
        }

    stages["after_preprocess"] = len(preprocess.get_feature_names_out())
    stages["classifier_input"] = len(scale.get_feature_names_out())

    return {
        "family": family,
        "stages": stages,
        "rf_selection": rf_selection,
        "hub_interactions": hub_interactions,
    }


def extract_shared_artifacts(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, Any]:
    """Cluster config and Spearman example from reference linear pipeline."""
    preprocess: ColumnTransformer = pipeline.named_steps["preprocess"]
    sensor_pipe = _sensor_branch_pipeline(preprocess)
    branch_cols = _sensor_branch_columns(sensor_pipe)
    X_sensors = X_train[branch_cols]

    impute = sensor_pipe.named_steps["impute"]
    X_imp = pd.DataFrame(
        impute.transform(X_sensors),
        columns=branch_cols,
        index=X_sensors.index,
    )
    cluster = sensor_pipe.named_steps["cluster"]
    smart_corr = cluster.named_steps["smart_corr"]
    X_pre_corr = _dataframe_before_smart_corr(cluster, X_imp)
    spearman_example = _spearman_cluster_example(smart_corr, X_pre_corr)

    return {
        "cluster_config": {
            "library": "feature_engine",
            "method": CORRELATED_SELECTION_METHOD,
            "threshold": float(CORRELATED_SELECTION_THRESHOLD),
            "selection_method": CORRELATED_SELECTION_CRITERION,
        },
        "spearman_cluster_example": spearman_example,
    }


def collect_holdout_artifacts(
    pipelines: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    holdout_split: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build full artifacts payload from holdout-fit pipelines."""
    models: dict[str, dict[str, Any]] = {}
    for model_id, pipeline in pipelines.items():
        if model_id not in BENCHMARK_MODEL_IDS:
            continue
        models[model_id] = extract_model_artifacts(pipeline, model_id, X_train)

    ref_linear = pipelines[REFERENCE_MODELS["linear"]]
    shared = extract_shared_artifacts(ref_linear, X_train, y_train)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "holdout_split": holdout_split or {
            "test_size": float(TEST_SIZE),
            "split_mode": "temporal",
        },
        "reference_models": dict(REFERENCE_MODELS),
        "shared": shared,
        "models": models,
    }


def save_pipeline_artifacts(
    payload: dict[str, Any],
    path: Path = PIPELINE_ARTIFACTS_PATH,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    return payload
