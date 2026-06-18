"""Model explainability: global importance and holdout wafer-level breakdowns."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier

from secom.costs import DEFAULT_PROFILE_ID, resolve_threshold_profiles
from secom.dashboard.data import model_info
from secom.metrics import predict_with_threshold
from secom.pipelines import (
    BENCHMARK_MODEL_IDS,
    ID_COL,
    RANDOM_SEED,
    TARGET_COL,
    TIMESTAMP_COL,
    WEIGHTING_MODEL_IDS,
    feature_columns,
    load_mart,
    split_train_test,
    time_decay_weights,
)
from secom.utils import fitted_base_classifier, load_tuned_blocked_params
from secom.tuning.registry import build_tuned_pipeline, fit_pipeline_weighted

GLOBAL_TOP_N = 15
LOCAL_TOP_N = 5
SHAP_BACKGROUND_ROWS = 200
PERMUTATION_SAMPLE_ROWS = 120


@dataclass(frozen=True)
class HoldoutSplit:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_cols: list[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass(frozen=True)
class WaferExplanation:
    observation_id: object
    actual_label: int
    predicted_label: int
    fail_probability: float
    threshold: float
    local_df: pd.DataFrame
    method: str


@st.cache_data(show_spinner=False)
def load_holdout_split() -> HoldoutSplit:
    df = load_mart()
    cols = feature_columns(df)
    train_df, test_df = split_train_test(df)
    X_train = train_df[cols]
    X_test = test_df[cols]
    y_train = train_df[TARGET_COL].astype(int)
    y_test = test_df[TARGET_COL].astype(int)
    return HoldoutSplit(
        train_df=train_df,
        test_df=test_df,
        feature_cols=cols,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


@st.cache_resource(show_spinner="Fitting pipeline on train split…")
def fit_holdout_pipeline(model_id: str):
    if model_id not in BENCHMARK_MODEL_IDS:
        raise ValueError(f"Unknown model_id: {model_id}")
    split = load_holdout_split()
    tuned = load_tuned_blocked_params(model_id)
    pipeline = build_tuned_pipeline(model_id, tuned)
    decay_lambda = (
        float(tuned.get("decay_lambda", 0.0))
        if model_id in WEIGHTING_MODEL_IDS
        else 0.0
    )
    weights = None
    if decay_lambda:
        train_ts = split.train_df[TIMESTAMP_COL].reset_index(drop=True)
        weights = time_decay_weights(train_ts, decay_lambda)
    pipeline, _ = fit_pipeline_weighted(
        pipeline, split.X_train, split.y_train, weights
    )
    return pipeline, tuned


def scaled_matrix(pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Transform to scaled feature space; return (matrix, feature_names)."""
    preprocess = pipeline.named_steps["preprocess"]
    scale = pipeline.named_steps["scale"]
    X_pre = preprocess.transform(X)
    X_scaled = scale.transform(X_pre)
    names = scale.get_feature_names_out()
    return np.asarray(X_scaled, dtype=float), np.asarray(names, dtype=object)


def _deploy_threshold(tuned: dict) -> float:
    profiles = resolve_threshold_profiles(tuned)
    return float(profiles[DEFAULT_PROFILE_ID])


def global_importance_linear(pipeline) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top |coef| table and signed coef subset for chart."""
    split = load_holdout_split()
    _, names = scaled_matrix(pipeline, split.X_train.iloc[:1])
    coefs = fitted_base_classifier(pipeline).coef_.ravel()
    full = pd.DataFrame({"feature": names, "coefficient": coefs})
    full["abs_coefficient"] = np.abs(full["coefficient"])
    top = full.nlargest(GLOBAL_TOP_N, "abs_coefficient").copy()
    top["importance"] = top["abs_coefficient"]
    pos = full.loc[full["coefficient"] > 0].nlargest(8, "coefficient")
    neg = full.loc[full["coefficient"] < 0].nsmallest(8, "coefficient")
    signed = pd.concat([pos, neg], ignore_index=True)
    return top, signed


def global_importance_shap(pipeline, model_id: str) -> pd.DataFrame:
    import shap

    split = load_holdout_split()
    X_bg, names = scaled_matrix(pipeline, split.X_train)
    if len(X_bg) > SHAP_BACKGROUND_ROWS:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(len(X_bg), size=SHAP_BACKGROUND_ROWS, replace=False)
        X_bg = X_bg[idx]

    estimator = fitted_base_classifier(pipeline)
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_bg)
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[1])
    elif np.asarray(shap_values).ndim == 3:
        values = np.asarray(shap_values)[:, :, 1]
    else:
        values = np.asarray(shap_values)

    mean_abs = np.mean(np.abs(values), axis=0)
    full = pd.DataFrame({"feature": names, "importance": mean_abs})
    return full.nlargest(GLOBAL_TOP_N, "importance")


def global_importance_knn(pipeline) -> pd.DataFrame:
    split = load_holdout_split()
    n = min(PERMUTATION_SAMPLE_ROWS, len(split.X_test))
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(split.X_test), size=n, replace=False)
    X_sample = split.X_test.iloc[idx]
    y_sample = split.y_test.iloc[idx]

    X_scaled, names = scaled_matrix(pipeline, X_sample)
    X_df = pd.DataFrame(X_scaled, columns=list(names))
    estimator = fitted_base_classifier(pipeline)
    result = permutation_importance(
        estimator,
        X_df,
        y_sample,
        scoring="average_precision",
        n_repeats=8,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    full = pd.DataFrame(
        {
            "feature": names,
            "importance": result.importances_mean,
        }
    )
    full = full.dropna(subset=["importance"])
    full = full.loc[full["importance"] != 0]
    if full.empty:
        return pd.DataFrame(columns=["feature", "importance"])
    return full.nlargest(GLOBAL_TOP_N, "importance")


def global_importance(model_id: str, pipeline) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    """
    Returns (top_df, optional_signed_coef_df, method_caption).
    """
    kind = model_info(model_id).explainability
    if kind == "linear":
        top, signed = global_importance_linear(pipeline)
        return (
            top,
            signed,
            "Global view uses elastic-net coefficients on scaled features "
            "(underlying logistic inside shared pipeline calibration).",
        )
    if kind == "tree":
        top = global_importance_shap(pipeline, model_id)
        return top, None, "Global view uses mean |SHAP| from TreeExplainer on a train subsample."
    if kind == "knn":
        top = global_importance_knn(pipeline)
        return (
            top,
            None,
            "k-NN has no tree SHAP; global ranking uses permutation importance on a "
            "holdout subsample.",
        )
    raise ValueError(model_id)


def _local_linear(pipeline, row_scaled: np.ndarray, names: np.ndarray) -> pd.DataFrame:
    coefs = fitted_base_classifier(pipeline).coef_.ravel()
    contrib = coefs * row_scaled
    df = pd.DataFrame(
        {"feature": names, "contribution": contrib, "coefficient": coefs}
    )
    df["abs_contribution"] = np.abs(df["contribution"])
    return df.nlargest(LOCAL_TOP_N, "abs_contribution")


def _shap_positive_class_values(shap_output: object, n_features: int) -> np.ndarray:
    """Extract positive-class SHAP vector length n_features."""
    if isinstance(shap_output, list):
        return np.asarray(shap_output[1]).reshape(-1)[:n_features]
    arr = np.asarray(shap_output)
    if arr.ndim == 3:
        return arr[0, :, 1]
    if arr.ndim == 2:
        if arr.shape[0] == 1 and arr.shape[1] == n_features:
            return arr[0, :]
        if arr.shape[1] >= 2 and arr.shape[0] == n_features:
            return arr[:, 1]
    return arr.reshape(-1)[:n_features]


def _local_shap(pipeline, row_scaled: np.ndarray, names: np.ndarray) -> pd.DataFrame:
    import shap

    estimator = fitted_base_classifier(pipeline)
    explainer = shap.TreeExplainer(estimator)
    sv = explainer.shap_values(row_scaled.reshape(1, -1))
    values = _shap_positive_class_values(sv, len(names))
    df = pd.DataFrame({"feature": names, "contribution": values})
    df["abs_contribution"] = np.abs(df["contribution"])
    return df.nlargest(LOCAL_TOP_N, "abs_contribution")


def _local_knn(
    pipeline,
    row_scaled: np.ndarray,
    names: np.ndarray,
    y_train: pd.Series,
    X_train_scaled: np.ndarray,
) -> pd.DataFrame:
    estimator = fitted_base_classifier(pipeline)
    if not isinstance(estimator, KNeighborsClassifier):
        raise TypeError("Expected KNeighborsClassifier")
    n_neighbors = int(estimator.n_neighbors)
    distances, indices = estimator.kneighbors(
        row_scaled.reshape(1, -1), n_neighbors=n_neighbors
    )
    dist = distances[0]
    idx = indices[0]
    neighbor_labels = y_train.iloc[idx].to_numpy(dtype=float)
    weights = 1.0 / (dist + 1e-12) if estimator.weights == "distance" else np.ones_like(dist)
    fail_rate = float(np.average(neighbor_labels, weights=weights))

    diffs = row_scaled - X_train_scaled[idx].mean(axis=0)
    df = pd.DataFrame(
        {
            "feature": names,
            "contribution": diffs,
            "neighbor_fail_rate": fail_rate,
        }
    )
    df["abs_contribution"] = np.abs(df["contribution"])
    top = df.nlargest(LOCAL_TOP_N, "abs_contribution")
    top["contribution"] = top["contribution"]  # signed distance from neighbor centroid
    return top


def wafer_explanation(
    model_id: str,
    pipeline,
    tuned: dict,
    observation_id: object,
) -> WaferExplanation | None:
    split = load_holdout_split()
    test_df = split.test_df
    if observation_id not in test_df[ID_COL].values:
        return None

    mask = test_df[ID_COL] == observation_id
    row = test_df.loc[mask].iloc[0]
    X_row = split.X_test.loc[mask]
    y_true = int(row[TARGET_COL])

    fail_proba = float(pipeline.predict_proba(X_row)[0, 1])
    threshold = _deploy_threshold(tuned)
    y_pred = int(predict_with_threshold(np.array([fail_proba]), threshold)[0])

    X_scaled, names = scaled_matrix(pipeline, X_row)
    row_scaled = X_scaled[0]

    kind = model_info(model_id).explainability
    if kind == "linear":
        local_df = _local_linear(pipeline, row_scaled, names)
        method = "Local: coefficient × scaled feature value."
    elif kind == "tree":
        local_df = _local_shap(pipeline, row_scaled, names)
        method = "Local: SHAP values for this wafer (TreeExplainer)."
    elif kind == "knn":
        X_train_scaled, _ = scaled_matrix(pipeline, split.X_train)
        local_df = _local_knn(
            pipeline, row_scaled, names, split.y_train, X_train_scaled
        )
        method = (
            "Local: features farthest from k-NN neighbor centroid in scaled space "
            f"(neighbor fail-rate ≈ {local_df['neighbor_fail_rate'].iloc[0]:.1%})."
        )
    else:
        raise ValueError(model_id)

    return WaferExplanation(
        observation_id=observation_id,
        actual_label=y_true,
        predicted_label=y_pred,
        fail_probability=fail_proba,
        threshold=threshold,
        local_df=local_df,
        method=method,
    )


@st.cache_data(show_spinner=False)
def holdout_wafer_ids() -> list:
    split = load_holdout_split()
    ids = split.test_df[ID_COL].tolist()
    return sorted(ids, key=str)


@st.cache_data(show_spinner="Computing global feature importance…")
def cached_global_importance(model_id: str) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    pipeline, _tuned = fit_holdout_pipeline(model_id)
    return global_importance(model_id, pipeline)


@st.cache_data(show_spinner="Building wafer explanation…")
def cached_wafer_explanation(model_id: str, observation_id: object) -> WaferExplanation | None:
    pipeline, tuned = fit_holdout_pipeline(model_id)
    return wafer_explanation(model_id, pipeline, tuned, observation_id)
