"""Model explainability: global importance and holdout wafer-level breakdowns."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from secom.costs import DEFAULT_PROFILE_ID, resolve_threshold_profiles
from secom.dashboard.data import model_info, report_entry
from secom.metrics import predict_with_threshold
from secom.pipelines import (
    BENCHMARK_MODEL_IDS,
    ID_COL,
    TARGET_COL,
    TIMESTAMP_COL,
    WEIGHTING_MODEL_IDS,
    feature_columns,
    load_mart,
    split_train_test,
    split_train_test_random,
    time_decay_weights,
)
from secom.reporting import _bayes_mean_coef, scaled_matrix
from secom.utils import (
    fitted_base_classifier,
    load_tuned_blocked_params,
    load_tuned_params,
)
from secom.tuning.registry import build_tuned_pipeline, fit_pipeline_weighted

LOCAL_TOP_N = 5

DEFAULT_TRACK = "extrapolation"


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
def load_holdout_split(track: str = DEFAULT_TRACK) -> HoldoutSplit:
    df = load_mart()
    cols = feature_columns(df)
    if track == "interpolation":
        train_df, test_df = split_train_test_random(df)
    else:
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


def _tuned_for(model_id: str, track: str) -> dict:
    if track == "interpolation":
        return load_tuned_params(model_id)
    return load_tuned_blocked_params(model_id)


@st.cache_resource(show_spinner="Fitting pipeline on train split…")
def fit_holdout_pipeline(model_id: str, track: str = DEFAULT_TRACK):
    if model_id not in BENCHMARK_MODEL_IDS:
        raise ValueError(f"Unknown model_id: {model_id}")
    split = load_holdout_split(track)
    tuned = _tuned_for(model_id, track)
    pipeline = build_tuned_pipeline(model_id, tuned)
    decay_lambda = (
        float(tuned.get("decay_lambda", 0.0))
        if (track == "extrapolation" and model_id in WEIGHTING_MODEL_IDS)
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


def _deploy_threshold(tuned: dict) -> float:
    profiles = resolve_threshold_profiles(tuned)
    return float(profiles[DEFAULT_PROFILE_ID])


def _local_linear(row_scaled: np.ndarray, names: np.ndarray, coefs: np.ndarray) -> pd.DataFrame:
    contrib = np.asarray(coefs).ravel() * row_scaled
    df = pd.DataFrame(
        {"feature": names, "contribution": contrib, "coefficient": np.asarray(coefs).ravel()}
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


def wafer_explanation(
    model_id: str,
    pipeline,
    tuned: dict,
    observation_id: object,
    track: str = DEFAULT_TRACK,
) -> WaferExplanation | None:
    split = load_holdout_split(track)
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

    kind = model_info(model_id).explainability
    X_scaled, names = scaled_matrix(pipeline, X_row)
    row_scaled = X_scaled[0]

    if kind == "linear":
        local_df = _local_linear(row_scaled, names, fitted_base_classifier(pipeline).coef_.ravel())
        method = "Local: coefficient × scaled feature value."
    elif kind == "bayesian":
        local_df = _local_linear(row_scaled, names, _bayes_mean_coef(pipeline))
        method = "Local: posterior-mean coefficient × scaled feature value."
    elif kind == "tree":
        local_df = _local_shap(pipeline, row_scaled, names)
        method = "Local: SHAP values for this wafer (TreeExplainer)."
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
def holdout_wafer_ids(track: str = DEFAULT_TRACK) -> list:
    split = load_holdout_split(track)
    ids = split.test_df[ID_COL].tolist()
    return sorted(ids, key=str)


@st.cache_data(show_spinner=False)
def cached_global_importance(
    model_id: str, track: str = DEFAULT_TRACK
) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    """Read the frozen global-importance payload written by ``secom.benchmark``."""
    entry = report_entry(track, model_id)
    importance = (entry or {}).get("global_importance")
    if not importance:
        raise FileNotFoundError(
            f"No frozen global importance for {model_id!r} ({track}). "
            "Run: python -m secom.benchmark"
        )
    top = pd.DataFrame(importance.get("top") or [])
    signed_records = importance.get("signed")
    signed = pd.DataFrame(signed_records) if signed_records else None
    return top, signed, str(importance.get("caption", ""))


@st.cache_data(show_spinner="Building wafer explanation…")
def cached_wafer_explanation(
    model_id: str, observation_id: object, track: str = DEFAULT_TRACK
) -> WaferExplanation | None:
    pipeline, tuned = fit_holdout_pipeline(model_id, track)
    return wafer_explanation(model_id, pipeline, tuned, observation_id, track)
