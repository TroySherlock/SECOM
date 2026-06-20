"""Model explainability: global importance and holdout wafer-level breakdowns."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

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
    split_train_test_random,
    time_decay_weights,
)
from secom.utils import (
    fitted_base_classifier,
    load_tuned_blocked_params,
    load_tuned_params,
)
from secom.tuning.registry import build_tuned_pipeline, fit_pipeline_weighted

GLOBAL_TOP_N = 15
LOCAL_TOP_N = 5
SHAP_BACKGROUND_ROWS = 200

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


def _bayes_mean_coef(pipeline) -> np.ndarray:
    """Posterior-mean elastic-net coefficients averaged over the calibration copies."""
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "estimator_"):  # FixedThresholdClassifier
        classifier = classifier.estimator_
    calibrated = getattr(classifier, "calibrated_classifiers_", None)
    if calibrated:
        means = [c.estimator.coef_summary()["mean"].to_numpy() for c in calibrated]
        return np.mean(means, axis=0)
    return fitted_base_classifier(pipeline).coef_summary()["mean"].to_numpy()


def global_importance_linear(pipeline) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top |coef| table and signed coef subset for chart (elastic-net LR)."""
    split = load_holdout_split()
    _, names = scaled_matrix(pipeline, split.X_train.iloc[:1])
    coefs = fitted_base_classifier(pipeline).coef_.ravel()
    return _coef_tables(names, coefs)


def global_importance_bayesian(pipeline) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Posterior-mean coefficient table + signed subset for the Bayesian head."""
    split = load_holdout_split()
    _, names = scaled_matrix(pipeline, split.X_train.iloc[:1])
    coefs = _bayes_mean_coef(pipeline)
    return _coef_tables(names, coefs)


def _coef_tables(names, coefs) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = pd.DataFrame({"feature": names, "coefficient": np.asarray(coefs).ravel()})
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


def global_importance(model_id: str, pipeline) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    """Returns (top_df, optional_signed_coef_df, method_caption)."""
    kind = model_info(model_id).explainability
    if kind == "bayesian":
        top, signed = global_importance_bayesian(pipeline)
        return (
            top,
            signed,
            "Global view uses posterior-mean elastic-net coefficients (averaged over "
            "the calibration copies) on the scaled design (Laplace L1 + ridge L2 priors).",
        )
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
    raise ValueError(model_id)


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


@st.cache_data(show_spinner="Computing global feature importance…")
def cached_global_importance(
    model_id: str, track: str = DEFAULT_TRACK
) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    pipeline, _tuned = fit_holdout_pipeline(model_id, track)
    return global_importance(model_id, pipeline)


@st.cache_data(show_spinner="Building wafer explanation…")
def cached_wafer_explanation(
    model_id: str, observation_id: object, track: str = DEFAULT_TRACK
) -> WaferExplanation | None:
    pipeline, tuned = fit_holdout_pipeline(model_id, track)
    return wafer_explanation(model_id, pipeline, tuned, observation_id, track)
