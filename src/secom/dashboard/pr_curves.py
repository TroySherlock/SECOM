"""Cached CV OOF and holdout scores for PR curve charts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone

from secom.costs import resolve_threshold_profiles
from secom.cv import make_blocked_time_cv
from secom.dashboard.explainability import (
    DEFAULT_TRACK,
    fit_holdout_pipeline,
    load_holdout_split,
)
from secom.metrics import PRCurve, pr_curve_points, threshold_pr_point
from secom.pipelines import (
    BENCHMARK_MODEL_IDS,
    TIMESTAMP_COL,
    WEIGHTING_MODEL_IDS,
    make_repeated_stratified_cv,
    time_decay_weights,
)
from secom.tuning.registry import build_tuned_pipeline, fit_pipeline_weighted
from secom.utils import load_tuned_blocked_params, load_tuned_params


def _tuned_for(model_id: str, track: str) -> dict:
    """Track-appropriate tuned params: blocked for extrapolation, stratified otherwise."""
    if track == "extrapolation":
        return load_tuned_blocked_params(model_id)
    return load_tuned_params(model_id)


@st.cache_data(show_spinner="Computing CV out-of-fold scores…")
def collect_cv_oof_proba(
    model_id: str, track: str = DEFAULT_TRACK
) -> tuple[pd.Series, np.ndarray]:
    if model_id not in BENCHMARK_MODEL_IDS:
        raise ValueError(f"Unknown model_id: {model_id}")
    split = load_holdout_split(track)
    tuned = _tuned_for(model_id, track)
    pipeline = build_tuned_pipeline(model_id, tuned)
    if track == "extrapolation":
        cv = make_blocked_time_cv(split.train_df)
        decay_lambda = (
            float(tuned.get("decay_lambda", 0.0))
            if model_id in WEIGHTING_MODEL_IDS
            else 0.0
        )
    else:
        cv = make_repeated_stratified_cv()
        decay_lambda = 0.0
    train_ts = split.train_df[TIMESTAMP_COL].reset_index(drop=True)
    proba = np.full(len(split.y_train), np.nan)
    for train_idx, val_idx in cv.split(split.X_train, split.y_train):
        weights = (
            time_decay_weights(train_ts.iloc[train_idx], decay_lambda)
            if decay_lambda
            else None
        )
        fold_pipe, _ = fit_pipeline_weighted(
            clone(pipeline),
            split.X_train.iloc[train_idx],
            split.y_train.iloc[train_idx],
            weights,
        )
        proba[val_idx] = fold_pipe.predict_proba(split.X_train.iloc[val_idx])[:, 1]
    mask = ~np.isnan(proba)
    y_oof = split.y_train.iloc[mask].reset_index(drop=True)
    return y_oof, np.asarray(proba[mask], dtype=float)


@st.cache_data(show_spinner="Computing holdout scores…")
def collect_holdout_proba(
    model_id: str, track: str = DEFAULT_TRACK
) -> tuple[pd.Series, np.ndarray]:
    if model_id not in BENCHMARK_MODEL_IDS:
        raise ValueError(f"Unknown model_id: {model_id}")
    split = load_holdout_split(track)
    pipeline, _ = fit_holdout_pipeline(model_id, track)
    proba = pipeline.predict_proba(split.X_test)[:, 1]
    return split.y_test.reset_index(drop=True), np.asarray(proba, dtype=float)


@st.cache_data(show_spinner="Building PR curves…")
def load_pr_curves(
    model_id: str, track: str = DEFAULT_TRACK
) -> tuple[PRCurve | None, PRCurve | None, tuple[float, float] | None]:
    """CV OOF + holdout PR curves and a BER operating point on the chosen protocol
    (extrapolation: blocked CV + temporal holdout; interpolation: stratified CV +
    random holdout)."""
    y_cv, score_cv = collect_cv_oof_proba(model_id, track)
    y_ho, score_ho = collect_holdout_proba(model_id, track)
    cv_curve = pr_curve_points(y_cv, score_cv)
    ho_curve = pr_curve_points(y_ho, score_ho)
    ber_point: tuple[float, float] | None = None
    try:
        tuned = _tuned_for(model_id, track)
        profiles = resolve_threshold_profiles(tuned)
        ber_thr = profiles.get("ber")
        if ber_thr is not None:
            ber_point = threshold_pr_point(y_ho, score_ho, ber_thr)
    except (FileNotFoundError, ValueError, KeyError):
        pass
    return cv_curve, ho_curve, ber_point
