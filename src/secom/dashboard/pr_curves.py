"""Cached CV OOF and holdout scores for PR curve charts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone

from secom.costs import resolve_threshold_profiles
from secom.cv import make_blocked_time_cv
from secom.dashboard.explainability import fit_holdout_pipeline, load_holdout_split
from secom.metrics import PRCurve, pr_curve_points, threshold_pr_point
from secom.pipelines import (
    BENCHMARK_MODEL_IDS,
    TIMESTAMP_COL,
    WEIGHTING_MODEL_IDS,
    time_decay_weights,
)
from secom.tuning.registry import build_tuned_pipeline
from secom.utils import load_tuned_blocked_params


@st.cache_data(show_spinner="Computing CV out-of-fold scores…")
def collect_cv_oof_proba(model_id: str) -> tuple[pd.Series, np.ndarray]:
    if model_id not in BENCHMARK_MODEL_IDS:
        raise ValueError(f"Unknown model_id: {model_id}")
    split = load_holdout_split()
    tuned = load_tuned_blocked_params(model_id)
    pipeline = build_tuned_pipeline(model_id, tuned)
    cv = make_blocked_time_cv(split.train_df)
    decay_lambda = (
        float(tuned.get("decay_lambda", 0.0))
        if model_id in WEIGHTING_MODEL_IDS
        else 0.0
    )
    train_ts = split.train_df[TIMESTAMP_COL].reset_index(drop=True)
    proba = np.full(len(split.y_train), np.nan)
    for train_idx, val_idx in cv.split(split.X_train, split.y_train):
        fold_pipe = clone(pipeline)
        fit_kwargs: dict = {}
        if decay_lambda:
            fit_kwargs["classifier__sample_weight"] = time_decay_weights(
                train_ts.iloc[train_idx], decay_lambda
            )
        fold_pipe.fit(
            split.X_train.iloc[train_idx], split.y_train.iloc[train_idx], **fit_kwargs
        )
        proba[val_idx] = fold_pipe.predict_proba(split.X_train.iloc[val_idx])[:, 1]
    mask = ~np.isnan(proba)
    y_oof = split.y_train.iloc[mask].reset_index(drop=True)
    score_oof = proba[mask]
    return y_oof, np.asarray(score_oof, dtype=float)


@st.cache_data(show_spinner="Computing holdout scores…")
def collect_holdout_proba(model_id: str) -> tuple[pd.Series, np.ndarray]:
    if model_id not in BENCHMARK_MODEL_IDS:
        raise ValueError(f"Unknown model_id: {model_id}")
    split = load_holdout_split()
    pipeline, _ = fit_holdout_pipeline(model_id)
    proba = pipeline.predict_proba(split.X_test)[:, 1]
    return split.y_test.reset_index(drop=True), np.asarray(proba, dtype=float)


@st.cache_data(show_spinner="Building PR curves…")
def load_pr_curves(
    model_id: str,
) -> tuple[PRCurve | None, PRCurve | None, tuple[float, float] | None]:
    """Blocked CV OOF, temporal holdout PR curves, and BER point on holdout."""
    y_cv, score_cv = collect_cv_oof_proba(model_id)
    y_ho, score_ho = collect_holdout_proba(model_id)
    cv_curve = pr_curve_points(y_cv, score_cv)
    ho_curve = pr_curve_points(y_ho, score_ho)
    ber_point: tuple[float, float] | None = None
    try:
        tuned = load_tuned_blocked_params(model_id)
        profiles = resolve_threshold_profiles(tuned)
        ber_thr = profiles.get("ber")
        if ber_thr is not None:
            ber_point = threshold_pr_point(y_ho, score_ho, ber_thr)
    except (FileNotFoundError, ValueError, KeyError):
        pass
    return cv_curve, ho_curve, ber_point
