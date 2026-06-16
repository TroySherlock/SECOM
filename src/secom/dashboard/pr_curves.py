"""Cached CV OOF and holdout scores for PR curve charts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import cross_val_predict

from secom.costs import resolve_threshold_profiles
from secom.dashboard.explainability import fit_holdout_pipeline, load_holdout_split
from secom.metrics import PRCurve, pr_curve_points, threshold_pr_point
from secom.pipelines import (
    BENCHMARK_MODEL_IDS,
    CV_N_JOBS,
    make_stratified_kfold_for_oof,
)
from secom.tuning.registry import build_tuned_pipeline
from secom.utils import load_tuned_params


@st.cache_data(show_spinner="Computing CV out-of-fold scores…")
def collect_cv_oof_proba(model_id: str) -> tuple[pd.Series, np.ndarray]:
    if model_id not in BENCHMARK_MODEL_IDS:
        raise ValueError(f"Unknown model_id: {model_id}")
    split = load_holdout_split()
    tuned = load_tuned_params(model_id)
    pipeline = build_tuned_pipeline(model_id, tuned)
    cv = make_stratified_kfold_for_oof()
    proba = cross_val_predict(
        pipeline,
        split.X_train,
        split.y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=CV_N_JOBS,
    )[:, 1]
    return split.y_train.reset_index(drop=True), np.asarray(proba, dtype=float)


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
    """CV OOF, holdout PR curves, and holdout (recall, precision) at BER threshold."""
    y_cv, score_cv = collect_cv_oof_proba(model_id)
    y_ho, score_ho = collect_holdout_proba(model_id)
    cv_curve = pr_curve_points(y_cv, score_cv)
    ho_curve = pr_curve_points(y_ho, score_ho)
    ber_point: tuple[float, float] | None = None
    try:
        tuned = load_tuned_params(model_id)
        profiles = resolve_threshold_profiles(tuned)
        ber_thr = profiles.get("ber")
        if ber_thr is not None:
            ber_point = threshold_pr_point(y_ho, score_ho, ber_thr)
    except (FileNotFoundError, ValueError, KeyError):
        pass
    return cv_curve, ho_curve, ber_point
