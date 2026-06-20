"""Shared SECOM foundation: paths, IO, splits, CV, shared builders, frozen config.

This module is track-agnostic. The unified model grid, front-ends, classifier
builders, gate config, and ``build_model_pipeline`` live in
:mod:`secom.pipelines`. Foundation modules (``cv``, ``gates``, ``bayes``) import
from here directly to avoid import cycles through the pipelines module.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from feature_engine.selection import SmartCorrelatedSelection
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    fbeta_score,
    make_scorer,
    recall_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from secom.paths import REPO_ROOT

# --- Paths / data contract ---------------------------------------------------
DB_PATH = REPO_ROOT / "data" / "secom.duckdb"
SOURCE_RELATION = "public.mart_secom_features"
OUTPUT_DIR = REPO_ROOT / "data" / "processed"
TUNED_PARAMS_DIR = OUTPUT_DIR / "tuned"
TUNED_BLOCKED_PARAMS_DIR = OUTPUT_DIR / "tuned_blocked"
BENCHMARK_RESULTS_PATH = OUTPUT_DIR / "secom_pipeline_benchmark.json"
PIPELINE_ARTIFACTS_PATH = OUTPUT_DIR / "secom_pipeline_artifacts.json"
# Pre-generated wafer narratives for the extrapolation narrative model.
NARRATIVES_PATH = OUTPUT_DIR / "extrap_wafer_narratives.json"
LINEAR_LR_NARRATIVES_PATH = NARRATIVES_PATH

TARGET_COL = "target"
TIMESTAMP_COL = "measurement_ts"
ID_COL = "observation_id"
N_SENSORS = 591

RANDOM_SEED = 42
TEST_SIZE = 0.20
HOLDOUT_SPLIT_MODE = "temporal"
N_SPLITS = 5
N_REPEATS = 1
N_BLOCKED_SPLITS = 5
BLOCKED_MIN_VAL_FAILS = 4

GRID_SEARCH_VERBOSE = 1

MODEL_NAME = "secom_linear_elastic_net"

N_MISSING_SENSORS_COL = "n_missing_sensors"
CHAMPION_IMPUTATION_METHOD = "median"
KNN_IMPUTE_NEIGHBORS = 5
ELASTIC_NET_MAX_ITER = 50000

# Base RF hyperparameters (the shared classifier builder + hub selector use these).
RF_N_ESTIMATORS = 1000
RF_MAX_DEPTH = 5
RF_MIN_SAMPLES_LEAF = 10

CORRELATED_SELECTION_THRESHOLD = 0.9
CORRELATED_SELECTION_THRESHOLD_GRID = [0.9, 0.95]
CORRELATED_SELECTION_METHOD = "spearman"
CORRELATED_SELECTION_CRITERION = "corr_with_target"

# Gate config (EFA_GATE_* and BAYES_GATE_*) lives in secom.pipelines; both gates
# share build_gate_feature_pipeline (below) for their impute -> cluster front-end.

CV_N_JOBS = -1
ESTIMATOR_N_JOBS = 1

PRIMARY_TUNING_METRIC = "pr_auc"
THRESHOLD_GRID = np.linspace(0.001, 0.999, num=1000)

CLASSIFIER_CALIBRATION_METHOD = "isotonic"
CLASSIFIER_CALIBRATION_CV = 3

HOLDOUT_BOOTSTRAP_N = 1000
HOLDOUT_BOOTSTRAP_CI = 0.95

CV_SCORING = {
    "balanced_accuracy": "balanced_accuracy",
    "true_positive_rate": "recall",
    "true_negative_rate": make_scorer(recall_score, pos_label=0),
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}

_CALENDAR_PATTERN = (
    r"^(?:is_weekend|month_sin|month_cos|dow_sin|dow_cos|hour_sin|hour_cos)$"
)
_MISSING_FLAG_PATTERN = r"^c_\d+__missing$"
_SENSOR_VALUE_PATTERN = r"^c_\d+$"
# Extrapolation track also consumes causal rolling-Z columns (c_<id>_rz).
_SENSOR_VALUE_PATTERN_EXTRAP = r"^c_\d+(?:_rz)?$"


# --- Estimator + imputer builders (shared) -----------------------------------
median_imputer = partial(
    SimpleImputer,
    strategy="median",
)

elastic_net_lr = partial(
    LogisticRegression,
    solver="saga",
    class_weight="balanced",
    max_iter=ELASTIC_NET_MAX_ITER,
    random_state=RANDOM_SEED,
)

random_forest_classifier = partial(
    RandomForestClassifier,
    n_estimators=RF_N_ESTIMATORS,
    max_depth=RF_MAX_DEPTH,
    min_samples_leaf=RF_MIN_SAMPLES_LEAF,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_SEED,
    n_jobs=ESTIMATOR_N_JOBS,
)


# --- IO + splits -------------------------------------------------------------
def load_mart(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as con:
        return con.execute(f"select * from {SOURCE_RELATION}").df()


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {ID_COL, TIMESTAMP_COL, TARGET_COL}
    return [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]


def split_train_test(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    timestamp_col: str = TIMESTAMP_COL,
    test_size: float = TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal holdout: earliest (1 - test_size) for train, latest test_size for test."""
    work = df.copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work = work.sort_values([timestamp_col, ID_COL], kind="mergesort")
    n_test = max(1, int(round(len(work) * test_size)))
    test_df = work.iloc[-n_test:].copy()
    train_df = work.iloc[:-n_test].copy()
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def split_train_test_random(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random stratified holdout (in-distribution / interpolation contrast)."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        shuffle=True,
        random_state=seed,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def holdout_split_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    split_mode: str = HOLDOUT_SPLIT_MODE,
    target_col: str = TARGET_COL,
    timestamp_col: str = TIMESTAMP_COL,
    test_size: float = TEST_SIZE,
) -> dict[str, float | int | str | None]:
    """Metadata for benchmark JSON: temporal bounds and fail rates."""
    train_ts = pd.to_datetime(train_df[timestamp_col], errors="coerce")
    test_ts = pd.to_datetime(test_df[timestamp_col], errors="coerce")

    def _fail_rate(frame: pd.DataFrame) -> float | None:
        if frame.empty or target_col not in frame.columns:
            return None
        return float(frame[target_col].astype(int).mean())

    return {
        "split_mode": split_mode,
        "test_size": float(test_size),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_fail_rate": _fail_rate(train_df),
        "holdout_fail_rate": _fail_rate(test_df),
        "train_ts_min": train_ts.min().isoformat() if not train_ts.empty else None,
        "train_ts_max": train_ts.max().isoformat() if not train_ts.empty else None,
        "holdout_ts_min": test_ts.min().isoformat() if not test_ts.empty else None,
        "holdout_ts_max": test_ts.max().isoformat() if not test_ts.empty else None,
    }


def make_repeated_stratified_cv() -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_SEED,
    )


def make_stratified_kfold_for_oof() -> StratifiedKFold:
    """Partitioning CV for out-of-fold predict_proba (one score per row)."""
    return StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )


def time_decay_weights(timestamps, decay_lambda: float) -> np.ndarray:
    """Exponential recency weights from timestamps (recent = heavier).

    Age is normalized to [0, 1] within the rows passed in (0 = newest,
    1 = oldest). ``decay_lambda=0`` yields uniform weights. Retained for the
    interpolation/sklearn paths; the Bayesian extrapolation models handle drift
    structurally via a random-walk intercept instead.
    """
    ts = np.asarray(
        pd.to_datetime(np.asarray(timestamps), errors="coerce").astype("int64"),
        dtype=float,
    )
    n = len(ts)
    if n == 0:
        return np.ones(0)
    span = ts.max() - ts.min()
    if span <= 0 or not decay_lambda:
        return np.ones(n)
    age = (ts.max() - ts) / span
    w = np.exp(-float(decay_lambda) * age)
    total = w.sum()
    if total <= 0:
        return np.ones(n)
    return w * (n / total)


# --- Shared preprocess scaffolding -------------------------------------------
def _cluster_step() -> Pipeline:
    return Pipeline(
        steps=[
            ("variance_threshold", VarianceThreshold(threshold=0)),
            (
                "smart_corr",
                SmartCorrelatedSelection(
                    method=CORRELATED_SELECTION_METHOD,
                    threshold=CORRELATED_SELECTION_THRESHOLD,
                    missing_values="ignore",
                    selection_method=CORRELATED_SELECTION_CRITERION,
                ),
            ),
        ],
    ).set_output(transform="pandas")


def _auxiliary_transformers() -> list[tuple[str, str, object]]:
    return [
        (
            "n_missing",
            "passthrough",
            make_column_selector(pattern=r"^n_missing_sensors$"),
        ),
        (
            "calendar",
            "passthrough",
            make_column_selector(pattern=_CALENDAR_PATTERN),
        ),
        (
            "missing_flags",
            "passthrough",
            make_column_selector(pattern=_MISSING_FLAG_PATTERN),
        ),
    ]


def _sensor_preprocess_column(
    sensor_steps: list[tuple[str, object]],
    sensor_pattern: str = _SENSOR_VALUE_PATTERN,
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "sensor_branch",
                Pipeline(steps=sensor_steps).set_output(transform="pandas"),
                make_column_selector(pattern=sensor_pattern),
            ),
            *_auxiliary_transformers(),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_gate_feature_pipeline(corr_threshold: float | None = None) -> Pipeline:
    """Impute -> cluster only (sensor branch through SmartCorrelatedSelection)."""
    cluster = _cluster_step()
    if corr_threshold is not None:
        cluster.set_params(smart_corr__threshold=float(corr_threshold))
    return Pipeline(
        steps=[
            ("impute", median_imputer()),
            ("cluster", cluster),
        ]
    ).set_output(transform="pandas")


def calibrated_classifier(
    estimator, calib_cv: int = CLASSIFIER_CALIBRATION_CV
) -> CalibratedClassifierCV:
    """Wrap the base estimator with probability calibration (shared by all models).

    ``calib_cv`` lets the expensive Bayesian head calibrate with fewer folds
    (each fold is a full ADVI refit) than the cheap LR/RF cells.
    """
    return CalibratedClassifierCV(
        estimator=estimator,
        method=CLASSIFIER_CALIBRATION_METHOD,
        cv=int(calib_cv),
    )


def feature_pipeline(
    classifier,
    preprocess: ColumnTransformer,
    *,
    calib_cv: int = CLASSIFIER_CALIBRATION_CV,
) -> Pipeline:
    """Preprocess -> RobustScaler -> calibrated classifier."""
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("scale", RobustScaler()),
            ("classifier", calibrated_classifier(classifier, calib_cv=calib_cv)),
        ]
    ).set_output(transform="pandas")


# --- Threshold profile sweep (pure, shared by both tracks) -------------------
def threshold_profile_sweep(
    y_true: np.ndarray | pd.Series,
    probs: np.ndarray | pd.Series,
    *,
    grid: np.ndarray = THRESHOLD_GRID,
) -> dict[str, dict]:
    """Sweep ``grid`` to pick a best threshold per profile from scores alone.

    Returns ``{profile_id: {"best_threshold", "best_score", "objective"}}`` in
    the shape ``secom.costs.resolve_threshold_profiles`` consumes. Pure in
    ``(y_true, probs)`` so the sklearn and Bayesian harnesses share it.
    """
    from secom.costs import THRESHOLD_PROFILES

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probs, dtype=float)
    grid = np.asarray(grid, dtype=float)
    out: dict[str, dict] = {}
    for pid, profile in THRESHOLD_PROFILES.items():
        best_thr = 0.5
        if profile.objective == "ber":
            best_score = np.inf
            for thr in grid:
                preds = (p >= thr).astype(int)
                # balanced error rate = 1 - balanced accuracy
                ber = 1.0 - balanced_accuracy_score(y, preds)
                if ber < best_score:
                    best_score, best_thr = ber, float(thr)
        else:
            best_score = -np.inf
            for thr in grid:
                preds = (p >= thr).astype(int)
                score = fbeta_score(
                    y, preds, beta=float(profile.beta), zero_division=0
                )
                if score > best_score:
                    best_score, best_thr = float(score), float(thr)
        out[str(pid)] = {
            "best_threshold": float(best_thr),
            "best_score": float(best_score),
            "objective": profile.objective,
        }
    return out


# --- Frozen config aggregator ------------------------------------------------
def frozen_config() -> dict:
    """Reproducibility snapshot (lazy import of pipelines to avoid a cycle)."""
    from secom.costs import threshold_profile_config
    from secom.pipelines import MODEL_IDS, pipelines_frozen_config_fragment

    config = {
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "holdout_split_mode": HOLDOUT_SPLIT_MODE,
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "n_blocked_splits": N_BLOCKED_SPLITS,
        "blocked_min_val_fails": BLOCKED_MIN_VAL_FAILS,
        "model_name": MODEL_NAME,
        "champion_imputation_method": CHAMPION_IMPUTATION_METHOD,
        "knn_impute_neighbors": int(KNN_IMPUTE_NEIGHBORS),
        "tuning_protocol": "sequential_pr_auc_hyperparams_multi_threshold",
        "primary_tuning_metric": PRIMARY_TUNING_METRIC,
        "threshold_tuning_profiles": ["f0_5", "f2", "f4", "ber"],
        "threshold_grid": [float(t) for t in THRESHOLD_GRID],
        "classifier_calibration_method": str(CLASSIFIER_CALIBRATION_METHOD),
        "classifier_calibration_cv": int(CLASSIFIER_CALIBRATION_CV),
        "elastic_net_max_iter": int(ELASTIC_NET_MAX_ITER),
        "rf_n_estimators": int(RF_N_ESTIMATORS),
        "rf_max_depth": int(RF_MAX_DEPTH),
        "correlated_selection_threshold": float(CORRELATED_SELECTION_THRESHOLD),
        "correlated_selection_threshold_grid": [
            float(t) for t in CORRELATED_SELECTION_THRESHOLD_GRID
        ],
        "correlated_selection_method": CORRELATED_SELECTION_METHOD,
        "correlated_selection_criterion": CORRELATED_SELECTION_CRITERION,
        "holdout_bootstrap_n": int(HOLDOUT_BOOTSTRAP_N),
        "holdout_bootstrap_ci": float(HOLDOUT_BOOTSTRAP_CI),
        "benchmark_model_ids": list(MODEL_IDS),
    }
    config.update(pipelines_frozen_config_fragment())
    config.update(threshold_profile_config())
    return config
