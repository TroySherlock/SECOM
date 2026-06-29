"""Unified SECOM foundation + scikit-learn model universe (single module).

This module is the single home for both the shared foundation (paths, IO,
splits, CV factories, estimator/imputer builders, preprocess scaffolding,
time-decay weighting, gate feature pipeline, frozen config) and the unified
3x3 model grid that runs on BOTH protocols:

    front-end {HSIC+hubs, RF-selection+hubs, sPLS}
        x classifier {RF, elastic-net LR, Bayesian elastic-net LR}

is tuned once on the random-stratified in-distribution CV and scored on both the
random holdout (interpolation) and a temporal forward holdout (extrapolation).
Every cell is a plain scikit-learn ``Pipeline`` built by ``build_model_pipeline``:

    raw c_id + rolling-Z c_id_rz
        -> median impute -> SmartCorrelatedSelection cluster
        -> front-end (selection + hub interactions, or sPLS)
        -> RobustScaler (full design)
        -> CalibratedClassifierCV(sigmoid)( classifier )

The Bayesian head (``BayesianElasticNetLogistic``) is a drop-in sklearn
estimator, so it tunes/benchmarks through the exact same machinery as LR/RF.
Two standalone risk-coverage gates (``PCAGate``, ``BayesGate``) live in
:mod:`secom.gates`; classifiers no longer ingest gate features.

Foundation modules (``cv``, ``gates``, ``bayes``) import from here directly.
``secom.hub_interactions`` lazily imports ``random_forest_classifier`` from here
inside its ``fit`` to keep module load acyclic.
"""
from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import duckdb
import joblib
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
    make_scorer,
    recall_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from secom.bayes.model import BayesianElasticNetLogistic
from secom.hub_interactions import (  # noqa: F401
    HSICSelectHubBlock,
    LinearSelectT2HubBlock,
    PLSFeatures,
)
from secom.paths import REPO_ROOT

# --- Paths / data contract ---------------------------------------------------
DB_PATH = REPO_ROOT / "data" / "secom.duckdb"
SOURCE_RELATION = "public.mart_secom_features"
OUTPUT_DIR = REPO_ROOT / "data" / "processed"
# joblib.Memory cache for fitted preprocess steps (incl. the O(n^2) HSIC-Lasso
# front-end), reused across the classifier-head grid; cleared via env or CLI.
PIPELINE_CACHE_DIR = OUTPUT_DIR / ".pipeline_cache"
TUNED_PARAMS_DIR = OUTPUT_DIR / "tuned"
BENCHMARK_RESULTS_PATH = OUTPUT_DIR / "secom_pipeline_benchmark.json"
PIPELINE_ARTIFACTS_PATH = OUTPUT_DIR / "secom_pipeline_artifacts.json"
# Frozen PR-curve + global-importance cache the dashboard reads (per track/model).
REPORT_CACHE_PATH = OUTPUT_DIR / "secom_report_cache.json"
# Pre-generated wafer narratives. The narrative model is track-dependent:
# extrapolation/temporal -> pls_bayes, interpolation/random -> hsic_rf.
NARRATIVES_PATH = OUTPUT_DIR / "extrap_wafer_narratives.json"
INTERP_NARRATIVES_PATH = OUTPUT_DIR / "interp_wafer_narratives.json"

TARGET_COL = "target"
TIMESTAMP_COL = "measurement_ts"
ID_COL = "observation_id"
N_SENSORS = 590

RANDOM_SEED = 42
TEST_SIZE = 0.20
HOLDOUT_SPLIT_MODE = "temporal"
N_SPLITS = 5
N_REPEATS = 2

GRID_SEARCH_VERBOSE = 1

MODEL_NAME = "secom_linear_elastic_net"

CHAMPION_IMPUTATION_METHOD = "median"
KNN_IMPUTE_NEIGHBORS = 5
ELASTIC_NET_MAX_ITER = 100000

# Base RF hyperparameters (the shared classifier builder + hub selector use these).
RF_N_ESTIMATORS = 1000
RF_MAX_DEPTH = 5
RF_MIN_SAMPLES_LEAF = 10

CORRELATED_SELECTION_THRESHOLD = 0.99
CORRELATED_SELECTION_THRESHOLD_GRID = [0.9, 0.95, 0.99]
CORRELATED_SELECTION_METHOD = "spearman"
CORRELATED_SELECTION_CRITERION = "corr_with_target"

CV_N_JOBS = -1
ESTIMATOR_N_JOBS = 1

PRIMARY_TUNING_METRIC = "pr_auc"
THRESHOLD_GRID = np.linspace(0.0001, 0.2, num=3000)

CLASSIFIER_CALIBRATION_METHOD = "sigmoid"
CLASSIFIER_CALIBRATION_CV = 5

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


def time_decay_weights(timestamps, decay_lambda: float) -> np.ndarray:
    """Exponential recency weights from timestamps (recent = heavier).

    Age is normalized to [0, 1] within the rows passed in (0 = newest,
    1 = oldest). ``decay_lambda=0`` yields uniform weights. Used by the temporal-
    holdout decay sweep; every head accepts the weights (the Bayesian ones fold
    them into their likelihood, LR/RF forward them via CalibratedClassifierCV).
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


# Cache fitted preprocess/scale steps so the heavy HSIC-Lasso front-end is
# computed once per (front-end config, CV fold) and reused across the whole
# classifier-head grid. Keyed on params + data, not source, so editing a
# front-end transformer requires a clear (env var, helper, or CLI flag).
# Only the HSIC cells use it: hashing the full design (~0.7s/lookup) only pays
# off against the O(n^2) HSIC recompute, not the cheap PLS / RF-selection fronts.
_PIPELINE_MEMORY = (
    None
    if os.environ.get("SECOM_DISABLE_PIPELINE_CACHE")
    else joblib.Memory(location=str(PIPELINE_CACHE_DIR), verbose=0)
)


def clear_pipeline_cache() -> None:
    """Drop cached preprocess fits (run after editing the HSIC front-end transformer)."""
    if _PIPELINE_MEMORY is not None:
        _PIPELINE_MEMORY.clear(warn=False)


def feature_pipeline(
    classifier,
    preprocess: ColumnTransformer,
    *,
    calib_cv: int = CLASSIFIER_CALIBRATION_CV,
    memory=None,
) -> Pipeline:
    """Preprocess -> RobustScaler -> calibrated classifier.

    ``memory`` caches the fitted preprocess/scale steps; pass it only for the
    expensive HSIC front-end (the input-hashing cost outweighs the savings for
    the cheap PLS / RF-selection fronts).
    """
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("scale", RobustScaler()),
            ("classifier", calibrated_classifier(classifier, calib_cv=calib_cv)),
        ],
        memory=memory,
    ).set_output(transform="pandas")


# --- Model grid: 3 front-ends x 3 classifiers, both protocols ----------------
FRONT_ENDS = ("hsic", "rfsel", "pls")
CLASSIFIER_KINDS = ("enet", "rf", "bayes")

MODEL_IDS: tuple[str, ...] = (
    "hsic_enet",
    "hsic_rf",
    "hsic_bayes",
    "rfsel_enet",
    "rfsel_rf",
    "rfsel_bayes",
    "pls_enet",
    "pls_rf",
    "pls_bayes",
)

#: (front_end, classifier_kind) for each neutral model id.
MODEL_CELLS: dict[str, tuple[str, str]] = {
    "hsic_enet": ("hsic", "enet"),
    "hsic_rf": ("hsic", "rf"),
    "hsic_bayes": ("hsic", "bayes"),
    "rfsel_enet": ("rfsel", "enet"),
    "rfsel_rf": ("rfsel", "rf"),
    "rfsel_bayes": ("rfsel", "bayes"),
    "pls_enet": ("pls", "enet"),
    "pls_rf": ("pls", "rf"),
    "pls_bayes": ("pls", "bayes"),
}

# Backwards-compat alias used across the benchmark / dashboard / utils.
BENCHMARK_MODEL_IDS = MODEL_IDS

# Reference models for the dashboard reduction widget + shared cluster example
# (both RF-selection front-ends so the stage breakdown extracts cleanly).
REFERENCE_MODELS = {"linear": "rfsel_enet", "topk": "rfsel_rf"}

# --- Front-end (selection / aggregation) grids -------------------------------
# Selection front-ends (HSIC / RF) screen K sensors then expand the top n_hubs
# into pairwise interaction features; sPLS aggregates into K latent components.
TOP_K_DEFAULT = 35
TOP_K_GRID = [25, 30, 35, 50, 60]
N_HUBS_DEFAULT = 5
N_HUBS_GRID = [0, 5, 10]
PLS_N_COMPONENTS_DEFAULT = 25 
PLS_N_COMPONENTS_GRID = [20, 25, 30, 35]

# --- Classifier head grids ---------------------------------------------------
# Elastic-net LR (saga) slope prior.
C_GRID = [0.01, 0.05, 0.1]
L1_RATIO_GRID = [0.2, 0.3, 0.5]
# RF classifier depth.
RF_MAX_DEPTH_GRID = [5, 8, 10, 12]

# Bayesian elastic-net head: kept deliberately tiny (each grid point is a full
# ADVI fit x calibration folds x CV folds x 2 protocols).
BAYES_C_GRID = [0.01, 0.05, 0.1]
BAYES_L1_RATIO_GRID = [0.2, 0.3, 0.5]
BAYES_POS_WEIGHT_GRID = [15.0]
BAYES_TOP_K = 50
BAYES_N_HUBS = 10
BAYES_PLS_COMPONENTS = 25
BAYES_SVI_STEPS = 1000
# Each calibration fold is a full ADVI refit -> use fewer folds than LR/RF.
BAYES_CALIB_CV = 3

# Exponential time-decay (recency weighting) grid for the temporal-holdout
# diagnostic sweep (lambda=0 recovers the unweighted headline model). Decay is
# no longer tuned; the benchmark sweeps these lambdas and scores the temporal
# holdout so the effect of recency weighting can be read off directly.
DECAY_LAMBDA_GRID = [0.0]

# --- Standalone gates (risk-coverage tools, not pipeline steps) --------------
# PCA gate (fab-standard baseline): PCA-MSPC -> Hotelling T2 + Q (SPE).
PCA_GATE_N_COMPONENTS = 10
PCA_GATE_T2_ALPHA = 0.03
PCA_GATE_Q_ALPHA = 0.005
PCA_GATE_LOGIC = "or"
PCA_GATE_CLIP = 5.0

# Bayes gate: sparse Bayesian factor analysis -> BGM density + Q (SPE).
BAYES_GATE_N_FACTORS = 8
BAYES_GATE_BGM_COMPONENTS = 5
BAYES_GATE_LOADING_SCALE = 0.3
BAYES_GATE_DENSITY_ALPHA = 0.03
BAYES_GATE_Q_ALPHA = 0.005
BAYES_GATE_LOGIC = "or"
BAYES_GATE_SVI_STEPS = 2000
BAYES_GATE_N_SEEDS = 5
BAYES_GATE_CLIP = 5.0

# Both gates score the raw post-cluster sensor space (impute -> cluster).
GATE_CORR_THRESHOLD = CORRELATED_SELECTION_THRESHOLD

# Risk-coverage sweep: keep the least-suspicious fraction of holdout wafers at
# each coverage and rescore (1.0 == global holdout metric).
RISK_COVERAGE_GRID = [1.0, 0.99, 0.95, 0.9, 0.85, 0.8]


# --- Estimator builders ------------------------------------------------------
def bayesian_elastic_net(
    C: float = float(BAYES_C_GRID[0]),
    l1_ratio: float = float(BAYES_L1_RATIO_GRID[0]),
    pos_weight: float = float(BAYES_POS_WEIGHT_GRID[0]),
) -> BayesianElasticNetLogistic:
    return BayesianElasticNetLogistic(
        C=C,
        l1_ratio=l1_ratio,
        pos_weight=pos_weight,
        inference="advi",
        svi_steps=BAYES_SVI_STEPS,
        seed=RANDOM_SEED,
    )


_CLASSIFIER_BUILDERS = {
    "enet": partial(elastic_net_lr, C=float(C_GRID[0]), l1_ratio=float(L1_RATIO_GRID[0])),
    "rf": random_forest_classifier,
    "bayes": bayesian_elastic_net,
}


def _front_end_step(front_end: str, *, top_k: int, n_hubs: int, pls_n_components: int):
    if front_end == "hsic":
        return ("front_end", HSICSelectHubBlock(top_k=int(top_k), n_hubs=int(n_hubs)))
    if front_end == "rfsel":
        return (
            "front_end",
            LinearSelectT2HubBlock(top_k=int(top_k), n_hubs=int(n_hubs)),
        )
    if front_end == "pls":
        return ("front_end", PLSFeatures(n_components=int(pls_n_components)))
    raise ValueError(f"front_end must be one of {FRONT_ENDS}, got {front_end!r}")


def build_model_pipeline(
    front_end: str,
    classifier_kind: str,
    *,
    top_k: int = TOP_K_DEFAULT,
    n_hubs: int = N_HUBS_DEFAULT,
    pls_n_components: int = PLS_N_COMPONENTS_DEFAULT,
):
    """Build a calibrated sklearn pipeline for one (front-end, classifier) cell.

    raw+rz sensors -> median impute -> cluster -> front-end -> RobustScaler ->
    CalibratedClassifierCV(sigmoid)(classifier).
    """
    if classifier_kind not in _CLASSIFIER_BUILDERS:
        raise ValueError(
            f"classifier_kind must be one of {CLASSIFIER_KINDS}, got {classifier_kind!r}"
        )
    sensor_steps: list[tuple[str, object]] = [
        ("impute", median_imputer()),
        ("cluster", _cluster_step()),
        _front_end_step(
            front_end,
            top_k=top_k,
            n_hubs=n_hubs,
            pls_n_components=pls_n_components,
        ),
    ]
    preprocess = _sensor_preprocess_column(
        sensor_steps, sensor_pattern=_SENSOR_VALUE_PATTERN_EXTRAP
    )
    calib_cv = BAYES_CALIB_CV if classifier_kind == "bayes" else CLASSIFIER_CALIBRATION_CV
    classifier = _CLASSIFIER_BUILDERS[classifier_kind]()
    # Cache only the HSIC front-end: the input-hashing cost only pays off against
    # the O(n^2) HSIC recompute, not the cheap PLS / RF-selection fronts.
    memory = _PIPELINE_MEMORY if front_end == "hsic" else None
    return feature_pipeline(classifier, preprocess, calib_cv=calib_cv, memory=memory)


def is_bayesian(model_id: str) -> bool:
    """True for the three Bayesian-head cells (still plain sklearn pipelines)."""
    return MODEL_CELLS.get(model_id, ("", ""))[1] == "bayes"


def pipelines_frozen_config_fragment() -> dict:
    """Model-grid + gate entries for the frozen-config snapshot."""
    return {
        "model_ids": list(MODEL_IDS),
        "model_cells": {k: list(v) for k, v in MODEL_CELLS.items()},
        "front_ends": list(FRONT_ENDS),
        "classifier_kinds": list(CLASSIFIER_KINDS),
        "top_k_default": int(TOP_K_DEFAULT),
        "top_k_grid": [int(k) for k in TOP_K_GRID],
        "n_hubs_default": int(N_HUBS_DEFAULT),
        "n_hubs_grid": [int(k) for k in N_HUBS_GRID],
        "pls_n_components_default": int(PLS_N_COMPONENTS_DEFAULT),
        "pls_n_components_grid": [int(k) for k in PLS_N_COMPONENTS_GRID],
        "c_grid": [float(c) for c in C_GRID],
        "l1_ratio_grid": [float(r) for r in L1_RATIO_GRID],
        "rf_max_depth_grid": [int(d) for d in RF_MAX_DEPTH_GRID],
        "bayes_c_grid": [float(c) for c in BAYES_C_GRID],
        "bayes_l1_ratio_grid": [float(r) for r in BAYES_L1_RATIO_GRID],
        "bayes_pos_weight_grid": [float(w) for w in BAYES_POS_WEIGHT_GRID],
        "bayes_top_k": int(BAYES_TOP_K),
        "bayes_n_hubs": int(BAYES_N_HUBS),
        "bayes_pls_components": int(BAYES_PLS_COMPONENTS),
        "bayes_svi_steps": int(BAYES_SVI_STEPS),
        "bayes_calib_cv": int(BAYES_CALIB_CV),
        "decay_lambda_sweep_grid": [float(x) for x in DECAY_LAMBDA_GRID],
        "pca_gate_n_components": int(PCA_GATE_N_COMPONENTS),
        "pca_gate_t2_alpha": float(PCA_GATE_T2_ALPHA),
        "pca_gate_q_alpha": float(PCA_GATE_Q_ALPHA),
        "pca_gate_logic": str(PCA_GATE_LOGIC),
        "bayes_gate_n_factors": int(BAYES_GATE_N_FACTORS),
        "bayes_gate_bgm_components": int(BAYES_GATE_BGM_COMPONENTS),
        "bayes_gate_loading_scale": float(BAYES_GATE_LOADING_SCALE),
        "bayes_gate_density_alpha": float(BAYES_GATE_DENSITY_ALPHA),
        "bayes_gate_q_alpha": float(BAYES_GATE_Q_ALPHA),
        "bayes_gate_logic": str(BAYES_GATE_LOGIC),
        "bayes_gate_svi_steps": int(BAYES_GATE_SVI_STEPS),
        "bayes_gate_n_seeds": int(BAYES_GATE_N_SEEDS),
        "bayes_gate_clip": float(BAYES_GATE_CLIP),
        "gate_corr_threshold": float(GATE_CORR_THRESHOLD),
        "risk_coverage_grid": [float(c) for c in RISK_COVERAGE_GRID],
    }


# --- Frozen config aggregator ------------------------------------------------
def frozen_config() -> dict:
    """Reproducibility snapshot of the full pipeline configuration."""
    from secom.costs import PROFILE_IDS, threshold_profile_config

    config = {
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "holdout_split_mode": HOLDOUT_SPLIT_MODE,
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "model_name": MODEL_NAME,
        "champion_imputation_method": CHAMPION_IMPUTATION_METHOD,
        "knn_impute_neighbors": int(KNN_IMPUTE_NEIGHBORS),
        "tuning_protocol": "sequential_pr_auc_hyperparams_multi_threshold",
        "primary_tuning_metric": PRIMARY_TUNING_METRIC,
        "threshold_tuning_profiles": list(PROFILE_IDS),
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
