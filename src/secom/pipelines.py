"""Shared SECOM pipeline builders, data loading, and tuning constants."""
from __future__ import annotations

from functools import partial
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from feature_engine.selection import (
    DropConstantFeatures,
    DropDuplicateFeatures,
    SmartCorrelatedSelection,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from secom.hub_interactions import LinearSelectT2HubBlock
from secom.paths import REPO_ROOT

DB_PATH = REPO_ROOT / "data" / "secom.duckdb"
SOURCE_RELATION = "public.mart_secom_features"
OUTPUT_DIR = REPO_ROOT / "data" / "processed"
# Dashboard data contract (regenerate: python -m secom.cli.benchmark after tuning):
#   tuned/<model_id>.json           — frozen hyperparameters from tuning notebooks
#   secom_pipeline_benchmark.json   — CV leaderboard + holdout metrics
#   secom_pipeline_artifacts.json   — holdout-fit pipeline reporting (feature counts, RF, clusters)
#   linear_lr_wafer_narratives.json — pre-generated Gemma summaries (python -m secom.cli.build_narratives)
TUNED_PARAMS_DIR = OUTPUT_DIR / "tuned"
TUNED_BLOCKED_PARAMS_DIR = OUTPUT_DIR / "tuned_blocked"
BENCHMARK_RESULTS_PATH = OUTPUT_DIR / "secom_pipeline_benchmark.json"
PIPELINE_ARTIFACTS_PATH = OUTPUT_DIR / "secom_pipeline_artifacts.json"
LINEAR_LR_NARRATIVES_PATH = OUTPUT_DIR / "linear_lr_wafer_narratives.json"

BENCHMARK_MODEL_IDS = (
    "linear_lr",
    "topk_rf",
    "topk_knn",
    "topk_xgb",
)

TARGET_COL = "target"
TIMESTAMP_COL = "measurement_ts"
ID_COL = "observation_id"
N_SENSORS = 591

RANDOM_SEED = 42
TEST_SIZE = 0.20
HOLDOUT_SPLIT_MODE = "temporal"
N_SPLITS = 5
N_REPEATS = 1
N_BLOCKED_SPLITS = 3
BLOCKED_MIN_VAL_FAILS = 4
BASELINE_NORM_EPS = 1e-6
GRID_SEARCH_VERBOSE = 1

C_GRID = [0.0075, 0.01, 0.05]
L1_RATIO_GRID = [0.3, 0.5, 0.7]

MODEL_NAME = "secom_linear_elastic_net"

N_MISSING_SENSORS_COL = "n_missing_sensors"
CHAMPION_IMPUTATION_METHOD = "median"
KNN_IMPUTE_NEIGHBORS = 5
ELASTIC_NET_MAX_ITER = 50000

KNN_CLASSIFIER_NEIGHBORS = 10
KNN_CLASSIFIER_WEIGHTS = "uniform"
KNN_NEIGHBORS_GRID = [10, 30]

RF_N_ESTIMATORS = 1000
RF_MAX_DEPTH = 3
RF_MAX_DEPTH_GRID = [3, 4, 5]
RF_MIN_SAMPLES_LEAF = 10
RF_SELECT_TOP_K = 35
RF_SELECT_TOP_K_GRID = [35, 100]

N_HUBS_DEFAULT = 5
N_HUBS_GRID = [5]

CORRELATED_SELECTION_THRESHOLD = 0.85
CORRELATED_SELECTION_THRESHOLD_GRID = [0.85, 0.9, 0.95]
CORRELATED_SELECTION_METHOD = "spearman"
CORRELATED_SELECTION_CRITERION = "corr_with_target"

XGB_N_ESTIMATORS = 1000
XGB_MAX_DEPTH = 3
XGB_MAX_DEPTH_GRID = [3, 4, 5]
XGB_LEARNING_RATE = 0.05
XGB_LEARNING_RATE_GRID = [0.1]
XGB_SCALE_POS_WEIGHT = 14.151515

CV_N_JOBS = -1
ESTIMATOR_N_JOBS = 1

PRIMARY_TUNING_METRIC = "pr_auc"
THRESHOLD_GRID = np.linspace(0.001, 0.999, num=500)

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


knn_imputer = partial(
    KNNImputer,
    n_neighbors=KNN_IMPUTE_NEIGHBORS,
    weights="distance",
)

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

knn_classifier = partial(
    KNeighborsClassifier,
    n_neighbors=KNN_CLASSIFIER_NEIGHBORS,
    weights=KNN_CLASSIFIER_WEIGHTS,
    n_jobs=ESTIMATOR_N_JOBS,
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

xgboost_classifier = partial(
    XGBClassifier,
    n_estimators=XGB_N_ESTIMATORS,
    max_depth=XGB_MAX_DEPTH,
    learning_rate=XGB_LEARNING_RATE,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,                  # NEW: Increases L2 regularization on weights to smooth probabilities
    reg_alpha=0.5,
    scale_pos_weight=XGB_SCALE_POS_WEIGHT,
    random_state=RANDOM_SEED,
    n_jobs=ESTIMATOR_N_JOBS,
    verbosity=0,
    eval_metric="logloss",
)


def frozen_config() -> dict:
    from secom.costs import threshold_profile_config

    return {
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "holdout_split_mode": HOLDOUT_SPLIT_MODE,
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "n_blocked_splits": N_BLOCKED_SPLITS,
        "blocked_min_val_fails": BLOCKED_MIN_VAL_FAILS,
        "c_grid": [float(c) for c in C_GRID],
        "l1_ratio_grid": [float(r) for r in L1_RATIO_GRID],
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
        "knn_classifier_neighbors": int(KNN_CLASSIFIER_NEIGHBORS),
        "knn_neighbors_grid": [int(k) for k in KNN_NEIGHBORS_GRID],
        "rf_n_estimators": int(RF_N_ESTIMATORS),
        "rf_max_depth": int(RF_MAX_DEPTH),
        "rf_max_depth_grid": [int(d) for d in RF_MAX_DEPTH_GRID],
        "xgb_n_estimators": int(XGB_N_ESTIMATORS),
        "xgb_max_depth": int(XGB_MAX_DEPTH),
        "xgb_max_depth_grid": [int(d) for d in XGB_MAX_DEPTH_GRID],
        "xgb_learning_rate_grid": [float(x) for x in XGB_LEARNING_RATE_GRID],
        "xgb_scale_pos_weight": float(XGB_SCALE_POS_WEIGHT),
        "rf_select_top_k": int(RF_SELECT_TOP_K),
        "rf_select_top_k_grid": [int(k) for k in RF_SELECT_TOP_K_GRID],
        "n_hubs_default": int(N_HUBS_DEFAULT),
        "n_hubs_grid": [int(k) for k in N_HUBS_GRID],
        "benchmark_model_ids": list(BENCHMARK_MODEL_IDS),
        "correlated_selection_threshold": float(CORRELATED_SELECTION_THRESHOLD),
        "correlated_selection_threshold_grid": [
            float(t) for t in CORRELATED_SELECTION_THRESHOLD_GRID
        ],
        "correlated_selection_method": CORRELATED_SELECTION_METHOD,
        "correlated_selection_criterion": CORRELATED_SELECTION_CRITERION,
        "holdout_bootstrap_n": int(HOLDOUT_BOOTSTRAP_N),
        "holdout_bootstrap_ci": float(HOLDOUT_BOOTSTRAP_CI),
        **threshold_profile_config(),
    }


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


class SensorBaselineNormalizer(BaseEstimator, TransformerMixin):
    """Per-sensor robust z-score vs train-fold baseline (IQR scale).

    When ``enabled=False`` (default), passthrough — used for in-distribution
  evaluation. Enable for extrapolation / temporal holdout pipelines.
    """

    _sklearn_auto_wrap_output_keys = ("transform",)

    def __init__(self, enabled: bool = False, eps: float = BASELINE_NORM_EPS):
        self.enabled = enabled
        self.eps = eps

    def fit(self, X, y=None):
        cols = list(X.columns) if isinstance(X, pd.DataFrame) else []
        self.feature_names_in_ = np.asarray(cols, dtype=object)
        if not self.enabled or not cols:
            self.median_ = {}
            self.scale_ = {}
            return self

        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=cols)
        numeric = frame.astype("float64", copy=False)
        self.median_ = numeric.median(numeric_only=True).to_dict()
        q75 = numeric.quantile(0.75, numeric_only=True)
        q25 = numeric.quantile(0.25, numeric_only=True)
        iqr = (q75 - q25).to_dict()
        std = numeric.std(numeric_only=True, ddof=0).to_dict()
        self.scale_ = {}
        for col in cols:
            scale = float(iqr.get(col, 0.0))
            if not np.isfinite(scale) or scale < self.eps:
                scale = float(std.get(col, 0.0))
            if not np.isfinite(scale) or scale < self.eps:
                scale = 1.0
            self.scale_[col] = scale
        return self

    def transform(self, X):
        if not self.enabled:
            return X
        if not isinstance(X, pd.DataFrame):
            return X
        out = X.copy()
        for col in out.columns:
            if col in self.median_:
                out[col] = (out[col] - self.median_[col]) / (
                    self.scale_[col] + self.eps
                )
        return out

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return self.feature_names_in_


def _cluster_step() -> Pipeline:
    return Pipeline(
        steps=[
            ("drop_constant", DropConstantFeatures(tol=1, missing_values="ignore")),
            ("drop_duplicates", DropDuplicateFeatures()),
            ("drop_low_variance", VarianceThreshold(threshold=0)),
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


def _sensor_preprocess_column(
    sensor_steps: list[tuple[str, object]],
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "sensor_branch",
                Pipeline(steps=sensor_steps).set_output(transform="pandas"),
                make_column_selector(pattern=_SENSOR_VALUE_PATTERN),
            ),
            *_auxiliary_transformers(),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def linear_preprocess(
    top_k: int = RF_SELECT_TOP_K,
    n_hubs: int = N_HUBS_DEFAULT,
) -> ColumnTransformer:
    """Baseline norm (off default) → impute → cluster → T² + hub pairs; passthrough aux."""
    sensor_steps: list[tuple[str, object]] = [
        #("baseline_norm", SensorBaselineNormalizer()),
        ("impute", median_imputer()),
        ("cluster", _cluster_step()),
        (
            "select_t2_hubs",
            LinearSelectT2HubBlock(top_k=top_k, n_hubs=n_hubs),
        ),
    ]
    return _sensor_preprocess_column(sensor_steps)


def calibrated_classifier(estimator) -> CalibratedClassifierCV:
    """Wrap the base estimator with probability calibration (shared by all models)."""
    return CalibratedClassifierCV(
        estimator=estimator,
        method=CLASSIFIER_CALIBRATION_METHOD,
        cv=int(CLASSIFIER_CALIBRATION_CV),
    )


def feature_pipeline(
    classifier,
    preprocess: ColumnTransformer,
) -> Pipeline:
    """Preprocess → RobustScaler → calibrated classifier."""
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("scale", RobustScaler()),
            ("classifier", calibrated_classifier(classifier)),
        ]
    ).set_output(transform="pandas")

