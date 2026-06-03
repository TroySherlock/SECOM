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
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from scripts.hub_interactions import LinearSelectT2HubBlock
REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "secom.duckdb"
SOURCE_RELATION = "public.mart_secom_features"
OUTPUT_DIR = REPO_ROOT / "data" / "processed"
# Dashboard data contract (regenerate: python -m scripts.benchmark_models after tuning):
#   tuned/<model_id>.json           — frozen hyperparameters from tuning notebooks
#   secom_pipeline_benchmark.json   — CV leaderboard + holdout metrics
#   secom_pipeline_artifacts.json   — holdout-fit pipeline reporting (feature counts, RF, clusters)
TUNED_PARAMS_DIR = OUTPUT_DIR / "tuned"
BENCHMARK_RESULTS_PATH = OUTPUT_DIR / "secom_pipeline_benchmark.json"
PIPELINE_ARTIFACTS_PATH = OUTPUT_DIR / "secom_pipeline_artifacts.json"

BENCHMARK_MODEL_IDS = (
    "linear_lr",
    "topk_rf",
    "topk_knn",
    "topk_xgb",
)

TARGET_COL = "target"
TIMESTAMP_COL = "measurement_ts"
ID_COL = "observation_id"

RANDOM_SEED = 42
TEST_SIZE = 0.20
N_SPLITS = 5
N_REPEATS = 2
GRID_SEARCH_VERBOSE = 1

C_GRID = [0.005, 0.0075]
L1_RATIO_GRID = [0.3, 0.4, 0.5]

MODEL_NAME = "secom_linear_elastic_net"

N_MISSING_SENSORS_COL = "n_missing_sensors"
CHAMPION_IMPUTATION_METHOD = "median"
KNN_IMPUTE_NEIGHBORS = 5
ELASTIC_NET_MAX_ITER = 20000

KNN_CLASSIFIER_NEIGHBORS = 10
KNN_CLASSIFIER_WEIGHTS = "uniform"
KNN_NEIGHBORS_GRID = [35]

RF_N_ESTIMATORS = 1000
RF_MAX_DEPTH = 6
RF_MAX_DEPTH_GRID = [8, 10, 12, 16]
RF_MIN_SAMPLES_LEAF = 10
RF_SELECT_TOP_K = 35
RF_SELECT_TOP_K_GRID = [35]

N_HUBS_DEFAULT = 5
N_HUBS_GRID = [5]

CORRELATED_SELECTION_THRESHOLD = 0.7
CORRELATED_SELECTION_THRESHOLD_GRID = [0.85]
CORRELATED_SELECTION_METHOD = "spearman"
CORRELATED_SELECTION_CRITERION = "corr_with_target"

XGB_N_ESTIMATORS = 1000
XGB_MAX_DEPTH = 4
XGB_MAX_DEPTH_GRID = [8, 12, 16, 18]
XGB_LEARNING_RATE = 0.05
XGB_LEARNING_RATE_GRID = [0.005, 0.01, 0.03, 0.1]
XGB_SCALE_POS_WEIGHT = 14.151515

CV_N_JOBS = -1
ESTIMATOR_N_JOBS = 1

PRIMARY_TUNING_METRIC = "pr_auc"
THRESHOLD_GRID = np.linspace(0.001, 0.999, num=1000)

LINEAR_CALIBRATION_METHOD = "isotonic"
LINEAR_CALIBRATION_CV = 3

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
    from scripts.secom_costs import threshold_profile_config

    return {
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "c_grid": [float(c) for c in C_GRID],
        "l1_ratio_grid": [float(r) for r in L1_RATIO_GRID],
        "model_name": MODEL_NAME,
        "champion_imputation_method": CHAMPION_IMPUTATION_METHOD,
        "knn_impute_neighbors": int(KNN_IMPUTE_NEIGHBORS),
        "tuning_protocol": "sequential_pr_auc_hyperparams_multi_threshold",
        "primary_tuning_metric": PRIMARY_TUNING_METRIC,
        "threshold_tuning_profiles": ["f1", "f2", "f3"],
        "threshold_grid": [float(t) for t in THRESHOLD_GRID],
        "linear_calibration_method": str(LINEAR_CALIBRATION_METHOD),
        "linear_calibration_cv": int(LINEAR_CALIBRATION_CV),
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
    test_size: float = TEST_SIZE,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=df[target_col],
        shuffle=True,
    )
    return train_df.sort_values(ID_COL).copy(), test_df.sort_values(ID_COL).copy()


def make_repeated_stratified_cv() -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
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


def _cluster_step() -> Pipeline:
    return Pipeline(
        steps=[
            ("drop_constant", DropConstantFeatures(tol=1)),
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
    """Impute → cluster → T² + hub pairs; passthrough aux."""
    sensor_steps: list[tuple[str, object]] = [
        ("impute", median_imputer()),
        ("cluster", _cluster_step()),
        (
            "select_t2_hubs",
            LinearSelectT2HubBlock(top_k=top_k, n_hubs=n_hubs),
        ),
    ]
    return _sensor_preprocess_column(sensor_steps)


def feature_pipeline(
    classifier,
    preprocess: ColumnTransformer,
) -> Pipeline:
    """Preprocess → RobustScaler → classifier."""
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("scale", RobustScaler()),
            ("classifier", classifier),
        ]
    ).set_output(transform="pandas")

