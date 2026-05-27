"""Shared SECOM pipeline builders, data loading, and tuning constants."""
from __future__ import annotations

from functools import partial
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures, SmartCorrelatedSelection
from xgboost import XGBClassifier

from scripts.mspc_features import MahalanobisT2Features, PLSWithQFeatures

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "secom.duckdb"
SOURCE_RELATION = "public.mart_secom_features"
OUTPUT_DIR = REPO_ROOT / "data" / "processed"
CHAMPION_PARAMS_PATH = OUTPUT_DIR / "secom_champion_params.json"
TUNED_PARAMS_DIR = OUTPUT_DIR / "tuned"
BENCHMARK_RESULTS_PATH = OUTPUT_DIR / "secom_pipeline_benchmark.json"

BENCHMARK_MODEL_IDS = (
    "mspc_lr",
    "mspc_rf",
    "xgb_mspc",
    "rf_k_lr",
    "rf_k_rf",
    "rf_k_knn",
)

TARGET_COL = "target"
TIMESTAMP_COL = "measurement_ts"
ID_COL = "observation_id"

RANDOM_SEED = 42
TEST_SIZE = 0.20
N_SPLITS = 5
N_REPEATS = 5
GRID_SEARCH_VERBOSE = 1

C_GRID = [0.001, 0.01]
L1_RATIO_GRID = [0.05, 0.5, 0.95]

MODEL_NAME = "mspc_elastic_net_logistic"

N_MISSING_SENSORS_COL = "n_missing_sensors"
CHAMPION_IMPUTATION_METHOD = "knn"
KNN_IMPUTE_NEIGHBORS = 5
ELASTIC_NET_MAX_ITER = 500_000

PLS_N_COMPONENTS = 15
PLS_N_COMPONENTS_GRID = [5, 10, 15]

KNN_CLASSIFIER_NEIGHBORS = 10
KNN_CLASSIFIER_WEIGHTS = "uniform"
KNN_NEIGHBORS_GRID = [5, 10, 15]

RF_N_ESTIMATORS = 1000
RF_MAX_DEPTH = 4
RF_MAX_DEPTH_GRID = [3, 4, 6, 8]
RF_MIN_SAMPLES_LEAF = 10
RF_SELECT_TOP_K = 15
RF_SELECT_TOP_K_GRID = [20, 30, 40, 50]

CORRELATED_SELECTION_THRESHOLD = 0.80
CORRELATED_SELECTION_METHOD = "spearman"
CORRELATED_SELECTION_CRITERION = "corr_with_target"

XGB_N_ESTIMATORS = 500
XGB_MAX_DEPTH = 6
XGB_MAX_DEPTH_GRID = [4, 6, 8]
XGB_LEARNING_RATE = 0.05
XGB_LEARNING_RATE_GRID = [0.03, 0.05, 0.1]
XGB_SCALE_POS_WEIGHT = 13.0

CV_N_JOBS = -1
ESTIMATOR_N_JOBS = 1

PRIMARY_TUNING_METRIC = "pr_auc"
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.05)

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

elastic_net = partial(
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
    scale_pos_weight=XGB_SCALE_POS_WEIGHT,
    random_state=RANDOM_SEED,
    n_jobs=ESTIMATOR_N_JOBS,
    verbosity=0,
    eval_metric="logloss",
)


def frozen_config() -> dict:
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
        "tuning_protocol": "sequential_pr_auc_hyperparams_ber_threshold",
        "primary_tuning_metric": PRIMARY_TUNING_METRIC,
        "threshold_tuning_metric": "ber",
        "threshold_grid": [float(t) for t in THRESHOLD_GRID],
        "elastic_net_max_iter": int(ELASTIC_NET_MAX_ITER),
        "pls_n_components": int(PLS_N_COMPONENTS),
        "pls_n_components_grid": [int(k) for k in PLS_N_COMPONENTS_GRID],
        "knn_classifier_neighbors": int(KNN_CLASSIFIER_NEIGHBORS),
        "knn_neighbors_grid": [int(k) for k in KNN_NEIGHBORS_GRID],
        "rf_n_estimators": int(RF_N_ESTIMATORS),
        "rf_max_depth": int(RF_MAX_DEPTH),
        "rf_max_depth_grid": [int(k) for k in RF_MAX_DEPTH_GRID],
        "xgb_n_estimators": int(XGB_N_ESTIMATORS),
        "xgb_max_depth": int(XGB_MAX_DEPTH),
        "xgb_max_depth_grid": [int(k) for k in XGB_MAX_DEPTH_GRID],
        "xgb_learning_rate_grid": [float(x) for x in XGB_LEARNING_RATE_GRID],
        "xgb_scale_pos_weight": float(XGB_SCALE_POS_WEIGHT),
        "rf_select_top_k": int(RF_SELECT_TOP_K),
        "rf_select_top_k_grid": [int(k) for k in RF_SELECT_TOP_K_GRID],
        "benchmark_model_ids": list(BENCHMARK_MODEL_IDS),
        "correlated_selection_threshold": float(CORRELATED_SELECTION_THRESHOLD),
        "correlated_selection_method": CORRELATED_SELECTION_METHOD,
        "correlated_selection_criterion": CORRELATED_SELECTION_CRITERION,
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


def _mspc_auxiliary_transformers() -> list[tuple[str, str, object]]:
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
            (
                "smart_corr",
                SmartCorrelatedSelection(
                    method=CORRELATED_SELECTION_METHOD,
                    threshold=CORRELATED_SELECTION_THRESHOLD,
                    selection_method=CORRELATED_SELECTION_CRITERION,
                    missing_values="ignore",
                ),
            ),
        ],
    ).set_output(transform="pandas")


def mspc_preprocess(n_components: int = PLS_N_COMPONENTS) -> ColumnTransformer:
    """Impute c_* → cluster → PLS+Q → Mahalanobis T² on pls scores; passthrough aux."""
    return ColumnTransformer(
        transformers=[
            (
                "sensor_mspc",
                Pipeline(
                    steps=[
                        ("impute", median_imputer()),
                        ("cluster", _cluster_step()),
                        ("pls", PLSWithQFeatures(n_components=n_components)),
                        (
                            "t2",
                            MahalanobisT2Features(score_columns=r"^pls_\d+$"),
                        ),
                    ]
                ).set_output(transform="pandas"),
                make_column_selector(pattern=_SENSOR_VALUE_PATTERN),
            ),
            *_mspc_auxiliary_transformers(),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def rf_top_k_preprocess(top_k: int = RF_SELECT_TOP_K) -> ColumnTransformer:
    """Impute c_* → cluster → RF SelectFromModel top_k → Mahalanobis T²; passthrough aux."""
    return ColumnTransformer(
        transformers=[
            (
                "sensor_mspc",
                Pipeline(
                    steps=[
                        ("impute", median_imputer()),
                        ("cluster", _cluster_step()),
                        (
                            "select",
                            SelectFromModel(
                                random_forest_classifier(),
                                max_features=top_k,
                                threshold=-np.inf,
                            ),
                        ),
                        ("t2", MahalanobisT2Features()),
                    ]
                ).set_output(transform="pandas"),
                make_column_selector(pattern=_SENSOR_VALUE_PATTERN),
            ),
            *_mspc_auxiliary_transformers(),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def feature_pipeline(classifier, preprocess: ColumnTransformer) -> Pipeline:
    """Preprocess → RobustScaler → classifier."""
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("scale", RobustScaler()),
            ("classifier", classifier),
        ]
    ).set_output(transform="pandas")


def secom_pipeline(
    classifier,
    *,
    n_components: int = PLS_N_COMPONENTS,
    preprocess: ColumnTransformer | None = None,
) -> Pipeline:
    """Default: MSPC preprocess → scale → classifier."""
    preprocess = preprocess or mspc_preprocess(n_components)
    return feature_pipeline(classifier, preprocess)

