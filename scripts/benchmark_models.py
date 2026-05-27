#!/usr/bin/env python3
"""Compare six tuned pipelines via repeated stratified CV.

Models: mspc_lr, mspc_rf, xgb_mspc, rf_k_lr, rf_k_rf, rf_k_knn.
Each uses frozen hyperparameters from data/processed/tuned/<model_id>.json.
Primary objective: maximize PR AUC on CV; tuned threshold for BER/TPR/TNR.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.progress import tqdm_joblib_context  # noqa: E402
from scripts.secom_metrics import compute_holdout_metrics  # noqa: E402
from scripts.secom_pipelines import (  # noqa: E402
    BENCHMARK_MODEL_IDS,
    BENCHMARK_RESULTS_PATH,
    CV_N_JOBS,
    CV_SCORING,
    RANDOM_SEED,
    RF_MAX_DEPTH,
    RF_N_ESTIMATORS,
    CORRELATED_SELECTION_CRITERION,
    CORRELATED_SELECTION_METHOD,
    CORRELATED_SELECTION_THRESHOLD,
    TARGET_COL,
    TEST_SIZE,
    TUNED_PARAMS_DIR,
    XGB_MAX_DEPTH,
    XGB_N_ESTIMATORS,
    XGB_SCALE_POS_WEIGHT,
    feature_columns,
    load_mart,
    make_repeated_stratified_cv,
    split_train_test,
)
from scripts.secom_utils import (  # noqa: E402
    json_safe,
    load_all_tuned_params,
    score_row_from_cv_result,
)
from scripts.tuning.registry import build_tuned_pipeline  # noqa: E402

PRIMARY_METRIC = "pr_auc"
CV_SORT_COL = "mean_pr_auc"
HOLDOUT_SORT_COL = "pr_auc"
RANKING = "descending_higher_is_better"


def _sensor_mspc_pipeline(preprocess: ColumnTransformer) -> Pipeline:
    for name, trans, _ in preprocess.transformers:
        if name == "sensor_mspc":
            return trans
    raise KeyError("preprocess has no sensor_mspc transformer")


def validate_benchmark_pipelines(
    pipelines: dict,
    X_sample: pd.DataFrame | None = None,
    y_sample: pd.Series | None = None,
) -> None:
    expected = set(BENCHMARK_MODEL_IDS)
    if set(pipelines.keys()) != expected:
        raise RuntimeError(
            f"benchmark must include exactly {sorted(expected)}, got {sorted(pipelines)}"
        )

    for name, pipeline in pipelines.items():
        if "preprocess" not in pipeline.named_steps:
            raise RuntimeError(f"{name} pipeline has no 'preprocess' step")
        if "classifier" not in pipeline.named_steps:
            raise RuntimeError(f"{name} pipeline has no 'classifier' step")
        if "smote" in pipeline.named_steps:
            raise RuntimeError(f"{name} must not include SMOTE")
        if not isinstance(pipeline, Pipeline):
            raise TypeError(f"{name} must be sklearn.pipeline.Pipeline")

        preprocess = pipeline.named_steps["preprocess"]
        if not isinstance(preprocess, ColumnTransformer):
            raise TypeError(f"{name} preprocess must be ColumnTransformer")

        sensor_pipe = _sensor_mspc_pipeline(preprocess)
        if "cluster" not in sensor_pipe.named_steps:
            raise RuntimeError(f"{name} preprocess missing 'cluster' step")
        is_rf_k = name.startswith("rf_k")
        if is_rf_k:
            if "select" not in sensor_pipe.named_steps:
                raise RuntimeError(f"{name} RF-K preprocess missing 'select' step")
            if "t2" not in sensor_pipe.named_steps:
                raise RuntimeError(f"{name} RF-K preprocess missing 't2' step")
            if "pls" in sensor_pipe.named_steps:
                raise RuntimeError(f"{name} RF-K preprocess must not include PLS step")
        else:
            if "pls" not in sensor_pipe.named_steps:
                raise RuntimeError(f"{name} MSPC preprocess missing 'pls' step")
            if "t2" not in sensor_pipe.named_steps:
                raise RuntimeError(f"{name} MSPC preprocess missing 't2' step")

    if X_sample is None or y_sample is None:
        return
    missing_in = [c for c in X_sample.columns if c.endswith("__missing")]
    if not missing_in:
        return
    mspc_ref = pipelines["mspc_lr"]
    preprocess = mspc_ref.named_steps["preprocess"]
    preprocess.fit(X_sample, y_sample)
    out = {str(n) for n in preprocess.get_feature_names_out()}
    if not any(n.endswith("__missing") for n in out):
        raise RuntimeError(
            "Mart has c_*__missing columns but preprocess did not passthrough missing_flags."
        )
    if not {"hotelling_t2", "q_statistic"}.issubset(out):
        raise RuntimeError(
            "MSPC preprocess did not emit hotelling_t2 and q_statistic."
        )

    rf_preprocess = pipelines["rf_k_lr"].named_steps["preprocess"]
    rf_preprocess.fit(X_sample, y_sample)
    rf_out = {str(n) for n in rf_preprocess.get_feature_names_out()}
    if "hotelling_t2" not in rf_out:
        raise RuntimeError("RF-K preprocess did not emit hotelling_t2.")
    if "q_statistic" in rf_out:
        raise RuntimeError("RF-K preprocess must not emit q_statistic.")


def build_benchmark_pipelines(tuned: dict[str, dict] | None = None) -> dict[str, Pipeline]:
    tuned = tuned or load_all_tuned_params()
    pipelines: dict[str, Pipeline] = {}
    for model_id in BENCHMARK_MODEL_IDS:
        if model_id not in tuned:
            raise KeyError(f"Missing tuned payload for {model_id}")
        pipelines[model_id] = build_tuned_pipeline(model_id, tuned[model_id])
    return pipelines


def run_pipeline_benchmark(
    pipelines: dict,
    X: pd.DataFrame,
    y: pd.Series,
    cv=None,
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    validate_benchmark_pipelines(pipelines, X, y)
    cv = cv or make_repeated_stratified_cv()
    n_folds = cv.get_n_splits(X, y)
    rows = []

    if show_progress:
        print(f"Benchmark: {len(pipelines)} pipelines x {n_folds} folds")

    iterator = pipelines.items()
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(
                iterator,
                total=len(pipelines),
                desc="Benchmark pipeline",
                position=0,
            )
        except ImportError:
            pass

    for name, pipeline in iterator:
        with tqdm_joblib_context(n_folds, f"CV {name}", leave=False):
            result = cross_validate(
                pipeline,
                X,
                y,
                cv=cv,
                scoring=CV_SCORING,
                n_jobs=CV_N_JOBS,
                error_score="raise",
            )
        row = {"pipeline": name, **score_row_from_cv_result(result)}
        rows.append(row)
        if show_progress:
            print(
                f"  {name}: mean PR AUC {row['mean_pr_auc']:.3f} "
                f"(±{row['std_pr_auc']:.3f}), ROC AUC {row['mean_roc_auc']:.3f}"
            )

    leaderboard = pd.DataFrame(rows).sort_values(
        CV_SORT_COL, ascending=False, kind="mergesort"
    )
    return leaderboard.reset_index(drop=True)


def run_holdout_benchmark(
    pipelines: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    validate_benchmark_pipelines(pipelines)
    rows = []

    if show_progress:
        print(f"Holdout: {len(pipelines)} pipelines (reporting only)")

    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_score = pipeline.predict_proba(X_test)[:, 1]
        row = {"pipeline": name, **compute_holdout_metrics(y_test, y_pred, y_score)}
        rows.append(row)
        if show_progress:
            print(
                f"  {name}: holdout PR AUC {row['pr_auc']:.3f}, "
                f"BER {row['ber_percent']:.1f}%"
            )

    holdout = pd.DataFrame(rows).sort_values(
        HOLDOUT_SORT_COL, ascending=False, kind="mergesort"
    )
    return holdout.reset_index(drop=True)


def save_benchmark_results(
    tuned: dict[str, dict],
    leaderboard: pd.DataFrame,
    holdout: pd.DataFrame | None = None,
    *,
    train_rows: int | None = None,
    test_rows: int | None = None,
    path: Path = BENCHMARK_RESULTS_PATH,
) -> dict:
    payload = {
        "primary_metric": PRIMARY_METRIC,
        "ranking": RANKING,
        "tuned_params_dir": str(TUNED_PARAMS_DIR),
        "model_ids": list(BENCHMARK_MODEL_IDS),
        "tuned_hyperparameters": {
            model_id: tuned[model_id].get("grid_search_best_params", {})
            for model_id in BENCHMARK_MODEL_IDS
        },
        "correlated_selection": {
            "library": "feature_engine",
            "steps": ["DropConstantFeatures", "DropDuplicateFeatures", "SmartCorrelatedSelection"],
            "drop_constant_tol": 1,
            "method": CORRELATED_SELECTION_METHOD,
            "threshold": float(CORRELATED_SELECTION_THRESHOLD),
            "selection_method": CORRELATED_SELECTION_CRITERION,
            "missing_values": "ignore",
        },
        "benchmark_rf_n_estimators": int(RF_N_ESTIMATORS),
        "benchmark_rf_max_depth": int(RF_MAX_DEPTH),
        "benchmark_xgb_n_estimators": int(XGB_N_ESTIMATORS),
        "benchmark_xgb_max_depth": int(XGB_MAX_DEPTH),
        "benchmark_xgb_scale_pos_weight": float(XGB_SCALE_POS_WEIGHT),
        "holdout_is_reporting_only": True,
        "holdout_split": {
            "test_size": float(TEST_SIZE),
            "random_seed": int(RANDOM_SEED),
            "train_rows": train_rows,
            "test_rows": test_rows,
        },
        "leaderboard": leaderboard.to_dict(orient="records"),
    }
    if holdout is not None:
        payload["holdout"] = holdout.to_dict(orient="records")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    return payload


def main() -> None:
    tuned = load_all_tuned_params()
    df = load_mart()
    cols = feature_columns(df)
    train_df, test_df = split_train_test(df)
    X_train = train_df[cols]
    y_train = train_df[TARGET_COL].astype(int)
    X_test = test_df[cols]
    y_test = test_df[TARGET_COL].astype(int)

    pipelines = build_benchmark_pipelines(tuned)
    leaderboard = run_pipeline_benchmark(pipelines, X_train, y_train, show_progress=True)
    holdout = run_holdout_benchmark(
        pipelines, X_train, y_train, X_test, y_test, show_progress=True
    )
    save_benchmark_results(
        tuned,
        leaderboard,
        holdout,
        train_rows=len(train_df),
        test_rows=len(test_df),
    )

    print(f"\nWrote {BENCHMARK_RESULTS_PATH}\n")
    print("CV leaderboard:")
    print(leaderboard.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nHoldout (reporting only):")
    print(holdout.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
