#!/usr/bin/env python3
"""Run sequential PR-AUC hyperparameter + multi-profile threshold tuning for all models.

Runs two CV protocols per model:
  - repeated_stratified_5x2 → data/processed/tuned/
  - blocked_time_local_strat → data/processed/tuned_blocked/

After changing classifier calibration settings, re-tune all models and run benchmark.
"""
from __future__ import annotations

from secom.cv import make_blocked_time_cv
from secom.pipelines import (
    DECAY_LAMBDA_DEFAULT,
    TARGET_COL,
    TIMESTAMP_COL,
    TUNED_BLOCKED_PARAMS_DIR,
    TUNED_PARAMS_DIR,
    WEIGHTING_MODEL_IDS,
    feature_columns,
    load_mart,
    make_repeated_stratified_cv,
    split_train_test,
)
from secom.tuning.registry import (
    MODEL_SPECS,
    fit_with_progress,
    run_grid_search,
    save_tuned_params,
    summarize_cv_search,
    tune_classifier_threshold_profiles,
    tune_time_decay_lambda,
)
from secom.utils import tuned_blocked_params_path, tuned_params_path

CV_PASSES = (
    ("repeated_stratified_5x2", make_repeated_stratified_cv, TUNED_PARAMS_DIR, tuned_params_path),
    (
        "blocked_time_local_strat",
        make_blocked_time_cv,
        TUNED_BLOCKED_PARAMS_DIR,
        tuned_blocked_params_path,
    ),
)


def _tune_pass(
    model_id: str,
    spec,
    train_df,
    X_train,
    y_train,
    cv_protocol: str,
    cv_factory,
    out_path_fn,
) -> None:
    if cv_protocol == "blocked_time_local_strat":
        cv = cv_factory(train_df)
    else:
        cv = cv_factory()

    print(f"\n=== {model_id} ({cv_protocol}) ===", flush=True)
    extrapolation = cv_protocol == "blocked_time_local_strat"
    weight_capable = extrapolation and model_id in WEIGHTING_MODEL_IDS
    timestamps = train_df[TIMESTAMP_COL].reset_index(drop=True) if weight_capable else None

    search, n_cand, n_splits, total = run_grid_search(
        spec, X_train, y_train, cv=cv
    )
    print(f"GridSearch: {n_cand} x {n_splits} = {total} fits", flush=True)
    search = fit_with_progress(search, X_train, y_train)
    cv_summary, fold_results, aggregated = summarize_cv_search(search, spec)
    print(f"Stage 1: mean_pr_auc={cv_summary['mean_pr_auc']:.4f}", flush=True)

    decay_lambda = DECAY_LAMBDA_DEFAULT
    decay_search = None
    if weight_capable:
        decay_search = tune_time_decay_lambda(
            spec, X_train, y_train, timestamps, cv_summary, cv
        )
        decay_lambda = decay_search["best_decay_lambda"]
        print(
            f"Stage 1b: decay_lambda={decay_lambda:.2f} "
            f"(mean_pr_auc={decay_search['best_mean_pr_auc']:.4f})",
            flush=True,
        )

    threshold_result = tune_classifier_threshold_profiles(
        spec,
        X_train,
        y_train,
        cv_summary,
        cv=cv,
        timestamps=timestamps,
        decay_lambda=decay_lambda,
    )
    for pid, prof in threshold_result["profiles"].items():
        if prof.get("objective") == "ber":
            print(
                f"  {pid}: threshold={prof['best_threshold']:.4f}, "
                f"mean_ber={prof['mean_ber_percent']:.2f}%",
                flush=True,
            )
        else:
            print(
                f"  {pid}: threshold={prof['best_threshold']:.4f}, "
                f"mean_fbeta={prof['mean_fbeta']:.4f}, "
                f"mean_ber={prof['mean_ber_percent']:.2f}%",
                flush=True,
            )
    save_tuned_params(
        spec,
        cv_summary,
        fold_results,
        aggregated,
        threshold_result=threshold_result,
        path=out_path_fn(model_id),
        cv_protocol=cv_protocol,
        decay_lambda=decay_lambda,
        decay_lambda_search=decay_search,
    )
    print(f"Wrote {out_path_fn(model_id)}", flush=True)


def main() -> int:
    df = load_mart()
    feature_cols = feature_columns(df)
    train_df, _ = split_train_test(df)
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].astype(int)

    for model_id, spec in MODEL_SPECS.items():
        for cv_protocol, cv_factory, _out_dir, out_path_fn in CV_PASSES:
            _tune_pass(
                model_id,
                spec,
                train_df,
                X_train,
                y_train,
                cv_protocol,
                cv_factory,
                out_path_fn,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
