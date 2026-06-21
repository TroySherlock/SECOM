#!/usr/bin/env python3
"""Run sequential PR-AUC hyperparameter + multi-profile threshold tuning.

Protocol is routed by each model's track:
  - interpolation track → repeated_stratified_5x2 → data/processed/tuned/
  - extrapolation track → blocked_time_local_strat → data/processed/tuned_blocked/

Use --model ID to retune a single pipeline, or --track {interpolation,extrapolation}
to retune one track. With no flag, all models are tuned.

After changing classifier calibration settings, re-tune all models and run benchmark.
"""
from __future__ import annotations

import argparse

from secom.cv import make_blocked_time_cv
from secom.pipelines import (
    DECAY_LAMBDA_DEFAULT,
    TARGET_COL,
    TIMESTAMP_COL,
    WEIGHTING_MODEL_IDS,
    clear_pipeline_cache,
    feature_columns,
    load_mart,
    make_repeated_stratified_cv,
    split_train_test,
    split_train_test_random,
)
from secom.tuning.registry import (
    ALL_MODEL_IDS,
    MODEL_SPECS,
    TRACKS,
    fit_with_progress,
    model_ids_for_track,
    run_grid_search,
    save_tuned_params,
    summarize_cv_search,
    tune_classifier_threshold_profiles,
    tune_time_decay_lambda,
)
from secom.utils import tuned_blocked_params_path, tuned_params_path

# CV pass per track: (protocol name, cv factory, output path fn). Each of the 9
# model cells is tuned on BOTH protocols.
CV_PASS_BY_TRACK = {
    "interpolation": (
        "repeated_stratified_5x2",
        make_repeated_stratified_cv,
        tuned_params_path,
    ),
    "extrapolation": (
        "blocked_time_local_strat",
        make_blocked_time_cv,
        tuned_blocked_params_path,
    ),
}


def _train_for_track(df, track: str):
    """Track-appropriate training split: random-stratified vs temporal."""
    if track == "extrapolation":
        train_df, _ = split_train_test(df)
    else:
        train_df, _ = split_train_test_random(df)
    return train_df


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


def _select_model_ids(model: str | None, track: str | None) -> list[str]:
    if model is not None:
        if model not in ALL_MODEL_IDS:
            raise SystemExit(
                f"Unknown --model {model!r}; choose from {sorted(ALL_MODEL_IDS)}"
            )
        return [model]
    return model_ids_for_track(track)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=sorted(ALL_MODEL_IDS),
        help="Tune a single pipeline by model id.",
    )
    parser.add_argument(
        "--track",
        choices=sorted(TRACKS),
        help="Tune only models in this track (ignored if --model is given).",
    )
    parser.add_argument(
        "--clear-pipeline-cache",
        action="store_true",
        help="Clear the joblib preprocess cache before tuning (after editing front-ends).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.clear_pipeline_cache:
        clear_pipeline_cache()
    df = load_mart()
    feature_cols = feature_columns(df)

    tracks = [args.track] if args.track else list(TRACKS)
    for track in tracks:
        cv_protocol, cv_factory, out_path_fn = CV_PASS_BY_TRACK[track]
        train_df = _train_for_track(df, track)
        X_train = train_df[feature_cols]
        y_train = train_df[TARGET_COL].astype(int)
        for model_id in _select_model_ids(args.model, track):
            _tune_pass(
                model_id,
                MODEL_SPECS[model_id],
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
