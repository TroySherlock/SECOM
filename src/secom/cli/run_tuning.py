#!/usr/bin/env python3
"""Run sequential PR-AUC hyperparameter + multi-profile threshold tuning.

Every model is tuned once on the in-distribution repeated-stratified 5x2 CV
(data/processed/tuned/). Those hyperparameters are reused for both the random
and temporal holdouts in the benchmark; time-decay is no longer tuned (it is a
post-hoc temporal-holdout diagnostic sweep computed by the benchmark).

Use --model ID to retune a single pipeline. With no flag, all models are tuned.

After changing classifier calibration settings, re-tune all models and run benchmark.
"""
from __future__ import annotations

import argparse

from secom.pipelines import (
    TARGET_COL,
    clear_pipeline_cache,
    feature_columns,
    load_mart,
    make_repeated_stratified_cv,
    split_train_test_random,
)
from secom.tuning.registry import (
    ALL_MODEL_IDS,
    MODEL_SPECS,
    fit_with_progress,
    patch_threshold_profiles,
    run_grid_search,
    save_tuned_params,
    summarize_cv_search,
    tune_classifier_threshold_profiles,
)
from secom.utils import tuned_params_path

CV_PROTOCOL = "repeated_stratified_5x2"


def _tune_model(model_id: str, spec, X_train, y_train) -> None:
    cv = make_repeated_stratified_cv()
    print(f"\n=== {model_id} ({CV_PROTOCOL}) ===", flush=True)

    search, n_cand, n_splits, total = run_grid_search(spec, X_train, y_train, cv=cv)
    print(f"GridSearch: {n_cand} x {n_splits} = {total} fits", flush=True)
    search = fit_with_progress(search, X_train, y_train)
    cv_summary, fold_results, aggregated = summarize_cv_search(search, spec)
    print(f"Stage 1: mean_pr_auc={cv_summary['mean_pr_auc']:.4f}", flush=True)

    threshold_result = tune_classifier_threshold_profiles(
        spec, X_train, y_train, cv_summary, cv=cv
    )
    for pid, prof in threshold_result["profiles"].items():
        print(
            f"  {pid}: threshold={prof['best_threshold']:.4f}, "
            f"mean_ber={prof['mean_ber_percent']:.2f}%",
            flush=True,
        )
    out_path = tuned_params_path(model_id)
    save_tuned_params(
        spec,
        cv_summary,
        fold_results,
        aggregated,
        threshold_result=threshold_result,
        path=out_path,
        cv_protocol=CV_PROTOCOL,
    )
    print(f"Wrote {out_path}", flush=True)


def _patch_thresholds(model_id: str, spec, X_train, y_train) -> None:
    """Stage-2-only refresh: rewrite threshold profiles in the existing JSON."""
    cv = make_repeated_stratified_cv()
    print(f"\n=== {model_id} (threshold-only refresh) ===", flush=True)
    out_path = tuned_params_path(model_id)
    payload = patch_threshold_profiles(spec, X_train, y_train, cv=cv, path=out_path)
    for pid, prof in (payload.get("threshold_profiles") or {}).items():
        print(
            f"  {pid}: threshold={float(prof['best_threshold']):.4f}, "
            f"mean_ber={float(prof['mean_ber_percent']):.2f}%",
            flush=True,
        )
    print(f"Patched {out_path}", flush=True)


def _select_model_ids(model: str | None) -> list[str]:
    if model is not None:
        if model not in ALL_MODEL_IDS:
            raise SystemExit(
                f"Unknown --model {model!r}; choose from {sorted(ALL_MODEL_IDS)}"
            )
        return [model]
    return list(ALL_MODEL_IDS)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=sorted(ALL_MODEL_IDS),
        help="Tune a single pipeline by model id.",
    )
    parser.add_argument(
        "--clear-pipeline-cache",
        action="store_true",
        help="Clear the joblib preprocess cache before tuning (after editing front-ends).",
    )
    parser.add_argument(
        "--threshold-only",
        action="store_true",
        help=(
            "Skip the Stage-1 grid search; only recompute the threshold profiles "
            "from each model's frozen cv_summary and merge them into the existing "
            "tuned JSON. Use this to bake new profiles (e.g. economic) without "
            "moving the Stage-1 best params."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.clear_pipeline_cache:
        clear_pipeline_cache()
    df = load_mart()
    feature_cols = feature_columns(df)

    train_df, _ = split_train_test_random(df)
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].astype(int)
    for model_id in _select_model_ids(args.model):
        if args.threshold_only:
            _patch_thresholds(model_id, MODEL_SPECS[model_id], X_train, y_train)
        else:
            _tune_model(model_id, MODEL_SPECS[model_id], X_train, y_train)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
