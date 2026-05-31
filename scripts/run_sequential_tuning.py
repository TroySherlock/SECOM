#!/usr/bin/env python3
"""Run sequential PR-AUC hyperparameter + BER threshold tuning for all four models."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.secom_pipelines import TARGET_COL, feature_columns, load_mart, split_train_test
from scripts.tuning.registry import (
    MODEL_SPECS,
    fit_with_progress,
    run_grid_search,
    save_tuned_params,
    summarize_cv_search,
    tune_classifier_threshold,
    tuned_params_path,
)


def main() -> None:
    df = load_mart()
    feature_cols = feature_columns(df)
    train_df, _ = split_train_test(df)
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].astype(int)

    for model_id in MODEL_SPECS:
        print(f"\n=== {model_id} ===", flush=True)
        spec = MODEL_SPECS[model_id]
        search, n_cand, n_splits, total = run_grid_search(spec, X_train, y_train)
        print(f"GridSearch: {n_cand} x {n_splits} = {total} fits", flush=True)
        search = fit_with_progress(search, X_train, y_train)
        cv_summary, fold_results, aggregated = summarize_cv_search(search, spec)
        print(f"Stage 1: mean_pr_auc={cv_summary['mean_pr_auc']:.4f}", flush=True)
        threshold_result = tune_classifier_threshold(spec, X_train, y_train, cv_summary)
        print(
            f"Stage 2: threshold={threshold_result['best_threshold']:.2f}, "
            f"mean_ber={threshold_result['mean_ber_percent']:.2f}%",
            flush=True,
        )
        save_tuned_params(
            spec,
            cv_summary,
            fold_results,
            aggregated,
            threshold_result=threshold_result,
        )
        print(f"Wrote {tuned_params_path(model_id)}", flush=True)


if __name__ == "__main__":
    main()
