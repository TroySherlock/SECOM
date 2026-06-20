#!/usr/bin/env python3
"""Benchmark interpolation and extrapolation track pipelines.

Interpolation track (intrap_*): stratified CV + random holdout, from
data/processed/tuned/<model_id>.json.
Extrapolation track (extrap_*): blocked CV + temporal holdout + process gate +
time-decay, from data/processed/tuned_blocked/<model_id>.json.

Use --model ID or --track {interpolation,extrapolation} to benchmark a subset;
subset runs merge into the existing benchmark JSON instead of overwriting it.
Primary objective: maximize PR AUC on CV; F-beta thresholds (F0.5 / F2 / F4) plus BER-min.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from secom.progress import tqdm_joblib_context
from secom.costs import (
    DEFAULT_PROFILE_ID,
    PROFILE_IDS,
    THRESHOLD_PROFILES,
    fbeta_at_threshold,
    resolve_threshold_profiles,
    threshold_profile_config,
)
from secom.metrics import (
    compute_holdout_metrics,
    predict_with_threshold,
    stratified_bootstrap_holdout_metrics,
)
from secom.artifacts import (
    collect_holdout_artifacts,
    save_pipeline_artifacts,
)
from secom.cv import make_blocked_time_cv
from secom.gates import BayesGate, EFAGate
from secom.pipelines import (
    BENCHMARK_MODEL_IDS,
    BENCHMARK_RESULTS_PATH,
    MODEL_IDS,
    PIPELINE_ARTIFACTS_PATH,
    CV_N_JOBS,
    CV_SCORING,
    DECAY_LAMBDA_DEFAULT,
    RANDOM_SEED,
    RF_MAX_DEPTH,
    RF_N_ESTIMATORS,
    RISK_COVERAGE_GRID,
    CORRELATED_SELECTION_CRITERION,
    CORRELATED_SELECTION_METHOD,
    CORRELATED_SELECTION_THRESHOLD,
    HOLDOUT_BOOTSTRAP_CI,
    HOLDOUT_BOOTSTRAP_N,
    TARGET_COL,
    TEST_SIZE,
    TIMESTAMP_COL,
    TUNED_PARAMS_DIR,
    TUNED_BLOCKED_PARAMS_DIR,
    WEIGHTING_MODEL_IDS,
    feature_columns,
    frozen_config,
    holdout_split_summary,
    load_mart,
    make_repeated_stratified_cv,
    split_train_test,
    split_train_test_random,
    time_decay_weights,
)
from secom.utils import (
    json_safe,
    load_all_tuned_blocked_params,
    load_all_tuned_params,
    score_row_from_cv_result,
)
from secom.tuning.registry import build_tuned_pipeline, fit_pipeline_weighted

MIN_CONDITIONAL_POSITIVES = 5

PRIMARY_METRIC = "pr_auc"
CV_SORT_COL = "mean_pr_auc"
HOLDOUT_SORT_COL = "pr_auc"
RANKING = "descending_higher_is_better"

CV_PROTOCOL_IN_DIST = "repeated_stratified_5x2"
CV_PROTOCOL_EXTRAP = "blocked_time_local_strat"


def build_benchmark_pipelines(
    tuned: dict[str, dict] | None = None,
    *,
    extrapolation: bool = False,
    model_ids=None,
) -> dict[str, Pipeline]:
    if tuned is None:
        tuned = (
            load_all_tuned_blocked_params() if extrapolation else load_all_tuned_params()
        )
    if model_ids is None:
        model_ids = list(tuned.keys())
    pipelines: dict[str, Pipeline] = {}
    for model_id in model_ids:
        if model_id not in tuned:
            raise KeyError(f"Missing tuned payload for {model_id}")
        pipelines[model_id] = build_tuned_pipeline(model_id, tuned[model_id])
    return pipelines


def _model_decay_lambda(name: str, tuned: dict[str, dict] | None) -> float:
    """Tuned decay lambda for a weight-capable model; 0 otherwise."""
    if name not in WEIGHTING_MODEL_IDS:
        return 0.0
    return float((tuned or {}).get(name, {}).get("decay_lambda", DECAY_LAMBDA_DEFAULT))


def _holdout_sample_weights(
    name: str,
    tuned: dict[str, dict] | None,
    train_timestamps: pd.Series | None,
) -> np.ndarray | None:
    """Time-decay sample weights for a full-train fit (None if unweighted).

    Returns ``None`` when there is no weighting so lambda=0 reproduces the
    unweighted fit exactly and k-NN (no sample_weight support) is never weighted.
    """
    if train_timestamps is None:
        return None
    decay_lambda = _model_decay_lambda(name, tuned)
    if not decay_lambda:
        return None
    return time_decay_weights(train_timestamps, decay_lambda)


def run_pipeline_benchmark(
    pipelines: dict,
    X: pd.DataFrame,
    y: pd.Series,
    cv=None,
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
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


def run_weighted_blocked_leaderboard(
    pipelines: dict,
    X: pd.DataFrame,
    y: pd.Series,
    train_timestamps: pd.Series,
    cv,
    tuned_blocked: dict[str, dict] | None = None,
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Blocked-CV leaderboard with per-fold time-decay weights.

    Mirrors ``cross_validate`` over ``CV_SCORING`` but fits each fold with
    time-decay sample weights computed from that fold's own train rows, so it
    matches the deployment-time weighted fit. Unweighted models (k-NN, lambda=0)
    reproduce the plain blocked-CV numbers exactly.
    """
    splits = list(cv.split(X, y))
    rows = []
    if show_progress:
        print(f"Weighted blocked CV: {len(pipelines)} pipelines x {len(splits)} folds")

    for name, pipeline in pipelines.items():
        decay_lambda = _model_decay_lambda(name, tuned_blocked)
        per_fold: dict[str, list[float]] = {
            "balanced_accuracy": [],
            "true_positive_rate": [],
            "true_negative_rate": [],
            "roc_auc": [],
            "pr_auc": [],
        }
        for train_idx, val_idx in splits:
            weights = None
            if decay_lambda and name in WEIGHTING_MODEL_IDS:
                weights = time_decay_weights(
                    train_timestamps.iloc[train_idx], decay_lambda
                )
            fold_pipe, threshold = fit_pipeline_weighted(
                clone(pipeline), X.iloc[train_idx], y.iloc[train_idx], weights
            )
            y_val = y.iloc[val_idx]
            proba = fold_pipe.predict_proba(X.iloc[val_idx])[:, 1]
            y_pred = (
                predict_with_threshold(proba, threshold)
                if threshold is not None
                else fold_pipe.predict(X.iloc[val_idx])
            )
            per_fold["balanced_accuracy"].append(
                float(balanced_accuracy_score(y_val, y_pred))
            )
            per_fold["true_positive_rate"].append(
                float(recall_score(y_val, y_pred, zero_division=0))
            )
            per_fold["true_negative_rate"].append(
                float(recall_score(y_val, y_pred, pos_label=0, zero_division=0))
            )
            per_fold["roc_auc"].append(float(roc_auc_score(y_val, proba)))
            per_fold["pr_auc"].append(float(average_precision_score(y_val, proba)))

        result = {f"test_{k}": np.asarray(v) for k, v in per_fold.items()}
        row = {"pipeline": name, **score_row_from_cv_result(result)}
        rows.append(row)
        if show_progress:
            print(
                f"  {name}: mean PR AUC {row['mean_pr_auc']:.3f} "
                f"(±{row['std_pr_auc']:.3f}), ROC AUC {row['mean_roc_auc']:.3f} "
                f"[lambda={decay_lambda:.2f}]"
            )

    leaderboard = pd.DataFrame(rows).sort_values(
        CV_SORT_COL, ascending=False, kind="mergesort"
    )
    return leaderboard.reset_index(drop=True)


def fit_holdout_pipelines(
    pipelines: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tuned: dict[str, dict] | None,
    train_timestamps: pd.Series | None = None,
) -> dict[str, Pipeline]:
    """Fit each tuned pipeline once on the full train (decay-weighted if capable)."""
    fitted: dict[str, Pipeline] = {}
    for name, pipeline in pipelines.items():
        f, _ = fit_pipeline_weighted(
            pipeline,
            X_train,
            y_train,
            _holdout_sample_weights(name, tuned, train_timestamps),
        )
        fitted[name] = f
    return fitted


def holdout_scores(fitted: dict[str, Pipeline], X_test: pd.DataFrame) -> dict[str, np.ndarray]:
    return {name: f.predict_proba(X_test)[:, 1] for name, f in fitted.items()}


def gate_conditional_report(
    scores: dict[str, np.ndarray],
    y_test: pd.Series,
    gate,
    X_test: pd.DataFrame,
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Conditional metrics on the gate's in-control wafers + coverage.

    Works for either standalone gate (``EFAGate`` / ``BayesGate``): the gate
    abstains under its configured ``logic``; per-statistic flag counts come from
    ``gate.flag_breakdown``.
    """
    in_control = gate.is_in_control(X_test)
    breakdown = gate.flag_breakdown(X_test)
    y_arr = np.asarray(y_test).astype(int)
    n_total = int(len(y_arr))
    n_in_control = int(in_control.sum())
    n_fails_total = int(y_arr.sum())
    n_fails_in_control = int(y_arr[in_control].sum())

    shared = {
        "coverage": (n_in_control / n_total) if n_total else None,
        "n_total": n_total,
        "n_in_control": n_in_control,
        "n_flagged_ooc": n_total - n_in_control,
        "n_fails_total": n_fails_total,
        "n_fails_in_control": n_fails_in_control,
        "n_fails_flagged_ooc": n_fails_total - n_fails_in_control,
        **{k: v for k, v in breakdown.items() if k.startswith("n_flagged")},
    }

    if show_progress:
        coverage = n_in_control / n_total if n_total else 0.0
        print(
            f"  gate {gate.config().get('method')} ({gate.logic.upper()}): "
            f"coverage {coverage:.1%} ({n_in_control}/{n_total}); "
            f"fails in-control {n_fails_in_control}/{n_fails_total}"
        )

    rows = []
    for name, y_score in scores.items():
        row: dict = {
            "pipeline": name,
            **shared,
            "global_pr_auc": float(average_precision_score(y_arr, y_score)),
            "global_roc_auc": float(roc_auc_score(y_arr, y_score)),
        }
        y_ic = y_arr[in_control]
        score_ic = y_score[in_control]
        enough = (
            n_fails_in_control >= MIN_CONDITIONAL_POSITIVES
            and len(np.unique(y_ic)) > 1
        )
        if enough:
            row["conditional_pr_auc"] = float(average_precision_score(y_ic, score_ic))
            row["conditional_roc_auc"] = float(roc_auc_score(y_ic, score_ic))
            boot = stratified_bootstrap_holdout_metrics(
                y_ic,
                score_ic,
                threshold=0.5,
                n_bootstrap=HOLDOUT_BOOTSTRAP_N,
                ci_level=HOLDOUT_BOOTSTRAP_CI,
                rng=np.random.default_rng(RANDOM_SEED),
            )
            row["conditional_pr_auc_ci_low"] = boot.get("pr_auc_ci_low")
            row["conditional_pr_auc_ci_high"] = boot.get("pr_auc_ci_high")
        else:
            row["conditional_pr_auc"] = None
            row["conditional_roc_auc"] = None
            row["conditional_pr_auc_ci_low"] = None
            row["conditional_pr_auc_ci_high"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def gate_risk_coverage(
    scores: dict[str, np.ndarray],
    y_test: pd.Series,
    gate,
    X_test: pd.DataFrame,
    coverage_grid=RISK_COVERAGE_GRID,
) -> pd.DataFrame:
    """Rank holdout wafers by the gate's OOC severity; rescore retained subsets.

    For each target ``coverage`` keep the least-suspicious ``round(c * n)`` wafers
    and rescore every model. ``coverage == 1.0`` keeps all wafers (== global).
    AUCs are NaN when the kept set has too few fails or is single-class. Returns
    ``[pipeline, coverage, n_kept, n_fails_kept, pr_auc, roc_auc]``.
    """
    severity = np.asarray(gate.ooc_severity(X_test), dtype="float64")
    y_arr = np.asarray(y_test).astype(int)
    n_total = int(len(y_arr))
    order = np.argsort(severity, kind="mergesort")  # least-suspicious first

    rows = []
    for c in coverage_grid:
        c = float(c)
        n_keep = max(0, min(n_total, int(round(c * n_total))))
        keep_idx = order[:n_keep]
        y_keep = y_arr[keep_idx]
        n_fails_kept = int(y_keep.sum())
        enough = n_fails_kept >= MIN_CONDITIONAL_POSITIVES and len(np.unique(y_keep)) > 1
        for name, y_score in scores.items():
            s_keep = np.asarray(y_score)[keep_idx]
            pr_auc = float(average_precision_score(y_keep, s_keep)) if enough else float("nan")
            roc_auc = float(roc_auc_score(y_keep, s_keep)) if enough else float("nan")
            rows.append(
                {
                    "pipeline": name,
                    "coverage": c,
                    "n_kept": int(n_keep),
                    "n_fails_kept": n_fails_kept,
                    "pr_auc": pr_auc,
                    "roc_auc": roc_auc,
                }
            )
    return pd.DataFrame(rows)


def _profile_holdout_columns(
    y_test: pd.Series,
    y_score: np.ndarray,
    threshold: float,
    profile_id: str,
) -> dict:
    profile = THRESHOLD_PROFILES[profile_id]  # type: ignore[index]
    y_pred = predict_with_threshold(y_score, threshold)
    metrics = compute_holdout_metrics(y_test, y_pred)
    cols = {
        f"{profile_id}_threshold": float(threshold),
        f"{profile_id}_ber_percent": metrics["ber_percent"],
        f"{profile_id}_true_positive_percent": metrics["true_positive_percent"],
        f"{profile_id}_true_negative_percent": metrics["true_negative_percent"],
        f"{profile_id}_confusion_matrix": metrics["confusion_matrix"],
    }
    if profile.objective == "fbeta" and profile.beta is not None:
        cols[f"{profile_id}_fbeta"] = fbeta_at_threshold(
            y_test, y_pred, beta=profile.beta
        )
    return cols


def run_holdout_benchmark(
    scores: dict[str, np.ndarray],
    y_test: pd.Series,
    tuned: dict[str, dict] | None = None,
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Report holdout metrics from precomputed positive-class scores.

    Scores come from ``holdout_scores`` of pipelines fit once on the protocol's
    train (decay-weighted where capable), so the expensive Bayesian heads are
    not refit per report.
    """
    rows = []

    if show_progress:
        print(f"Holdout: {len(scores)} pipelines (reporting only)")

    for name, y_score in scores.items():
        row: dict = {
            "pipeline": name,
            "pr_auc": float(average_precision_score(y_test, y_score)),
            "roc_auc": float(roc_auc_score(y_test, y_score)),
        }

        model_tuned = (tuned or {}).get(name, {})
        thresholds = resolve_threshold_profiles(model_tuned) if model_tuned else {}
        for profile_id in PROFILE_IDS:
            if profile_id not in thresholds:
                continue
            row.update(
                _profile_holdout_columns(
                    y_test, y_score, thresholds[profile_id], profile_id
                )
            )

        default_thr = thresholds.get(DEFAULT_PROFILE_ID, thresholds.get("ber", 0.5))
        y_pred_default = predict_with_threshold(y_score, default_thr)
        ber_metrics = compute_holdout_metrics(y_test, y_pred_default)
        row.update(
            {
                "ber_percent": ber_metrics["ber_percent"],
                "true_positive_percent": ber_metrics["true_positive_percent"],
                "true_negative_percent": ber_metrics["true_negative_percent"],
                "confusion_matrix": ber_metrics["confusion_matrix"],
            }
        )
        row.update(
            stratified_bootstrap_holdout_metrics(
                y_test,
                y_score,
                y_pred_default,
                n_bootstrap=HOLDOUT_BOOTSTRAP_N,
                ci_level=HOLDOUT_BOOTSTRAP_CI,
                rng=np.random.default_rng(RANDOM_SEED),
            )
        )
        rows.append(row)
        if show_progress:
            pr_lo, pr_hi = row.get("pr_auc_ci_low"), row.get("pr_auc_ci_high")
            roc_lo, roc_hi = row.get("roc_auc_ci_low"), row.get("roc_auc_ci_high")
            ber_lo, ber_hi = row.get("ber_percent_ci_low"), row.get("ber_percent_ci_high")
            pr_ci = (
                f" [{pr_lo:.3f}, {pr_hi:.3f}]"
                if pr_lo is not None and pr_hi is not None
                else ""
            )
            roc_ci = (
                f" [{roc_lo:.3f}, {roc_hi:.3f}]"
                if roc_lo is not None and roc_hi is not None
                else ""
            )
            ber_ci = (
                f" [{ber_lo:.1f}%, {ber_hi:.1f}%]"
                if ber_lo is not None and ber_hi is not None
                else ""
            )
            print(
                f"  {name}: holdout PR AUC {row['pr_auc']:.3f}{pr_ci}, "
                f"ROC AUC {row['roc_auc']:.3f}{roc_ci}, "
                f"BER (F2 threshold) {row['ber_percent']:.1f}%{ber_ci}"
            )

    holdout = pd.DataFrame(rows).sort_values(
        HOLDOUT_SORT_COL, ascending=False, kind="mergesort"
    )
    return holdout.reset_index(drop=True)


def _merge_model_rows(
    existing: list | None,
    new_rows: list,
    model_ids,
) -> list:
    """Replace rows for the given pipelines, preserving rows for other models."""
    touched = set(model_ids)
    kept = [r for r in (existing or []) if r.get("pipeline") not in touched]
    return kept + list(new_rows or [])


def save_benchmark_results(
    *,
    model_ids,
    tuned: dict[str, dict] | None = None,
    tuned_blocked: dict[str, dict] | None = None,
    leaderboard: pd.DataFrame | None = None,
    leaderboard_blocked: pd.DataFrame | None = None,
    holdout: pd.DataFrame | None = None,
    holdout_random: pd.DataFrame | None = None,
    holdout_conditional: pd.DataFrame | None = None,
    holdout_conditional_random: pd.DataFrame | None = None,
    risk_coverage: pd.DataFrame | None = None,
    risk_coverage_random: pd.DataFrame | None = None,
    gate_reports: dict | None = None,
    holdout_split: dict | None = None,
    holdout_split_random: dict | None = None,
    process_gate: dict | None = None,
    interp_process_gate: dict | None = None,
    merge: bool = False,
    path: Path = BENCHMARK_RESULTS_PATH,
) -> dict:
    """Write benchmark JSON. When merge=True, update only the touched models'
    sections in the existing file (used for single-model runs)."""
    interp_ids = list(model_ids)
    extrap_ids = list(model_ids)
    payload: dict = {}
    if merge and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))

    payload.update(
        {
            "primary_metric": PRIMARY_METRIC,
            "ranking": RANKING,
            "tuned_params_dir": str(TUNED_PARAMS_DIR),
            "tuned_blocked_params_dir": str(TUNED_BLOCKED_PARAMS_DIR),
            "cv_protocol": {
                "in_distribution": CV_PROTOCOL_IN_DIST,
                "extrapolation": CV_PROTOCOL_EXTRAP,
            },
            "model_ids": list(MODEL_IDS),
            "correlated_selection": {
                "library": "feature_engine",
                "steps": [
                    "VarianceThreshold",
                    "SmartCorrelatedSelection",
                ],
                "method": CORRELATED_SELECTION_METHOD,
                "threshold": float(CORRELATED_SELECTION_THRESHOLD),
                "selection_method": CORRELATED_SELECTION_CRITERION,
                "missing_values": "ignore",
            },
            "benchmark_rf_n_estimators": int(RF_N_ESTIMATORS),
            "benchmark_rf_max_depth": int(RF_MAX_DEPTH),
            "holdout_is_reporting_only": True,
            "holdout_bootstrap": {
                "n": int(HOLDOUT_BOOTSTRAP_N),
                "ci_level": float(HOLDOUT_BOOTSTRAP_CI),
                "method": "stratified",
            },
            "threshold_profile_ids": list(PROFILE_IDS),
            "threshold_profile_config": threshold_profile_config(),
        }
    )

    if holdout_split is not None:
        payload["holdout_split"] = holdout_split
    elif "holdout_split" not in payload:
        payload["holdout_split"] = {
            "test_size": float(TEST_SIZE),
            "split_mode": "temporal",
        }
    if holdout_split_random is not None:
        payload["holdout_split_random"] = holdout_split_random

    if tuned is not None:
        hp = dict(payload.get("tuned_hyperparameters") or {})
        for mid in interp_ids:
            hp[mid] = tuned[mid].get("grid_search_best_params", {})
        payload["tuned_hyperparameters"] = hp

    if tuned_blocked is not None:
        hp_b = dict(payload.get("tuned_hyperparameters_blocked") or {})
        time_decay = dict(payload.get("time_decay") or {})
        for mid in extrap_ids:
            hp_b[mid] = tuned_blocked[mid].get("grid_search_best_params", {})
            time_decay[mid] = {
                "decay_lambda": float(tuned_blocked[mid].get("decay_lambda", 0.0)),
                "weight_capable": mid in WEIGHTING_MODEL_IDS,
                "search": tuned_blocked[mid].get("decay_lambda_search"),
            }
        payload["tuned_hyperparameters_blocked"] = hp_b
        payload["time_decay"] = time_decay

    if leaderboard is not None:
        payload["leaderboard"] = _merge_model_rows(
            payload.get("leaderboard"), leaderboard.to_dict(orient="records"), interp_ids
        )
    if leaderboard_blocked is not None:
        payload["leaderboard_blocked"] = _merge_model_rows(
            payload.get("leaderboard_blocked"),
            leaderboard_blocked.to_dict(orient="records"),
            extrap_ids,
        )
    if holdout is not None:
        payload["holdout"] = _merge_model_rows(
            payload.get("holdout"), holdout.to_dict(orient="records"), extrap_ids
        )
    if holdout_random is not None:
        payload["holdout_random"] = _merge_model_rows(
            payload.get("holdout_random"),
            holdout_random.to_dict(orient="records"),
            interp_ids,
        )
    if holdout_conditional is not None:
        payload["holdout_conditional"] = _merge_model_rows(
            payload.get("holdout_conditional"),
            holdout_conditional.to_dict(orient="records"),
            extrap_ids,
        )
    if holdout_conditional_random is not None:
        payload["holdout_conditional_random"] = _merge_model_rows(
            payload.get("holdout_conditional_random"),
            holdout_conditional_random.to_dict(orient="records"),
            interp_ids,
        )
    if risk_coverage is not None:
        # Multi-row-per-model (one row per coverage), so replace wholesale rather
        # than merging by model id like the single-row leaderboard sections.
        payload["risk_coverage"] = risk_coverage.to_dict(orient="records")
    if risk_coverage_random is not None:
        payload["risk_coverage_random"] = risk_coverage_random.to_dict(orient="records")
    if gate_reports is not None:
        payload["gate_reports"] = gate_reports
    if process_gate is not None:
        payload["process_gate"] = process_gate
    if interp_process_gate is not None:
        payload["interp_process_gate"] = interp_process_gate

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    return payload


def _select_model_ids(model: str | None) -> list[str]:
    if model is not None:
        if model not in MODEL_IDS:
            raise SystemExit(
                f"Unknown --model {model!r}; choose from {list(MODEL_IDS)}"
            )
        return [model]
    return list(MODEL_IDS)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=list(MODEL_IDS),
        help="Benchmark a single pipeline by model id (merges into existing JSON).",
    )
    return parser.parse_args(argv)


def _fit_gate(gate_cls, X_train, y_train):
    return gate_cls().fit(X_train, y_train)


def _protocol_gate_reports(
    scores: dict[str, np.ndarray],
    y_test: pd.Series,
    X_test: pd.DataFrame,
    efa_gate: EFAGate,
    bayes_gate: BayesGate,
) -> dict:
    """Conditional + risk-coverage for BOTH standalone gates on one protocol."""
    return {
        "efa": {
            "config": efa_gate.config(),
            "conditional": gate_conditional_report(
                scores, y_test, efa_gate, X_test
            ).to_dict(orient="records"),
            "risk_coverage": gate_risk_coverage(
                scores, y_test, efa_gate, X_test
            ).to_dict(orient="records"),
        },
        "bayes": {
            "config": bayes_gate.config(),
            "conditional": gate_conditional_report(
                scores, y_test, bayes_gate, X_test
            ).to_dict(orient="records"),
            "risk_coverage": gate_risk_coverage(
                scores, y_test, bayes_gate, X_test
            ).to_dict(orient="records"),
        },
    }


def main(argv=None) -> None:
    """Re-run after tuning. Every cell runs on BOTH protocols:
    interpolation = stratified CV + random holdout; extrapolation = blocked CV +
    temporal holdout + time-decay. Both standalone gates (EFA, Bayes) are scored
    on each protocol's holdout. A --model subset merges into the existing JSON."""
    args = _parse_args(argv)
    selected = _select_model_ids(args.model)
    merge = set(selected) != set(MODEL_IDS)

    df = load_mart()
    cols = feature_columns(df)

    # --- Interpolation protocol: random stratified split ---------------------
    rand_train_df, rand_test_df = split_train_test_random(df)
    split_meta_random = holdout_split_summary(
        rand_train_df, rand_test_df, split_mode="random"
    )
    Xr_train = rand_train_df[cols]
    yr_train = rand_train_df[TARGET_COL].astype(int)
    Xr_test = rand_test_df[cols]
    yr_test = rand_test_df[TARGET_COL].astype(int)

    tuned = load_all_tuned_params(model_ids=selected)
    print("Stratified CV leaderboard (in-distribution protocol):")
    leaderboard = run_pipeline_benchmark(
        build_benchmark_pipelines(tuned, model_ids=selected),
        Xr_train,
        yr_train,
        cv=make_repeated_stratified_cv(),
        show_progress=True,
    )
    print("\nFitting random-holdout pipelines (interp):")
    fitted_random = fit_holdout_pipelines(
        build_benchmark_pipelines(tuned, model_ids=selected),
        Xr_train,
        yr_train,
        tuned,
        train_timestamps=None,
    )
    scores_random = holdout_scores(fitted_random, Xr_test)
    print("\nRandom holdout (in-distribution):")
    holdout_random = run_holdout_benchmark(scores_random, yr_test, tuned)

    print("\nStandalone gates on the random holdout:")
    efa_gate_random = _fit_gate(EFAGate, Xr_train, yr_train)
    bayes_gate_random = _fit_gate(BayesGate, Xr_train, yr_train)
    gate_random = _protocol_gate_reports(
        scores_random, yr_test, Xr_test, efa_gate_random, bayes_gate_random
    )
    holdout_conditional_random = pd.DataFrame(gate_random["efa"]["conditional"])
    risk_coverage_random = pd.DataFrame(gate_random["efa"]["risk_coverage"])

    # --- Extrapolation protocol: temporal split -----------------------------
    train_df, test_df = split_train_test(df)
    split_meta = holdout_split_summary(train_df, test_df, split_mode="temporal")
    X_train = train_df[cols]
    y_train = train_df[TARGET_COL].astype(int)
    X_test = test_df[cols]
    y_test = test_df[TARGET_COL].astype(int)
    train_ts = train_df[TIMESTAMP_COL].reset_index(drop=True)

    tuned_blocked = load_all_tuned_blocked_params(model_ids=selected)
    print("\nWeighted blocked-CV leaderboard (extrapolation protocol):")
    leaderboard_blocked = run_weighted_blocked_leaderboard(
        build_benchmark_pipelines(tuned_blocked, extrapolation=True, model_ids=selected),
        X_train,
        y_train,
        train_ts,
        make_blocked_time_cv(train_df),
        tuned_blocked,
        show_progress=True,
    )
    print("\nFitting temporal-holdout pipelines (extrap):")
    fitted_temporal = fit_holdout_pipelines(
        build_benchmark_pipelines(tuned_blocked, extrapolation=True, model_ids=selected),
        X_train,
        y_train,
        tuned_blocked,
        train_timestamps=train_ts,
    )
    scores_temporal = holdout_scores(fitted_temporal, X_test)
    print("\nTemporal holdout (extrapolation):")
    holdout = run_holdout_benchmark(scores_temporal, y_test, tuned_blocked)

    print("\nStandalone gates on the temporal holdout:")
    efa_gate_temporal = _fit_gate(EFAGate, X_train, y_train)
    bayes_gate_temporal = _fit_gate(BayesGate, X_train, y_train)
    gate_temporal = _protocol_gate_reports(
        scores_temporal, y_test, X_test, efa_gate_temporal, bayes_gate_temporal
    )
    holdout_conditional = pd.DataFrame(gate_temporal["bayes"]["conditional"])
    risk_coverage = pd.DataFrame(gate_temporal["bayes"]["risk_coverage"])

    gate_reports = {"random": gate_random, "temporal": gate_temporal}

    # Pipeline artifacts (reduction stages, coef summaries) are a full-run product
    # keyed off the reference models; a single-model subset can't rebuild the
    # shared section, so only regenerate on a full run.
    artifacts = (
        None
        if merge
        else collect_holdout_artifacts(
            fitted_temporal, X_train, y_train, holdout_split=split_meta
        )
    )

    save_benchmark_results(
        model_ids=selected,
        tuned=tuned,
        tuned_blocked=tuned_blocked,
        leaderboard=leaderboard,
        leaderboard_blocked=leaderboard_blocked,
        holdout=holdout,
        holdout_random=holdout_random,
        holdout_conditional=holdout_conditional,
        holdout_conditional_random=holdout_conditional_random,
        risk_coverage=risk_coverage,
        risk_coverage_random=risk_coverage_random,
        gate_reports=gate_reports,
        holdout_split=split_meta,
        holdout_split_random=split_meta_random,
        process_gate=bayes_gate_temporal.config(),
        interp_process_gate=efa_gate_random.config(),
        merge=merge,
    )
    if artifacts is not None:
        save_pipeline_artifacts(artifacts)

    print(f"\nWrote {BENCHMARK_RESULTS_PATH}")
    if artifacts is not None:
        print(f"Wrote {PIPELINE_ARTIFACTS_PATH}")
    print("\nStratified CV leaderboard:")
    print(leaderboard.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nBlocked CV leaderboard:")
    print(leaderboard_blocked.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nTemporal holdout (extrapolation):")
    print(holdout.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nRandom holdout (in-distribution):")
    print(holdout_random.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
