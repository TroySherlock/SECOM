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
from secom.pipelines import frozen_config
from secom.artifacts import (
    collect_holdout_artifacts,
    save_pipeline_artifacts,
)
from secom.cv import make_blocked_time_cv
from secom.gate import ProcessGate
from secom.pipelines import (
    BENCHMARK_MODEL_IDS,
    BENCHMARK_RESULTS_PATH,
    EXTRAP_MODEL_IDS,
    INTERP_MODEL_IDS,
    PIPELINE_ARTIFACTS_PATH,
    CV_N_JOBS,
    CV_SCORING,
    DECAY_LAMBDA_DEFAULT,
    GATE_CORR_THRESHOLD,
    GATE_LOGIC,
    IF_GATE_ALPHA,
    IF_GATE_MAX_SAMPLES,
    IF_GATE_N_ESTIMATORS,
    RANDOM_SEED,
    RF_MAX_DEPTH,
    RF_N_ESTIMATORS,
    CORRELATED_SELECTION_CRITERION,
    CORRELATED_SELECTION_METHOD,
    CORRELATED_SELECTION_THRESHOLD,
    HOLDOUT_BOOTSTRAP_CI,
    HOLDOUT_BOOTSTRAP_N,
    T2_GATE_ALPHA,
    TARGET_COL,
    TEST_SIZE,
    TIMESTAMP_COL,
    TUNED_PARAMS_DIR,
    TUNED_BLOCKED_PARAMS_DIR,
    WEIGHTING_MODEL_IDS,
    XGB_MAX_DEPTH,
    XGB_N_ESTIMATORS,
    XGB_SCALE_POS_WEIGHT,
    feature_columns,
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


def run_gate_conditional_benchmark(
    pipelines: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    gate: ProcessGate,
    tuned: dict[str, dict] | None = None,
    *,
    train_timestamps: pd.Series | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Conditional metrics on in-control wafers + coverage (T² + IF gate).

    The gate abstains under its configured ``logic``: ``"or"`` when either
    post-cluster T² or Isolation Forest flags OOC, ``"and"`` only when both do.
  """
    masks = gate.flag_masks(X_test)
    in_control = gate.is_in_control(X_test)
    y_test_arr = np.asarray(y_test).astype(int)
    n_total = int(len(y_test_arr))
    n_in_control = int(in_control.sum())
    n_fails_total = int(y_test_arr.sum())
    n_fails_in_control = int(y_test_arr[in_control].sum())

    n_flagged_t2 = int(masks["t2_ooc"].sum())
    n_flagged_if = int(masks["if_ooc"].sum())
    n_flagged_both = int(masks["both_ooc"].sum())
    fails = y_test_arr.astype(bool)
    n_fails_flagged_t2 = int((masks["t2_ooc"] & fails).sum())
    n_fails_flagged_if = int((masks["if_ooc"] & fails).sum())
    n_fails_flagged_both = int((masks["both_ooc"] & fails).sum())

    shared = {
        "coverage": (n_in_control / n_total) if n_total else None,
        "n_total": n_total,
        "n_in_control": n_in_control,
        "n_flagged_ooc": n_total - n_in_control,
        "n_flagged_t2": n_flagged_t2,
        "n_flagged_if": n_flagged_if,
        "n_flagged_both": n_flagged_both,
        "n_fails_total": n_fails_total,
        "n_fails_in_control": n_fails_in_control,
        "n_fails_flagged_ooc": n_fails_total - n_fails_in_control,
        "n_fails_flagged_t2": n_fails_flagged_t2,
        "n_fails_flagged_if": n_fails_flagged_if,
        "n_fails_flagged_both": n_fails_flagged_both,
    }

    rows = []
    if show_progress:
        coverage = n_in_control / n_total if n_total else 0.0
        print(
            f"Process gate (T² {gate.logic.upper()} IF): coverage {coverage:.1%} "
            f"({n_in_control}/{n_total}); T²={n_flagged_t2} IF={n_flagged_if} "
            f"both={n_flagged_both}; fails in-control "
            f"{n_fails_in_control}/{n_fails_total}"
        )

    for name, pipeline in pipelines.items():
        fitted, _threshold = fit_pipeline_weighted(
            pipeline,
            X_train,
            y_train,
            _holdout_sample_weights(name, tuned, train_timestamps),
        )
        y_score = fitted.predict_proba(X_test)[:, 1]

        row: dict = {
            "pipeline": name,
            **shared,
            "global_pr_auc": float(average_precision_score(y_test_arr, y_score)),
            "global_roc_auc": float(roc_auc_score(y_test_arr, y_score)),
        }

        y_ic = y_test_arr[in_control]
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
        if show_progress:
            cond = row["conditional_pr_auc"]
            cond_str = f"{cond:.3f}" if cond is not None else "n/a (<5 fails)"
            print(
                f"  {name}: conditional PR AUC {cond_str} "
                f"vs global {row['global_pr_auc']:.3f}"
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
    pipelines: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tuned: dict[str, dict] | None = None,
    *,
    train_timestamps: pd.Series | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Fit on train, report holdout metrics.

    When ``train_timestamps`` is given, weight-capable models are fit with
    time-decay sample weights at their tuned ``decay_lambda`` (extrapolation
    path). The random/in-distribution path leaves this None (unweighted).
    """
    rows = []

    if show_progress:
        print(f"Holdout: {len(pipelines)} pipelines (reporting only)")

    for name, pipeline in pipelines.items():
        fitted, _threshold = fit_pipeline_weighted(
            pipeline,
            X_train,
            y_train,
            _holdout_sample_weights(name, tuned, train_timestamps),
        )
        y_score = fitted.predict_proba(X_test)[:, 1]
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
    interp_ids,
    extrap_ids,
    tuned: dict[str, dict] | None = None,
    tuned_blocked: dict[str, dict] | None = None,
    leaderboard: pd.DataFrame | None = None,
    leaderboard_blocked: pd.DataFrame | None = None,
    holdout: pd.DataFrame | None = None,
    holdout_random: pd.DataFrame | None = None,
    holdout_conditional: pd.DataFrame | None = None,
    holdout_split: dict | None = None,
    holdout_split_random: dict | None = None,
    process_gate: dict | None = None,
    merge: bool = False,
    path: Path = BENCHMARK_RESULTS_PATH,
) -> dict:
    """Write benchmark JSON. When merge=True, update only the touched models'
    sections in the existing file (used for single-model / single-track runs)."""
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
            "model_ids": list(BENCHMARK_MODEL_IDS),
            "interpolation_model_ids": list(INTERP_MODEL_IDS),
            "extrapolation_model_ids": list(EXTRAP_MODEL_IDS),
            "correlated_selection": {
                "library": "feature_engine",
                "steps": [
                    "DropConstantFeatures",
                    "DropDuplicateFeatures",
                    "VarianceThreshold",
                    "SmartCorrelatedSelection",
                ],
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
    if process_gate is not None:
        payload["process_gate"] = process_gate

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    return payload


def _select_model_ids(model: str | None, track: str | None) -> list[str]:
    if model is not None:
        if model not in BENCHMARK_MODEL_IDS:
            raise SystemExit(
                f"Unknown --model {model!r}; choose from {list(BENCHMARK_MODEL_IDS)}"
            )
        return [model]
    if track == "interpolation":
        return list(INTERP_MODEL_IDS)
    if track == "extrapolation":
        return list(EXTRAP_MODEL_IDS)
    return list(BENCHMARK_MODEL_IDS)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=list(BENCHMARK_MODEL_IDS),
        help="Benchmark a single pipeline by model id (merges into existing JSON).",
    )
    parser.add_argument(
        "--track",
        choices=["interpolation", "extrapolation"],
        help="Benchmark only one track (ignored if --model is given).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Re-run after tuning. Interpolation track = stratified CV + random holdout;
    extrapolation track = blocked CV + temporal holdout + process gate + decay.
    A --model/--track subset merges into the existing benchmark JSON."""
    args = _parse_args(argv)
    selected = _select_model_ids(args.model, args.track)
    interp_ids = [m for m in INTERP_MODEL_IDS if m in selected]
    extrap_ids = [m for m in EXTRAP_MODEL_IDS if m in selected]
    merge = set(selected) != set(BENCHMARK_MODEL_IDS)

    df = load_mart()
    cols = feature_columns(df)

    train_df, test_df = split_train_test(df)
    split_meta = holdout_split_summary(train_df, test_df, split_mode="temporal")
    X_train = train_df[cols]
    y_train = train_df[TARGET_COL].astype(int)
    X_test = test_df[cols]
    y_test = test_df[TARGET_COL].astype(int)
    train_ts = train_df[TIMESTAMP_COL].reset_index(drop=True)

    tuned = leaderboard = holdout_random = split_meta_random = None
    tuned_blocked = leaderboard_blocked = holdout = holdout_conditional = None
    gate = extrapolation_pipelines = None

    if interp_ids:
        tuned = load_all_tuned_params(model_ids=interp_ids)
        print("Stratified CV leaderboard (in-distribution protocol):")
        leaderboard = run_pipeline_benchmark(
            build_benchmark_pipelines(tuned, model_ids=interp_ids),
            X_train,
            y_train,
            cv=make_repeated_stratified_cv(),
            show_progress=True,
        )

        rand_train_df, rand_test_df = split_train_test_random(df)
        split_meta_random = holdout_split_summary(
            rand_train_df, rand_test_df, split_mode="random"
        )
        Xr_train = rand_train_df[cols]
        yr_train = rand_train_df[TARGET_COL].astype(int)
        Xr_test = rand_test_df[cols]
        yr_test = rand_test_df[TARGET_COL].astype(int)
        print("\nRandom holdout (stratified-tuned, in-distribution):")
        holdout_random = run_holdout_benchmark(
            build_benchmark_pipelines(tuned, model_ids=interp_ids),
            Xr_train,
            yr_train,
            Xr_test,
            yr_test,
            tuned,
            show_progress=True,
        )

    if extrap_ids:
        tuned_blocked = load_all_tuned_blocked_params(model_ids=extrap_ids)
        print("\nBlocked time CV leaderboard (extrapolation, time-weighted):")
        leaderboard_blocked = run_weighted_blocked_leaderboard(
            build_benchmark_pipelines(
                tuned_blocked, model_ids=extrap_ids, extrapolation=True
            ),
            X_train,
            y_train,
            train_ts,
            make_blocked_time_cv(train_df),
            tuned_blocked,
            show_progress=True,
        )

        extrapolation_pipelines = build_benchmark_pipelines(
            tuned_blocked, model_ids=extrap_ids, extrapolation=True
        )
        print("\nTemporal holdout (blocked-tuned, time-weighted):")
        holdout = run_holdout_benchmark(
            extrapolation_pipelines,
            X_train,
            y_train,
            X_test,
            y_test,
            tuned_blocked,
            train_timestamps=train_ts,
            show_progress=True,
        )

        print(
            f"\nProcess gate (post-cluster T² {GATE_LOGIC.upper()} IF, "
            "passing-train reference):"
        )
        gate = ProcessGate(
            t2_alpha=T2_GATE_ALPHA,
            if_alpha=IF_GATE_ALPHA,
            if_n_estimators=IF_GATE_N_ESTIMATORS,
            if_max_samples=IF_GATE_MAX_SAMPLES,
            gate_corr_threshold=GATE_CORR_THRESHOLD,
            logic=GATE_LOGIC,
        ).fit(X_train, y_train)
        holdout_conditional = run_gate_conditional_benchmark(
            build_benchmark_pipelines(
                tuned_blocked, model_ids=extrap_ids, extrapolation=True
            ),
            X_train,
            y_train,
            X_test,
            y_test,
            gate,
            tuned_blocked,
            train_timestamps=train_ts,
            show_progress=True,
        )

    save_benchmark_results(
        interp_ids=interp_ids,
        extrap_ids=extrap_ids,
        tuned=tuned,
        tuned_blocked=tuned_blocked,
        leaderboard=leaderboard,
        leaderboard_blocked=leaderboard_blocked,
        holdout=holdout,
        holdout_random=holdout_random,
        holdout_conditional=holdout_conditional,
        holdout_split=split_meta,
        holdout_split_random=split_meta_random,
        process_gate=gate.config() if gate is not None else None,
        merge=merge,
    )

    if extrapolation_pipelines is not None:
        # Artifacts report pipeline structure (fitted feature stages); fit on the
        # full temporal train unweighted so the wrapped pipelines are populated
        # regardless of the weighted-fit path used for scoring.
        for pipe in extrapolation_pipelines.values():
            pipe.fit(X_train, y_train)
        artifacts = collect_holdout_artifacts(
            extrapolation_pipelines,
            X_train,
            y_train,
            holdout_split=split_meta,
        )
        save_pipeline_artifacts(artifacts)

    print(f"\nWrote {BENCHMARK_RESULTS_PATH}")
    if extrapolation_pipelines is not None:
        print(f"Wrote {PIPELINE_ARTIFACTS_PATH}")
    if leaderboard is not None:
        print("\nStratified CV leaderboard:")
        print(leaderboard.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if leaderboard_blocked is not None:
        print("\nBlocked CV leaderboard:")
        print(
            leaderboard_blocked.to_string(index=False, float_format=lambda x: f"{x:.3f}")
        )
    if holdout is not None:
        print("\nTemporal holdout (extrapolation):")
        print(holdout.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if holdout_random is not None:
        print("\nRandom holdout (in-distribution):")
        print(holdout_random.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
