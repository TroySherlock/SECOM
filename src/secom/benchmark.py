#!/usr/bin/env python3
"""Benchmark the unified model grid on both evaluation protocols.

Every cell (hsic_* / rfsel_* / pls_* x enet/rf/bayes) is tuned once on the
in-distribution stratified CV (data/processed/tuned/<model_id>.json) and scored on:
  - interpolation: stratified CV + random holdout
  - extrapolation: a forward temporal holdout of the same in-distribution params
    (unweighted), plus a time-decay-vs-lambda diagnostic sweep on that holdout

Use --model ID to benchmark a single cell; subset runs merge into the existing
benchmark JSON instead of overwriting it.
Primary objective: maximize PR AUC on CV; F-beta thresholds (F0.5 / F2 / F4) plus BER-min.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from secom.artifacts import (
    collect_holdout_artifacts,
    save_pipeline_artifacts,
)
from secom.costs import (
    DEFAULT_PROFILE_ID,
    PROFILE_IDS,
    THRESHOLD_PROFILES,
    fbeta_at_threshold,
    resolve_threshold_profiles,
    threshold_profile_config,
)
from secom.gates import BayesGate, EFAGate
from secom.metrics import (
    compute_holdout_metrics,
    predict_with_threshold,
    stratified_bootstrap_holdout_metrics,
)
from secom.pipelines import (
    BENCHMARK_RESULTS_PATH,
    CORRELATED_SELECTION_CRITERION,
    CORRELATED_SELECTION_METHOD,
    CORRELATED_SELECTION_THRESHOLD,
    CV_N_JOBS,
    CV_SCORING,
    HOLDOUT_BOOTSTRAP_CI,
    HOLDOUT_BOOTSTRAP_N,
    MODEL_IDS,
    PIPELINE_ARTIFACTS_PATH,
    RANDOM_SEED,
    REPORT_CACHE_PATH,
    RF_MAX_DEPTH,
    RF_N_ESTIMATORS,
    RISK_COVERAGE_GRID,
    TARGET_COL,
    TEST_SIZE,
    TIMESTAMP_COL,
    TUNED_PARAMS_DIR,
    clear_pipeline_cache,
    feature_columns,
    holdout_split_summary,
    load_mart,
    make_repeated_stratified_cv,
    split_train_test,
    split_train_test_random,
)
from secom.reporting import (
    collect_cv_oof_proba,
    compute_global_importance,
    pr_curve_payload,
    scores_payload,
)
from secom.tuning.registry import build_tuned_pipeline, fit_pipeline_weighted
from secom.utils import (
    json_safe,
    load_all_tuned_params,
    score_row_from_cv_result,
    tqdm_joblib_context,
)

MIN_CONDITIONAL_POSITIVES = 5

PRIMARY_METRIC = "pr_auc"
CV_SORT_COL = "mean_pr_auc"
HOLDOUT_SORT_COL = "pr_auc"
RANKING = "descending_higher_is_better"

CV_PROTOCOL_IN_DIST = "repeated_stratified_5x2"


def build_benchmark_pipelines(
    tuned: dict[str, dict] | None = None,
    *,
    model_ids=None,
) -> dict[str, Pipeline]:
    if tuned is None:
        tuned = load_all_tuned_params()
    if model_ids is None:
        model_ids = list(tuned.keys())
    pipelines: dict[str, Pipeline] = {}
    for model_id in model_ids:
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


def fit_holdout_pipelines(
    pipelines: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, Pipeline]:
    """Fit each tuned pipeline once on the full train (unweighted)."""
    fitted: dict[str, Pipeline] = {}
    for name, pipeline in pipelines.items():
        f, _ = fit_pipeline_weighted(pipeline, X_train, y_train, None)
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
                f"BER (BER-min threshold) {row['ber_percent']:.1f}%{ber_ci}"
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
    leaderboard: pd.DataFrame | None = None,
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
            "cv_protocol": {
                "in_distribution": CV_PROTOCOL_IN_DIST,
                "extrapolation": "in_distribution_tuned_temporal_holdout",
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

    if leaderboard is not None:
        payload["leaderboard"] = _merge_model_rows(
            payload.get("leaderboard"), leaderboard.to_dict(orient="records"), interp_ids
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
    parser.add_argument(
        "--clear-pipeline-cache",
        action="store_true",
        help="Clear the joblib preprocess cache before benchmarking (after editing front-ends).",
    )
    return parser.parse_args(argv)


def _fit_gate(gate_cls, X_train, y_train):
    return gate_cls().fit(X_train, y_train)


def _bayes_gate_diagnostics(
    bayes_gate: BayesGate,
    y_test: pd.Series,
    X_test: pd.DataFrame,
    timestamps: pd.Series | None,
    *,
    include_sbfa: bool = False,
) -> dict:
    """Per-wafer BGM control statistics for the gate-monitor charts (Tier 1/2).

    The reference distribution is the passing-train density/Q the gate stored at
    fit; the holdout block carries per-wafer density/Q/flags (+ timestamps on the
    temporal track) so the dashboard can plot the distribution shift and the
    time-ordered MSPC control chart without refitting the gate. When
    ``include_sbfa`` is set (temporal track only) an ``sbfa`` block freezes the
    member-0 latent-factor scores, sparse loadings and BGM envelope for the
    Tier-2 factor-space / root-cause visuals.
    """
    diag = bayes_gate.diagnostics(X_test)
    holdout = {
        "density": diag["density"],
        "q": diag["q"],
        "y_true": np.asarray(y_test).astype(int),
        "density_ooc": diag["density_ooc"],
        "q_ooc": diag["q_ooc"],
        "ooc": diag["ooc"],
    }
    if timestamps is not None:
        holdout["ts"] = [str(t) for t in pd.Series(timestamps).to_numpy()]
    result = {
        "reference": {
            "density": bayes_gate.ref_density_,
            "q": bayes_gate.ref_q_,
        },
        "holdout": holdout,
        "limits": {
            "density_lcl": float(bayes_gate.density_lcl_),
            "q_ucl": float(bayes_gate.q_ucl_),
        },
    }
    if include_sbfa:
        bgm = bayes_gate.bgm_params()
        result["sbfa"] = {
            "loadings": bayes_gate.loadings(),
            "feature_names": list(bayes_gate.feature_names_),
            "bgm": {
                "means": bgm["means"],
                "covariances": bgm["covariances"],
                "weights": bgm["weights"],
            },
            "reference_scores": bayes_gate.ref_scores_,
            "holdout_scores": bayes_gate.factor_scores(X_test),
            "y_true": np.asarray(y_test).astype(int),
            "flagged": diag["ooc"],
        }
    return result


def _efa_gate_diagnostics(
    efa_gate: EFAGate,
    y_test: pd.Series,
    X_test: pd.DataFrame,
    timestamps: pd.Series | None,
    *,
    include_factor: bool = False,
) -> dict:
    """Per-wafer Hotelling T2 / Q control statistics for the 5.2 gate-monitor charts.

    Mirrors ``_bayes_gate_diagnostics``: reference is the passing-train T2/Q the
    gate stored at fit; the holdout block carries per-wafer T2/Q/flags (+ temporal
    timestamps). When ``include_factor`` is set (temporal track) a ``factor`` block
    freezes the EFA factor scores, dense loadings and score-Gaussian for the
    factor-space (Hotelling ellipse) and loadings root-cause visuals.
    """
    diag = efa_gate.diagnostics(X_test)
    holdout = {
        "t2": diag["t2"],
        "q": diag["q"],
        "y_true": np.asarray(y_test).astype(int),
        "t2_ooc": diag["t2_ooc"],
        "q_ooc": diag["q_ooc"],
        "ooc": diag["ooc"],
    }
    if timestamps is not None:
        holdout["ts"] = [str(t) for t in pd.Series(timestamps).to_numpy()]
    result = {
        "reference": {
            "t2": efa_gate.efa_.t2_ref_,
            "q": efa_gate.efa_.q_ref_,
        },
        "holdout": holdout,
        "limits": {
            "t2_ucl": float(efa_gate.t2_ucl_),
            "q_ucl": float(efa_gate.q_ucl_),
        },
    }
    if include_factor:
        score_mean, score_cov = efa_gate.score_gaussian()
        result["factor"] = {
            "loadings": efa_gate.loadings(),
            "feature_names": list(efa_gate.feature_names_),
            "score_mean": score_mean,
            "score_cov": score_cov,
            "t2_alpha": float(efa_gate.t2_alpha),
            "reference_scores": efa_gate.ref_scores_,
            "holdout_scores": efa_gate.factor_scores(X_test),
            "y_true": np.asarray(y_test).astype(int),
            "flagged": diag["ooc"],
        }
    return result


def _protocol_gate_reports(
    scores: dict[str, np.ndarray],
    y_test: pd.Series,
    X_test: pd.DataFrame,
    efa_gate: EFAGate,
    bayes_gate: BayesGate,
    *,
    timestamps: pd.Series | None = None,
    include_sbfa: bool = False,
) -> dict:
    """Conditional + risk-coverage for BOTH standalone gates on one protocol.

    Both blocks also carry frozen per-wafer ``diagnostics`` (EFA T2/Q and BGM
    drift monitors); ``timestamps`` (temporal track only) enables the control
    charts and ``include_sbfa`` freezes the Tier-2 EFA/sBFA latent + loadings
    artifacts.
    """
    return {
        "efa": {
            "config": efa_gate.config(),
            "conditional": gate_conditional_report(
                scores, y_test, efa_gate, X_test
            ).to_dict(orient="records"),
            "risk_coverage": gate_risk_coverage(
                scores, y_test, efa_gate, X_test
            ).to_dict(orient="records"),
            "diagnostics": _efa_gate_diagnostics(
                efa_gate, y_test, X_test, timestamps, include_factor=include_sbfa
            ),
        },
        "bayes": {
            "config": bayes_gate.config(),
            "conditional": gate_conditional_report(
                scores, y_test, bayes_gate, X_test
            ).to_dict(orient="records"),
            "risk_coverage": gate_risk_coverage(
                scores, y_test, bayes_gate, X_test
            ).to_dict(orient="records"),
            "diagnostics": _bayes_gate_diagnostics(
                bayes_gate, y_test, X_test, timestamps, include_sbfa=include_sbfa
            ),
        },
    }


def _report_entries_for_track(
    model_ids,
    tuned_map: dict[str, dict],
    fitted: dict[str, Pipeline],
    scores: dict[str, np.ndarray],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv,
    *,
    show_progress: bool = True,
) -> dict[str, dict]:
    """Per-model frozen PR-curve + global-importance payload for one protocol.

    Holdout PR data reuses the already-computed ``scores``. When ``cv`` is given
    (interpolation) the CV-OOF curve is added too; for the temporal track ``cv``
    is ``None`` (no temporal CV), so the deep-dive shows a holdout-only PR curve.
    """
    entries: dict[str, dict] = {}
    for name in model_ids:
        if cv is not None:
            oof_pipeline = build_tuned_pipeline(name, tuned_map[name])
            y_cv, score_cv = collect_cv_oof_proba(oof_pipeline, X_train, y_train, cv)
        else:
            y_cv, score_cv = None, None
        try:
            ber_threshold = resolve_threshold_profiles(tuned_map[name]).get("ber")
        except (KeyError, ValueError):
            ber_threshold = None
        pr_curve = pr_curve_payload(
            y_cv, score_cv, y_test, scores[name], ber_threshold
        )
        score_block = scores_payload(y_cv, score_cv, y_test, scores[name])
        importance = compute_global_importance(name, fitted[name], X_train)
        entries[name] = {
            "pr_curve": pr_curve,
            "global_importance": importance,
            "scores": score_block,
        }
        if show_progress:
            print(f"  report cache: {name}")
    return entries


def save_report_cache(
    entries_by_track: dict[str, dict],
    *,
    merge: bool = False,
    path: Path = REPORT_CACHE_PATH,
) -> dict:
    """Write the frozen dashboard report cache, keyed ``[track][model_id]``.

    On a ``--model`` subset run (merge=True) only the touched models are replaced
    within each track; other models' frozen reports are preserved.
    """
    payload: dict = {}
    if merge and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    for track, entries in entries_by_track.items():
        track_block = dict(payload.get(track) or {})
        track_block.update(entries)
        payload[track] = track_block
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    return payload


def main(argv=None) -> None:
    """Re-run after tuning. Every cell is tuned once on the in-distribution
    stratified CV (data/processed/tuned/) and scored on two holdouts:
    interpolation = stratified CV + random holdout; extrapolation = a forward
    temporal holdout of those same params (unweighted), plus a diagnostic
    time-decay sweep. Both standalone gates (EFA, Bayes) are scored on each
    protocol's holdout. A --model subset merges into the existing JSON."""
    args = _parse_args(argv)
    if args.clear_pipeline_cache:
        clear_pipeline_cache()
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

    # Reuse the single in-distribution `tuned/` params; the temporal holdout is
    # an unweighted forward test of those hyperparameters (no separate tuning).
    print("\nFitting temporal-holdout pipelines (extrap, in-dist tuned params):")
    fitted_temporal = fit_holdout_pipelines(
        build_benchmark_pipelines(tuned, model_ids=selected),
        X_train,
        y_train,
    )
    scores_temporal = holdout_scores(fitted_temporal, X_test)
    print("\nTemporal holdout (extrapolation):")
    holdout = run_holdout_benchmark(scores_temporal, y_test, tuned)

    print("\nStandalone gates on the temporal holdout:")
    efa_gate_temporal = _fit_gate(EFAGate, X_train, y_train)
    bayes_gate_temporal = _fit_gate(BayesGate, X_train, y_train)
    gate_temporal = _protocol_gate_reports(
        scores_temporal,
        y_test,
        X_test,
        efa_gate_temporal,
        bayes_gate_temporal,
        timestamps=test_df[TIMESTAMP_COL],
        include_sbfa=True,
    )
    holdout_conditional = pd.DataFrame(gate_temporal["bayes"]["conditional"])
    risk_coverage = pd.DataFrame(gate_temporal["bayes"]["risk_coverage"])

    gate_reports = {"random": gate_random, "temporal": gate_temporal}

    # Frozen dashboard report cache: PR curves (CV-OOF + holdout) and global
    # importance, computed once here so the dashboard never refits a model.
    print("\nFreezing dashboard report cache (PR curves + global importance):")
    report_entries = {
        "interpolation": _report_entries_for_track(
            selected,
            tuned,
            fitted_random,
            scores_random,
            Xr_train,
            yr_train,
            yr_test,
            make_repeated_stratified_cv(),
        ),
        "extrapolation": _report_entries_for_track(
            selected,
            tuned,
            fitted_temporal,
            scores_temporal,
            X_train,
            y_train,
            y_test,
            None,
        ),
    }
    save_report_cache(report_entries, merge=merge)

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
        leaderboard=leaderboard,
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
    print(f"Wrote {REPORT_CACHE_PATH}")
    if artifacts is not None:
        print(f"Wrote {PIPELINE_ARTIFACTS_PATH}")
    print("\nStratified CV leaderboard:")
    print(leaderboard.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nTemporal holdout (extrapolation, in-dist tuned params):")
    print(holdout.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nRandom holdout (in-distribution):")
    print(holdout_random.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
