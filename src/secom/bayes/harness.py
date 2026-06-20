"""Bayesian extrapolation harness: tuning + benchmarking (parallel to the sklearn path).

Emits the same on-disk shapes the sklearn path does so ``benchmark.py`` and the
dashboard consume the results unchanged:
- tuned JSON under ``data/processed/tuned_blocked/<id>.json`` with
  ``grid_search_best_params``, ``classifier_threshold``, ``decay_lambda``,
  ``threshold_profiles``, ``cv_summary``, ``frozen_config``.
- ``leaderboard_blocked`` / ``holdout`` / ``holdout_conditional`` row frames.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm

from secom.core import (
    HOLDOUT_BOOTSTRAP_CI,
    HOLDOUT_BOOTSTRAP_N,
    RANDOM_SEED,
    TARGET_COL,
    TIMESTAMP_COL,
    frozen_config,
    threshold_profile_sweep,
)
from secom.costs import (
    DEFAULT_PROFILE_ID,
    PROFILE_IDS,
    THRESHOLD_PROFILES,
    fbeta_at_threshold,
    resolve_threshold_profiles,
)
from secom.cv import make_blocked_time_cv
from secom.metrics import (
    compute_holdout_metrics,
    predict_with_threshold,
    stratified_bootstrap_holdout_metrics,
)
from secom.bayes.calibration import IsotonicCalibrator
from secom.bayes.model import BayesianElasticNetLogistic, make_block_index
from secom.bayes.representation import ExtrapRepresentation
from secom.utils import json_safe, score_row_from_cv_result, tuned_blocked_params_path

CV_PROTOCOL_EXTRAP = "blocked_time_local_strat"


# --- Config / grid -----------------------------------------------------------
def _spec(model_id: str) -> dict:
    from secom.extrap_pipelines import BAYES_MODEL_SPECS

    if model_id not in BAYES_MODEL_SPECS:
        raise KeyError(f"Unknown Bayesian model id {model_id!r}")
    return BAYES_MODEL_SPECS[model_id]


def _param_grid(model_id: str) -> list[dict]:
    from secom.extrap_pipelines import (
        EXTRAP_C_GRID,
        EXTRAP_K_GRID,
        EXTRAP_L1_RATIO_GRID,
        EXTRAP_N_HUBS_GRID,
        EXTRAP_POS_WEIGHT_GRID,
        EXTRAP_RW_BLOCKS,
        EXTRAP_SPLS_COMPONENTS_GRID,
    )

    spec = _spec(model_id)
    method = spec["method"]
    rw = bool(spec["rw_intercept"])
    grid: list[dict] = []
    if method == "spls":
        for k, c, l1r, pw in product(
            EXTRAP_SPLS_COMPONENTS_GRID,
            EXTRAP_C_GRID,
            EXTRAP_L1_RATIO_GRID,
            EXTRAP_POS_WEIGHT_GRID,
        ):
            grid.append(
                {
                    "method": method,
                    "k": int(k),
                    "n_hubs": 0,
                    "C": float(c),
                    "l1_ratio": float(l1r),
                    "pos_weight": float(pw),
                    "rw_intercept": rw,
                    "n_blocks": int(EXTRAP_RW_BLOCKS),
                }
            )
    else:
        for k, n_hubs, c, l1r, pw in product(
            EXTRAP_K_GRID,
            EXTRAP_N_HUBS_GRID,
            EXTRAP_C_GRID,
            EXTRAP_L1_RATIO_GRID,
            EXTRAP_POS_WEIGHT_GRID,
        ):
            grid.append(
                {
                    "method": method,
                    "k": int(k),
                    "n_hubs": int(n_hubs),
                    "C": float(c),
                    "l1_ratio": float(l1r),
                    "pos_weight": float(pw),
                    "rw_intercept": rw,
                    "n_blocks": int(EXTRAP_RW_BLOCKS),
                }
            )
    return grid


# --- Fit one (representation + model) ----------------------------------------
def _fit_one(
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_ts: pd.Series,
    *,
    inference: str,
    draws: int,
    tune: int,
    svi_steps: int,
) -> tuple[ExtrapRepresentation, BayesianElasticNetLogistic]:
    rep = ExtrapRepresentation(
        params["method"], k=params["k"], n_hubs=params["n_hubs"]
    ).fit(X_train, y_train)
    design = rep.transform(X_train)

    block_idx = None
    if params["rw_intercept"]:
        block_idx = make_block_index(train_ts, params["n_blocks"])

    C, l1_ratio = _resolve_prior_params(params)
    model = BayesianElasticNetLogistic(
        rw_intercept=params["rw_intercept"],
        n_blocks=params["n_blocks"],
        C=C,
        l1_ratio=l1_ratio,
        pos_weight=float(params.get("pos_weight", 1.0)),
        inference=inference,
        draws=draws,
        tune=tune,
        svi_steps=svi_steps,
        seed=RANDOM_SEED,
    ).fit(design.to_numpy(dtype=float), np.asarray(y_train, dtype=int), block_idx)
    return rep, model


def _resolve_prior_params(params: dict) -> tuple[float, float]:
    """Read (C, l1_ratio); translate legacy (l1_scale, l2) tuned payloads in place."""
    if "C" in params:
        return float(params["C"]), float(params["l1_ratio"])
    # Back-compat: stale tuned JSON stored the raw (l1_scale, l2) prior knobs.
    l1_scale = float(params["l1_scale"])
    l2 = float(params["l2"])
    C = 1.0 / (1.0 / l1_scale + 2.0 * l2)
    l1_ratio = C / l1_scale
    return C, l1_ratio


def _fold_metrics(y_val: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    y_pred = (proba >= threshold).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_val, y_pred)),
        "true_positive_rate": float(recall_score(y_val, y_pred, zero_division=0)),
        "true_negative_rate": float(
            recall_score(y_val, y_pred, pos_label=0, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_val, proba)) if len(np.unique(y_val)) > 1 else 0.5,
        "pr_auc": float(average_precision_score(y_val, proba)),
    }


def _cv_oof(
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    train_df: pd.DataFrame,
    *,
    inference: str,
    svi_steps: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Run blocked CV for one config; return concatenated (y, proba) + per-fold metrics."""
    cv = make_blocked_time_cv(train_df)
    ts = train_df[TIMESTAMP_COL].reset_index(drop=True)
    y_arr = np.asarray(y, dtype=int)
    oof_y: list[np.ndarray] = []
    oof_p: list[np.ndarray] = []
    per_fold: list[dict] = []
    for train_idx, val_idx in cv.split(X, y):
        rep, model = _fit_one(
            params,
            X.iloc[train_idx],
            y.iloc[train_idx],
            ts.iloc[train_idx],
            inference=inference,
            draws=200,
            tune=200,
            svi_steps=svi_steps,
        )
        proba = model.predict_proba_pos(rep.transform(X.iloc[val_idx]))
        y_val = y_arr[val_idx]
        oof_y.append(y_val)
        oof_p.append(proba)
        per_fold.append(_fold_metrics(y_val, proba, 0.5))
    return (
        np.concatenate(oof_y) if oof_y else np.zeros(0),
        np.concatenate(oof_p) if oof_p else np.zeros(0),
        per_fold,
    )


# --- Tuning ------------------------------------------------------------------
def tune_bayes_model(model_id: str, train_df: pd.DataFrame, cols: list[str]) -> dict:
    from secom.extrap_pipelines import EXTRAP_BAYES_INFERENCE

    X = train_df[cols]
    y = train_df[TARGET_COL].astype(int)
    grid = _param_grid(model_id)
    inf = EXTRAP_BAYES_INFERENCE

    print(f"\n=== {model_id} (bayesian {CV_PROTOCOL_EXTRAP}) ===", flush=True)
    print(f"GridSearch: {len(grid)} configs x blocked CV folds", flush=True)

    best = None
    for params in tqdm(grid, desc=f"Tuning {model_id} ({len(grid)} cfgs)", unit="cfg"):
        oof_y, oof_p, per_fold = _cv_oof(
            params, X, y, train_df,
            inference="advi", svi_steps=inf["search_svi_steps"],
        )
        mean_pr = float(np.mean([f["pr_auc"] for f in per_fold])) if per_fold else 0.0
        tqdm.write(
            f"  k={params['k']} n_hubs={params['n_hubs']} "
            f"C={params['C']} l1r={params['l1_ratio']} "
            f"pw={params['pos_weight']}: "
            f"mean_pr_auc={mean_pr:.4f}"
        )
        cand = {"params": params, "mean_pr": mean_pr, "oof_y": oof_y, "oof_p": oof_p,
                "per_fold": per_fold}
        if best is None or mean_pr > best["mean_pr"]:
            best = cand

    assert best is not None
    # OOF isotonic calibration on the best config's ADVI out-of-fold predictions:
    # pos_weight inflates the raw posterior probabilities, so we sweep the
    # cost-optimal thresholds on the *calibrated* scale and persist the map for
    # predict time. Monotonic -> PR-AUC/ROC unchanged.
    cal = IsotonicCalibrator().fit(best["oof_p"], best["oof_y"])
    cal_oof_p = cal.transform(best["oof_p"])
    profiles = threshold_profile_sweep(best["oof_y"], cal_oof_p)
    default_thr = profiles.get(DEFAULT_PROFILE_ID, profiles.get("ber", {})).get(
        "best_threshold", 0.5
    )

    result = {f"test_{k}": np.asarray([f[k] for f in best["per_fold"]])
              for k in ("balanced_accuracy", "true_positive_rate",
                        "true_negative_rate", "roc_auc", "pr_auc")}
    cv_summary = {
        "mean_pr_auc": best["mean_pr"],
        "std_pr_auc": float(np.std([f["pr_auc"] for f in best["per_fold"]], ddof=0))
        if best["per_fold"] else 0.0,
        "mean_roc_auc": float(np.mean([f["roc_auc"] for f in best["per_fold"]]))
        if best["per_fold"] else 0.0,
        "n_folds": len(best["per_fold"]),
        **score_row_from_cv_result(result),
    }

    payload = {
        "model_id": model_id,
        "track": "extrapolation",
        "cv_protocol": CV_PROTOCOL_EXTRAP,
        "classifier_threshold": float(default_thr),
        "grid_search_best_params": best["params"],
        "decay_lambda": 0.0,
        "decay_lambda_search": None,
        "cv_summary": cv_summary,
        "threshold_profiles": profiles,
        "calibration": cal.to_dict(),
        "bayesian_inference": {
            "sampler": "numpyro_nuts",
            "search_inference": "advi",
            "final_draws": inf["final_draws"],
            "final_tune": inf["final_tune"],
            "search_svi_steps": inf["search_svi_steps"],
            "n_blocks": best["params"]["n_blocks"],
        },
        "frozen_config": frozen_config(),
        "tuned_at": datetime.now(timezone.utc).isoformat(),
    }
    path = tuned_blocked_params_path(model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    print(f"Wrote {path}", flush=True)
    return payload


# --- Benchmark ---------------------------------------------------------------
def _fit_final(
    model_id: str,
    tuned: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_ts: pd.Series,
) -> tuple[ExtrapRepresentation, BayesianElasticNetLogistic]:
    from secom.extrap_pipelines import EXTRAP_BAYES_INFERENCE

    inf = EXTRAP_BAYES_INFERENCE
    params = tuned[model_id]["grid_search_best_params"]
    rep, model = _fit_one(
        params, X_train, y_train, train_ts,
        inference="nuts", draws=inf["final_draws"], tune=inf["final_tune"],
        svi_steps=inf["search_svi_steps"],
    )
    # Reattach the OOF isotonic map (fit on ADVI OOF during tuning) to the final
    # NUTS model so holdout / gate-conditional / risk-coverage all consume
    # calibrated probabilities. ADVI-OOF map on the NUTS final is the accepted
    # documented approximation.
    cal = IsotonicCalibrator.from_dict(tuned[model_id].get("calibration"))
    model.set_calibrator(cal)
    return rep, model


def run_bayes_blocked_leaderboard(
    model_ids: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_df: pd.DataFrame,
    tuned_blocked: dict[str, dict],
) -> pd.DataFrame:
    from secom.extrap_pipelines import EXTRAP_BAYES_INFERENCE

    # Stays uncalibrated: this board reports PR-AUC/ROC (calibration-invariant)
    # plus diagnostic TP/TN at 0.5; fitting a per-fold isotonic map here adds cost
    # for no leaderboard change.
    inf = EXTRAP_BAYES_INFERENCE
    rows = []
    print(f"Bayesian blocked CV: {len(model_ids)} models")
    for name in tqdm(model_ids, desc="Blocked CV leaderboard", unit="model"):
        params = tuned_blocked[name]["grid_search_best_params"]
        thr = float(tuned_blocked[name].get("classifier_threshold", 0.5))
        _, _, per_fold = _cv_oof(
            params, X_train, y_train, train_df,
            inference="advi", svi_steps=inf["search_svi_steps"],
        )
        result = {f"test_{k}": np.asarray([f[k] for f in per_fold])
                  for k in ("balanced_accuracy", "true_positive_rate",
                            "true_negative_rate", "roc_auc", "pr_auc")}
        row = {"pipeline": name, **score_row_from_cv_result(result)}
        rows.append(row)
        tqdm.write(f"  {name}: mean PR AUC {row['mean_pr_auc']:.3f} (±{row['std_pr_auc']:.3f})")
    return pd.DataFrame(rows).sort_values(
        "mean_pr_auc", ascending=False, kind="mergesort"
    ).reset_index(drop=True)


def _profile_holdout_columns(y_test, y_score, threshold, profile_id) -> dict:
    profile = THRESHOLD_PROFILES[profile_id]
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
        cols[f"{profile_id}_fbeta"] = fbeta_at_threshold(y_test, y_pred, beta=profile.beta)
    return cols


def run_bayes_holdout(
    fitted: dict[str, tuple],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tuned_blocked: dict[str, dict],
) -> pd.DataFrame:
    rows = []
    print(f"Bayesian holdout: {len(fitted)} models (reporting only)")
    for name, (rep, model) in tqdm(
        fitted.items(), total=len(fitted), desc="Holdout", unit="model"
    ):
        y_score = model.predict_proba_pos(rep.transform(X_test))
        row: dict = {
            "pipeline": name,
            "pr_auc": float(average_precision_score(y_test, y_score)),
            "roc_auc": float(roc_auc_score(y_test, y_score)),
        }
        thresholds = resolve_threshold_profiles(tuned_blocked.get(name, {}))
        for profile_id in PROFILE_IDS:
            if profile_id in thresholds:
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
                y_test, y_score, y_pred_default,
                n_bootstrap=HOLDOUT_BOOTSTRAP_N,
                ci_level=HOLDOUT_BOOTSTRAP_CI,
                rng=np.random.default_rng(RANDOM_SEED),
            )
        )
        rows.append(row)
        tqdm.write(f"  {name}: holdout PR AUC {row['pr_auc']:.3f}, ROC AUC {row['roc_auc']:.3f}")
    return pd.DataFrame(rows).sort_values(
        "pr_auc", ascending=False, kind="mergesort"
    ).reset_index(drop=True)


def run_bayes_gate_conditional(
    fitted: dict[str, tuple],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    gate,
) -> pd.DataFrame:
    from secom.benchmark import MIN_CONDITIONAL_POSITIVES

    masks = gate.flag_masks(X_test)
    in_control = gate.is_in_control(X_test)
    y_arr = np.asarray(y_test).astype(int)
    n_total = int(len(y_arr))
    n_in_control = int(in_control.sum())
    n_fails_total = int(y_arr.sum())
    n_fails_in_control = int(y_arr[in_control].sum())
    fails = y_arr.astype(bool)
    shared = {
        "coverage": (n_in_control / n_total) if n_total else None,
        "n_total": n_total,
        "n_in_control": n_in_control,
        "n_flagged_ooc": n_total - n_in_control,
        "n_flagged_density": int(masks["density_ooc"].sum()),
        "n_flagged_q": int(masks["q_ooc"].sum()),
        "n_flagged_both": int(masks["both_ooc"].sum()),
        "n_fails_total": n_fails_total,
        "n_fails_in_control": n_fails_in_control,
        "n_fails_flagged_ooc": n_fails_total - n_fails_in_control,
        "n_fails_flagged_density": int((masks["density_ooc"] & fails).sum()),
        "n_fails_flagged_q": int((masks["q_ooc"] & fails).sum()),
        "n_fails_flagged_both": int((masks["both_ooc"] & fails).sum()),
    }
    rows = []
    for name, (rep, model) in tqdm(
        fitted.items(), total=len(fitted), desc="Gate-conditional", unit="model"
    ):
        y_score = model.predict_proba_pos(rep.transform(X_test))
        row: dict = {
            "pipeline": name,
            **shared,
            "global_pr_auc": float(average_precision_score(y_arr, y_score)),
            "global_roc_auc": float(roc_auc_score(y_arr, y_score)),
        }
        y_ic = y_arr[in_control]
        score_ic = y_score[in_control]
        enough = n_fails_in_control >= MIN_CONDITIONAL_POSITIVES and len(np.unique(y_ic)) > 1
        if enough:
            row["conditional_pr_auc"] = float(average_precision_score(y_ic, score_ic))
            row["conditional_roc_auc"] = float(roc_auc_score(y_ic, score_ic))
            boot = stratified_bootstrap_holdout_metrics(
                y_ic, score_ic, threshold=0.5,
                n_bootstrap=HOLDOUT_BOOTSTRAP_N, ci_level=HOLDOUT_BOOTSTRAP_CI,
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


def run_bayes_risk_coverage(
    fitted: dict[str, tuple],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    gate,
    coverage_grid,
) -> pd.DataFrame:
    """Rank-based risk-coverage sweep for the extrapolation gate.

    Rank holdout wafers by the gate's single OOC severity, then for each target
    ``coverage`` keep the least-suspicious ``round(c * n)`` wafers and rescore
    every fitted model on that retained subset. ``coverage == 1.0`` keeps all
    wafers, so its PR/ROC AUC equals the global holdout metric. AUCs are NaN when
    the kept set has fewer than ``MIN_CONDITIONAL_POSITIVES`` fails or is
    single-class. Returns a long frame
    ``[pipeline, coverage, n_kept, n_fails_kept, pr_auc, roc_auc]``.
    """
    from secom.benchmark import MIN_CONDITIONAL_POSITIVES

    severity = np.asarray(gate.ooc_severity(X_test), dtype="float64")
    y_arr = np.asarray(y_test).astype(int)
    n_total = int(len(y_arr))
    # Ascending severity -> least-suspicious wafers come first.
    order = np.argsort(severity, kind="mergesort")

    # Precompute each model's holdout scores once; the sweep just subsets them.
    scores = {
        name: model.predict_proba_pos(rep.transform(X_test))
        for name, (rep, model) in fitted.items()
    }

    rows = []
    for c in coverage_grid:
        c = float(c)
        n_keep = int(round(c * n_total))
        n_keep = max(0, min(n_total, n_keep))
        keep_idx = order[:n_keep]
        y_keep = y_arr[keep_idx]
        n_fails_kept = int(y_keep.sum())
        enough = n_fails_kept >= MIN_CONDITIONAL_POSITIVES and len(np.unique(y_keep)) > 1
        for name in fitted:
            s_keep = scores[name][keep_idx]
            if enough:
                pr_auc = float(average_precision_score(y_keep, s_keep))
                roc_auc = float(roc_auc_score(y_keep, s_keep))
            else:
                pr_auc = float("nan")
                roc_auc = float("nan")
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


def collect_bayes_artifacts(
    fitted: dict[str, tuple],
    tuned_blocked: dict[str, dict],
    holdout_split: dict,
) -> dict:
    from datetime import datetime, timezone

    from secom.artifacts import REFERENCE_MODELS

    models: dict[str, dict] = {}
    for name, (rep, model) in fitted.items():
        coef = model.coef_summary()
        coef_sorted = coef.reindex(
            coef["mean"].abs().sort_values(ascending=False).index
        )
        models[name] = {
            "model_id": name,
            "family": "bayesian",
            "representation": rep.config(),
            "rw_intercept": bool(model.rw_intercept),
            "n_design_features": int(model.n_features_),
            "top_coefficients": coef_sorted.head(20).to_dict(orient="records"),
            "grid_search_best_params": tuned_blocked[name]["grid_search_best_params"],
            # Empty stages block so the dashboard reduction widget degrades gracefully.
            "stages": {},
        }
    return {
        "kind": "bayesian_extrapolation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "holdout_split": holdout_split,
        "reference_models": dict(REFERENCE_MODELS),
        "models": models,
    }


def cv_oof_scores(
    model_id: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_df: pd.DataFrame,
    tuned: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Blocked-CV out-of-fold (y, proba) for the tuned config (dashboard PR curves)."""
    from secom.extrap_pipelines import EXTRAP_BAYES_INFERENCE

    params = tuned["grid_search_best_params"]
    oof_y, oof_p, _ = _cv_oof(
        params, X_train, y_train, train_df,
        inference="advi", svi_steps=EXTRAP_BAYES_INFERENCE["search_svi_steps"],
    )
    return oof_y, oof_p


def fit_bayes_models(
    model_ids: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_ts: pd.Series,
    tuned_blocked: dict[str, dict],
) -> dict[str, tuple]:
    """Fit each model once on full train (final NUTS) for reuse across reports."""
    fitted: dict[str, tuple] = {}
    for name in tqdm(model_ids, desc="Final NUTS fits", unit="model"):
        tqdm.write(f"  fitting {name} on full train (NUTS)...")
        fitted[name] = _fit_final(name, tuned_blocked, X_train, y_train, train_ts)
    return fitted
