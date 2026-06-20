"""Cross-protocol diagnostic: retune each model under the *other* track's protocol.

Standalone and additive. Fills the off-diagonal of the
{model family} x {evaluation protocol} 2x2:

    - Direction A (bayes_on_interp): the Bayesian extrapolation models retuned
      under the interpolation protocol (random-stratified holdout + stratified
      OOF CV). If a Bayesian model recovers ~0.70 ROC here, the model is correct
      and the weak extrapolation result is a drift effect, not a model defect.
    - Direction B (sklearn_on_extrap): the sklearn interpolation models retuned
      under the extrapolation protocol (temporal holdout + blocked-time CV). If
      these strong models also collapse to ~0.55, that independently confirms the
      gap is drift.

Nothing here writes to ``data/processed/tuned/``, ``tuned_blocked/`` or the main
benchmark JSON. All output lands under ``data/processed/cross_eval/`` so the real
tracks are untouched. Run one model per invocation via
``python -m secom.cli.run_cross_eval``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm

from secom.core import (
    BENCHMARK_RESULTS_PATH,
    OUTPUT_DIR,
    RANDOM_SEED,
    TARGET_COL,
    TIMESTAMP_COL,
    feature_columns,
    load_mart,
    make_stratified_kfold_for_oof,
    split_train_test,
    split_train_test_random,
)
from secom.cv import make_blocked_time_cv
from secom.utils import json_safe

CROSS_EVAL_DIR = OUTPUT_DIR / "cross_eval"
BAYES_ON_INTERP_DIR = CROSS_EVAL_DIR / "bayes_on_interp"
SKLEARN_ON_EXTRAP_DIR = CROSS_EVAL_DIR / "sklearn_on_extrap"

DIRECTIONS = ("bayes_on_interp", "sklearn_on_extrap")


def _write(path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    print(f"Wrote {path}", flush=True)


# --- Direction A: Bayesian models on the interpolation protocol --------------
def retune_bayes_on_interp(model_id: str) -> dict:
    """Retune one Bayesian model under random-stratified OOF CV (interp regime)."""
    from secom.bayes.harness import _fit_one, _param_grid
    from secom.extrap_pipelines import (
        BAYES_MODEL_SPECS,
        EXTRAP_BAYES_INFERENCE,
        is_bayesian,
    )

    if not is_bayesian(model_id):
        raise SystemExit(f"--model {model_id!r} is not a Bayesian (extrapolation) model")

    df = load_mart()
    cols = feature_columns(df)
    train_df, _ = split_train_test_random(df)
    X = train_df[cols]
    y = train_df[TARGET_COL].astype(int)
    ts = train_df[TIMESTAMP_COL].reset_index(drop=True)
    y_arr = y.to_numpy()

    rw = bool(BAYES_MODEL_SPECS[model_id]["rw_intercept"])
    grid = _param_grid(model_id)
    cv = make_stratified_kfold_for_oof()
    svi_steps = EXTRAP_BAYES_INFERENCE["search_svi_steps"]

    print(f"\n=== {model_id} (bayes_on_interp: random_stratified_oof) ===", flush=True)
    if rw:
        print(
            "  NOTE: RW-intercept under shuffled folds is not a meaningful drift "
            "test; interpret with care.",
            flush=True,
        )
    print(f"GridSearch: {len(grid)} configs x {cv.get_n_splits()} folds", flush=True)

    best: dict | None = None
    for params in tqdm(grid, desc=f"CrossEval {model_id} ({len(grid)} cfgs)", unit="cfg"):
        fold_pr: list[float] = []
        fold_roc: list[float] = []
        for tr, val in cv.split(X, y_arr):
            rep, model = _fit_one(
                params,
                X.iloc[tr],
                y.iloc[tr],
                ts.iloc[tr],
                inference="advi",
                draws=0,
                tune=0,
                svi_steps=svi_steps,
            )
            proba = model.predict_proba_pos(rep.transform(X.iloc[val]))
            y_val = y_arr[val]
            fold_pr.append(float(average_precision_score(y_val, proba)))
            fold_roc.append(
                float(roc_auc_score(y_val, proba)) if len(np.unique(y_val)) > 1 else 0.5
            )
        mean_pr = float(np.mean(fold_pr))
        tqdm.write(
            f"  k={params['k']} n_hubs={params['n_hubs']} C={params['C']} "
            f"l1r={params['l1_ratio']} pw={params['pos_weight']}: "
            f"mean_pr_auc={mean_pr:.4f}"
        )
        cand = {"params": params, "fold_pr": fold_pr, "fold_roc": fold_roc, "mean_pr": mean_pr}
        if best is None or mean_pr > best["mean_pr"]:
            best = cand

    assert best is not None
    payload = {
        "model_id": model_id,
        "direction": "bayes_on_interp",
        "protocol": "random_stratified_oof",
        "rw_under_random_cv": rw,
        "grid_search_best_params": best["params"],
        "mean_pr_auc": float(np.mean(best["fold_pr"])),
        "std_pr_auc": float(np.std(best["fold_pr"], ddof=0)),
        "mean_roc_auc": float(np.mean(best["fold_roc"])),
        "std_roc_auc": float(np.std(best["fold_roc"], ddof=0)),
        "per_fold": {"pr_auc": best["fold_pr"], "roc_auc": best["fold_roc"]},
        "n_configs": len(grid),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write(BAYES_ON_INTERP_DIR / f"{model_id}.json", payload)
    print(
        f"  best mean_pr_auc={payload['mean_pr_auc']:.4f} "
        f"mean_roc_auc={payload['mean_roc_auc']:.4f}",
        flush=True,
    )
    return payload


# --- Direction B: sklearn models on the extrapolation protocol ---------------
def retune_sklearn_on_extrap(model_id: str) -> dict:
    """Retune one sklearn model under blocked-time CV (extrapolation regime)."""
    from secom.tuning.registry import (
        MODEL_SPECS,
        fit_with_progress,
        run_grid_search,
        summarize_cv_search,
    )

    if model_id not in MODEL_SPECS:
        raise SystemExit(f"--model {model_id!r} is not a sklearn (interpolation) model")

    df = load_mart()
    cols = feature_columns(df)
    train_df, _ = split_train_test(df)
    X = train_df[cols]
    y = train_df[TARGET_COL].astype(int)

    spec = MODEL_SPECS[model_id]
    cv = make_blocked_time_cv(train_df)

    print(f"\n=== {model_id} (sklearn_on_extrap: blocked_time_local_strat) ===", flush=True)
    search, n_cand, n_splits, total = run_grid_search(spec, X, y, cv=cv)
    print(f"GridSearch: {n_cand} x {n_splits} = {total} fits", flush=True)
    search = fit_with_progress(search, X, y)
    summary, _fold_results, _aggregated = summarize_cv_search(search, spec)

    payload = {
        "model_id": model_id,
        "direction": "sklearn_on_extrap",
        "protocol": "blocked_time_local_strat",
        "grid_search_best_params": spec.build_grid_search_best_params(summary),
        "mean_pr_auc": float(summary["mean_pr_auc"]),
        "std_pr_auc": float(summary["std_pr_auc"]),
        "mean_roc_auc": float(summary["mean_roc_auc"]),
        "std_roc_auc": float(summary["std_roc_auc"]),
        "n_configs": int(n_cand),
        "n_folds": int(n_splits),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write(SKLEARN_ON_EXTRAP_DIR / f"{model_id}.json", payload)
    print(
        f"  mean_pr_auc={payload['mean_pr_auc']:.4f} "
        f"mean_roc_auc={payload['mean_roc_auc']:.4f}",
        flush=True,
    )
    return payload


def run_direction(direction: str, model_id: str) -> dict:
    if direction == "bayes_on_interp":
        return retune_bayes_on_interp(model_id)
    if direction == "sklearn_on_extrap":
        return retune_sklearn_on_extrap(model_id)
    raise SystemExit(f"Unknown --direction {direction!r}; choose from {DIRECTIONS}")


# --- Reporting: the filled 2x2 ----------------------------------------------
def _load_dir(path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for f in sorted(path.glob("*.json")):
        out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
    return out


def _diagonal_rows(section_key: str) -> dict[str, dict]:
    """Read a leaderboard section from the existing benchmark JSON (read-only)."""
    if not BENCHMARK_RESULTS_PATH.exists():
        return {}
    payload = json.loads(BENCHMARK_RESULTS_PATH.read_text(encoding="utf-8"))
    rows = payload.get(section_key) or []
    return {r["pipeline"]: r for r in rows if "pipeline" in r}


def _fmt(row: dict | None) -> str:
    if not row:
        return "        --        "
    return (
        f"PR {row['mean_pr_auc']:.3f}+/-{row.get('std_pr_auc', 0.0):.3f} | "
        f"ROC {row['mean_roc_auc']:.3f}+/-{row.get('std_roc_auc', 0.0):.3f}"
    )


def summarize_cross_eval() -> dict:
    """Print the filled 2x2 (off-diagonal from cross_eval, diagonal from benchmark)."""
    bayes_interp = _load_dir(BAYES_ON_INTERP_DIR)          # off-diag
    sklearn_extrap = _load_dir(SKLEARN_ON_EXTRAP_DIR)      # off-diag
    interp_diag = _diagonal_rows("leaderboard")            # sklearn + random
    extrap_diag = _diagonal_rows("leaderboard_blocked")    # bayes + blocked

    print("\n================ Cross-evaluation 2x2 ================", flush=True)
    print("sklearn (interpolation) models", flush=True)
    for mid in sorted(set(interp_diag) | set(sklearn_extrap)):
        print(f"  {mid}", flush=True)
        print(f"    random stratified CV : {_fmt(interp_diag.get(mid))}", flush=True)
        print(f"    blocked temporal  CV : {_fmt(sklearn_extrap.get(mid))}", flush=True)
    print("Bayesian (extrapolation) models", flush=True)
    for mid in sorted(set(extrap_diag) | set(bayes_interp)):
        rw = bayes_interp.get(mid, {}).get("rw_under_random_cv")
        flag = "  [RW: not meaningful under random CV]" if rw else ""
        print(f"  {mid}", flush=True)
        print(f"    random stratified CV : {_fmt(bayes_interp.get(mid))}{flag}", flush=True)
        print(f"    blocked temporal  CV : {_fmt(extrap_diag.get(mid))}", flush=True)
    print("=====================================================", flush=True)

    return {
        "bayes_on_interp": bayes_interp,
        "sklearn_on_extrap": sklearn_extrap,
        "interp_diagonal": interp_diag,
        "extrap_diagonal": extrap_diag,
    }
