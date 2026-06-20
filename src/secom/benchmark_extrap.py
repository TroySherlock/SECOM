"""Extrapolation benchmark flow (Bayesian track).

Parallel to the sklearn interpolation flow in ``benchmark.py``: blocked-CV
leaderboard + temporal holdout + process-gate-conditional report, all driven by
the custom NumPyro harness in :mod:`secom.bayes.harness`. The gate-conditional
section uses the extrapolation gate (sparse Bayesian factor analysis -> a
BayesianGaussianMixture density + Q/SPE statistic) from
:mod:`secom.bayes.gate`.
"""
from __future__ import annotations

import pandas as pd

from secom.bayes.gate import ExtrapProcessGate
from secom.extrap_pipelines import (
    EXTRAP_GATE_BGM_COMPONENTS,
    EXTRAP_GATE_CORR_THRESHOLD,
    EXTRAP_GATE_DENSITY_ALPHA,
    EXTRAP_GATE_LOGIC,
    EXTRAP_GATE_N_FACTORS,
    EXTRAP_GATE_N_SEEDS,
    EXTRAP_GATE_Q_ALPHA,
    EXTRAP_GATE_SVI_STEPS,
    EXTRAP_RISK_COVERAGE_GRID,
)
from secom.utils import load_all_tuned_blocked_params


def run_extrapolation_benchmark(
    extrap_ids: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    train_df: pd.DataFrame,
    train_ts: pd.Series,
    holdout_split: dict,
) -> dict:
    """Run the full Bayesian extrapolation report; returns frames + gate + artifacts."""
    from secom.bayes.harness import (
        collect_bayes_artifacts,
        fit_bayes_models,
        run_bayes_blocked_leaderboard,
        run_bayes_gate_conditional,
        run_bayes_holdout,
        run_bayes_risk_coverage,
    )

    tuned_blocked = load_all_tuned_blocked_params(model_ids=extrap_ids)

    print("\nBayesian blocked time CV leaderboard (extrapolation):")
    leaderboard_blocked = run_bayes_blocked_leaderboard(
        extrap_ids, X_train, y_train, train_df, tuned_blocked
    )

    print("\nFitting final Bayesian models on full train (NUTS) for holdout/artifacts:")
    fitted = fit_bayes_models(extrap_ids, X_train, y_train, train_ts, tuned_blocked)

    print("\nTemporal holdout (Bayesian posterior predictive):")
    holdout = run_bayes_holdout(fitted, X_test, y_test, tuned_blocked)

    print(
        f"\nExtrapolation gate (post-cluster raw sBFA -> BGM density "
        f"{EXTRAP_GATE_LOGIC.upper()} Q/SPE, passing-train reference):"
    )
    gate = ExtrapProcessGate(
        n_factors=EXTRAP_GATE_N_FACTORS,
        n_mixture_components=EXTRAP_GATE_BGM_COMPONENTS,
        density_alpha=EXTRAP_GATE_DENSITY_ALPHA,
        q_alpha=EXTRAP_GATE_Q_ALPHA,
        svi_steps=EXTRAP_GATE_SVI_STEPS,
        gate_corr_threshold=EXTRAP_GATE_CORR_THRESHOLD,
        n_seeds=EXTRAP_GATE_N_SEEDS,
        logic=EXTRAP_GATE_LOGIC,
    ).fit(X_train, y_train)
    holdout_conditional = run_bayes_gate_conditional(fitted, X_test, y_test, gate)

    print("\nGate risk-coverage sweep (rank by OOC severity):")
    risk_coverage = run_bayes_risk_coverage(
        fitted, X_test, y_test, gate, EXTRAP_RISK_COVERAGE_GRID
    )

    artifacts = collect_bayes_artifacts(fitted, tuned_blocked, holdout_split)

    return {
        "tuned_blocked": tuned_blocked,
        "leaderboard_blocked": leaderboard_blocked,
        "holdout": holdout,
        "holdout_conditional": holdout_conditional,
        "risk_coverage": risk_coverage,
        "process_gate": gate.config(),
        "artifacts": artifacts,
    }
