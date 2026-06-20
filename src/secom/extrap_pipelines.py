"""Extrapolation track: Bayesian yield line (blocked CV + temporal holdout).

3-stage pipeline per model: (1) HSIC-Lasso / random-forest nonlinear screening or
sPLS aggregation on the rolling-Z sensors, (2) a physical interaction frame (mains
+ pairs + squares on ``n_hubs`` anchors), (3) a custom NumPyro elastic-net logistic
head (static or random-walk intercept) with tunable positive-class weighting and
OOF isotonic calibration. All grids/specs live here; the heavy NumPyro code is
imported lazily by :mod:`secom.bayes.harness`.

Drift is handled structurally by the random-walk intercept, so the sklearn
time-decay sample-weight path is retired for this track (``WEIGHTING_MODEL_IDS``
is empty).
"""
from __future__ import annotations

from secom.core import CORRELATED_SELECTION_THRESHOLD

# 5 Bayesian models: {HSIC, RF} x {static, RW-intercept} + sPLS RW comparator.
EXTRAP_MODEL_IDS = (
    "extrap_hsic_static",
    "extrap_hsic_rw",
    "extrap_rf_static",
    "extrap_rf_rw",
    "extrap_spls_rw",
)

# Stage 1 selection size (physical anchors) and Stage 2 interaction expansion.
# n_hubs >= 5 so the HSIC/RF screening models always build the interaction frame
# (squares + pairs); sPLS ignores n_hubs (no interactions on latent scores).
EXTRAP_K_GRID = [35]
EXTRAP_N_HUBS_GRID = [5]
EXTRAP_SPLS_COMPONENTS_GRID = [15, 20]

# Stage 3 elastic-net slope prior in sklearn terms: C (inverse total penalty) x
# l1_ratio (L1/L2 mix). C grid fills the middle so the class-weighted likelihood
# can land a usable slope strength instead of being forced to near-zero.
EXTRAP_C_GRID = [0.01, 0.1]
EXTRAP_L1_RATIO_GRID = [0.3, 0.7]
# Positive-class likelihood weight (renormalized to preserve effective N); lets
# the minority class pull harder so larger C earns its keep.
EXTRAP_POS_WEIGHT_GRID = [15]
EXTRAP_RW_BLOCKS = 3

# Inference budget: ADVI in the CV grid search, NUTS for the final fits.
EXTRAP_BAYES_INFERENCE = {
    "search_svi_steps": 1500,
    "final_draws":400,
    "final_tune": 400,
}

# Extrapolation process gate: sparse Bayesian factor analysis (NumPyro ADVI) ->
# BayesianGaussianMixture density + Q (SPE) reconstruction stat. Scored on the
# RAW post-cluster sensor space (impute -> SmartCorrelatedSelection, same
# front-end as the interp gate) so it stays orthogonal to the rolling-Z yield
# models and flags drift/excursions they normalise away. Fit once on
# passing-train wafers; no Hotelling T2, no Isolation Forest. Gate UCLs use
# empirical passing-wafer quantiles (alpha tail each side).
EXTRAP_GATE_N_FACTORS = 8
EXTRAP_GATE_BGM_COMPONENTS = 5
EXTRAP_GATE_LOADING_SCALE = 0.3
EXTRAP_GATE_DENSITY_ALPHA = 0.03
EXTRAP_GATE_Q_ALPHA = 0.005
EXTRAP_GATE_LOGIC = "or"
EXTRAP_GATE_SVI_STEPS = 2000
EXTRAP_GATE_CORR_THRESHOLD = CORRELATED_SELECTION_THRESHOLD
# Robust-scaled gate features are clipped to +/-this bound before the Gaussian
# sBFA/BGM/SPE, so un-clipped SECOM spikes cannot dominate the squared SPE and
# log-density (RobustScaler resists outliers in the estimate but does not bound
# the tails).
EXTRAP_GATE_CLIP = 5.0
# Seed-ensemble: average density/Q over this many independent sBFA+BGM fits so the
# ADVI run-to-run variance does not flip the gate's operating point.
EXTRAP_GATE_N_SEEDS = 5

# Risk-coverage sweep: keep the least-suspicious fraction of holdout wafers at each
# coverage and rescore, tracing conditional PR/ROC AUC vs coverage (1.0 == global).
EXTRAP_RISK_COVERAGE_GRID = [1.0, 0.99, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]

BAYES_MODEL_SPECS: dict[str, dict] = {
    "extrap_hsic_static": {"method": "hsic", "rw_intercept": False},
    "extrap_hsic_rw": {"method": "hsic", "rw_intercept": True},
    "extrap_rf_static": {"method": "rf", "rw_intercept": False},
    "extrap_rf_rw": {"method": "rf", "rw_intercept": True},
    "extrap_spls_rw": {"method": "spls", "rw_intercept": True},
}

# Drift handled structurally by the RW intercept -> no sklearn time-decay weights.
WEIGHTING_MODEL_IDS: tuple[str, ...] = ()
DECAY_LAMBDA_DEFAULT = 0.0
DECAY_LAMBDA_GRID = [0.0]


def is_bayesian(model_id: str) -> bool:
    return model_id in BAYES_MODEL_SPECS


def extrap_frozen_config_fragment() -> dict:
    """Extrapolation-specific entries for the frozen-config snapshot."""
    return {
        "extrapolation_model_ids": list(EXTRAP_MODEL_IDS),
        "extrap_track": "bayesian",
        "extrap_k_grid": [int(k) for k in EXTRAP_K_GRID],
        "extrap_n_hubs_grid": [int(k) for k in EXTRAP_N_HUBS_GRID],
        "extrap_spls_components_grid": [int(k) for k in EXTRAP_SPLS_COMPONENTS_GRID],
        "extrap_c_grid": [float(x) for x in EXTRAP_C_GRID],
        "extrap_l1_ratio_grid": [float(x) for x in EXTRAP_L1_RATIO_GRID],
        "extrap_pos_weight_grid": [float(x) for x in EXTRAP_POS_WEIGHT_GRID],
        "extrap_rw_blocks": int(EXTRAP_RW_BLOCKS),
        "extrap_bayes_inference": dict(EXTRAP_BAYES_INFERENCE),
        "extrap_bayes_model_specs": {k: dict(v) for k, v in BAYES_MODEL_SPECS.items()},
        "extrap_gate_method": "sbfa_bgm_q",
        "extrap_gate_feature_stage": "post_cluster_sbfa",
        "extrap_gate_n_factors": int(EXTRAP_GATE_N_FACTORS),
        "extrap_gate_bgm_components": int(EXTRAP_GATE_BGM_COMPONENTS),
        "extrap_gate_loading_scale": float(EXTRAP_GATE_LOADING_SCALE),
        "extrap_gate_density_alpha": float(EXTRAP_GATE_DENSITY_ALPHA),
        "extrap_gate_q_alpha": float(EXTRAP_GATE_Q_ALPHA),
        "extrap_gate_logic": str(EXTRAP_GATE_LOGIC),
        "extrap_gate_svi_steps": int(EXTRAP_GATE_SVI_STEPS),
        "extrap_gate_corr_threshold": float(EXTRAP_GATE_CORR_THRESHOLD),
        "extrap_gate_n_seeds": int(EXTRAP_GATE_N_SEEDS),
        "extrap_gate_clip": float(EXTRAP_GATE_CLIP),
        "extrap_risk_coverage_grid": [float(c) for c in EXTRAP_RISK_COVERAGE_GRID],
        "weighting_model_ids": list(WEIGHTING_MODEL_IDS),
        "decay_lambda_default": float(DECAY_LAMBDA_DEFAULT),
    }
