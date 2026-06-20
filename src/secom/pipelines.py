"""Unified scikit-learn model universe for both protocols.

A single 3x3 model grid is run on BOTH the random-stratified (interpolation) and
blocked-temporal (extrapolation) protocols:

    front-end {HSIC+hubs, RF-selection+hubs, sPLS}
        x classifier {RF, elastic-net LR, Bayesian elastic-net LR}

Every cell is a plain scikit-learn ``Pipeline`` built by ``build_model_pipeline``:

    raw c_id + rolling-Z c_id_rz
        -> median impute -> SmartCorrelatedSelection cluster
        -> front-end (selection + hub interactions, or sPLS)
        -> RobustScaler (full design)
        -> CalibratedClassifierCV(isotonic)( classifier )

The Bayesian head (``BayesianElasticNetLogistic``) is a drop-in sklearn
estimator, so it tunes/benchmarks through the exact same machinery as LR/RF.
Two standalone risk-coverage gates (``EFAGate``, ``BayesGate``) live in
:mod:`secom.gates`; classifiers no longer ingest gate features.

Shared foundation (paths, IO, splits, CV, estimator builders) is re-exported from
:mod:`secom.core`.
"""
from __future__ import annotations

from functools import partial

from secom.core import *  # noqa: F401,F403
from secom.core import (  # noqa: F401  (underscore names used by other modules)
    _CALENDAR_PATTERN,
    _MISSING_FLAG_PATTERN,
    _SENSOR_VALUE_PATTERN,
    _SENSOR_VALUE_PATTERN_EXTRAP,
    _auxiliary_transformers,
    _cluster_step,
    _sensor_preprocess_column,
    CLASSIFIER_CALIBRATION_CV,
    CORRELATED_SELECTION_THRESHOLD,
    RANDOM_SEED,
    calibrated_classifier,
    elastic_net_lr,
    feature_pipeline,
    frozen_config,
    median_imputer,
    random_forest_classifier,
)
from secom.bayes.model import BayesianElasticNetLogistic
from secom.hub_interactions import (  # noqa: F401
    HSICSelectHubBlock,
    LinearSelectT2HubBlock,
    PLSFeatures,
)

# --- Model grid: 3 front-ends x 3 classifiers, both protocols ----------------
FRONT_ENDS = ("hsic", "rfsel", "pls")
CLASSIFIER_KINDS = ("enet", "rf", "bayes")

MODEL_IDS: tuple[str, ...] = (
    "hsic_enet",
    "hsic_rf",
    "hsic_bayes",
    "rfsel_enet",
    "rfsel_rf",
    "rfsel_bayes",
    "pls_enet",
    "pls_rf",
    "pls_bayes",
)

#: (front_end, classifier_kind) for each neutral model id.
MODEL_CELLS: dict[str, tuple[str, str]] = {
    "hsic_enet": ("hsic", "enet"),
    "hsic_rf": ("hsic", "rf"),
    "hsic_bayes": ("hsic", "bayes"),
    "rfsel_enet": ("rfsel", "enet"),
    "rfsel_rf": ("rfsel", "rf"),
    "rfsel_bayes": ("rfsel", "bayes"),
    "pls_enet": ("pls", "enet"),
    "pls_rf": ("pls", "rf"),
    "pls_bayes": ("pls", "bayes"),
}

# Backwards-compat alias used across the benchmark / dashboard / utils.
BENCHMARK_MODEL_IDS = MODEL_IDS

# --- Front-end (selection / aggregation) grids -------------------------------
# Selection front-ends (HSIC / RF) screen K sensors then expand the top n_hubs
# into pairwise interaction features; sPLS aggregates into K latent components.
TOP_K_DEFAULT = 35
TOP_K_GRID = [35, 60]
N_HUBS_DEFAULT = 5
N_HUBS_GRID = [0, 5, 10]
PLS_N_COMPONENTS_DEFAULT = 20
PLS_N_COMPONENTS_GRID = [15, 25, 35, 50]

# --- Classifier head grids ---------------------------------------------------
# Elastic-net LR (saga) slope prior.
C_GRID = [0.0075]
L1_RATIO_GRID = [0.3]
# RF classifier depth.
RF_MAX_DEPTH_GRID = [3, 5]

# Bayesian elastic-net head: kept deliberately tiny (each grid point is a full
# ADVI fit x calibration folds x CV folds x 2 protocols).
BAYES_C_GRID = [0.01, 0.1]
BAYES_L1_RATIO_GRID = [0.3]
BAYES_POS_WEIGHT_GRID = [15.0]
BAYES_TOP_K = 35
BAYES_N_HUBS = 5
BAYES_PLS_COMPONENTS = 20
BAYES_SVI_STEPS = 1500
# Each calibration fold is a full ADVI refit -> use fewer folds than LR/RF.
BAYES_CALIB_CV = 2

# Exponential time-decay (recency weighting): grid-tuned for the weight-capable
# (LR/RF) cells on the temporal protocol only.
DECAY_LAMBDA_DEFAULT = 0.0
DECAY_LAMBDA_GRID = [0.0, 1.0, 2.0, 4.0]
#: Weight-capable cells (LR/RF). The Bayesian cells skip decay (compute), and the
#: decay path only runs on the temporal protocol.
WEIGHTING_MODEL_IDS: tuple[str, ...] = (
    "hsic_enet",
    "hsic_rf",
    "rfsel_enet",
    "rfsel_rf",
    "pls_enet",
    "pls_rf",
)

# --- Standalone gates (risk-coverage tools, not pipeline steps) --------------
# EFA gate: Regularized EFA -> Hotelling T2 + Q (SPE).
EFA_GATE_N_FACTORS = 10
EFA_GATE_T2_ALPHA = 0.05
EFA_GATE_Q_ALPHA = 0.05
EFA_GATE_LOGIC = "or"

# Bayes gate: sparse Bayesian factor analysis -> BGM density + Q (SPE).
BAYES_GATE_N_FACTORS = 8
BAYES_GATE_BGM_COMPONENTS = 5
BAYES_GATE_LOADING_SCALE = 0.3
BAYES_GATE_DENSITY_ALPHA = 0.03
BAYES_GATE_Q_ALPHA = 0.005
BAYES_GATE_LOGIC = "or"
BAYES_GATE_SVI_STEPS = 2000
BAYES_GATE_N_SEEDS = 5
BAYES_GATE_CLIP = 5.0

# Both gates score the raw post-cluster sensor space (impute -> cluster).
GATE_CORR_THRESHOLD = CORRELATED_SELECTION_THRESHOLD

# Risk-coverage sweep: keep the least-suspicious fraction of holdout wafers at
# each coverage and rescore (1.0 == global holdout metric).
RISK_COVERAGE_GRID = [1.0, 0.99, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]


# --- Estimator builders ------------------------------------------------------
def bayesian_elastic_net(
    C: float = float(BAYES_C_GRID[0]),
    l1_ratio: float = float(BAYES_L1_RATIO_GRID[0]),
    pos_weight: float = float(BAYES_POS_WEIGHT_GRID[0]),
) -> BayesianElasticNetLogistic:
    return BayesianElasticNetLogistic(
        C=C,
        l1_ratio=l1_ratio,
        pos_weight=pos_weight,
        inference="advi",
        svi_steps=BAYES_SVI_STEPS,
        seed=RANDOM_SEED,
    )


_CLASSIFIER_BUILDERS = {
    "enet": partial(elastic_net_lr, C=float(C_GRID[0]), l1_ratio=float(L1_RATIO_GRID[0])),
    "rf": random_forest_classifier,
    "bayes": bayesian_elastic_net,
}


def _front_end_step(front_end: str, *, top_k: int, n_hubs: int, pls_n_components: int):
    if front_end == "hsic":
        return ("front_end", HSICSelectHubBlock(top_k=int(top_k), n_hubs=int(n_hubs)))
    if front_end == "rfsel":
        return (
            "front_end",
            LinearSelectT2HubBlock(top_k=int(top_k), n_hubs=int(n_hubs)),
        )
    if front_end == "pls":
        return ("front_end", PLSFeatures(n_components=int(pls_n_components)))
    raise ValueError(f"front_end must be one of {FRONT_ENDS}, got {front_end!r}")


def build_model_pipeline(
    front_end: str,
    classifier_kind: str,
    *,
    top_k: int = TOP_K_DEFAULT,
    n_hubs: int = N_HUBS_DEFAULT,
    pls_n_components: int = PLS_N_COMPONENTS_DEFAULT,
):
    """Build a calibrated sklearn pipeline for one (front-end, classifier) cell.

    raw+rz sensors -> median impute -> cluster -> front-end -> RobustScaler ->
    CalibratedClassifierCV(isotonic)(classifier).
    """
    if classifier_kind not in _CLASSIFIER_BUILDERS:
        raise ValueError(
            f"classifier_kind must be one of {CLASSIFIER_KINDS}, got {classifier_kind!r}"
        )
    sensor_steps: list[tuple[str, object]] = [
        ("impute", median_imputer()),
        ("cluster", _cluster_step()),
        _front_end_step(
            front_end,
            top_k=top_k,
            n_hubs=n_hubs,
            pls_n_components=pls_n_components,
        ),
    ]
    preprocess = _sensor_preprocess_column(
        sensor_steps, sensor_pattern=_SENSOR_VALUE_PATTERN_EXTRAP
    )
    calib_cv = BAYES_CALIB_CV if classifier_kind == "bayes" else CLASSIFIER_CALIBRATION_CV
    classifier = _CLASSIFIER_BUILDERS[classifier_kind]()
    return feature_pipeline(classifier, preprocess, calib_cv=calib_cv)


def is_bayesian(model_id: str) -> bool:
    """True for the three Bayesian-head cells (still plain sklearn pipelines)."""
    return MODEL_CELLS.get(model_id, ("", ""))[1] == "bayes"


def pipelines_frozen_config_fragment() -> dict:
    """Model-grid + gate entries for the frozen-config snapshot."""
    return {
        "model_ids": list(MODEL_IDS),
        "model_cells": {k: list(v) for k, v in MODEL_CELLS.items()},
        "front_ends": list(FRONT_ENDS),
        "classifier_kinds": list(CLASSIFIER_KINDS),
        "top_k_default": int(TOP_K_DEFAULT),
        "top_k_grid": [int(k) for k in TOP_K_GRID],
        "n_hubs_default": int(N_HUBS_DEFAULT),
        "n_hubs_grid": [int(k) for k in N_HUBS_GRID],
        "pls_n_components_default": int(PLS_N_COMPONENTS_DEFAULT),
        "pls_n_components_grid": [int(k) for k in PLS_N_COMPONENTS_GRID],
        "c_grid": [float(c) for c in C_GRID],
        "l1_ratio_grid": [float(r) for r in L1_RATIO_GRID],
        "rf_max_depth_grid": [int(d) for d in RF_MAX_DEPTH_GRID],
        "bayes_c_grid": [float(c) for c in BAYES_C_GRID],
        "bayes_l1_ratio_grid": [float(r) for r in BAYES_L1_RATIO_GRID],
        "bayes_pos_weight_grid": [float(w) for w in BAYES_POS_WEIGHT_GRID],
        "bayes_top_k": int(BAYES_TOP_K),
        "bayes_n_hubs": int(BAYES_N_HUBS),
        "bayes_pls_components": int(BAYES_PLS_COMPONENTS),
        "bayes_svi_steps": int(BAYES_SVI_STEPS),
        "bayes_calib_cv": int(BAYES_CALIB_CV),
        "decay_lambda_default": float(DECAY_LAMBDA_DEFAULT),
        "decay_lambda_grid": [float(x) for x in DECAY_LAMBDA_GRID],
        "weighting_model_ids": list(WEIGHTING_MODEL_IDS),
        "efa_gate_n_factors": int(EFA_GATE_N_FACTORS),
        "efa_gate_t2_alpha": float(EFA_GATE_T2_ALPHA),
        "efa_gate_q_alpha": float(EFA_GATE_Q_ALPHA),
        "efa_gate_logic": str(EFA_GATE_LOGIC),
        "bayes_gate_n_factors": int(BAYES_GATE_N_FACTORS),
        "bayes_gate_bgm_components": int(BAYES_GATE_BGM_COMPONENTS),
        "bayes_gate_loading_scale": float(BAYES_GATE_LOADING_SCALE),
        "bayes_gate_density_alpha": float(BAYES_GATE_DENSITY_ALPHA),
        "bayes_gate_q_alpha": float(BAYES_GATE_Q_ALPHA),
        "bayes_gate_logic": str(BAYES_GATE_LOGIC),
        "bayes_gate_svi_steps": int(BAYES_GATE_SVI_STEPS),
        "bayes_gate_n_seeds": int(BAYES_GATE_N_SEEDS),
        "bayes_gate_clip": float(BAYES_GATE_CLIP),
        "gate_corr_threshold": float(GATE_CORR_THRESHOLD),
        "risk_coverage_grid": [float(c) for c in RISK_COVERAGE_GRID],
    }
