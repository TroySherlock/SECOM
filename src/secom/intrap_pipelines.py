"""Interpolation track: stratified CV + random holdout + Regularized-EFA gate.

2x2 yield line: {RF-selection, PLS} x {elastic-net LR, RF classifier}. All
interpolation-specific constants, tuning grids, and preprocess builders live
here; shared scaffolding comes from :mod:`secom.core`.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline

from secom.core import (
    _SENSOR_VALUE_PATTERN,
    _auxiliary_transformers,
    _cluster_step,
    median_imputer,
)
from secom.hub_interactions import LinearSelectT2HubBlock, PLSFeatures

# 2x2 yield line: {RF-selection, PLS} x {elastic-net LR, RF classifier}.
INTERP_MODEL_IDS = (
    "intrap_linear_lr",
    "intrap_topk_rf",
    "intrap_pls_enet",
    "intrap_pls_rf",
)

# Elastic-net LR tuning grid (interpolation).
C_GRID = [0.0075]
L1_RATIO_GRID = [0.3]

# RF classifier / RF-selection tuning grids.
RF_MAX_DEPTH_GRID = [3, 5]
RF_SELECT_TOP_K = 35
RF_SELECT_TOP_K_GRID = [35, 60]

# Hub interaction expansion (number of selected sensors expanded into pairs).
N_HUBS_DEFAULT = 5
N_HUBS_GRID = [0, 5, 10]

# Legacy KNN / XGB grids (kept for frozen-config continuity).
KNN_NEIGHBORS_GRID = [30]
XGB_MAX_DEPTH_GRID = [3, 5]
XGB_LEARNING_RATE_GRID = [0.1]

# PLS reduction front-end (interpolation yield line).
PLS_N_COMPONENTS_DEFAULT = 10
PLS_N_COMPONENTS_GRID = [15, 25, 35, 50]

# Interpolation process gate (post-cluster Regularized EFA -> Hotelling T2 + Q/SPE).
INTERP_EFA_N_FACTORS = 10
INTERP_T2_GATE_ALPHA = 0.05
INTERP_Q_GATE_ALPHA = 0.05
INTERP_GATE_LOGIC = "or"
from secom.core import CORRELATED_SELECTION_THRESHOLD  # noqa: E402

INTERP_GATE_CORR_THRESHOLD = CORRELATED_SELECTION_THRESHOLD


def linear_preprocess(
    top_k: int = RF_SELECT_TOP_K,
    n_hubs: int = N_HUBS_DEFAULT,
) -> ColumnTransformer:
    """Impute -> cluster -> T2 + hub pairs; passthrough aux."""
    from secom.core import _sensor_preprocess_column

    sensor_steps: list[tuple[str, object]] = [
        ("impute", median_imputer()),
        ("cluster", _cluster_step()),
        ("select_t2_hubs", LinearSelectT2HubBlock(top_k=top_k, n_hubs=n_hubs)),
    ]
    return _sensor_preprocess_column(sensor_steps)


def _efa_gate_branch() -> Pipeline:
    """Gate branch: impute -> cluster -> Regularized-EFA T2/Q monitor features.

    Imported lazily because ``secom.gates.efa`` imports constants from this
    module (avoids a circular import at module load).
    """
    from secom.gates.efa import EFAMonitorFeatures

    return Pipeline(
        steps=[
            ("impute", median_imputer()),
            ("cluster", _cluster_step()),
            (
                "efa_monitor",
                EFAMonitorFeatures(
                    n_factors=INTERP_EFA_N_FACTORS,
                    t2_alpha=INTERP_T2_GATE_ALPHA,
                    q_alpha=INTERP_Q_GATE_ALPHA,
                ),
            ),
        ]
    ).set_output(transform="pandas")


def interp_preprocess(
    front_end: str = "rf",
    *,
    top_k: int = RF_SELECT_TOP_K,
    n_hubs: int = N_HUBS_DEFAULT,
    pls_n_components: int = PLS_N_COMPONENTS_DEFAULT,
    with_gate: bool = True,
) -> ColumnTransformer:
    """Interpolation preprocess: a yield front-end + the EFA T2/Q gate branch.

    ``front_end="rf"`` runs the RF top-k + T2 + hub block; ``"pls"`` runs PLS
    reduction. When ``with_gate`` is true a parallel branch appends the
    Regularized-EFA ``gate_t2``/``gate_q`` statistics as classifier features.
    """
    if front_end == "rf":
        front_step: tuple[str, object] = (
            "select_t2_hubs",
            LinearSelectT2HubBlock(top_k=top_k, n_hubs=n_hubs),
        )
    elif front_end == "pls":
        front_step = ("pls", PLSFeatures(n_components=pls_n_components))
    else:
        raise ValueError(f"front_end must be 'rf' or 'pls', got {front_end!r}")

    sensor_steps: list[tuple[str, object]] = [
        ("impute", median_imputer()),
        ("cluster", _cluster_step()),
        front_step,
    ]
    transformers: list[tuple[str, object, object]] = [
        (
            "sensor_branch",
            Pipeline(steps=sensor_steps).set_output(transform="pandas"),
            make_column_selector(pattern=_SENSOR_VALUE_PATTERN),
        ),
    ]
    if with_gate:
        transformers.append(
            (
                "gate_branch",
                _efa_gate_branch(),
                make_column_selector(pattern=_SENSOR_VALUE_PATTERN),
            )
        )
    transformers.extend(_auxiliary_transformers())
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def intrap_frozen_config_fragment() -> dict:
    """Interpolation-specific entries for the frozen-config snapshot."""
    return {
        "interpolation_model_ids": list(INTERP_MODEL_IDS),
        "c_grid": [float(c) for c in C_GRID],
        "l1_ratio_grid": [float(r) for r in L1_RATIO_GRID],
        "rf_max_depth_grid": [int(d) for d in RF_MAX_DEPTH_GRID],
        "rf_select_top_k": int(RF_SELECT_TOP_K),
        "rf_select_top_k_grid": [int(k) for k in RF_SELECT_TOP_K_GRID],
        "n_hubs_default": int(N_HUBS_DEFAULT),
        "n_hubs_grid": [int(k) for k in N_HUBS_GRID],
        "knn_neighbors_grid": [int(k) for k in KNN_NEIGHBORS_GRID],
        "xgb_max_depth_grid": [int(d) for d in XGB_MAX_DEPTH_GRID],
        "xgb_learning_rate_grid": [float(x) for x in XGB_LEARNING_RATE_GRID],
        "pls_n_components_default": int(PLS_N_COMPONENTS_DEFAULT),
        "pls_n_components_grid": [int(k) for k in PLS_N_COMPONENTS_GRID],
        "interp_efa_n_factors": int(INTERP_EFA_N_FACTORS),
        "interp_t2_gate_alpha": float(INTERP_T2_GATE_ALPHA),
        "interp_q_gate_alpha": float(INTERP_Q_GATE_ALPHA),
        "interp_gate_logic": str(INTERP_GATE_LOGIC),
        "interp_gate_corr_threshold": float(INTERP_GATE_CORR_THRESHOLD),
    }
