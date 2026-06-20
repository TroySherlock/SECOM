"""Backward-compatible facade. Source of truth now lives in:

- :mod:`secom.core` - shared paths/IO, splits, CV, builders, frozen-config aggregator
- :mod:`secom.intrap_pipelines` - interpolation ids, preprocess, grids
- :mod:`secom.extrap_pipelines` - extrapolation (Bayesian) ids, specs, grids

Existing imports of ``secom.pipelines`` keep working via the re-exports below.
New code should import from the track-specific modules directly.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer

from secom.core import *  # noqa: F401,F403
from secom.core import (  # noqa: F401  (underscore names used by other modules)
    _CALENDAR_PATTERN,
    _MISSING_FLAG_PATTERN,
    _SENSOR_VALUE_PATTERN,
    _SENSOR_VALUE_PATTERN_EXTRAP,
    _auxiliary_transformers,
    _cluster_step,
    _sensor_preprocess_column,
    frozen_config,
    median_imputer,
)
from secom.intrap_pipelines import *  # noqa: F401,F403
from secom.intrap_pipelines import (  # noqa: F401
    INTERP_MODEL_IDS,
    _efa_gate_branch,
)
from secom.extrap_pipelines import *  # noqa: F401,F403
from secom.extrap_pipelines import EXTRAP_MODEL_IDS  # noqa: F401
from secom.hub_interactions import LinearSelectT2HubBlock  # noqa: F401

BENCHMARK_MODEL_IDS = INTERP_MODEL_IDS + EXTRAP_MODEL_IDS


def extrap_preprocess(top_k: int = 35, n_hubs: int = 5) -> ColumnTransformer:
    """Legacy sklearn extrapolation preprocess (raw + rolling-Z -> impute -> cluster -> T2/hubs).

    Retained for backward compatibility; the Bayesian extrapolation track builds
    its own representation in :mod:`secom.bayes.representation`.
    """
    sensor_steps: list[tuple[str, object]] = [
        ("impute", median_imputer()),
        ("cluster", _cluster_step()),
        ("select_t2_hubs", LinearSelectT2HubBlock(top_k=top_k, n_hubs=n_hubs)),
    ]
    return _sensor_preprocess_column(
        sensor_steps, sensor_pattern=_SENSOR_VALUE_PATTERN_EXTRAP
    )
