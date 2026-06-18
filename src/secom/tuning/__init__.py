"""Tuning registry and GridSearchCV helpers (used by benchmark)."""

from secom.tuning.registry import (
    MODEL_SPECS,
    build_tuned_pipeline,
    fit_with_progress,
    model_ids_for_track,
    run_grid_search,
    save_tuned_params,
    summarize_cv_search,
    tune_classifier_threshold_profiles,
)
from secom.utils import tuned_params_path

__all__ = [
    "MODEL_SPECS",
    "build_tuned_pipeline",
    "fit_with_progress",
    "model_ids_for_track",
    "run_grid_search",
    "save_tuned_params",
    "summarize_cv_search",
    "tune_classifier_threshold_profiles",
    "tuned_params_path",
]
