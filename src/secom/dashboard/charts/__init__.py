"""Plotly chart builders for the SECOM Streamlit dashboard.

Split into thematic submodules (eda / pipeline / benchmark / gates / explain)
over a shared ``_base`` palette. Public ``fig_*`` builders and palette constants
are re-exported here so callers keep importing from ``secom.dashboard.charts``.
"""
from __future__ import annotations

from secom.dashboard.charts._base import (
    C_AQUA,
    C_BLUE,
    C_GREEN,
    C_ORANGE,
    C_PURPLE,
    C_RED,
    C_YELLOW,
    CHART_BG,
    CI_YELLOW_RGBA,
    COLORSCALE_CORRELATION,
    COLORSCALE_DRIFT_DIVERGING,
    COLORSCALE_LOW_GREEN_HIGH_RED,
    CV_ERROR_PURPLE_RGBA,
    C,
)
from secom.dashboard.charts.benchmark import (
    fig_benchmark_leaderboard,
    fig_calibration,
    fig_catch_overkill_curve,
    fig_cv_vs_holdout_validation,
    fig_delta_bar,
    fig_expected_cost_curve,
    fig_holdout_confusion,
    fig_pr_curve_clean,
    fig_risk_coverage,
)
from secom.dashboard.charts.eda import (
    fig_class_donut,
    fig_fails_over_time,
    fig_missing_rate_distribution,
    fig_missingness_structure,
    fig_sensor_drift_heatmap,
    fig_sensor_histogram,
    fig_sensor_multicollinearity,
    fig_two_track_schematic,
    fig_wafer_drift_spikes,
)
from secom.dashboard.charts.explain import (
    fig_local_contributions,
    fig_posterior_forest,
    fig_spc_distribution,
    fig_top_features_bar,
)
from secom.dashboard.charts.gates import (
    fig_bgm_mode_weights,
    fig_contribution_comparison,
    fig_factor_drift,
    fig_gate_control_chart,
    fig_gate_disagreement_scatter,
    fig_gate_statistic_distributions,
    fig_pca_component_space,
    fig_sbfa_factor_space,
    fig_sbfa_loadings_heatmap,
    fig_sensor_noise_spectrum,
)
from secom.dashboard.charts.pipeline import (
    fig_hsic_dependence_intuition,
    fig_hsic_selected_rank,
    fig_pipeline_feature_funnel,
    fig_pls_score_scatter,
    fig_rf_topk_selection,
    fig_rf_topk_selection_example,
    fig_spearman_cluster,
    fig_spearman_cluster_example,
)

__all__ = [
    # palette
    "CHART_BG",
    "CI_YELLOW_RGBA",
    "COLORSCALE_CORRELATION",
    "COLORSCALE_DRIFT_DIVERGING",
    "COLORSCALE_LOW_GREEN_HIGH_RED",
    "CV_ERROR_PURPLE_RGBA",
    "C",
    "C_AQUA",
    "C_BLUE",
    "C_GREEN",
    "C_ORANGE",
    "C_PURPLE",
    "C_RED",
    "C_YELLOW",
    # eda
    "fig_class_donut",
    "fig_fails_over_time",
    "fig_missing_rate_distribution",
    "fig_missingness_structure",
    "fig_sensor_drift_heatmap",
    "fig_sensor_histogram",
    "fig_sensor_multicollinearity",
    "fig_two_track_schematic",
    "fig_wafer_drift_spikes",
    # pipeline
    "fig_hsic_dependence_intuition",
    "fig_hsic_selected_rank",
    "fig_pipeline_feature_funnel",
    "fig_pls_score_scatter",
    "fig_rf_topk_selection",
    "fig_rf_topk_selection_example",
    "fig_spearman_cluster",
    "fig_spearman_cluster_example",
    # benchmark
    "fig_benchmark_leaderboard",
    "fig_calibration",
    "fig_catch_overkill_curve",
    "fig_cv_vs_holdout_validation",
    "fig_delta_bar",
    "fig_expected_cost_curve",
    "fig_holdout_confusion",
    "fig_pr_curve_clean",
    "fig_risk_coverage",
    # gates
    "fig_bgm_mode_weights",
    "fig_contribution_comparison",
    "fig_factor_drift",
    "fig_gate_control_chart",
    "fig_gate_disagreement_scatter",
    "fig_gate_statistic_distributions",
    "fig_pca_component_space",
    "fig_sbfa_factor_space",
    "fig_sbfa_loadings_heatmap",
    "fig_sensor_noise_spectrum",
    # explain
    "fig_local_contributions",
    "fig_posterior_forest",
    "fig_spc_distribution",
    "fig_top_features_bar",
]
