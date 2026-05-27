"""Preprocessing and feature engineering page."""
from __future__ import annotations

import streamlit as st

from scripts.dashboard_app import ensure_repo_on_path
from scripts.dashboard_charts import (
    fig_hotelling_t2_intuition,
    fig_pls_compression_example,
    fig_reduction_impact,
    fig_rf_topk_selection_example,
    fig_spearman_cluster_example,
)
from scripts.dashboard_theme import plotly_chart, render_callout
from scripts.secom_pipelines import (
    N_REPEATS,
    N_SPLITS,
    PRIMARY_TUNING_METRIC,
    RF_SELECT_TOP_K,
)

ensure_repo_on_path()

METHOD_PLS = "Method A: PLS + Q"
METHOD_RF = "Method B: RF top-k"


def illustrative_reduction_profile() -> dict[str, int]:
    return {
        "start": 591,
        "drop_constant": 18,
        "drop_duplicate": 39,
        "drop_corr": 284,
        "retained_for_selection": 250,
        "pls_q_final": 16,
        "rf_top_k_final": RF_SELECT_TOP_K,
    }


def _architecture_selector() -> str:
    options = [METHOD_PLS, METHOD_RF]
    if hasattr(st, "segmented_control"):
        return st.segmented_control(
            "Choose feature extraction method",
            options=options,
            default=METHOD_PLS,
            key="p2_architecture",
        )
    return st.radio(
        "Choose feature extraction method",
        options=options,
        horizontal=True,
        key="p2_architecture_radio",
    )


def main() -> None:
    st.title("Preprocessing and feature engineering")
    st.caption(
        "How SECOM features are prepared before model training. "
        "No model leaderboard on this page."
    )

    f1, f2, f3 = st.columns(3)
    f1.metric("Fold-safe preprocessing", "In each CV fold")
    f2.metric("CV protocol", f"{N_SPLITS}x{N_REPEATS} repeated stratified")
    f3.metric("Comparison metric", PRIMARY_TUNING_METRIC.upper())
    st.info(
        "All feature engineering is fit on training folds only (no leakage), "
        "then compared with ROC AUC across repeated CV."
    )

    st.subheader("Shared preprocessing foundation")

    with st.expander("Step 1: dbt preprocessing context", expanded=True):
        st.markdown(
            """
`stg_secom` → `int_secom_features` → `int_secom_column_metadata` → `mart_secom_features`

- Cyclical time features (`month/dow/hour` sin/cos + `is_weekend`)
- Missing indicators (`c_*__missing`) and `n_missing_sensors`
- Profile sensors; drop **44** with missing rate >10%
            """
        )

    with st.expander("Step 2: Robust median imputation"):
        st.markdown(
            """
- **Action:** fill missing `c_*` with per-sensor median (training-fold baseline).
- **Impact:** stabilizes sparse sensors without outlier skew.
            """
        )

    with st.expander("Step 3: Hierarchical Spearman clustering"):
        st.markdown(
            """
- **Action:** `DropConstantFeatures` → `DropDuplicateFeatures` → `SmartCorrelatedSelection` (Spearman, threshold 0.80).
- **Impact:** collapses redundant sensors before matrix-heavy steps.
            """
        )
        plotly_chart(fig_spearman_cluster_example(), key="p2_spearman_cluster")
        st.caption(
            "Example cluster: `c_340`, `c_204`, `c_67` grouped by high Spearman ρ (≈0.89–0.93)."
        )

    with st.expander("Step 4: Hotelling T² (Mahalanobis)"):
        st.markdown(
            """
- **Action:** multivariate distance from normal operating structure.
- **Impact:** detects joint drifts invisible to single sensors.
            """
        )
        st.latex(r"T^2 = (\mathbf{x}-\boldsymbol{\mu})^\top \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})")
        plotly_chart(fig_hotelling_t2_intuition(), key="p2_hotelling_intuition")

    st.divider()
    st.subheader("Model architecture selector")
    architecture = _architecture_selector()

    chart_col, note_col = st.columns([2, 1], gap="large")
    with chart_col:
        if METHOD_PLS in architecture:
            plotly_chart(fig_pls_compression_example(), key="p2_pls_compression")
        else:
            plotly_chart(
                fig_rf_topk_selection_example(top_k=RF_SELECT_TOP_K),
                key="p2_rf_topk",
            )
    with note_col:
        if METHOD_PLS in architecture:
            st.info(
                "**Method A:** Projects sensors into target-aligned latent components (`pls_*`). "
                "**Q statistic** captures residual anomaly signal omitted by the latent subspace."
            )
        else:
            st.info(
                f"**Method B:** Random forest ranks features; top **{RF_SELECT_TOP_K}** retained. "
                "Emphasizes high-importance predictors and nonlinear interactions."
            )

    st.divider()
    st.subheader("Illustrative reduction profile")
    reduction = illustrative_reduction_profile()
    a, b, c = st.columns(3)
    a.metric("Shared preprocess output", f"{reduction['retained_for_selection']:,}")
    b.metric("PLS + Q output size", f"{reduction['pls_q_final']:,}")
    c.metric("RF top-k output size", f"{reduction['rf_top_k_final']:,}")

    d1, d2, d3 = st.columns(3)
    d1.metric("Dropped constant", f"{reduction['drop_constant']:,}")
    d2.metric("Dropped duplicate", f"{reduction['drop_duplicate']:,}")
    d3.metric("Dropped correlated", f"{reduction['drop_corr']:,}")

    stage_counts = [
        ("Start sensors", reduction["start"]),
        ("After constant drop", reduction["start"] - reduction["drop_constant"]),
        (
            "After duplicate drop",
            reduction["start"] - reduction["drop_constant"] - reduction["drop_duplicate"],
        ),
        ("After correlated drop", reduction["retained_for_selection"]),
        ("RF top-k final", reduction["rf_top_k_final"]),
    ]
    plotly_chart(fig_reduction_impact(stage_counts), key="p2_reduction_impact")
    st.caption(
        "Illustrative example feature-set sizes (communication only), not live transformer execution."
    )


main()
