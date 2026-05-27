"""Preprocessing and feature engineering pipeline page."""
from __future__ import annotations

import streamlit as st

from scripts.dashboard_app import ensure_repo_on_path
from scripts.dashboard_artifacts import (
    artifacts_available,
    build_reduction_profile,
    get_reference_artifacts,
    load_pipeline_artifacts,
)
from scripts.dashboard_charts import (
    fig_hotelling_t2_intuition,
    fig_pls_explained_variance,
    fig_reduction_impact,
    fig_reduction_impact_from_stages,
    fig_rf_topk_selection,
    fig_rf_topk_selection_example,
    fig_spearman_cluster,
    fig_spearman_cluster_example,
)
from scripts.dashboard_pipeline import render_preprocessing_flowchart
from scripts.dashboard_theme import plotly_chart
from scripts.secom_pipelines import (
    N_REPEATS,
    N_SPLITS,
    PLS_N_COMPONENTS_GRID,
    PRIMARY_TUNING_METRIC,
    RF_SELECT_TOP_K,
    RF_SELECT_TOP_K_GRID,
)

ensure_repo_on_path()

METHOD_PLS = "Method A: PLS + Q"
METHOD_RF = "Method B: RF top-k"

HYPERPARAM_NOTE = (
    "`n_components` (PLS) and `max_features` / top-k (RF) are **hyperparameters** tuned "
    f"with {N_SPLITS}×{N_REPEATS} repeated stratified CV grid search "
    f"(PLS grid: {', '.join(str(k) for k in PLS_N_COMPONENTS_GRID)}; "
    f"RF top-k grid: {', '.join(str(k) for k in RF_SELECT_TOP_K_GRID)}), "
    "not fixed pipeline defaults."
)


def illustrative_reduction_profile() -> dict[str, int]:
    return {
        "stg_sensors": 591,
        "mart_sensors": 422,
        "dbt_dropped_sensors": 169,
        "after_cluster": 219,
        "auxiliary_features": 33,
        "classifier_input": 52,
        "drop_correlated": 203,
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
    st.title("Pipeline")
    st.caption(
        "How SECOM features are prepared before model training. "
        "See the Models page for benchmark results."
    )

    artifacts: dict | None = None
    artifact_caption = ""
    if artifacts_available():
        artifacts = load_pipeline_artifacts()
        generated = artifacts.get("generated_at", "")[:19].replace("T", " ")
        artifact_caption = (
            f"From holdout training fit (`secom_pipeline_artifacts.json`, {generated} UTC)."
        )
    else:
        st.warning(
            "Pipeline artifacts not found. Run `python -m scripts.benchmark_models` "
            "after tuning to populate reporting charts. Showing illustrative fallbacks."
        )

    mspc_ref = get_reference_artifacts(artifacts, "mspc") if artifacts else None
    rf_ref = get_reference_artifacts(artifacts, "rf_k") if artifacts else None
    shared = (artifacts or {}).get("shared", {})
    cluster_example = shared.get("spearman_cluster_example")

    f1, f2, f3 = st.columns(3)
    f1.metric("Fold-safe preprocessing", "In each CV fold")
    f2.metric("CV protocol", f"{N_SPLITS}x{N_REPEATS} repeated stratified")
    f3.metric("Comparison metric", PRIMARY_TUNING_METRIC.upper())
    st.info(
        "All feature engineering is fit on training folds only (no leakage), "
        "then compared with ROC AUC across repeated CV."
    )

    st.subheader("Preprocessing flow")
    render_preprocessing_flowchart()
    st.caption(
        "dbt profiles sensors before sklearn; shared steps run in-fold during CV and benchmark."
    )

    st.divider()
    st.subheader("Shared preprocessing foundation")

    with st.expander("Step 1: dbt preprocessing context", expanded=True):
        st.markdown(
            """
`stg_secom` → `int_secom_features` → `int_secom_column_metadata` → `mart_secom_features`

- Cyclical time features (`month/dow/hour` sin/cos + `is_weekend`)
- Missing indicators (`c_*__missing`) and `n_missing_sensors`
- Profile sensors; drop high-missing (>10%) and zero-variance columns before the mart
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
- **Note:** constant/duplicate drops are kept for fold-safe parity with tuning; after dbt profiling they usually remove **no additional** sensors.
            """
        )
        if cluster_example:
            plotly_chart(fig_spearman_cluster(cluster_example), key="p2_spearman_cluster")
            members = cluster_example.get("members", [])
            st.caption(
                f"Correlated cluster from holdout training fit: {', '.join(members)}."
            )
        else:
            plotly_chart(fig_spearman_cluster_example(), key="p2_spearman_cluster")
            st.caption(
                "Example cluster: `c_340`, `c_204`, `c_67` grouped by high Spearman ρ (illustrative)."
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
    st.caption(HYPERPARAM_NOTE)

    chart_col, note_col = st.columns([2, 1], gap="large")
    with chart_col:
        if METHOD_PLS in architecture:
            pls = (mspc_ref or {}).get("pls")
            if pls:
                plotly_chart(
                    fig_pls_explained_variance(
                        pls,
                        n_components_grid=list(PLS_N_COMPONENTS_GRID),
                    ),
                    key="p2_pls_variance",
                )
            else:
                st.info("PLS variance chart requires pipeline artifacts from benchmark run.")
        else:
            if rf_ref:
                plotly_chart(fig_rf_topk_selection(rf_ref), key="p2_rf_topk")
            else:
                plotly_chart(
                    fig_rf_topk_selection_example(top_k=RF_SELECT_TOP_K),
                    key="p2_rf_topk",
                )
    with note_col:
        active_ref = mspc_ref if METHOD_PLS in architecture else rf_ref
        if active_ref:
            clf_in = active_ref.get("stages", {}).get("classifier_input", "—")
            st.metric("Classifier input features", f"{clf_in:,}")
        if METHOD_PLS in architecture:
            st.info(
                "**Method A:** Target-aligned latent scores (`pls_*`) plus **Q statistic** "
                "for residual structure. Components are not per-sensor groupings."
            )
        else:
            top_k = (rf_ref or {}).get("rf_selection", {}).get("top_k", RF_SELECT_TOP_K)
            st.info(
                f"**Method B:** Random forest ranks clustered sensors; top **{top_k}** retained "
                f"(CV grid: {', '.join(str(k) for k in RF_SELECT_TOP_K_GRID)})."
            )

    st.divider()
    st.subheader("Feature reduction profile")
    profile = (
        build_reduction_profile(artifacts) if artifacts else illustrative_reduction_profile()
    )

    sensors_cluster = profile["after_cluster"]
    aux = profile["auxiliary_features"]
    a, b, c = st.columns(3)
    a.metric(
        "Sensors after clustering",
        f"{sensors_cluster:,}",
        help="Sensors retained after Spearman correlated selection (sklearn, in-fold).",
    )
    b.metric(
        "Auxiliary features",
        f"{aux:,}",
        help="Calendar, missing flags, and n_missing_sensors passed through preprocess.",
    )
    c.metric(
        "Classifier input features",
        f"{profile['classifier_input']:,}",
        help="Total columns after preprocess + scale (MSPC reference model, holdout fit).",
    )
    st.caption(
        f"After clustering: **{sensors_cluster:,} sensors + {aux:,} auxiliary features** "
        f"(calendar, missing indicators, `n_missing_sensors`)."
    )

    plotly_chart(fig_reduction_impact_from_stages(profile), key="p2_reduction_impact")

    st.info(HYPERPARAM_NOTE)

    if artifact_caption:
        st.caption(artifact_caption)
    else:
        st.caption(
            "Illustrative stage counts (communication only), not live transformer execution."
        )


main()
