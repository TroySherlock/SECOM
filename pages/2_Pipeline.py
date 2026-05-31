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
    fig_reduction_impact_from_stages,
    fig_rf_topk_selection,
    fig_rf_topk_selection_example,
    fig_spearman_cluster,
    fig_spearman_cluster_example,
)
from scripts.dashboard_pipeline import render_preprocessing_flowchart
from scripts.dashboard_theme import plotly_chart
from scripts.secom_pipelines import (
    ENABLE_ISOLATION_FOREST,
    ENABLE_NEIGHBOR_FAIL_RATE,
    N_REPEATS,
    N_SPLITS,
    N_HUBS_GRID,
    PRIMARY_TUNING_METRIC,
    RF_SELECT_TOP_K,
    RF_SELECT_TOP_K_GRID,
)

ensure_repo_on_path()

METHOD_LINEAR = "Linear: elastic-net LR"
METHOD_TOPK = "Nonlinear: RF / k-NN / XGBoost"

HYPERPARAM_NOTE = (
    "`top_k` and `n_hubs` are **hyperparameters** tuned "
    f"with {N_SPLITS}×{N_REPEATS} repeated stratified CV grid search "
    f"(top-k grid: {', '.join(str(k) for k in RF_SELECT_TOP_K_GRID)}; "
    f"n_hubs grid: {', '.join(str(k) for k in N_HUBS_GRID)}), "
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
    options = [METHOD_LINEAR, METHOD_TOPK]
    if hasattr(st, "segmented_control"):
        return st.segmented_control(
            "Classifier family",
            options=options,
            default=METHOD_LINEAR,
            key="p2_architecture",
        )
    return st.radio(
        "Classifier family",
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

    linear_ref = get_reference_artifacts(artifacts, "linear") if artifacts else None
    topk_ref = get_reference_artifacts(artifacts, "topk") if artifacts else None
    shared = (artifacts or {}).get("shared", {})
    cluster_example = shared.get("spearman_cluster_example")

    f1, f2, f3 = st.columns(3)
    f1.metric("Fold-safe preprocessing", "In each CV fold")
    f2.metric("CV protocol", f"{N_SPLITS}x{N_REPEATS} repeated stratified")
    f3.metric("Comparison metric", PRIMARY_TUNING_METRIC.upper())
    st.info(
        "All feature engineering is fit on training folds only (no leakage), "
        "then compared with PR AUC across repeated CV."
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

    with st.expander("Step 3: Spearman correlated selection"):
        st.markdown(
            """
- **Action:** `DropConstantFeatures` → `DropDuplicateFeatures` → `VarianceThreshold` → `SmartCorrelatedSelection` (Spearman, threshold from `CORRELATED_SELECTION_THRESHOLD`; keep the feature with highest |ρ| to target per correlated group).
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

    with st.expander("Step 4: RF top-k selection"):
        st.markdown(
            """
- **Action:** `SelectFromModel` with a random forest ranks clustered sensors; keep top-k (CV-tuned).
- **Impact:** focuses downstream steps on the most target-aligned sensors.
            """
        )
        ref_for_rf = topk_ref or linear_ref
        if ref_for_rf and ref_for_rf.get("rf_selection"):
            plotly_chart(fig_rf_topk_selection(ref_for_rf), key="p2_rf_topk")
        else:
            plotly_chart(
                fig_rf_topk_selection_example(top_k=RF_SELECT_TOP_K),
                key="p2_rf_topk",
            )

    with st.expander("Step 5: Hotelling T² (Mahalanobis)"):
        st.markdown(
            """
- **Action:** multivariate distance from normal operating structure on selected `c_*` sensors.
- **Impact:** detects joint drifts invisible to single sensors.
            """
        )
        st.latex(r"T^2 = (\mathbf{x}-\boldsymbol{\mu})^\top \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})")
        plotly_chart(fig_hotelling_t2_intuition(), key="p2_hotelling_intuition")

    st.divider()
    st.subheader("Classifier family")
    architecture = _architecture_selector()
    st.caption(HYPERPARAM_NOTE)

    chart_col, note_col = st.columns([2, 1], gap="large")
    with chart_col:
        active_ref = linear_ref if METHOD_LINEAR in architecture else topk_ref
        hub = (active_ref or linear_ref or topk_ref or {}).get("hub_interactions")
        if hub:
            n_pairs = hub.get("n_interaction_features", 0)
            n_hubs = hub.get("n_hubs_selected", 0)
            st.info(
                f"All models share hub×hub products from the top {n_hubs} RF-importance "
                f"sensors ({n_pairs} interaction columns), on top of all top-k `c_*` and T²."
            )
        else:
            meta_parts = []
            if ENABLE_NEIGHBOR_FAIL_RATE:
                meta_parts.append("KNN neighbor fail-rate")
            if ENABLE_ISOLATION_FOREST:
                meta_parts.append("isolation forest score")
            meta_tail = (
                " → ".join(meta_parts) + " → " if meta_parts else ""
            )
            st.info(
                f"All models share RF top-k → T² → hub×hub products → {meta_tail}"
                "scale → classifier. Optional meta steps are toggled via "
                "`ENABLE_NEIGHBOR_FAIL_RATE` and `ENABLE_ISOLATION_FOREST` in "
                "`scripts/secom_pipelines.py`."
            )

    with note_col:
        active_ref = linear_ref if METHOD_LINEAR in architecture else topk_ref
        if active_ref:
            clf_in = active_ref.get("stages", {}).get("classifier_input", "—")
            st.metric("Classifier input features", f"{clf_in:,}")
        if METHOD_LINEAR in architecture:
            st.info(
                "**Linear:** shared hub preprocess → scale → "
                "elastic-net logistic regression (saga)."
            )
        else:
            st.info(
                "**Nonlinear:** same hub preprocess → scale → "
                "random forest, k-NN, or XGBoost (per model)."
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
        help="Total columns after preprocess + scale (linear reference model, holdout fit).",
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
