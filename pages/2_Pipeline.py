"""Preprocessing and feature engineering pipeline page."""
from __future__ import annotations

import streamlit as st

from scripts.dashboard import ensure_repo_on_path, render_blue_note
from scripts.dashboard.data import (
    artifacts_available,
    build_reduction_profile,
    get_reference_artifacts,
    load_pipeline_artifacts,
)
from scripts.dashboard.charts import (
    fig_hotelling_t2_intuition,
    fig_reduction_impact_from_stages,
    fig_rf_topk_selection,
    fig_rf_topk_selection_example,
    fig_spearman_cluster,
    fig_spearman_cluster_example,
)
from scripts.dashboard.pipeline import render_preprocessing_flowchart
from scripts.secom_pipelines import (
    N_REPEATS,
    N_SPLITS,
    N_HUBS_GRID,
    PRIMARY_TUNING_METRIC,
    RF_SELECT_TOP_K,
    RF_SELECT_TOP_K_GRID,
)

ensure_repo_on_path()

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


def main() -> None:
    st.title("Pipeline")
    st.caption(
        "How SECOM features are prepared before model training. "
        "See the Models page for benchmark results and the **Threshold profiles (F1/F2/F3)** tab "
        "(F1 conservative / F2 neutral / F3 aggressive threshold trade-offs)."
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

    render_blue_note(
        "All feature engineering is fit on training folds only (no leakage), "
        "then compared with PR AUC across repeated CV."
    )

    st.subheader("Preprocessing flow")
    render_preprocessing_flowchart()
    st.caption(
        "dbt profiles sensors before sklearn; shared steps run in-fold during CV and benchmark. "
        "All four benchmark models use the same preprocess ending in scale → classifier."
    )

    st.divider()
    st.subheader("Shared preprocessing foundation")

    with st.expander("Step 1: dbt preprocessing context", expanded=True):
        render_blue_note(
            "`stg_secom` → `int_secom_features` → `int_secom_column_metadata` → "
            "`mart_secom_features`\n\n"
            "Training and benchmarking read **`public.mart_secom_features`**."
        )
        st.markdown(
            """
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
            st.plotly_chart(
                fig_spearman_cluster(cluster_example),
                width="stretch",
                theme="streamlit",
                key="p2_spearman_cluster",
            )
            members = cluster_example.get("members", [])
            st.caption(
                f"Correlated cluster from holdout training fit: {', '.join(members)}."
            )
        else:
            st.plotly_chart(
                fig_spearman_cluster_example(),
                width="stretch",
                theme="streamlit",
                key="p2_spearman_cluster",
            )
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
            st.plotly_chart(
                fig_rf_topk_selection(ref_for_rf),
                width="stretch",
                theme="streamlit",
                key="p2_rf_topk",
            )
        else:
            st.plotly_chart(
                fig_rf_topk_selection_example(top_k=RF_SELECT_TOP_K),
                width="stretch",
                theme="streamlit",
                key="p2_rf_topk",
            )

    with st.expander("Step 5: Hotelling T² and hub pairs"):
        st.markdown(
            """
- **Action:** Hotelling T² on RF-selected `c_*` sensors; hub×hub product features from the top `n_hubs` ranked sensors.
- **Impact:** joint drift score plus nonlinear interactions among the strongest sensors.
            """
        )
        st.latex(r"T^2 = (\mathbf{x}-\boldsymbol{\mu})^\top \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})")
        st.plotly_chart(
            fig_hotelling_t2_intuition(),
            width="stretch",
            theme="streamlit",
            key="p2_hotelling_intuition",
        )

    st.divider()
    st.subheader("Feature reduction profile")
    profile = (
        build_reduction_profile(artifacts) if artifacts else illustrative_reduction_profile()
    )

    sensors_cluster = profile["after_cluster"]
    aux = profile["auxiliary_features"]
    st.caption(
        f"After clustering: **{sensors_cluster:,} sensors + {aux:,} auxiliary features** "
        f"(calendar, missing indicators, `n_missing_sensors`)."
    )

    st.plotly_chart(
        fig_reduction_impact_from_stages(profile),
        width="stretch",
        theme="streamlit",
        key="p2_reduction_impact",
    )

    render_blue_note(HYPERPARAM_NOTE)

    if artifact_caption:
        st.caption(artifact_caption)
    else:
        st.caption(
            "Illustrative stage counts (communication only), not live transformer execution."
        )


main()
