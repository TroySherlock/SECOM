"""Preprocessing and feature engineering pipeline page."""
from __future__ import annotations

import streamlit as st

from secom.dashboard import render_blue_note
from secom.dashboard.data import (
    artifacts_available,
    build_reduction_profile,
    get_reference_artifacts,
    load_pipeline_artifacts,
)
from secom.dashboard.charts import (
    fig_hotelling_t2_intuition,
    fig_reduction_impact_from_stages,
    fig_reduction_sankey,
    fig_rf_topk_selection,
    fig_rf_topk_selection_example,
    fig_spearman_cluster,
    fig_spearman_cluster_example,
)
from secom.dashboard.pipeline import render_preprocessing_flowchart
from secom.pipelines import (
    N_REPEATS,
    N_SPLITS,
    N_HUBS_GRID,
    TOP_K_DEFAULT,
    TOP_K_GRID,
)


HYPERPARAM_NOTE = (
    "`top_k` and `n_hubs` are **hyperparameters** tuned "
    f"with {N_SPLITS}×{N_REPEATS} repeated stratified CV grid search "
    f"(top-k grid: {', '.join(str(k) for k in TOP_K_GRID)}; "
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
        "See the Interpolation / Extrapolation pages' **Deep-dive** and **Thresholding** tabs for "
        "PR curves and the conservative / BER-min / aggressive / economic (cost-optimal) operating points."
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
            "Pipeline artifacts not found. Run `python -m secom.cli.benchmark` "
            "after tuning to populate reporting charts. Showing illustrative fallbacks."
        )

    linear_ref = get_reference_artifacts(artifacts, "linear") if artifacts else None
    topk_ref = get_reference_artifacts(artifacts, "topk") if artifacts else None
    shared = (artifacts or {}).get("shared", {})
    cluster_example = shared.get("spearman_cluster_example")
    models = (artifacts or {}).get("models", {})

    render_blue_note(
        "Every cell of the 3×3 grid (3 front-ends × 3 classifier heads) runs the **same spine** — "
        "impute → cluster → scale → calibrate — but each **front-end** builds features differently. "
        "All steps are fit on training folds only (no leakage) and compared with PR AUC across "
        "repeated CV."
    )

    st.subheader("Preprocessing flow")
    render_preprocessing_flowchart()
    st.caption(
        "dbt profiles sensors before sklearn. The shared spine is common to all nine pipelines; "
        "the front-end (HSIC-Lasso / RF-selection / sPLS) is the only branching step before scale "
        "→ isotonic-calibrated classifier."
    )

    st.divider()
    st.subheader("Shared spine")
    st.caption(
        "The four steps every cell shares, regardless of front-end or classifier head."
    )

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

    with st.expander("Step 3: Variance + Spearman correlated selection"):
        st.markdown(
            """
- **Action:** `VarianceThreshold` → `SmartCorrelatedSelection` (Spearman, threshold from `CORRELATED_SELECTION_THRESHOLD`; keep the feature with highest |ρ| to target per correlated group).
- **Impact:** collapses redundant near-duplicate sensors before the front-end.
- **Note:** the correlation threshold is **CV-tuned**, so the number of sensors surviving this step differs per cell.
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

    with st.expander("Step 4: Scale + calibrate"):
        st.markdown(
            """
- **Action:** `RobustScaler` on the front-end output, then `CalibratedClassifierCV` (isotonic) wraps the classifier head.
- **Impact:** median/IQR scaling resists outliers; isotonic calibration makes predicted fail probabilities trustworthy for thresholding (pages 3-4).
            """
        )

    st.divider()
    st.subheader("Three front-ends")
    render_blue_note(
        "The front-end is where the pipelines diverge. **HSIC-Lasso** and **RF-selection** both "
        "pick a top-k sensor subset, then append a **Hotelling T²** score and hub-pair "
        "interactions; **sPLS** instead projects all clustered sensors onto a few supervised "
        "latent components (no T², no hubs).\n\n"
        "Note: this T² is an **engineered feature** inside the hsic/rfsel hub blocks — distinct "
        "from the standalone EFA → Hotelling T² monitoring **gate** on the Gates pages (5.2)."
    )

    fe_hsic, fe_rf, fe_pls = st.tabs(
        ["HSIC-Lasso → T² + hubs", "RF-selection → T² + hubs", "sPLS components"]
    )

    with fe_hsic:
        st.markdown(
            """
- **Mechanism:** HSIC-Lasso (kernel dependence) ranks clustered sensors by nonlinear association with the fail label and keeps the top-k; a Hotelling T² score and hub×hub product features are appended.
- **Used by:** `hsic_enet`, `hsic_rf`, `hsic_bayes`.
            """
        )
        hsic_art = models.get("hsic_enet") or {}
        hub = hsic_art.get("hub_interactions") or {}
        if hub:
            c1, c2, c3 = st.columns(3)
            c1.metric("Sensors selected", f"{hub.get('top_k_requested', 0):,}")
            c2.metric("Hub sensors", f"{hub.get('n_hubs_selected', 0):,}")
            c3.metric("Interaction features", f"{hub.get('n_interaction_features', 0):,}")
            hub_sensors = hub.get("hub_sensors") or []
            if hub_sensors:
                st.caption("Hub sensors (highest-leverage): " + ", ".join(f"`{s}`" for s in hub_sensors))
        else:
            st.info("HSIC front-end artifact not available; run `python -m secom.cli.benchmark`.")
        st.latex(r"T^2 = (\mathbf{x}-\boldsymbol{\mu})^\top \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})")
        st.plotly_chart(
            fig_hotelling_t2_intuition(),
            width="stretch",
            theme="streamlit",
            key="p2_hotelling_intuition",
        )

    with fe_rf:
        st.markdown(
            """
- **Mechanism:** `SelectFromModel` with a random forest ranks clustered sensors by impurity importance; keep top-k (CV-tuned), then append the same Hotelling T² + hub-pair interactions.
- **Used by:** `rfsel_enet`, `rfsel_rf`, `rfsel_bayes`.
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
                fig_rf_topk_selection_example(top_k=TOP_K_DEFAULT),
                width="stretch",
                theme="streamlit",
                key="p2_rf_topk",
            )

    with fe_pls:
        st.markdown(
            """
- **Mechanism:** sparse PLS (sPLS) projects **all** clustered sensors onto a small set of supervised latent components that maximise covariance with the fail label. No top-k selection, no Hotelling T², no hub interactions.
- **Why it matters:** aggregation across many weak sensors is more drift-robust than locking onto a handful of era-specific channels — which is why sPLS tends to hold up better on the extrapolation track.
- **Used by:** `pls_enet`, `pls_rf`, `pls_bayes`.
            """
        )
        pls_art = models.get("pls_enet") or {}
        pls_stages = pls_art.get("stages") or {}
        if pls_stages:
            after_cluster = int(pls_stages.get("after_cluster", 0))
            classifier_input = int(pls_stages.get("classifier_input", 0))
            aux = int(pls_stages.get("auxiliary_features", 0))
            n_components = max(0, classifier_input - aux)
            c1, c2, c3 = st.columns(3)
            c1.metric("Clustered sensors in", f"{after_cluster:,}")
            c2.metric("Latent components", f"{n_components:,}")
            c3.metric("Classifier input (+aux)", f"{classifier_input:,}")
            st.caption(
                "sPLS compresses the clustered sensor block into a handful of latent components; "
                "the classifier input adds the shared auxiliary features (calendar, missing flags, "
                "`n_missing_sensors`)."
            )
        else:
            st.info("sPLS front-end artifact not available; run `python -m secom.cli.benchmark`.")

    st.divider()
    st.subheader("Feature reduction funnel")
    profile = (
        build_reduction_profile(artifacts) if artifacts else illustrative_reduction_profile()
    )

    classifier_input = profile["classifier_input"]
    st.caption(
        f"Reference model `rfsel_enet`. The Sankey (left) is the dbt sensor drop, where counts "
        f"conserve: {profile['stg_sensors']:,} staged columns split into {profile['mart_sensors']:,} "
        f"kept and {profile['dbt_dropped_sensors']:,} dropped. The bar (right) is the downstream "
        f"feature snapshot ending in {classifier_input:,} classifier inputs; counts there change "
        "units (each sensor carries a raw + robust-z variant, and hub/T² plus auxiliary features "
        "are added), so it is a snapshot rather than a conserved flow. The funnel is reference-"
        "model specific because the correlation threshold is CV-tuned per cell. (The dbt stage "
        "counts 591 raw sensor columns; the cleaned `stg_secom` view exposes 590.)"
    )

    sankey_col, bar_col = st.columns(2, gap="large")
    with sankey_col:
        st.plotly_chart(
            fig_reduction_sankey(profile),
            width="stretch",
            theme="streamlit",
            key="p2_reduction_sankey",
        )
    with bar_col:
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
