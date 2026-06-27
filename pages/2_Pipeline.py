"""Preprocessing and feature engineering pipeline page."""
from __future__ import annotations

import streamlit as st

from secom.dashboard import render_blue_note
from secom.dashboard.charts import (
    fig_hsic_dependence_intuition,
    fig_hsic_selected_rank,
    fig_pipeline_feature_funnel,
    fig_pls_score_scatter,
    fig_rf_topk_selection,
    fig_rf_topk_selection_example,
    fig_spearman_cluster,
    fig_spearman_cluster_example,
)
from secom.dashboard.data import (
    artifacts_available,
    get_reference_artifacts,
    load_pipeline_artifacts,
)
from secom.dashboard.explainability import cached_pls_score_scatter
from secom.pipelines import (
    N_HUBS_GRID,
    N_REPEATS,
    N_SPLITS,
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
        "stg_sensors": 590,
        "mart_sensors": 422,
        "dbt_dropped_sensors": 168,
        "after_impute": 844,
        "after_cluster": 529,
        "drop_correlated": 303,
        "after_selection": 50,
        "after_hub_interactions": 51,
        "auxiliary_features": 35,
        "after_preprocess": 86,
        "classifier_input": 86,
    }


def _render_rz_explainer(linear_ref: dict | None) -> None:
    stages = (linear_ref or {}).get("stages") or {}
    mart = int(stages.get("mart_sensors", 422))
    after_impute = int(stages.get("after_impute", mart * 2))

    st.subheader("Robust-z (rz) twin features")
    st.caption(
        "Why the feature count *doubles* before selection: every kept raw sensor gets a "
        "causal rolling robust-z partner built in the dbt mart."
    )

    what, why, how = st.columns(3, gap="medium")
    with what:
        with st.container(border=True):
            st.markdown("#### 🧬 What")
            st.markdown(
                f"Each raw sensor `c_NNN` gets a twin `c_NNN_rz`: its value re-expressed as a "
                f"**robust z-score** (median / IQR) over a strictly-past rolling window. The mart "
                f"carries **both** — raw absolutes *and* the local-deviation view — so "
                f"`{mart:,}` sensors become `{after_impute:,}` columns."
            )
    with why:
        with st.container(border=True):
            st.markdown("#### 🎯 Why")
            st.markdown(
                "Raw levels drift across the fab's lifetime, so an absolute reading means "
                "different things in different eras. The rz twin says *how unusual a reading is "
                "relative to its own recent baseline* — a **drift-robust** signal that helps most "
                "on the extrapolation track, while the raw twin keeps the absolute level."
            )
    with how:
        with st.container(border=True):
            st.markdown("#### 🛠️ How")
            st.markdown(
                "Computed in `mart_secom_features.sql` with a windowed median/IQR over "
                "`rows between 50 preceding and 1 preceding` — **strictly past** rows only, so "
                "the current wafer never sees its own or future values. **No leakage.**"
            )

    render_blue_note(
        "The rz twins are the single **+** step in the feature-count chart below (the "
        f"`+{after_impute - mart:,}` jump from {mart:,} → {after_impute:,}). Variance + correlation "
        "selection then prunes whichever twin is redundant, so a sensor can survive as its raw "
        "form, its rz form, or both."
    )


def _render_shared_spine(linear_ref: dict | None, cluster_example: dict | None) -> None:
    stages = (linear_ref or {}).get("stages") or {}
    after_impute = int(stages.get("after_impute", 844))
    after_cluster = int(stages.get("after_cluster", 529))

    st.subheader("Shared spine")
    st.caption(
        "Four steps every cell shares, regardless of front-end or classifier head — "
        "all fit on training folds only (no leakage)."
    )
    render_blue_note(
        "Upstream, dbt (`stg_secom` → `int_secom_*` → **`mart_secom_features`**) profiles sensors, "
        "drops >10% missing / zero-variance columns, and adds the rz twins, cyclical calendar "
        "features and missing-flags. Everything below runs in sklearn on that mart."
    )

    cols = st.columns(4, gap="medium")
    cards = [
        ("1️⃣ Impute", f"{after_impute:,} features",
         "Per-sensor **median** fill (training-fold baseline).",
         "Fills missing `c_*` with the per-sensor median learned on the training fold, so "
         "sparse sensors stay usable without letting outliers skew the fill value."),
        ("2️⃣ Cluster", f"→ {after_cluster:,} kept",
         "`VarianceThreshold` → **Spearman** correlated drop.",
         "Drops near-zero-variance columns, then `SmartCorrelatedSelection` collapses each "
         "Spearman-correlated group to its single best member (highest |ρ| to the fail label). "
         "The threshold is **CV-tuned**, so the survivor count differs per cell."),
        ("3️⃣ Scale", "front-end output",
         "`RobustScaler` (median / IQR).",
         "Centres on the median and scales by the IQR so heavy-tailed sensor distributions "
         "and outliers don't dominate the downstream classifier."),
        ("4️⃣ Calibrate", "fail P(·)",
         "`CalibratedClassifierCV` (**Platt / sigmoid**).",
         "Wraps the fitted head and maps its raw scores to trustworthy fail probabilities via "
         "cross-validated sigmoid (Platt) calibration — what the operating-point thresholds on "
         "pages 3-4 rely on."),
    ]
    for col, (title, chip, action, detail) in zip(cols, cards):
        with col:
            with st.container(border=True):
                st.markdown(f"#### {title}")
                st.markdown(f"`{chip}`")
                st.markdown(action)
                with st.expander("Detail"):
                    st.markdown(detail)

    st.markdown("##### Spearman cluster (step 2 in action)")
    if cluster_example:
        st.plotly_chart(
            fig_spearman_cluster(cluster_example),
            width="stretch",
            theme="streamlit",
            key="p2_spearman_cluster",
        )
        members = cluster_example.get("members", [])
        st.caption(
            "One correlated cluster from the holdout training fit — these sensors move together "
            f"(high Spearman ρ), so only the best member survives: {', '.join(members)}."
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

    st.divider()
    _render_rz_explainer(linear_ref)

    st.divider()
    _render_shared_spine(linear_ref, cluster_example)

    st.divider()
    st.subheader("Three front-ends")
    render_blue_note(
        "The front-end is where the pipelines diverge. **HSIC-Lasso** and **RF-selection** both "
        "pick a top-k sensor subset, then append a **Hotelling T²** score and hub-pair "
        "interactions; **PLS** instead projects all clustered sensors onto a few supervised "
        "latent components (no T², no hubs).\n\n"
        "Note: this T² is an **engineered feature** inside the hsic/rfsel hub blocks — distinct "
        "from the standalone EFA → Hotelling T² monitoring **gate** on the Gates pages (5.2)."
    )

    fe_hsic, fe_rf, fe_pls = st.tabs(
        ["HSIC-Lasso → T² + hubs", "RF-selection → T² + hubs", "PLS components"]
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
            selected = hub.get("selected_features") or []
            if selected:
                st.plotly_chart(
                    fig_hsic_selected_rank(selected, top_k=20),
                    width="stretch",
                    theme="streamlit",
                    key="p2_hsic_rank",
                )
                st.caption(
                    "The actual sensors HSIC kept for `hsic_enet`, in the order it picked them "
                    "(earlier = stronger nonlinear dependence with the fail label). HSIC stores "
                    "only this order, not the kernel magnitudes, so bar height is selection rank. "
                    "Note how several **rz** twins are selected — the drift-robust view often "
                    "carries the signal."
                )
        else:
            st.info("HSIC front-end artifact not available; run `python -m secom.cli.benchmark`.")
        st.markdown(
            "**Why a kernel method?** HSIC measures *statistical dependence*, not just linear "
            "correlation, so it catches sensors whose relationship to fail is nonlinear (e.g. risk "
            "rising at both extremes) — exactly the links a Pearson/Spearman filter would miss."
        )
        st.plotly_chart(
            fig_hsic_dependence_intuition(),
            width="stretch",
            theme="streamlit",
            key="p2_hsic_intuition",
        )
        st.caption(
            "The appended **Hotelling T²** here is just one engineered summary feature inside the "
            "hub block — distinct from the standalone monitoring **gate** on the Gates pages (5.2)."
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
- **Mechanism:** **PLS** (`PLSRegression`, `scale=True`) projects **all** clustered sensors onto a small set of *supervised* latent components — directions chosen to maximise covariance with the fail label. No top-k selection, no Hotelling T², no hub interactions.
- **Why it matters:** aggregating many weak sensors is more drift-robust than locking onto a handful of era-specific channels — which is why PLS tends to hold up better on the extrapolation track.
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
                "PLS compresses the clustered sensor block into a handful of latent components; "
                "the classifier input adds the shared auxiliary features (calendar, missing flags, "
                "`n_missing_sensors`)."
            )
        else:
            st.info("PLS front-end artifact not available; run `python -m secom.cli.benchmark`.")

        render_blue_note(
            "**What is a latent component?** Instead of picking individual sensors, PLS builds a "
            "few new axes — each a weighted blend of *all* clustered sensors — chosen so that "
            "moving along the axis tracks the fail label as closely as possible. A 'component' is "
            "one such blended axis; a handful of them summarise hundreds of correlated sensors "
            "into the directions that matter for failure."
        )
        try:
            t1, t2, y_scatter = cached_pls_score_scatter(track="interpolation")
            st.plotly_chart(
                fig_pls_score_scatter(t1, t2, y_scatter),
                width="stretch",
                theme="streamlit",
                key="p2_pls_scatter",
            )
            st.caption(
                "Live fit of `pls_enet` on the **in-distribution train split**: each point is a "
                "wafer placed by its first two PLS components. Because the components are built to "
                "track the fail label, the fail wafers (red) separate from the pass cloud even in "
                "2-D — that supervised separation is what the classifier head then thresholds."
            )
        except Exception as exc:  # pragma: no cover - defensive UI fallback
            st.info(f"PLS latent-score scatter unavailable ({type(exc).__name__}).")

    st.divider()
    st.subheader("Feature count through the pipeline")
    stages = (linear_ref or {}).get("stages") if linear_ref else None
    stages = stages or illustrative_reduction_profile()

    st.plotly_chart(
        fig_pipeline_feature_funnel(stages),
        width="stretch",
        theme="streamlit",
        key="p2_feature_funnel",
    )
    st.caption(
        f"Reference model `rfsel_enet`, one conserved chain (every step is a signed delta, so the "
        f"flow balances end to end). Read it left to right: {stages['stg_sensors']:,} staged sensor "
        f"columns lose {stages['stg_sensors'] - stages['mart_sensors']:,} high-missing / zero-variance "
        f"columns at the dbt mart, then the count **doubles** to {stages.get('after_impute', 0):,} when "
        "every kept sensor gains a **robust-z (rz) twin** (see the rz section above — this is the one "
        "increase). Variance + Spearman selection then collapses correlated sensors to "
        f"{stages['after_cluster']:,}, the front-end keeps the top-k and appends Hotelling T² / hubs, "
        f"and the shared auxiliary features (calendar + missing flags) are added to reach "
        f"{stages.get('classifier_input', 0):,} classifier inputs. The chain is reference-model "
        "specific because the correlation threshold is CV-tuned per cell."
    )

    render_blue_note(HYPERPARAM_NOTE)

    if artifact_caption:
        st.caption(artifact_caption)
    else:
        st.caption(
            "Illustrative stage counts (communication only), not live transformer execution."
        )


main()
