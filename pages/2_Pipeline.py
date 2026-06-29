"""Preprocessing and feature engineering pipeline page."""
from __future__ import annotations

import streamlit as st

from secom.dashboard import render_blue_note
from secom.dashboard.charts import (
    fig_hsic_dependence_intuition,
    fig_hsic_selected_rank,
    fig_pipeline_stage_counts,
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


# --- step 1: rz twins --------------------------------------------------------
def _render_rz_explainer(linear_ref: dict | None) -> None:
    stages = (linear_ref or {}).get("stages") or {}
    mart = int(stages.get("mart_sensors", 422))
    after_impute = int(stages.get("after_impute", mart * 2))

    st.subheader("Step 1 — robust-z (rz) twin features")
    st.caption(
        "Why the feature count *doubles* before any selection: every kept raw sensor gets a "
        "causal rolling robust-z partner, built in the dbt mart."
    )

    what, why, how = st.columns(3, gap="medium")
    with what:
        with st.container(border=True, key="card_rz_what"):
            st.markdown("#### 🧬 What")
            st.markdown(
                f"Each raw sensor `c_NNN` gets a twin `c_NNN_rz`: its value re-expressed as a "
                f"**robust z-score** (median / IQR) over a strictly-past rolling window. The mart "
                f"carries **both** — raw absolutes *and* the local-deviation view — so "
                f"`{mart:,}` sensors become `{after_impute:,}` columns."
            )
    with why:
        with st.container(border=True, key="card_rz_why"):
            st.markdown("#### 🎯 Why")
            st.markdown(
                "Raw levels drift across the fab's lifetime, so an absolute reading means "
                "different things in different eras. The rz twin says *how unusual a reading is "
                "relative to its own recent baseline* — a **drift-robust** signal that helps most "
                "on the extrapolation track, while the raw twin keeps the absolute level."
            )
    with how:
        with st.container(border=True, key="card_rz_how"):
            st.markdown("#### 🛠️ How")
            st.markdown(
                "Computed in `mart_secom_features.sql` with a windowed median/IQR over "
                "`rows between 50 preceding and 1 preceding` — **strictly past** rows only, so "
                "the current wafer never sees its own or future values. **No leakage.**"
            )

    render_blue_note(
        "The rz twins are the single **increase** in the feature-count chart at the foot of this "
        f"page (the jump from {mart:,} → {after_impute:,}). The variance + correlation step then "
        "prunes whichever twin is redundant, so a sensor can survive as its raw form, its rz form, "
        "or both."
    )


# --- step 2: shared spine ----------------------------------------------------
def _render_shared_spine(linear_ref: dict | None, cluster_example: dict | None) -> None:
    stages = (linear_ref or {}).get("stages") or {}
    after_impute = int(stages.get("after_impute", 844))
    after_cluster = int(stages.get("after_cluster", 529))

    st.subheader("Step 2 — the shared spine")
    st.caption(
        "Four steps every cell of the 3×3 grid shares, regardless of front-end or classifier "
        "head. All are fit on training folds only — no leakage."
    )
    render_blue_note(
        "Upstream, dbt (`stg_secom` → `int_secom_*` → **`mart_secom_features`**) profiles sensors, "
        "drops >10% missing / zero-variance columns, and adds the rz twins, cyclical calendar "
        "features and missing-flags. Everything below runs in sklearn on that mart."
    )

    cols = st.columns(4, gap="medium")
    cards = [
        ("1 · Impute", f"{after_impute:,} cols",
         "Per-sensor **median** fill learned on the training fold, so sparse sensors stay usable "
         "without outliers skewing the fill value."),
        ("2 · Cluster", f"→ {after_cluster:,} kept",
         "`VarianceThreshold` drops near-constant columns, then **Spearman** "
         "`SmartCorrelatedSelection` collapses each correlated group to its single best member. "
         "The threshold is **CV-tuned**, so the survivor count differs per cell."),
        ("3 · Scale", "front-end output",
         "`RobustScaler` (median / IQR) so heavy-tailed sensors and outliers don't dominate the "
         "downstream classifier."),
        ("4 · Calibrate", "fail P(·)",
         "`CalibratedClassifierCV` (**Platt / sigmoid**) maps raw head scores to trustworthy fail "
         "probabilities — what the operating-point thresholds on pages 3–4 rely on."),
    ]
    for i, (col, (title, chip, detail)) in enumerate(zip(cols, cards)):
        with col:
            with st.container(border=True, key=f"card_spine_{i}"):
                st.markdown(f"#### {title}")
                st.markdown(f"`{chip}`")
                st.markdown(detail)

    with st.expander("Show the Spearman cluster (step 2 in action)"):
        if cluster_example:
            st.plotly_chart(
                fig_spearman_cluster(cluster_example),
                width="stretch",
                theme="streamlit",
                key="p2_spearman_cluster",
            )
            members = cluster_example.get("members", [])
            st.caption(
                "One correlated cluster from the holdout training fit — these sensors move "
                f"together (high Spearman ρ), so only the best member survives: "
                f"{', '.join(members)}."
            )
        else:
            st.plotly_chart(
                fig_spearman_cluster_example(),
                width="stretch",
                theme="streamlit",
                key="p2_spearman_cluster",
            )
            st.caption(
                "Example cluster: `c_340`, `c_204`, `c_67` grouped by high Spearman ρ "
                "(illustrative)."
            )


# --- step 3: three front-ends ------------------------------------------------
def _cross_method_agreement(models: dict) -> dict | None:
    """Sensors both the HSIC and RF selectors keep, and each method's #1 pick."""
    hub = (models.get("hsic_enet") or {}).get("hub_interactions") or {}
    rf = (models.get("rfsel_enet") or {}).get("rf_selection") or {}
    hsic_sel = list(hub.get("selected_features") or [])
    rf_sel = list(rf.get("selected_features") or [])
    rf_imp = list(rf.get("importances") or [])
    if not hsic_sel or not rf_sel:
        return None
    rf_set = set(rf_sel)
    shared = [s for s in hsic_sel if s in rf_set]
    hsic_top = hsic_sel[0]
    rf_top = str(rf_imp[0].get("feature")) if rf_imp else rf_sel[0]
    return {"shared": shared, "hsic_top": hsic_top, "rf_top": rf_top}


def _render_front_end_overview() -> None:
    st.subheader("Step 3 — the three front-ends")
    render_blue_note(
        "The front-end is the **only** place the nine pipelines diverge — everything else (the "
        "spine above, the calibrated head) is shared. Two front-ends *select* a small sensor "
        "subset; the third *projects* all sensors into a few supervised directions."
    )
    a, b, c = st.columns(3, gap="medium")
    with a:
        with st.container(border=True, key="card_fe_hsic"):
            st.markdown("#### HSIC-Lasso → T² + hubs")
            st.markdown(
                "Keeps the **top-k** sensors by *kernel statistical dependence* with the fail "
                "label, then appends a Hotelling T² score and hub×hub interactions.\n\n"
                "Catches **nonlinear** links a correlation filter misses.\n\n"
                "_Used by_ `hsic_enet`, `hsic_rf`, `hsic_bayes`."
            )
    with b:
        with st.container(border=True, key="card_fe_rf"):
            st.markdown("#### RF-selection → T² + hubs")
            st.markdown(
                "Keeps the **top-k** sensors by random-forest impurity importance, then appends "
                "the same T² + hub interactions.\n\n"
                "**Multivariate** — values sensors for their joint, interacting signal.\n\n"
                "_Used by_ `rfsel_enet`, `rfsel_rf`, `rfsel_bayes`."
            )
    with c:
        with st.container(border=True, key="card_fe_pls"):
            st.markdown("#### PLS components")
            st.markdown(
                "**No selection.** Projects *all* clustered sensors onto a few supervised latent "
                "components; no T², no hubs.\n\n"
                "Aggregating many weak sensors is **more drift-robust** → stronger on "
                "extrapolation.\n\n"
                "_Used by_ `pls_enet`, `pls_rf`, `pls_bayes`."
            )
    render_blue_note(
        "The T² here is an **engineered feature** inside the hsic/rfsel hub blocks — distinct from "
        "the standalone PCA → Hotelling T² monitoring **gate** on the Gates pages (5.2)."
    )


def _render_hsic_tab(models: dict) -> None:
    st.markdown(
        "**Mechanism:** HSIC-Lasso ranks the clustered sensors by nonlinear association with the "
        "fail label and keeps the top-k; a Hotelling T² score and hub×hub product features are "
        "then appended. **Used by** `hsic_enet`, `hsic_rf`, `hsic_bayes`."
    )
    render_blue_note(
        "**What is statistical dependence?** HSIC (Hilbert–Schmidt Independence Criterion) is a "
        "kernel test for whether two variables are related *in any way at all* — not just the "
        "straight-line / monotone trend a Pearson or Spearman correlation measures. It maps each "
        "variable through a kernel and asks whether their joint behaviour differs from what you'd "
        "see if they were independent. So a sensor whose fail-risk rises at **both** extremes "
        "(U-shaped) scores ~0 correlation but **high** HSIC dependence — exactly the link a "
        "correlation filter throws away. HSIC-Lasso adds a sparsity penalty so it keeps a small, "
        "non-redundant set of such sensors."
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
                fig_hsic_selected_rank(selected),
                width="stretch",
                theme="streamlit",
                key="p2_hsic_rank",
            )
            st.caption(
                "The actual sensors HSIC kept for `hsic_enet`, in the order it picked them "
                "(earlier = stronger nonlinear dependence with the fail label). HSIC stores only "
                "this order, not the kernel magnitudes, so bar length is selection rank. Several "
                "**rz** twins (pink) are selected — the drift-robust view often carries the signal."
            )
    else:
        st.info("HSIC front-end artifact not available; run `python -m secom.cli.benchmark`.")

    with st.expander("Why a kernel method beats correlation (intuition)"):
        st.plotly_chart(
            fig_hsic_dependence_intuition(),
            width="stretch",
            theme="streamlit",
            key="p2_hsic_intuition",
        )
        st.caption(
            "A U-shaped sensor→fail link: fails concentrate at both extremes, so the linear "
            "correlation is ≈ 0 and a Pearson/Spearman filter is blind to it — but kernel "
            "dependence (HSIC) detects it."
        )


def _render_rf_tab(models: dict, topk_ref: dict | None, linear_ref: dict | None) -> None:
    st.markdown(
        "**Mechanism:** `SelectFromModel` with a random forest ranks the clustered sensors, keeps "
        "the top-k (CV-tuned), then appends the same Hotelling T² + hub interactions. **Used by** "
        "`rfsel_enet`, `rfsel_rf`, `rfsel_bayes`."
    )
    render_blue_note(
        "**What is RF selection?** A random forest is fit on all clustered sensors, and each "
        "sensor is scored by its **impurity-decrease importance** — how much, on average, "
        "splitting on that sensor improves the trees' pass/fail separation. Because the forest "
        "splits on combinations of sensors, the score is **multivariate**: it credits sensors for "
        "the signal they carry *together with others*, including interactions HSIC's marginal test "
        "may underrate. The trade-off is that impurity importance can favour correlated / "
        "high-cardinality sensors, which is exactly why the Spearman clustering step runs first."
    )

    ref_for_rf = topk_ref or linear_ref
    if ref_for_rf and ref_for_rf.get("rf_selection"):
        st.plotly_chart(
            fig_rf_topk_selection(ref_for_rf),
            width="stretch",
            theme="streamlit",
            key="p2_rf_topk",
        )
        st.caption(
            "RF importance ranking from the holdout fit; the highlighted bars are the top-k kept "
            "by `SelectFromModel`. The longest bar is the forest's single most informative sensor."
        )
    else:
        st.plotly_chart(
            fig_rf_topk_selection_example(top_k=TOP_K_DEFAULT),
            width="stretch",
            theme="streamlit",
            key="p2_rf_topk",
        )


def _render_pls_tab(models: dict) -> None:
    st.markdown(
        "**Mechanism:** **PLS** (`PLSRegression`, `scale=True`) projects **all** clustered "
        "sensors onto a small set of *supervised* latent components — directions chosen to "
        "maximise covariance with the fail label. No top-k selection, no Hotelling T², no hub "
        "interactions. **Used by** `pls_enet`, `pls_rf`, `pls_bayes`."
    )
    render_blue_note(
        "**What is a latent component?** Instead of picking individual sensors, PLS builds a few "
        "new axes — each a weighted blend of *all* clustered sensors — chosen so that moving along "
        "the axis tracks the fail label as closely as possible. A 'component' is one such blended "
        "axis; a handful of them summarise hundreds of correlated sensors into the directions that "
        "matter for failure. Aggregating many weak sensors this way is more drift-robust than "
        "locking onto a handful of era-specific channels, which is why PLS tends to hold up better "
        "on the extrapolation track."
    )

    pls_art = models.get("pls_enet") or {}
    pls_stages = pls_art.get("stages") or {}
    if pls_stages:
        after_cluster = int(pls_stages.get("after_cluster", 0))
        classifier_input = int(pls_stages.get("classifier_input", 0))
        aux = int(pls_stages.get("auxiliary_features", 0))
        n_components = max(0, classifier_input - aux)
        c1, c2, c3 = st.columns(3)
        c1.metric("Clustered sensors fed to PLS", f"{after_cluster:,}")
        c2.metric("Compressed to components", f"{n_components:,}")
        c3.metric("Classifier input (+ aux)", f"{classifier_input:,}")
        st.caption(
            f"PLS takes the **whole** clustered sensor block ({after_cluster:,} columns — no "
            f"top-k) and compresses it into just {n_components:,} supervised latent components, "
            f"then adds {aux:,} shared auxiliary features (calendar, missing flags, "
            f"`n_missing_sensors`) to reach {classifier_input:,} classifier inputs. The clustered "
            "count is larger than the selection front-ends' (e.g. `rfsel_enet`'s 377) because the "
            "Spearman correlation threshold is **CV-tuned per cell** — PLS does better keeping a "
            "wider block to blend."
        )
    else:
        st.info("PLS front-end artifact not available; run `python -m secom.cli.benchmark`.")

    try:
        t1, t2, y_scatter = cached_pls_score_scatter(track="interpolation")
        st.plotly_chart(
            fig_pls_score_scatter(t1, t2, y_scatter),
            width="stretch",
            theme="streamlit",
            key="p2_pls_scatter",
        )
        st.caption(
            "Live fit of `pls_enet` on the **in-distribution train split**: each point is a wafer "
            "placed by its first two PLS components. Because the components are built to track the "
            "fail label, the fail wafers (red) separate from the pass cloud even in 2-D — that "
            "supervised separation is what the classifier head then thresholds."
        )
    except Exception as exc:  # pragma: no cover - defensive UI fallback
        st.info(f"PLS latent-score scatter unavailable ({type(exc).__name__}).")


def _render_agreement_callout(models: dict) -> None:
    agree = _cross_method_agreement(models)
    if not agree:
        return
    shared = agree["shared"]
    if agree["hsic_top"] == agree["rf_top"]:
        lead = (
            f"**Two independent selectors agree: `{agree['hsic_top']}` is the #1 sensor for "
            "*both* HSIC-Lasso and RF-selection.**"
        )
    else:
        lead = (
            f"**Top picks:** HSIC's #1 is `{agree['hsic_top']}`, RF's #1 is `{agree['rf_top']}`."
        )
    shared_txt = ", ".join(f"`{s}`" for s in shared[:10]) if shared else "—"
    st.success(
        f"{lead} A kernel-dependence filter and a tree-importance filter use completely different "
        f"maths, so when they converge on the same sensors that is strong, method-independent "
        f"evidence of a genuine fail signal — not an artifact of one technique. They jointly keep "
        f"**{len(shared)}** sensors: {shared_txt}."
    )


# --- main --------------------------------------------------------------------
def main() -> None:
    st.title("Pipeline")
    st.caption(
        "How SECOM features are prepared before model training. "
        "See the Interpolation / Extrapolation pages' **Deep-dive** and **Thresholding** tabs for "
        "PR curves and the conservative / BER-min / aggressive / economic (cost-optimal) operating "
        "points."
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
        "impute → cluster → scale → calibrate — but each **front-end** builds features "
        "differently. All steps are fit on training folds only (no leakage) and compared with "
        "PR AUC across repeated CV. The page follows the data: **rz twins → shared spine → "
        "front-ends → the end-to-end feature count.**"
    )

    st.divider()
    _render_rz_explainer(linear_ref)

    st.divider()
    _render_shared_spine(linear_ref, cluster_example)

    st.divider()
    _render_front_end_overview()
    fe_hsic, fe_rf, fe_pls = st.tabs(
        ["HSIC-Lasso → T² + hubs", "RF-selection → T² + hubs", "PLS components"]
    )
    with fe_hsic:
        _render_hsic_tab(models)
    with fe_rf:
        _render_rf_tab(models, topk_ref, linear_ref)
    with fe_pls:
        _render_pls_tab(models)

    _render_agreement_callout(models)

    st.divider()
    st.subheader("The whole journey — two champions, two routes")
    render_blue_note(
        "The two champion models take **different routes** to the classifier. `hsic_rf` "
        "(interpolation champion) **selects** a small sensor subset then appends Hotelling T² + "
        "hub interactions; `pls_bayes` (extrapolation champion) **projects** the whole clustered "
        "block into a handful of supervised latent components. Note the cluster step is now split "
        "into its two cuts — `VarianceThreshold` (near-constant columns) then Spearman "
        "`SmartCorrelatedSelection`. Because the correlation threshold is **CV-tuned per cell**, "
        "both champions keep 706 sensors; the more aggressive 844 → 377 cut you may have seen "
        "earlier was the old `rfsel_enet` reference, not these models."
    )

    hsic_stages = (models.get("hsic_rf") or {}).get("stages")
    pls_stages = (models.get("pls_bayes") or {}).get("stages")
    fallback = illustrative_reduction_profile()

    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("**`hsic_rf` — interpolation champion (select)**")
        s = hsic_stages or fallback
        st.plotly_chart(
            fig_pipeline_stage_counts(s, title="HSIC_RF feature count"),
            width="stretch",
            theme="streamlit",
            key="p2_journey_hsic",
        )
        st.caption(
            f"{s['stg_sensors']:,} staged → {s['mart_sensors']:,} mart → **rises** to "
            f"{s.get('after_impute', 0):,} with rz twins (green) → variance threshold trims "
            "near-constant columns → Spearman correlation collapses redundant sensors to "
            f"{s['after_cluster']:,} → top-k select + Hotelling T² / hubs → "
            f"{s.get('classifier_input', 0):,} classifier inputs (red)."
        )
    with right:
        st.markdown("**`pls_bayes` — extrapolation champion (project)**")
        s = pls_stages or fallback
        n_components = max(0, int(s.get("classifier_input", 0)) - int(s.get("auxiliary_features", 0)))
        st.plotly_chart(
            fig_pipeline_stage_counts(s, title="PLS_Bayes feature count"),
            width="stretch",
            theme="streamlit",
            key="p2_journey_pls",
        )
        st.caption(
            f"Same spine up to {s['after_cluster']:,} clustered sensors, then **no top-k**: PLS "
            f"compresses the whole block into ~{n_components:,} supervised latent components and "
            f"adds the shared auxiliary features to reach {s.get('classifier_input', 0):,} "
            "classifier inputs (red). Aggregating many weak sensors is what makes it more "
            "drift-robust on extrapolation."
        )

    render_blue_note(HYPERPARAM_NOTE)

    if artifact_caption:
        st.caption(artifact_caption)
    else:
        st.caption(
            "Illustrative stage counts (communication only), not live transformer execution."
        )


main()
