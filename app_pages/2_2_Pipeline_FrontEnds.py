"""Pipeline (2/2): the three front-ends and the two-champion feature journeys."""
from __future__ import annotations

import streamlit as st

from secom.dashboard import render_blue_note, render_verdict
from secom.dashboard.charts import (
    fig_hsic_selected_rank,
    fig_pipeline_stage_counts,
    fig_pls_score_scatter,
    fig_rf_topk_selection,
    fig_rf_topk_selection_example,
)
from secom.dashboard.data import PipelineContext, load_pipeline_context
from secom.dashboard.explainability import cached_pls_score_scatter
from secom.pipelines import (
    N_HUBS_GRID,
    N_REPEATS,
    N_SPLITS,
    TOP_K_DEFAULT,
    TOP_K_GRID,
)

ARTIFACTS_MISSING = (
    "Pipeline artifacts not found. Run `python -m secom.cli.benchmark` "
    "after tuning to populate reporting charts. Showing illustrative fallbacks."
)

HYPERPARAM_NOTE = (
    "`top_k` and `n_hubs` are **hyperparameters** tuned "
    f"with {N_SPLITS}×{N_REPEATS} repeated stratified CV grid search "
    f"(top-k grid: {', '.join(str(k) for k in TOP_K_GRID)}; "
    f"n_hubs grid: {', '.join(str(k) for k in N_HUBS_GRID)}), "
    "not fixed pipeline defaults."
)


def illustrative_reduction_profile() -> dict[str, int]:
    """Fallback stage counts when artifacts are missing (mirrors frozen rfsel_enet)."""
    return {
        "stg_sensors": 590,
        "mart_sensors": 422,
        "dbt_dropped_sensors": 168,
        "after_impute": 844,
        "after_cluster": 377,
        "drop_correlated": 455,
        "after_selection": 50,
        "after_hub_interactions": 61,
        "auxiliary_features": 35,
        "after_preprocess": 96,
        "classifier_input": 96,
    }


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
    st.subheader("🔱 Step 3 — the three front-ends")
    render_blue_note(
        "The front-end is the **only** place the nine pipelines diverge — everything else (the "
        "shared spine, the calibrated head) is shared. Two front-ends *select* a small sensor "
        "subset; the third *projects* all sensors into a few supervised directions."
    )
    a, b, c = st.columns(3, gap="medium")
    with a:
        with st.container(key="card_fe_hsic"):
            st.markdown("#### HSIC-Lasso → T² + hubs")
            st.markdown(
                "Keeps the **top-k** sensors by *kernel statistical dependence* with the fail "
                "label, then appends a Hotelling T² score and hub×hub interactions.\n\n"
                "Catches **nonlinear** links a correlation filter misses.\n\n"
                "_Used by_ `hsic_enet`, `hsic_rf`, `hsic_bayes`."
            )
    with b:
        with st.container(key="card_fe_rf"):
            st.markdown("#### RF-selection → T² + hubs")
            st.markdown(
                "Keeps the **top-k** sensors by random-forest impurity importance, then appends "
                "the same T² + hub interactions.\n\n"
                "**Multivariate** — values sensors for their joint, interacting signal.\n\n"
                "_Used by_ `rfsel_enet`, `rfsel_rf`, `rfsel_bayes`."
            )
    with c:
        with st.container(key="card_fe_pls"):
            st.markdown("#### PLS components")
            st.markdown(
                "**No selection.** Projects *all* clustered sensors onto a few supervised latent "
                "components; no T², no hubs.\n\n"
                "Aggregating many weak sensors is **more drift-robust** → stronger on "
                "extrapolation.\n\n"
                "_Used by_ `pls_enet`, `pls_rf`, `pls_bayes`."
            )
    st.caption(
        "The T² here is an **engineered feature** inside the hsic/rfsel hub blocks — distinct from "
        "the standalone PCA → Hotelling T² monitoring **gate** on the Gates pages."
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
        selection_stages = (models.get("rfsel_enet") or {}).get("stages") or {}
        selection_cluster = selection_stages.get("after_cluster")
        comparison = (
            f" For comparison, `rfsel_enet` keeps {int(selection_cluster):,} clustered sensors "
            "under its own tuned Spearman threshold."
            if selection_cluster is not None and int(selection_cluster) != after_cluster
            else ""
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Clustered sensors fed to PLS", f"{after_cluster:,}")
        c2.metric("Compressed to components", f"{n_components:,}")
        c3.metric("Classifier input (+ aux)", f"{classifier_input:,}")
        st.caption(
            f"PLS takes the **whole** clustered sensor block ({after_cluster:,} columns — no "
            f"top-k) and compresses it into just {n_components:,} supervised latent components, "
            f"then adds {aux:,} shared auxiliary features (calendar, missing flags, "
            f"`n_missing_sensors`) to reach {classifier_input:,} classifier inputs. The Spearman "
            "correlation threshold is **CV-tuned per cell**, so the clustered count can differ "
            f"between front-ends.{comparison}"
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
    render_verdict(
        f"{lead} A kernel-dependence filter and a tree-importance filter use completely different "
        f"maths, so when they converge on the same sensors that is strong, method-independent "
        f"evidence of a genuine fail signal — not an artifact of one technique. They jointly keep "
        f"**{len(shared)}** sensors: {shared_txt}."
    )


def _render_journey(ctx: PipelineContext) -> None:
    hsic_stages = (ctx.models.get("hsic_rf") or {}).get("stages")
    pls_stages = (ctx.models.get("pls_bayes") or {}).get("stages")
    fallback = illustrative_reduction_profile()
    hsic_cluster = int((hsic_stages or fallback).get("after_cluster", 0))
    pls_cluster = int((pls_stages or fallback).get("after_cluster", 0))
    if hsic_cluster and pls_cluster and hsic_cluster == pls_cluster:
        cluster_text = f"both champions keep {hsic_cluster:,} clustered sensors before diverging"
    elif hsic_cluster and pls_cluster:
        cluster_text = (
            f"`hsic_rf` keeps {hsic_cluster:,} clustered sensors and `pls_bayes` keeps "
            f"{pls_cluster:,} before diverging"
        )
    else:
        cluster_text = "each champion reaches its own CV-tuned clustered sensor block before diverging"

    with st.container(key="card_journey"):
        st.subheader("🛣️ The whole journey — two champions, two routes")
        st.markdown(
            "The two champion models take **different routes** to the classifier. `hsic_rf` "
            "(interpolation champion) **selects** a small sensor subset then appends Hotelling T² + "
            "hub interactions; `pls_bayes` (extrapolation champion) **projects** the whole clustered "
            "block into a handful of supervised latent components. The cluster step is split into "
            "two cuts — `VarianceThreshold` (near-constant columns) then Spearman "
            "`SmartCorrelatedSelection`; because the correlation threshold is **CV-tuned per cell**, "
            f"{cluster_text}."
        )

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

    if ctx.artifact_caption:
        st.caption(ctx.artifact_caption)
    else:
        st.caption(
            "Illustrative stage counts (communication only), not live transformer execution."
        )


def main() -> None:
    st.title("Pipeline — front-ends & journeys")
    st.caption(
        "Where the nine pipelines diverge: the three feature front-ends (HSIC-Lasso, RF-selection, "
        "PLS) and the end-to-end feature count for the two champions. See the Interpolation / "
        "Extrapolation pages' architecture subpage for PR curves and operating points."
    )

    ctx = load_pipeline_context()
    if not ctx.available:
        st.warning(ARTIFACTS_MISSING)

    _render_front_end_overview()
    fe_hsic, fe_rf, fe_pls = st.tabs(
        ["HSIC-Lasso → T² + hubs", "RF-selection → T² + hubs", "PLS components"]
    )
    with fe_hsic:
        _render_hsic_tab(ctx.models)
    with fe_rf:
        _render_rf_tab(ctx.models, ctx.topk_ref, ctx.linear_ref)
    with fe_pls:
        _render_pls_tab(ctx.models)

    _render_agreement_callout(ctx.models)

    st.divider()
    _render_journey(ctx)


main()
