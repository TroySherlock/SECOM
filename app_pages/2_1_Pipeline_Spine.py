"""Pipeline (1/2): the shared feature spine — rz twins and shared preprocessing."""
from __future__ import annotations

import streamlit as st

from secom.dashboard import render_blue_note
from secom.dashboard.charts import (
    fig_calibrate_example,
    fig_impute_example,
    fig_scale_example,
    fig_spearman_cluster,
    fig_spearman_cluster_example,
)
from secom.dashboard.data import load_pipeline_context

ARTIFACTS_MISSING = (
    "Pipeline artifacts not found. Run `python -m secom.cli.benchmark` "
    "after tuning to populate reporting charts. Showing illustrative fallbacks."
)


# --- step 1: rz twins --------------------------------------------------------
def _render_rz_explainer(linear_ref: dict | None) -> None:
    stages = (linear_ref or {}).get("stages") or {}
    mart = int(stages.get("mart_sensors", 422))
    after_impute = int(stages.get("after_impute", mart * 2))

    st.subheader("🧬 Step 1 — rolling-z (rz) twin features")
    st.caption(
        "Why the feature count *doubles* before any selection: every kept raw sensor gets a "
        "causal rolling-z partner, built in the dbt mart."
    )

    what, why, how = st.columns(3, gap="medium")
    with what:
        with st.container(key="card_rz_what"):
            st.markdown("#### 🧬 What")
            st.markdown(
                f"Each raw sensor `c_NNN` gets a twin `c_NNN_rz`: its value re-expressed as a "
                f"**causal rolling z-score** (trailing mean / sample SD over the previous "
                f"50 wafers). The mart carries **both** — raw absolutes *and* the "
                f"local-deviation view — so "
                f"`{mart:,}` sensors become `{after_impute:,}` columns."
            )
    with why:
        with st.container(key="card_rz_why"):
            st.markdown("#### 🎯 Why")
            st.markdown(
                "Raw levels drift across the fab's lifetime, so an absolute reading means "
                "different things in different eras. The rz twin says *how unusual a reading is "
                "relative to its own recent baseline* — a **drift-robust** signal that helps most "
                "on the extrapolation track, while the raw twin keeps the absolute level."
            )
    with how:
        with st.container(key="card_rz_how"):
            st.markdown("#### 🛠️ How")
            st.markdown(
                "Computed in `mart_secom_features.sql` with a windowed mean / sample SD "
                "(`avg` / `stddev_samp`) over "
                "`rows between 50 preceding and 1 preceding` — **strictly past** rows only, so "
                "the current wafer never sees its own or future values. **No leakage.** Empty "
                "early windows are filled with rz = 0; the first few dozen rows therefore have "
                "short, noisier baselines."
            )

    render_blue_note(
        "The rz twins are the single **increase** in the feature-count chart on the Front-ends & "
        f"journeys page (the jump from {mart:,} → {after_impute:,}). The variance + correlation "
        "step then prunes whichever twin is redundant, so a sensor can survive as its raw form, "
        "its rz form, or both."
    )


# --- step 2: shared spine ----------------------------------------------------
def _render_shared_spine(linear_ref: dict | None, cluster_example: dict | None) -> None:
    stages = (linear_ref or {}).get("stages") or {}
    after_impute = int(stages.get("after_impute", 844))
    after_cluster = int(stages.get("after_cluster", 529))

    st.subheader("🧱 Step 2 — the shared spine")
    st.caption(
        "Four steps every cell of the 3×3 grid shares, regardless of front-end or classifier "
        "head. All are fit on training folds only — read top to bottom, each with a worked example."
    )
    st.markdown(
        f":gray-background[{after_impute:,} cols] → :gray-background[{after_cluster:,} kept] → "
        ":gray-background[scaled] → :gray-background[calibrated P(fail)]"
    )
    render_blue_note(
        "Upstream, dbt (`stg_secom` → `int_secom_*` → **`mart_secom_features`**) profiles sensors, "
        "drops >10% missing / zero-variance columns, and adds the rz twins, cyclical calendar "
        "features and missing-flags. These keep/drop rules profile all 1,567 rows, including the "
        "holdout era; because they use unsupervised metadata only, the leakage risk is negligible, "
        "but train-only profiling would be the purist alternative. Everything below runs in sklearn "
        "on that mart."
    )

    if cluster_example:
        cluster_fig = fig_spearman_cluster(cluster_example)
        members = cluster_example.get("members", [])
        cluster_caption = (
            "Holdout-fit cluster — kept the single best member, dropped the rest: "
            + ", ".join(f"`{m}`" for m in members)
            if members
            else "One correlated cluster from the holdout training fit."
        )
    else:
        cluster_fig = fig_spearman_cluster_example()
        cluster_caption = "Illustrative cluster: `c_340`, `c_204`, `c_67` collapse to one survivor."

    steps = [
        {
            "slug": "impute",
            "title": "1 · Impute",
            "chip": f"{after_impute:,} cols",
            "detail": "Per-sensor **median** fill learned on the training fold, so sparse sensors "
                      "stay usable without outliers skewing the fill value.",
            "fig": fig_impute_example(),
            "caption": None,
        },
        {
            "slug": "cluster",
            "title": "2 · Cluster",
            "chip": f"→ {after_cluster:,} kept",
            "detail": "`VarianceThreshold` drops near-constant columns, then **Spearman** "
                      "`SmartCorrelatedSelection` collapses each correlated group to its single "
                      "best member. The threshold is **CV-tuned**, so the survivor count differs "
                      "per cell.",
            "fig": cluster_fig,
            "caption": cluster_caption,
        },
        {
            "slug": "scale",
            "title": "3 · Scale",
            "chip": "front-end output",
            "detail": "`RobustScaler` (median / IQR) so heavy-tailed sensors and outliers don't "
                      "dominate the downstream classifier.",
            "fig": fig_scale_example(),
            "caption": None,
        },
        {
            "slug": "calibrate",
            "title": "4 · Calibrate",
            "chip": "fail P(·)",
            "detail": "`CalibratedClassifierCV` (**Platt / sigmoid**) maps raw head scores to "
                      "trustworthy fail probabilities — what the operating-point thresholds on the "
                      "model pages rely on.",
            "fig": fig_calibrate_example(),
            "caption": None,
        },
    ]

    for step in steps:
        with st.container(key=f"card_spine_{step['slug']}"):
            text_col, chart_col = st.columns([1, 1.1], gap="large")
            with text_col:
                st.markdown(f"#### {step['title']}")
                st.markdown(f"`{step['chip']}`")
                st.markdown(step["detail"])
                if step["caption"]:
                    st.caption(step["caption"])
            with chart_col:
                st.plotly_chart(
                    step["fig"],
                    width="stretch",
                    theme="streamlit",
                    key=f"p2_spine_{step['slug']}",
                )


def main() -> None:
    st.title("Pipeline — feature spine")
    st.caption(
        "How raw SECOM sensors become the shared feature spine every model starts from: the "
        "rolling-z twin features and the impute → cluster → scale → calibrate steps shared across "
        "the whole 3×3 grid. The Front-ends & journeys page shows how each model diverges from here."
    )

    ctx = load_pipeline_context()
    if not ctx.available:
        st.warning(ARTIFACTS_MISSING)

    render_blue_note(
        "Every cell of the 3×3 grid (3 front-ends × 3 classifier heads) runs the **same spine** — "
        "impute → cluster → scale → calibrate — fit on training folds only (no leakage). This page "
        "follows the data through that shared spine: **rz twins → the shared spine.** The "
        "Front-ends & journeys page then shows how each front-end diverges."
    )
    _render_rz_explainer(ctx.linear_ref)
    _render_shared_spine(ctx.linear_ref, ctx.cluster_example)


main()
