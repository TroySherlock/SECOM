"""Gates 5.1 - introduction: what the MSPC abstention gates are and how to read them."""
from __future__ import annotations

import streamlit as st

from secom.dashboard import render_blue_note
from secom.dashboard.charts import fig_sensor_drift_heatmap
from secom.dashboard.stg import StgSnapshot, build_stg_snapshot, era_drift_summary
from secom.pipelines import TEST_SIZE, TIMESTAMP_COL


@st.cache_data(show_spinner="Loading stg_secom...")
def _load_stg_snapshot() -> StgSnapshot:
    return build_stg_snapshot()


def _render_sensor_drift() -> None:
    st.subheader("🌊 What the gates are up against: sensor drift")
    st.caption(
        "Before any gate, the problem: the sensors themselves drift over time. The models are fit "
        "on the training era, but later wafers are measured on a slowly shifting process - so a "
        "wafer can pass inspection yet sit far from anything the model (or the gate) was fit on."
    )
    snapshot = _load_stg_snapshot()
    df = snapshot.df
    sensor_cols = snapshot.sensor_cols
    drift = era_drift_summary(
        df,
        timestamp_col=TIMESTAMP_COL,
        sensor_cols=sensor_cols,
        test_size=TEST_SIZE,
    )
    d1, d2, d3 = st.columns(3)
    d1.metric(
        f"Sensors drifting >{drift['z_threshold']:.0f}σ",
        f"{drift['pct_drifted']:.0f}%",
        border=True,
    )
    d2.metric(
        "Drifting / evaluated",
        f"{drift['n_drifted']:,} / {drift['n_evaluated']:,}",
        border=True,
    )
    d3.metric(
        "Median |shift| (holdout)",
        f"{drift['median_abs_shift']:.2f}σ",
        border=True,
    )
    st.plotly_chart(
        fig_sensor_drift_heatmap(
            df,
            timestamp_col=TIMESTAMP_COL,
            sensor_cols=sensor_cols,
            test_size=TEST_SIZE,
        ),
        width="stretch",
        theme="streamlit",
        key="p51_sensor_drift",
    )
    st.caption(
        "Each row is one of the most drift-prone sensors; each column is a time window. Colour is "
        "the sensor's mean standardized against the training era, so blue/red cells show how far "
        f"later wafers drift from what the models were fit on. The {drift['pct_drifted']:.0f}% of "
        "sensors past their training-era baseline is exactly the excursion the gates exist to catch "
        "(same view as the Overview page's drift tab)."
    )


def main() -> None:
    st.title("Gates — introduction")
    st.caption(
        "Two standalone multivariate process-control (MSPC) gates, fit on passing-train wafers "
        "only and evaluated separately from the classifiers. Each is framed as a sensor-space "
        "drift / excursion monitor: it flags out-of-control wafers so the model can abstain."
    )

    render_blue_note(
        "A gate is **not** a recall booster. It is an abstention rule: rank wafers by a control "
        "statistic, drop the most out-of-control ones, and rescore the rest. The win is **higher "
        "conditional performance on the kept set** - and gains can come from dropping easy "
        "negatives, so always read conditional PR-AUC next to coverage and the flagged-fail counts."
    )

    _render_sensor_drift()

    render_blue_note(
        "**The monitoring loop.** Each gate fits its control limits on passing-train wafers, then "
        "scores new wafers in measurement-time order. When the rolling out-of-control rate climbs "
        "above its in-control baseline (equivalently, coverage drops), that is the **drift / "
        "retraining trigger** - the signal to refit on recent data. On SECOM this fires mainly "
        "through the BGM density and the shared Q channel; the PCA T² limit barely moves. **5.4** "
        "shows a concrete passing wafer the custom gate caught this way."
    )

    st.divider()
    st.subheader("🚪 The two gates")
    col_t2, col_bgm = st.columns(2)
    with col_t2:
        with st.container(key="card_gate_pca"):
            st.markdown("**5.2 - Hotelling T² gate (PCA, fab standard)**")
            st.markdown(
                "- Standard PCA-MSPC on the raw post-cluster sensors (the **fab-standard baseline**)\n"
                "- **Hotelling T²** flags in-subspace excursions (drift along known component directions)\n"
                "- **Q / SPE** residual flags structural breaks the components cannot explain\n"
                "- Frequentist, dense loadings; abstain on `T² > UCL` **or** `Q > UCL`"
            )
    with col_bgm:
        with st.container(key="card_gate_bgm"):
            st.markdown("**5.3 - sBFA → BGM gate (custom)**")
            st.markdown(
                "- Sparse Bayesian factor analysis (NumPyro, ADVI) with Laplace-sparse loadings\n"
                "- A **Bayesian Gaussian mixture** density on the factor scores replaces T²\n"
                "- Same **Q / SPE** residual for orthogonal novelty\n"
                "- Robust-scaled + clipped; abstain on low BGM log-density **or** high Q\n"
                "- The **custom** gate; 5.4 tests it against the PCA baseline at equal overkill"
            )

    with st.container(key="card_control_stats"):
        st.subheader("📐 The control statistics")
        st.markdown(
            "- **Hotelling T²** - Mahalanobis distance of the factor scores; flags excursions **along** "
            "the learned factor directions (5.2). \n"
            "- **Q / SPE** - squared reconstruction residual; flags structural breaks **orthogonal** to "
            "the factor model that T²/density cannot see (both gates). \n"
            "- **BGM log-density** - log-likelihood under a Bayesian Gaussian mixture on the factor "
            "scores; the multimodal analogue of T² (5.3). Low density trips the gate."
        )
        st.latex(r"T^2 = (\mathbf{x}-\boldsymbol{\mu})^\top \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})")

    with st.container(key="card_how_to_read"):
        st.subheader("🧭 How to read the gate pages")
        st.markdown(
            "- **5.2 Hotelling T² gate (PCA baseline)** - a four-section drift monitor: distribution "
            "shift, the time-ordered MSPC control chart (the retraining trigger), an honest KS/AUC "
            "drift scalar, and the PCA component space with its Hotelling ellipse.\n"
            "- **5.3 sBFA → BGM gate (custom)** - the same monitor plus the Bayesian extras: the BGM "
            "mixture weights (proof the in-control region is multimodal), a sparse-loadings root-cause "
            "view that points at the candidate drifting subsystem, and why that attribution is "
            "noise-weighted across heteroscedastic sensors.\n"
            "- **5.4 Gate comparison** - the complete head-to-head story: whether the custom gate fires "
            "under drift (coverage drift), a concrete passing wafer it caught, why PCA over-abstains on "
            "a multimodal in-control region, the risk-coverage trade-off, and a verdict on which gate to run."
        )

    render_blue_note(
        "**Honest framing.** The per-wafer drift evidence (distribution shift, control chart, KS) "
        "is computed over **all** wafers, so it is the statistically solid story - the gate is an "
        "MSPC drift/excursion monitor, not a way to manufacture recall. Conditional yield metrics "
        "(5.4) rest on only ~17-20 holdout fails, so read their direction, not their magnitude."
    )


main()
