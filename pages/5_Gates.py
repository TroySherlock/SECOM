"""Gates & risk-coverage: standalone MSPC abstention analysis for both tracks."""
from __future__ import annotations

import streamlit as st

from secom.dashboard import render_blue_note
from secom.dashboard.charts import fig_hotelling_t2_intuition
from secom.dashboard.data import DELTA_METRIC_COLS
from secom.dashboard.model_views import (
    load_payload,
    render_gate_lift,
    render_gate_section,
    render_risk_coverage,
)


def _render_gate_track(payload: dict, *, track: str) -> None:
    """Both gates on one track: lift deltas, gate-vs-gate, both risk-coverage curves."""
    metric = st.radio(
        "Metric",
        list(DELTA_METRIC_COLS),
        horizontal=True,
        key=f"gate_metric_{track}",
    )
    render_gate_lift(payload, track=track, metric=metric)

    st.markdown("**Risk–coverage curves** — keep the least-suspicious fraction, then rescore")
    st.caption(
        "Coverage runs high→low left→right, so abstention increases to the right; coverage=1.0 is "
        "the global metric. The dashed line marks each gate's actual operating coverage. This is "
        "where the Hotelling-T²/EFA curve appears on both tracks, beside the sBFA→BGM curve."
    )
    left, right = st.columns(2)
    with left:
        render_risk_coverage(payload, track=track, gate="efa", metric=metric)
    with right:
        render_risk_coverage(payload, track=track, gate="bayes", metric=metric)

    render_gate_section(payload, track=track, gate="efa")
    render_gate_section(payload, track=track, gate="bayes")


def main() -> None:
    st.title("Gates & risk-coverage")
    st.caption(
        "Two standalone multivariate process-control (MSPC) gates, fit on passing-train wafers "
        "only and evaluated separately from the classifiers. They flag out-of-control wafers so "
        "the model can abstain; we report conditional PR-AUC and coverage on the wafers they keep."
    )

    try:
        payload = load_payload()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    render_blue_note(
        "A gate is **not** a recall booster. It is an abstention rule: rank wafers by a control "
        "statistic (Hotelling T² and Q/SPE residual, or sparse-Bayesian-FA density and Q), drop "
        "the most out-of-control ones, and rescore the rest. The win is **higher conditional "
        "performance on the kept set** — gains can come from dropping easy negatives, so read "
        "conditional PR-AUC next to coverage and the flagged-fail counts."
    )

    with st.expander("Intuition: Hotelling T² control region", expanded=False):
        st.latex(r"T^2 = (\mathbf{x}-\boldsymbol{\mu})^\top \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})")
        st.plotly_chart(
            fig_hotelling_t2_intuition(),
            width="stretch",
            theme="streamlit",
            key="p5_hotelling_intuition",
        )

    st.caption(
        "Both gates (EFA → T²+Q and sBFA → BGM+Q) are fit and scored on **both** protocols, so "
        "you can compare each gate against no gate and against the other on the same track."
    )
    tab_interp, tab_extrap = st.tabs(
        ["Interpolation (random holdout)", "Extrapolation (temporal holdout)"]
    )
    with tab_interp:
        _render_gate_track(payload, track="interpolation")
    with tab_extrap:
        _render_gate_track(payload, track="extrapolation")

    render_blue_note(
        "**Honest conclusion.** With only ~17-20 holdout fails, conditional metrics have wide "
        "confidence intervals. The gate is best framed as an MSPC drift/excursion monitor that "
        "lifts conditional performance on the wafers it keeps — a deployment-realistic abstention "
        "tool, not a way to manufacture recall."
    )


main()
