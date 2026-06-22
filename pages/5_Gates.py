"""Gates & risk-coverage: standalone MSPC abstention analysis for both tracks."""
from __future__ import annotations

import streamlit as st

from secom.dashboard import render_blue_note
from secom.dashboard.charts import fig_hotelling_t2_intuition
from secom.dashboard.model_views import (
    load_payload,
    render_gate_section,
    render_risk_coverage,
)


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

    tab_interp, tab_extrap = st.tabs(
        ["Interpolation gate (EFA → T²+Q)", "Extrapolation gate (sBFA → BGM+Q)"]
    )
    with tab_interp:
        render_gate_section(payload, blocked=False)
    with tab_extrap:
        cond_df = render_gate_section(payload, blocked=True)
        if cond_df is not None:
            render_risk_coverage(payload, cond_df)

    render_blue_note(
        "**Honest conclusion.** With only ~17-20 holdout fails, conditional metrics have wide "
        "confidence intervals. The gate is best framed as an MSPC drift/excursion monitor that "
        "lifts conditional performance on the wafers it keeps — a deployment-realistic abstention "
        "tool, not a way to manufacture recall."
    )


main()
