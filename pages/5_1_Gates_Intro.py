"""Gates 5.1 - introduction: what the MSPC abstention gates are and how to read them."""
from __future__ import annotations

import streamlit as st

from secom.dashboard import render_blue_note


def main() -> None:
    st.title("5.1 Gates - introduction")
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

    st.subheader("The two gates")
    col_t2, col_bgm = st.columns(2)
    with col_t2:
        st.markdown("**5.2 - Hotelling T² gate (EFA)**")
        st.markdown(
            "- Regularized exploratory factor analysis on the raw post-cluster sensors\n"
            "- **Hotelling T²** flags in-subspace excursions (drift along known factor directions)\n"
            "- **Q / SPE** residual flags structural breaks the factor model cannot explain\n"
            "- Frequentist, dense loadings; abstain on `T² > UCL` **or** `Q > UCL`"
        )
    with col_bgm:
        st.markdown("**5.3 - sBFA → BGM gate**")
        st.markdown(
            "- Sparse Bayesian factor analysis (NumPyro, ADVI) with Laplace-sparse loadings\n"
            "- A **Bayesian Gaussian mixture** density on the factor scores replaces T²\n"
            "- Same **Q / SPE** residual for orthogonal novelty\n"
            "- Robust-scaled + clipped; abstain on low BGM log-density **or** high Q"
        )

    st.subheader("The control statistics")
    st.markdown(
        "- **Hotelling T²** - Mahalanobis distance of the factor scores; flags excursions **along** "
        "the learned factor directions (5.2). \n"
        "- **Q / SPE** - squared reconstruction residual; flags structural breaks **orthogonal** to "
        "the factor model that T²/density cannot see (both gates). \n"
        "- **BGM log-density** - log-likelihood under a Bayesian Gaussian mixture on the factor "
        "scores; the multimodal analogue of T² (5.3). Low density trips the gate."
    )
    st.latex(r"T^2 = (\mathbf{x}-\boldsymbol{\mu})^\top \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})")

    st.subheader("How to read the gate pages")
    st.markdown(
        "Pages **5.2** (Hotelling T²) and **5.3** (sBFA → BGM) share the same five-section drift "
        "monitor for their gate:\n"
        "1. **Distribution shift** - each control statistic, reference vs holdout, against its limit.\n"
        "2. **MSPC control chart** - the temporal holdout in measurement-time order with a rolling "
        "out-of-control rate (the retraining trigger).\n"
        "3. **Drift scalar** - KS distance + separability AUC on the passing wafers (in-control "
        "process drift, fails removed) - the statistically solid number.\n"
        "4. **Latent factor space** - the two top-drifting factors with the control envelope and "
        "the holdout split into pass / flagged pass / caught fail / missed fail.\n"
        "5. **Loadings root cause** - factors ranked by drift, with the heavy-loading sensors of "
        "the top factor as the candidate drifting subsystem.\n\n"
        "**5.4 Gate comparison** holds the head-to-head views: lift vs no gate, gate-vs-gate, "
        "coverage drift, and the conditional PR-AUC / risk-coverage CI detail."
    )

    render_blue_note(
        "**Honest framing.** The per-wafer drift evidence (sections 1-3) is computed over **all** "
        "wafers, so the distribution shift and out-of-control rate are the statistically solid "
        "story - the gate is an MSPC drift/excursion monitor, not a way to manufacture recall. "
        "Conditional yield metrics (5.4) rest on only ~17-20 holdout fails, so read their direction, "
        "not their magnitude."
    )


main()
