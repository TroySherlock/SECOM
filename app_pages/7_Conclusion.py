"""Conclusion: what the project built, what it found, and what was deliberately left out."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from secom.dashboard import render_verdict
from secom.dashboard.components import load_payload
from secom.dashboard.data import holdout_delta_df


def _champion_row(d: pd.DataFrame, pipeline: str) -> dict:
    if d.empty or "pipeline" not in d:
        return {}
    row = d[d["pipeline"] == pipeline]
    return {} if row.empty else {
        "interp": float(row["interpolation"].iloc[0]),
        "extrap": float(row["extrapolation"].iloc[0]),
    }


def _render_champion_comparison() -> None:
    """Lead with results: the interpolation champion vs the extrapolation champion."""
    try:
        d = holdout_delta_df(load_payload(), "pr_auc")
    except Exception:
        d = pd.DataFrame()
    hsic = _champion_row(d, "hsic_rf")     # interpolation champion
    pls = _champion_row(d, "pls_bayes")    # extrapolation champion

    with st.container(key="card_champs"):
        st.subheader("🏆 Two champions, two regimes")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(":orange[**Interpolation champion · `hsic_rf`**]")
            if hsic:
                st.metric(
                    "PR-AUC (in-distribution)",
                    f"{hsic['interp']:.3f}",
                    delta=f"{hsic['extrap'] - hsic['interp']:.3f} under drift",
                )
            else:
                st.metric("PR-AUC (in-distribution)", "n/a")
            st.caption(
                "Best when there is no drift — but selection-based, so it **collapses** on the "
                "forward holdout (down to the worst model)."
            )
        with c2:
            st.markdown(":green[**Extrapolation champion · `pls_bayes`**]")
            if pls:
                st.metric(
                    "PR-AUC (forward holdout)",
                    f"{pls['extrap']:.3f}",
                    delta=f"{pls['extrap'] - pls['interp']:.3f} under drift",
                )
            else:
                st.metric("PR-AUC (forward holdout)", "n/a")
            st.caption(
                "The deploy pick — PLS latent components aggregate many weak sensors and actually "
                "**gain** under drift."
            )
        st.caption(
            "The interpolation track is the optimistic in-distribution ceiling; the extrapolation "
            "track trains on the earliest 80% by time and tests on the latest 20%. The flip in "
            "champions is the whole point: the model that wins without drift is not the one you "
            "deploy into a drifting fab."
        )


def _render_what_we_built() -> None:
    with st.container(key="card_built"):
        st.subheader("🛠️ What this project built")
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown(
                "An end-to-end, drift-aware defect-detection system for the SECOM line, built as "
                "one honest narrative rather than a single accuracy number:\n"
                "- **Overview** — the problem: 6.6% fail rate, hundreds of weak/redundant sensors, "
                "structured missingness, and a process that drifts forward in time.\n"
                "- **Pipeline** — the shared feature spine (rz twins, variance/correlation pruning) "
                "and the three front-ends with the two champion journeys.\n"
                "- **Interpolation / Extrapolation** — every pipeline tuned once, then scored on an "
                "in-distribution ceiling **and** a forward temporal holdout, so the cost of drift "
                "is measured, not hidden."
            )
        with right:
            st.markdown(
                "- **Gates** — a PCA Hotelling-T² fab-standard baseline and a custom sBFA → BGM "
                "gate that abstains on out-of-control wafers instead of extrapolating onto a "
                "shifted process.\n"
                "- **Results** — calibrated probabilities, same-scale calibrated credible "
                "intervals, SPC context, a gate cross-check, and a grounded local-LLM narrative "
                "that turns the statistics into an engineer-readable disposition.\n"
                "- **Conclusion** — this page: the deploy verdict and an honest account of what "
                "was scoped out."
            )


def _render_takeaways() -> None:
    st.subheader("⭐ Key strengths and takeaways")
    row1_left, row1_right = st.columns(2)
    with row1_left:
        with st.container(key="card_takeaway_eval"):
            st.markdown(
                "🎯 **Honest evaluation is the backbone.** The two-track design (interpolation = "
                "ceiling, extrapolation = forward holdout) refuses to report a single optimistic "
                "number and instead quantifies the drop under drift."
            )
    with row1_right:
        with st.container(key="card_takeaway_abstain"):
            st.markdown(
                "🛡️ **Abstention beats false confidence.** The gates encode \"don't predict on a "
                "shifted process.\" The gate-comparison case rests on three independent structural "
                "arguments — matched in-control false-positive rate, different flagged members "
                "(multimodality), and a visible downward density drift on the holdout overlay — not "
                "noisy yield deltas."
            )
    row2_left, row2_right = st.columns(2)
    with row2_left:
        with st.container(key="card_takeaway_calibration"):
            st.markdown(
                "📏 **Calibration end-to-end.** A calibrated deploy probability and a same-scale "
                "calibrated credible interval, plus a deterministic uncertainty/borderline "
                "vocabulary, mean the narratives can no longer contradict themselves."
            )
    with row2_right:
        with st.container(key="card_takeaway_llm"):
            st.markdown(
                "🤖 **The LLM is grounded, not generative trivia.** The v8 prompt reports only "
                "derived facts (BGM gate, drift in SD, direction-counts, four-outcome actions). "
                "Across both tracks there are zero raw-token leaks and consistent, correct "
                "dispositions."
            )


def _render_what_we_found() -> None:
    with st.container(key="card_found"):
        st.subheader("🔬 What we found")
        st.markdown(
            "- **Drift is real and costly.** The champion ranking flips: `hsic_rf` wins "
            "in-distribution but :red[collapses] under drift, while `pls_bayes` :green[gains] — the "
            "central justification for everything downstream.\n"
            "- **The in-control region is multimodal.** The BGM mixture activates more than one "
            "mode, so a single PCA Hotelling ellipse over-abstains on healthy wafers that simply "
            "belong to a legitimate second mode.\n"
            "- **Sensor noise is heteroscedastic.** Noise variance spans orders of magnitude, so "
            "the custom gate's noise-weighted attribution points engineers at the right subsystem "
            "where equal-weight blame does not.\n"
            "- **The verdict:** run the custom sBFA → BGM gate as the primary drift/excursion "
            "monitor and keep the trusted PCA-MSPC gate as an always-on conservative sanity check."
        )


def _render_left_out() -> None:
    with st.container(key="card_left_out"):
        st.subheader("🚧 Deliberately left out (and why)")
        st.caption(
            "SECOM is a fixed benchmark with only ~104 fails total (~17-20 in the temporal "
            "holdout). The following were scoped out as honest consequences of that data limit, "
            "not as oversights."
        )
        st.markdown(
            "- **TabPFN.** Considered as an alternative classifier head but set aside: it would add "
            "a black-box in-context point model without advancing the project's thesis (honest "
            "drift evaluation, interpretable gates, end-to-end calibration), and it sits awkwardly "
            "with the high feature count and structured missingness here.\n"
            "- **Retraining + streaming data.** A live monitor with drift-triggered recalibration "
            "needs a continuous wafer stream and ongoing disposition labels that a static benchmark "
            "cannot provide. The SPC control chart and KS/AUC drift statistics are the offline "
            "stand-ins for that loop.\n"
            "- **Semi-supervised learning via sample statistics.** Leaning harder on the unlabeled "
            "passing-wafer distribution (self-training / distribution-matching) is partially already "
            "realized — the gates set their control limits and latent models from passing wafers — "
            "but full SSL was deferred because validating any gain needs more labels than SECOM has."
        )


def main() -> None:
    st.title("Conclusion — what we built, found, and left out")
    st.caption(
        "SECOM in one line: a rare-event (6.6% fail), temporally drifting, heavily-missing process "
        "where a single accuracy number is misleading. This project answers it with honest "
        "two-track evaluation, abstention gates that refuse to extrapolate onto a shifted process, "
        "and calibrated, grounded explanations."
    )

    _render_champion_comparison()
    _render_what_we_built()
    st.divider()
    _render_takeaways()
    _render_what_we_found()
    _render_left_out()

    render_verdict(
        "**The bottom line.** On a rare-event, drifting, heavily-missing process, the most valuable "
        "thing a model can do is know when not to predict. This system measures the true cost of "
        "drift, abstains on out-of-control wafers instead of guessing, and explains what survives "
        "in calibrated, engineer-readable terms — a monitoring system that is honest about what it "
        "does not know."
    )


main()
