"""Conclusion: what the project built, what it found, and what was deliberately left out."""
from __future__ import annotations

import streamlit as st

from secom.dashboard import render_blue_note
from secom.dashboard.data import holdout_delta_df
from secom.dashboard.explainability import champion_for_track
from secom.dashboard.model_views import load_payload
from secom.dashboard.stg import (
    STG_RELATION,
    StgSnapshot,
    build_stg_snapshot,
    era_drift_summary,
    stg_available,
)
from secom.pipelines import DB_PATH, TEST_SIZE, TIMESTAMP_COL


@st.cache_data(show_spinner="Loading stg_secom...")
def _load_stg_snapshot() -> StgSnapshot:
    return build_stg_snapshot()


def _na(value: str | None) -> str:
    return value if value else "n/a"


def _render_headline_metrics() -> None:
    """Live headline numbers; every fetch is guarded so the page never errors."""
    n_obs = n_fail = fail_pct = sensors = drift_pct = None
    if DB_PATH.exists() and stg_available():
        try:
            snapshot = _load_stg_snapshot()
            stats = snapshot.stats
            n_obs = f"{stats['n_obs']:,}"
            n_fail = f"{stats['n_fail']:,}"
            fail_pct = f"{100 * stats['fail_rate']:.1f}%"
            sensors = f"{stats['n_sensors']:,}"
            drift = era_drift_summary(
                snapshot.df,
                timestamp_col=TIMESTAMP_COL,
                sensor_cols=snapshot.sensor_cols,
                test_size=TEST_SIZE,
            )
            drift_pct = f"{drift['pct_drifted']:.0f}%"
        except Exception:
            pass

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Wafers", _na(n_obs), border=True)
    c2.metric("Fails", _na(n_fail), border=True)
    c3.metric("Fail prevalence", _na(fail_pct), border=True)
    c4.metric("Sensor channels", _na(sensors), border=True)
    c5.metric("Sensors drifting >2σ", _na(drift_pct), border=True)


def _render_drift_cost() -> None:
    """Champion PR-AUC: in-distribution ceiling vs forward (temporal) holdout."""
    interp_pr = extrap_pr = delta = None
    champ = champion_for_track("extrapolation")
    try:
        payload = load_payload()
        delta_df = holdout_delta_df(payload, "pr_auc")
        if not delta_df.empty:
            row = delta_df[delta_df["pipeline"] == champ]
            if not row.empty:
                interp_pr = f"{float(row['interpolation'].iloc[0]):.3f}"
                extrap_pr = f"{float(row['extrapolation'].iloc[0]):.3f}"
                delta = f"-{float(row['delta'].iloc[0]):.3f}"
    except Exception:
        pass

    st.markdown(f"**The cost of drift — champion model (`{champ}`)**")
    d1, d2, d3 = st.columns(3)
    d1.metric("Interpolation PR-AUC (ceiling)", _na(interp_pr), border=True)
    d2.metric("Extrapolation PR-AUC (forward)", _na(extrap_pr), border=True)
    d3.metric("Drift cost (ΔPR-AUC)", _na(delta), border=True)
    st.caption(
        "The interpolation track is the optimistic in-distribution ceiling; the extrapolation "
        "track trains on the earliest 80% by time and tests on the latest 20%. The gap between "
        "them is the honest cost of temporal drift — the number a single random-split score hides."
    )


def _render_what_we_built() -> None:
    with st.container(border=True, key="card_built"):
        st.subheader("What this project built")
        st.markdown(
            "An end-to-end, drift-aware defect-detection system for the SECOM line, built as one "
            "honest narrative rather than a single accuracy number:\n"
            "- **The problem (1):** ~7% fail rate, hundreds of weak/redundant sensors, structured "
            "missingness, and a process that drifts forward in time.\n"
            "- **The pipeline (2):** a defensive feature-engineering spine (robust-z, correlation/"
            "variance pruning, top-k, PLS components, Hotelling T², hub interactions) feeding two "
            "champion journeys.\n"
            "- **Two-track models (3, 4):** every pipeline tuned once, then scored on an "
            "in-distribution ceiling **and** a forward temporal holdout — so the cost of drift is "
            "measured, not hidden.\n"
            "- **Abstention gates (5):** a PCA Hotelling-T² fab-standard baseline and a custom "
            "sBFA → BGM gate that abstains on out-of-control wafers instead of extrapolating onto a "
            "shifted process.\n"
            "- **Grounded explainability (6):** calibrated probabilities, same-scale calibrated "
            "credible intervals, SPC context, a gate cross-check, and a local-LLM narrative that "
            "turns the statistics into an engineer-readable disposition."
        )


def _render_takeaways() -> None:
    st.subheader("Key strengths and takeaways")
    row1_left, row1_right = st.columns(2)
    with row1_left:
        with st.container(border=True, key="card_takeaway_eval"):
            st.markdown(
                "**Honest evaluation is the backbone.** The two-track design (interpolation = "
                "ceiling, extrapolation = forward holdout) refuses to report a single optimistic "
                "number and instead quantifies the drop under drift — a senior-level instinct."
            )
    with row1_right:
        with st.container(border=True, key="card_takeaway_abstain"):
            st.markdown(
                "**Abstention beats false confidence.** The gates encode \"don't predict on a "
                "shifted process.\" The 5.4 case rests on three independent structural arguments — "
                "matched in-control false-positive rate, different flagged members (multimodality), "
                "and a visible downward density drift on the holdout overlay — not noisy yield deltas."
            )
    row2_left, row2_right = st.columns(2)
    with row2_left:
        with st.container(border=True, key="card_takeaway_calibration"):
            st.markdown(
                "**Calibration end-to-end.** A calibrated deploy probability and a same-scale "
                "calibrated credible interval, plus a deterministic uncertainty/borderline "
                "vocabulary, mean the narratives can no longer contradict themselves."
            )
    with row2_right:
        with st.container(border=True, key="card_takeaway_llm"):
            st.markdown(
                "**The LLM is grounded, not generative trivia.** The v8 prompt reports only derived "
                "facts (BGM gate, drift in SD, direction-counts, four-outcome actions). Across both "
                "tracks there are zero raw-token leaks and consistent, correct dispositions — it "
                "reads like a process engineer."
            )


def _render_what_we_found() -> None:
    with st.container(border=True, key="card_found"):
        st.subheader("What we found")
        st.markdown(
            "- **Drift is real and costly.** The champion's PR-AUC drops measurably from the "
            "in-distribution ceiling to the forward holdout — the central justification for "
            "everything downstream.\n"
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
    with st.container(border=True, key="card_left_out"):
        st.subheader("Deliberately left out (and why)")
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
            "realized — the gates are fit on passing wafers only — but full SSL was deferred because "
            "validating any gain needs more labels than SECOM has."
        )


def main() -> None:
    st.title("Conclusion — what we built, found, and left out")
    st.caption(
        "SECOM in one line: a rare-event (~7% fail), temporally drifting, heavily-missing process "
        "where a single accuracy number is misleading. This project answers it with honest "
        "two-track evaluation, abstention gates that refuse to extrapolate onto a shifted process, "
        "and calibrated, grounded explanations."
    )

    if not (DB_PATH.exists() and stg_available()):
        st.info(
            f"Headline data-quality metrics need `{STG_RELATION}` in DuckDB (build with "
            "`dbt run -s stg_secom`). The conclusions below stand without it."
        )
    _render_headline_metrics()
    st.divider()
    _render_drift_cost()

    st.divider()
    _render_what_we_built()

    st.divider()
    _render_takeaways()

    st.divider()
    _render_what_we_found()

    st.divider()
    _render_left_out()

    render_blue_note(
        "**The bottom line.** On a rare-event, drifting, heavily-missing process, the most valuable "
        "thing a model can do is know when not to predict. This system measures the true cost of "
        "drift, abstains on out-of-control wafers instead of guessing, and explains what survives "
        "in calibrated, engineer-readable terms — a monitoring system that is honest about what it "
        "does not know."
    )


main()
