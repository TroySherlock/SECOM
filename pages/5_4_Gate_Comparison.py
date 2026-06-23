"""Gates 5.4 - head-to-head comparison: lift vs no gate, gate-vs-gate, coverage drift."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from secom.dashboard import render_blue_note
from secom.dashboard.data import DELTA_METRIC_COLS, gate_conditional_df
from secom.dashboard.model_views import (
    GATE_LABELS,
    load_payload,
    render_gate_lift,
    render_gate_section,
    render_risk_coverage,
)


def _operating_coverage(payload: dict, track: str, gate: str) -> float | None:
    cond = gate_conditional_df(payload, track, gate)
    if cond.empty or "coverage" not in cond.columns:
        return None
    cov = cond["coverage"].dropna()
    return float(cov.iloc[0]) if not cov.empty else None


def _coverage_drift_table(payload: dict) -> pd.DataFrame:
    rows = []
    for gate in ("efa", "bayes"):
        interp = _operating_coverage(payload, "interpolation", gate)
        temporal = _operating_coverage(payload, "extrapolation", gate)
        delta = (temporal - interp) if (interp is not None and temporal is not None) else None
        rows.append(
            {
                "Gate": GATE_LABELS.get(gate, gate),
                "In-distribution coverage": interp,
                "Temporal coverage": temporal,
                "Δ coverage (temporal − in-dist)": delta,
            }
        )
    return pd.DataFrame(rows)


def _render_track(payload: dict, *, track: str) -> None:
    metric = st.radio(
        "Metric",
        list(DELTA_METRIC_COLS),
        horizontal=True,
        key=f"cmp_metric_{track}",
    )
    render_gate_lift(payload, track=track, metric=metric)

    st.markdown("**Risk–coverage curves side by side** - Hotelling T² (left) vs sBFA → BGM (right)")
    st.caption(
        "Coverage runs high→low left→right; the dashed line marks each gate's actual operating "
        "coverage. Compare how each gate trades coverage for conditional performance on this track."
    )
    left, right = st.columns(2)
    with left:
        render_risk_coverage(payload, track=track, gate="efa", metric=metric)
    with right:
        render_risk_coverage(payload, track=track, gate="bayes", metric=metric)

    st.markdown("**Conditional performance on kept wafers** - CI detail per gate")
    render_gate_section(payload, track=track, gate="efa")
    render_gate_section(payload, track=track, gate="bayes")


def main() -> None:
    st.title("5.4 Gate comparison")
    st.caption(
        "The two MSPC gates head-to-head on the same track: conditional lift vs no gate, the "
        "EFA-vs-Bayes delta, and how each gate's operating coverage shifts from the in-distribution "
        "to the temporal protocol."
    )

    try:
        payload = load_payload()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    st.subheader("Operating-coverage drift (in-distribution → temporal)")
    st.caption(
        "Coverage = fraction of wafers each gate keeps at its fitted limits. A coverage drop into "
        "the temporal window means the gate is abstaining more - it is detecting the forward-window "
        "drift. This per-wafer count is the trustworthy signal (vs the noisy conditional metrics)."
    )
    cov_df = _coverage_drift_table(payload)
    st.dataframe(
        cov_df,
        width="stretch",
        hide_index=True,
        column_config={
            "In-distribution coverage": st.column_config.NumberColumn(format="%.1%"),
            "Temporal coverage": st.column_config.NumberColumn(format="%.1%"),
            "Δ coverage (temporal − in-dist)": st.column_config.NumberColumn(format="%+.1%"),
        },
    )

    render_blue_note(
        "Read **lift bars** for whether a gate helps conditional performance on the wafers it "
        "keeps, and the **coverage-drift table** for whether it actually triggers under drift. The "
        "EFA T² limit barely moves between protocols (most signal sits in Q), while the BGM gate's "
        "density tightens - see 5.3 for the per-wafer drift evidence."
    )

    tab_interp, tab_extrap = st.tabs(
        ["Interpolation (random holdout)", "Extrapolation (temporal holdout)"]
    )
    with tab_interp:
        _render_track(payload, track="interpolation")
    with tab_extrap:
        _render_track(payload, track="extrapolation")


main()
