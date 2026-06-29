"""Gates 5.4 - the complete gate-comparison story: PCA baseline vs custom sBFA -> BGM.

One linear narrative answering "is the custom gate worth running beside the fab
standard?" on the temporal (drift) protocol:
1. Does it fire under drift? - operating-coverage drift (the trustworthy per-wafer signal).
2. A wafer the custom gate caught - a concrete passing wafer flagged out-of-control.
3. Why PCA over-abstains - the in-control region is multimodal (disagreement quadrant).
4. The risk-coverage trade-off on the temporal holdout.
5. Verdict - which gate I would run on this line.

The noise-weighted-attribution case (sensors are heteroscedastic) lives on 5.3,
next to the sparse-loadings root cause it explains.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from secom.dashboard import render_blue_note
from secom.dashboard.charts import (
    fig_gate_disagreement_scatter,
    fig_wafer_drift_spikes,
)
from secom.dashboard.data import (
    DELTA_METRIC_COLS,
    bgm_ooc_wafers,
    gate_conditional_df,
    gate_contrast,
    gate_diagnostics,
    gate_disagreement_summary,
)
from secom.dashboard.explainability import wafer_drift_spikes
from secom.dashboard.model_views import (
    GATE_LABELS,
    load_payload,
    render_risk_coverage,
)

_EMPTY = "No frozen gate-contrast artifacts in benchmark JSON. Re-run `python -m secom.benchmark`."


def _operating_coverage(payload: dict, track: str, gate: str) -> float | None:
    cond = gate_conditional_df(payload, track, gate)
    if cond.empty or "coverage" not in cond.columns:
        return None
    cov = cond["coverage"].dropna()
    return float(cov.iloc[0]) if not cov.empty else None


def _coverage_drift_table(payload: dict) -> pd.DataFrame:
    rows = []
    for gate in ("pca", "bayes"):
        interp = _operating_coverage(payload, "interpolation", gate)
        temporal = _operating_coverage(payload, "extrapolation", gate)
        delta = (temporal - interp) if (interp is not None and temporal is not None) else None
        rows.append(
            {
                "Gate": GATE_LABELS.get(gate, gate),
                "In-distribution coverage": interp * 100 if interp is not None else None,
                "Temporal coverage": temporal * 100 if temporal is not None else None,
                "Δ coverage (temporal − in-dist)": delta * 100 if delta is not None else None,
            }
        )
    return pd.DataFrame(rows)


def _render_coverage_drift(payload: dict) -> None:
    st.subheader("1. Does the custom gate fire under drift?")
    st.caption(
        "Coverage = fraction of wafers each gate keeps at its fitted limits. A coverage drop into "
        "the temporal window means the gate is abstaining more - it is detecting the forward-window "
        "drift. This per-wafer count is the trustworthy signal (vs the noisy conditional metrics "
        "that rest on only ~17-20 holdout fails)."
    )
    cov_df = _coverage_drift_table(payload)
    st.dataframe(
        cov_df,
        width="stretch",
        hide_index=True,
        column_config={
            "In-distribution coverage": st.column_config.NumberColumn(format="%.1f%%"),
            "Temporal coverage": st.column_config.NumberColumn(format="%.1f%%"),
            "Δ coverage (temporal − in-dist)": st.column_config.NumberColumn(format="%+.1f%%"),
        },
    )
    render_blue_note(
        "**The asymmetry is the headline.** EDA confirms the forward window has drifted, yet the "
        "**PCA T² limit barely moves** between protocols - most of PCA's drift signal lives in its "
        "shared Q/SPE channel, so its T² coverage stays flat (or even rises). The **custom gate's "
        "BGM density tightens** and its coverage drops, because density is sensitive to wafers "
        "sliding *between* the in-control modes that T² cannot see. On this line the custom gate is "
        "the more responsive drift detector - the next two sections explain why, structurally."
    )


def _render_multimodality(payload: dict, contrast: dict) -> None:
    st.subheader("3. Why PCA over-abstains: the in-control region is multimodal")
    summary = gate_disagreement_summary(contrast)
    ref = contrast.get("reference") or {}
    t2 = np.asarray(ref.get("pca_t2", []), dtype=float)
    dens = np.asarray(ref.get("bgm_density", []), dtype=float)
    if t2.size == 0 or dens.size != t2.size:
        st.info(_EMPTY)
        return

    density_lcl = float(ref.get("bgm_density_lcl", float("-inf")))

    # Holdout per-wafer stats are already frozen, in matching X_test order across
    # the two gate diagnostics blocks (PCA T2 and BGM density).
    pca_hold = gate_diagnostics(payload, "extrapolation", "pca").get("holdout") or {}
    bgm_hold = gate_diagnostics(payload, "extrapolation", "bayes").get("holdout") or {}
    ho_t2 = np.asarray(pca_hold.get("t2", []), dtype=float)
    ho_dens = np.asarray(bgm_hold.get("density", []), dtype=float)
    has_holdout = ho_t2.size > 0 and ho_t2.size == ho_dens.size

    st.plotly_chart(
        fig_gate_disagreement_scatter(
            t2,
            dens,
            t2_ucl=float(ref.get("pca_t2_ucl", float("inf"))),
            density_lcl=density_lcl,
            holdout_t2=ho_t2 if has_holdout else None,
            holdout_density=ho_dens if has_holdout else None,
            title=(
                "Where the gates disagree, and how the holdout drifts down"
                if has_holdout
                else "Where the gates disagree (passing wafers)"
            ),
        ),
        width="stretch",
        theme="streamlit",
        key="p54_disagreement",
    )
    if summary:
        st.success(
            f"**{summary['pca_only']} of {summary['n_total']} healthy wafers** sit outside PCA's "
            "single ellipse (T² above its limit) but inside the BGM's modes (density in-control). "
            "The PCA baseline would overkill these; the custom gate does not."
        )
    if has_holdout:
        n_below = int((ho_dens < density_lcl).sum())
        n_ho = int(ho_dens.size)
        ref_pct = 100.0 * float((dens < density_lcl).mean()) if dens.size else 0.0
        st.info(
            f"Under the forward temporal holdout, **{n_below} of {n_ho} wafers "
            f"({100.0 * n_below / n_ho:.0f}%)** fall below the BGM density limit, versus the "
            f"~{ref_pct:.0f}% the gate allows in-control on the pre-drift reference - the cloud has "
            "drifted down, exactly the in-subspace shift the BGM gate is built to catch."
        )
    st.caption(
        "Grey points are the in-control pre-drift reference; coloured points are the temporal "
        "holdout, split by which gate would abstain. The green quadrant (high T², high density) is "
        "the cost of a single homoscedastic ellipse: wafers PCA abstains on that are genuinely "
        "normal for a multimodal line (multiple recipes / products / chambers / eras). The BGM "
        "wraps each mode separately, so a wafer in a legitimate second mode stays in-control. See "
        "**5.3** for the BGM mode weights that prove more than one mode is active."
    )


def _render_caught_wafer(payload: dict) -> None:
    st.subheader("2. A wafer the custom gate caught")
    wafers = bgm_ooc_wafers(payload, track="extrapolation")
    if wafers.empty:
        st.info(
            "No frozen BGM out-of-control wafers in benchmark JSON. Re-run `python -m secom.benchmark`."
        )
        return

    st.caption(
        "Passing (good-yield) wafers from the temporal holdout that the custom sBFA → BGM gate still "
        "flagged out-of-control. These are in-control by yield but out-of-control by process - pick "
        "one to see which of the globally drifting sensors are spiking on it."
    )
    labels = {
        int(r.observation_id): (
            f"Wafer {int(r.observation_id)}"
            + (f" · {str(r.ts)[:10]}" if r.ts else "")
            + (f" · density {r.density:.1f}" if r.density == r.density else "")
        )
        for r in wafers.itertuples()
    }
    choice = st.selectbox(
        "Out-of-control passing wafer (lowest density first)",
        options=list(labels),
        format_func=lambda i: labels[i],
        key="p54_wafer_pick",
    )

    spikes = wafer_drift_spikes(int(choice), track="extrapolation")
    if spikes.empty:
        st.info("No drifting-sensor readings available for this wafer.")
        return
    st.plotly_chart(
        fig_wafer_drift_spikes(
            spikes["sensor"].astype(str).tolist(),
            spikes["robust_z"].to_numpy(),
            spikes["drift_shift"].to_numpy(),
        ),
        width="stretch",
        theme="streamlit",
        key="p54_wafer_spikes",
    )
    n_shown = len(spikes)
    n_spiking = int((spikes["robust_z"].abs() > 2.0).sum())
    st.caption(
        f"This wafer passed inspection, yet across its known-drifting sensors ({n_shown} with a "
        f"reading here) **{n_spiking} sit beyond ±2σ** of the in-control distribution (red bars). "
        "Note no single sensor has to scream: the BGM gate fires on the **joint** density across "
        "these correlated, globally-drifted sensors, so a wafer can be far out-of-control even when "
        "each sensor is only mildly elevated - exactly the multivariate excursion per-sensor limits "
        "miss. The gate abstains on wafers like this so the classifier is not asked to extrapolate "
        "onto a shifted process. (Hover a bar for that sensor's global train→holdout drift.)"
    )


def _render_risk_coverage(payload: dict) -> None:
    st.subheader("4. The risk-coverage trade-off (temporal holdout)")
    metric = st.radio(
        "Metric",
        list(DELTA_METRIC_COLS),
        horizontal=True,
        key="cmp_metric_extrap",
    )
    st.caption(
        "Coverage runs high→low left→right; the dashed line marks each gate's actual operating "
        "coverage. Compare how each gate trades coverage for conditional performance on the kept "
        "wafers. These conditional curves rest on ~17-20 holdout fails, so read direction and "
        "shape, not the precise magnitude."
    )
    left, right = st.columns(2)
    with left:
        render_risk_coverage(payload, track="extrapolation", gate="pca", metric=metric)
    with right:
        render_risk_coverage(payload, track="extrapolation", gate="bayes", metric=metric)


def _render_verdict() -> None:
    st.subheader("5. Verdict: which gate I would run")
    with st.container(border=True, key="card_verdict"):
        st.markdown(
            "**I would run the custom sBFA → BGM as the primary drift / excursion monitor on this "
            "line, keeping the PCA-MSPC gate as an always-on conservative sanity check.** The case "
            "rests on three structural facts, not on the noisy conditional yield deltas:\n"
            "- **Drift response (section 1):** under the EDA-confirmed temporal drift the custom gate's "
            "coverage actually moves, while the PCA T² limit stays flat - the custom gate sees the "
            "in-subspace, between-mode drift PCA is blind to.\n"
            "- **Multimodality (section 3):** the in-control region has more than one mode, so PCA's "
            "single ellipse would needlessly overkill ~2% of perfectly healthy wafers that the BGM "
            "keeps in-control.\n"
            "- **Heteroscedasticity (5.3):** sensor noise spans orders of magnitude, so the custom "
            "gate's noise-weighted attribution (shown on **5.3**) points engineers at the right "
            "subsystem where PCA's equal-weight blame does not.\n\n"
            "PCA remains valuable precisely *because* it is the trusted, dense, fully-understood fab "
            "standard - it is the benchmark the custom gate has to beat, and a useful second opinion. "
            "But as the operating monitor for a multimodal, uneven-noise line that drifts forward in "
            "time, the custom gate is the better-specified detector."
        )


def main() -> None:
    st.title("Gate comparison")
    st.caption(
        "The complete head-to-head story: the custom sBFA → BGM gate vs the standard PCA baseline "
        "on the temporal (drift) protocol. Does the custom gate detect drift the baseline misses, "
        "and is its root-cause attribution better - at equal overkill?"
    )

    try:
        payload = load_payload()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    render_blue_note(
        "**The honest framing.** 5.1 showed the sensors drift; 5.2 and 5.3 showed each gate detects "
        "it (5.3 also shows why its noise-weighted attribution across **heteroscedastic** sensors is "
        "sharper). PCA is the trusted fab standard and drops nothing; the custom gate earns its place "
        "only where it is provably better. This page makes the case: it fires under drift (section "
        "1), here is a concrete passing wafer it caught (section 2), and it does not overkill a "
        "**multimodal** in-control region (section 3) - then it weighs the conditional trade-off "
        "(section 4) and gives the verdict."
    )

    contrast = gate_contrast(payload, track="extrapolation")

    _render_coverage_drift(payload)
    st.divider()
    _render_caught_wafer(payload)
    st.divider()
    if contrast:
        _render_multimodality(payload, contrast)
    else:
        st.info(_EMPTY)
    st.divider()
    _render_risk_coverage(payload)
    st.divider()
    _render_verdict()


main()
