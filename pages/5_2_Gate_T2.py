"""Gates 5.2 - Hotelling T² gate (regularized EFA → T² + Q) drift monitor.

Mirrors 5.3 (sBFA → BGM) for the frequentist EFA gate: distribution shift, MSPC
control chart, an honest drift scalar, an EFA factor-space scatter with a
Hotelling control ellipse, and a dense-loadings root-cause heatmap. Conditional
performance and risk-coverage live on 5.4 Gate comparison.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from secom.dashboard import render_blue_note
from secom.dashboard.charts import (
    fig_efa_factor_space,
    fig_factor_drift,
    fig_gate_control_chart,
    fig_gate_statistic_distributions,
    fig_sbfa_loadings_heatmap,
)
from secom.dashboard.data import (
    efa_factor_diagnostics,
    factor_drift_ranking,
    gate_diagnostics,
    gate_drift_stats,
)
from secom.dashboard.model_views import load_payload

_GATE = "efa"

# Statistic radio label -> (array key, limit key, limit side, OOC-mask key).
_STATISTIC_SPECS = {
    "Hotelling T²": ("t2", "t2_ucl", "upper", "t2_ooc"),
    "Q / SPE": ("q", "q_ucl", "upper", "q_ooc"),
}
# Drift-scalar statistics (holdout key -> label).
_DRIFT_STATS = [("t2", "Hotelling T²"), ("q", "Q / SPE")]


def _arr(block: dict, key: str) -> np.ndarray:
    return np.asarray((block or {}).get(key, []), dtype=float)


def _render_distribution_shift(payload: dict, *, stat_key: str, limit_key: str, limit_side: str) -> None:
    interp = gate_diagnostics(payload, "interpolation", _GATE)
    extrap = gate_diagnostics(payload, "extrapolation", _GATE)
    if not interp and not extrap:
        st.info("No frozen EFA diagnostics in benchmark JSON. Re-run `python -m secom.benchmark`.")
        return

    left, right = st.columns(2)
    with left:
        ref = interp.get("reference") or {}
        ho = interp.get("holdout") or {}
        st.plotly_chart(
            fig_gate_statistic_distributions(
                {"reference": _arr(ref, stat_key), "in_distribution": _arr(ho, stat_key)},
                statistic=stat_key,
                limit=(interp.get("limits") or {}).get(limit_key),
                limit_side=limit_side,
                title="Interpolation gate (in-distribution)",
            ),
            width="stretch",
            theme="streamlit",
            key=f"p52_dist_interp_{stat_key}",
        )
    with right:
        ref = extrap.get("reference") or {}
        ho = extrap.get("holdout") or {}
        st.plotly_chart(
            fig_gate_statistic_distributions(
                {"reference": _arr(ref, stat_key), "temporal": _arr(ho, stat_key)},
                statistic=stat_key,
                limit=(extrap.get("limits") or {}).get(limit_key),
                limit_side=limit_side,
                title="Extrapolation gate (temporal)",
            ),
            width="stretch",
            theme="streamlit",
            key=f"p52_dist_extrap_{stat_key}",
        )
    st.caption(
        "T² and Q are both upper-tail (high = out of control); each gate is compared to its own "
        "passing-train reference. The in-distribution holdout (left) hugs the reference; under "
        "drift the temporal holdout (right) pushes past the control limit - the well-sampled signal."
    )
    st.warning(
        "T² is a Mahalanobis distance in the **fitted factor subspace** - comparable only within "
        "one fit, not across the interpolation and extrapolation panels."
    )


def _render_control_chart(payload: dict, *, stat_key: str, limit_key: str, limit_side: str, ooc_key: str) -> None:
    extrap = gate_diagnostics(payload, "extrapolation", _GATE)
    ho = extrap.get("holdout") or {}
    values = _arr(ho, stat_key)
    if values.size == 0:
        st.info("No temporal EFA diagnostics in benchmark JSON. Re-run `python -m secom.benchmark`.")
        return
    limit = (extrap.get("limits") or {}).get(limit_key)
    ooc_mask = np.asarray(ho.get(ooc_key, []), dtype=bool)
    ts = ho.get("ts")
    st.plotly_chart(
        fig_gate_control_chart(
            values,
            limit=float(limit) if limit is not None else float("nan"),
            ooc_mask=ooc_mask,
            x=np.asarray(ts) if ts else None,
            statistic=stat_key,
            limit_side=limit_side,
            title=f"Temporal holdout control chart - {stat_key}",
        ),
        width="stretch",
        theme="streamlit",
        key=f"p52_control_{stat_key}",
    )
    st.caption(
        "Wafers in measurement-time order on the temporal holdout. Points beyond the upper limit "
        "are out-of-control; the solid line is the trailing rolling OOC rate (the faint dotted line "
        "is the cumulative rate). A rolling rate climbing above its baseline into the latest era is "
        "the gate's drift / retraining trigger - exactly how a fab uses MSPC."
    )
    st.warning(
        "Limits are passing-train quantiles (α = T² 0.03 / Q 0.005), so a ~3% / ~0.5% in-control "
        "false-alarm rate is expected **by construction** - read the trend and the excess over that "
        "baseline, not the raw out-of-control count."
    )


def _render_drift_scalar(payload: dict) -> None:
    stats = gate_drift_stats(payload, gate=_GATE, track="extrapolation", stats=_DRIFT_STATS)
    if stats.empty:
        st.info("No frozen EFA diagnostics in benchmark JSON. Re-run `python -m secom.benchmark`.")
        return
    st.caption(
        "An honest in-control drift scalar: the 2-sample Kolmogorov–Smirnov distance (max gap "
        "between the CDFs) and the separability AUC (how well the statistic alone tells the two "
        "apart; 0.5 = no drift), computed on the passing-train reference vs the **passing** wafers "
        "of the temporal holdout (fails removed), so it isolates sensor/process drift from yield. "
        "The per-statistic n below shows the wafer counts used."
    )
    cols = st.columns(len(stats))
    for col, (_, row) in zip(cols, stats.iterrows()):
        with col:
            st.markdown(f"**{row['statistic']}**")
            st.metric("KS distance", f"{row['ks_distance']:.3f}")
            st.metric("Separability AUC", f"{row['separability_auc']:.3f}")
            st.caption(f"KS p = {row['ks_pvalue']:.1e} · n = {int(row['n_reference'])}/{int(row['n_holdout'])}")
    st.warning(
        "This measures **sensor/process distribution drift, not yield-prediction skill** - a high KS "
        "does not imply the gate catches fails. With n this large the p-value is tiny even for small "
        "shifts, so report the effect size (KS / AUC). On SECOM most forward-window signal lands in "
        "Q, not T²."
    )


def _render_factor_space(factor: dict, ranking) -> None:
    if not factor or ranking is None or ranking.empty:
        st.info("No frozen EFA factor artifacts in benchmark JSON. Re-run `python -m secom.benchmark`.")
        return
    top = ranking["factor_idx"].tolist()
    dx = int(top[0])
    dy = int(top[1]) if len(top) > 1 else (dx + 1)
    st.plotly_chart(
        fig_efa_factor_space(
            np.asarray(factor.get("reference_scores", []), dtype=float),
            np.asarray(factor.get("holdout_scores", []), dtype=float),
            np.asarray(factor.get("y_true", []), dtype=int),
            np.asarray(factor.get("flagged", []), dtype=bool),
            dims=(dx, dy),
            score_mean=np.asarray(factor.get("score_mean", []), dtype=float) if factor.get("score_mean") else None,
            score_cov=np.asarray(factor.get("score_cov", []), dtype=float) if factor.get("score_cov") else None,
            t2_alpha=float(factor.get("t2_alpha", 0.03)),
            factor_labels=(f"Factor {dx + 1}", f"Factor {dy + 1}"),
        ),
        width="stretch",
        theme="streamlit",
        key="p52_factor_space",
    )
    st.caption(
        "The two top-drifting EFA factors. The ellipse is the bivariate χ² (Hotelling) control "
        "region at α; faint points are the passing-train reference, colored points are the temporal "
        "holdout split into pass, flagged pass, caught fail, and missed fail - so you can see which "
        "fails the gate's OOC region actually catches. Drift = the cloud sliding outside the region."
    )
    st.warning(
        "Frequentist EFA, **single fit**; factor axes are rotation- and sign-ambiguous and not "
        "comparable across runs or to the BGM gate. The 2-D ellipse approximates the full-k T² "
        "limit - it is a qualitative geometry view, not the gate's actual decision boundary."
    )


def _render_root_cause(factor: dict, ranking) -> None:
    if not factor or ranking is None or ranking.empty:
        st.info("No frozen EFA factor artifacts in benchmark JSON. Re-run `python -m secom.benchmark`.")
        return
    top = ranking.iloc[0]
    st.markdown(
        f"Drift concentrates in **factor {int(top['factor'])}** (KS = {top['ks_distance']:.3f}), "
        f"which loads most heavily on **{top['top_sensors'] or 'n/a'}**."
    )
    left, right = st.columns([2, 3])
    with left:
        st.plotly_chart(
            fig_factor_drift(ranking),
            width="stretch",
            theme="streamlit",
            key="p52_factor_drift",
        )
    with right:
        st.plotly_chart(
            fig_sbfa_loadings_heatmap(
                np.asarray(factor.get("loadings", []), dtype=float),
                list(factor.get("feature_names", [])),
                factor_order=ranking["factor_idx"].astype(int).tolist(),
                highlight_factor=int(top["factor_idx"]),
                title="Regularized EFA loadings (sensor × factor)",
            ),
            width="stretch",
            theme="streamlit",
            key="p52_loadings",
        )
    st.caption(
        "Factors ranked by in-distribution→temporal KS on their scores; the heatmap shows the EFA "
        "loadings (sensor × factor, strongest sensors only). The heavy-loading sensors of the "
        "top-drifting factor are the candidate drifting subsystem."
    )
    st.warning(
        "EFA loadings are **dense** (regularized, not Laplace-sparse like the sBFA gate), so several "
        "sensors load on each factor. They are correlational, not causal - read this as 'where to "
        "look first'; magnitude (|loading|) matters, sign is arbitrary."
    )


def main() -> None:
    st.title("5.2 Hotelling T² gate")
    st.caption(
        "Regularized exploratory factor analysis on the raw post-cluster sensors → Hotelling T² "
        "(in-subspace excursions) + Q/SPE residual (structural breaks), fit on passing-train "
        "wafers. Abstain when either statistic exceeds its upper control limit."
    )

    try:
        payload = load_payload()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    render_blue_note(
        "**Frequentist drift monitor.** T² captures drift **along** the learned factor directions; "
        "Q/SPE captures novelty **orthogonal** to them. The control statistics below are computed "
        "over **all** holdout wafers (not just the ~17-20 fails), so the distribution shift and "
        "out-of-control rate are statistically solid - the trustworthy evidence the forward window "
        "has drifted out of control."
    )

    statistic = st.radio(
        "Control statistic",
        list(_STATISTIC_SPECS),
        horizontal=True,
        key="p52_statistic",
    )
    stat_key, limit_key, limit_side, ooc_key = _STATISTIC_SPECS[statistic]

    st.subheader("1. Control-statistic distribution shift")
    _render_distribution_shift(payload, stat_key=stat_key, limit_key=limit_key, limit_side=limit_side)

    st.subheader("2. MSPC control chart (temporal holdout)")
    _render_control_chart(
        payload, stat_key=stat_key, limit_key=limit_key, limit_side=limit_side, ooc_key=ooc_key
    )

    st.divider()
    st.subheader("3. A real drift number")
    _render_drift_scalar(payload)

    st.divider()
    st.markdown("### The EFA latent space")
    factor = efa_factor_diagnostics(payload, track="extrapolation")
    ranking = factor_drift_ranking(factor)

    st.subheader("4. EFA latent factor space")
    _render_factor_space(factor, ranking)

    st.subheader("5. Which subsystem is drifting (root cause)")
    _render_root_cause(factor, ranking)

    render_blue_note(
        "**5.2 is a frequentist sensor-space drift monitor.** Trust the population drift evidence "
        "(sections 1-3); treat the latent-space and loadings views (sections 4-5) as interpretable "
        "diagnostics from a single EFA fit. Conditional yield lift lives on **5.4 Gate comparison**, "
        "where the confidence intervals are wide (~17-20 holdout fails)."
    )


main()
