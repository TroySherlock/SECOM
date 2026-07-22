"""Hotelling T² gate (standard PCA → T² + Q) drift monitor.

The fab-standard MSPC baseline (mirrors the custom sBFA → BGM gate): PCA on the
raw post-cluster sensors, then distribution shift, an MSPC control chart, an
honest drift scalar, and a PCA component-space scatter with a Hotelling control
ellipse. Conditional performance and risk-coverage live on Gate comparison.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from secom.dashboard import render_blue_note, render_caveat
from secom.dashboard.charts import (
    fig_gate_control_chart,
    fig_gate_statistic_distributions,
    fig_pca_component_space,
)
from secom.dashboard.components import PCA_VS_SBFA_ONE_LINER, load_payload
from secom.dashboard.data import (
    factor_drift_ranking,
    gate_config,
    gate_diagnostics,
    gate_drift_stats,
    pca_component_diagnostics,
)

_GATE = "pca"

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
        st.info("No frozen PCA diagnostics in benchmark JSON. Re-run `python -m secom.benchmark`.")
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
    render_caveat(
        "T² is a Mahalanobis distance in the **fitted PCA subspace** - comparable only within "
        "one fit, not across the interpolation and extrapolation panels."
    )


def _render_control_chart(payload: dict, *, stat_key: str, limit_key: str, limit_side: str, ooc_key: str) -> None:
    extrap = gate_diagnostics(payload, "extrapolation", _GATE)
    ho = extrap.get("holdout") or {}
    values = _arr(ho, stat_key)
    if values.size == 0:
        st.info("No temporal PCA diagnostics in benchmark JSON. Re-run `python -m secom.benchmark`.")
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
    cfg = gate_config(payload, "extrapolation", _GATE)
    t2_a = cfg.get("t2_alpha", 0.03)
    q_a = cfg.get("q_alpha", 0.005)
    render_caveat(
        f"Limits are passing-train quantiles (α = T² {t2_a:g} / Q {q_a:g}), so a "
        f"~{t2_a:.1%} / ~{q_a:.1%} in-control false-alarm rate is expected **by construction** - "
        "read the trend and the excess over that baseline, not the raw out-of-control count. The "
        "quantiles are in-sample (taken on the same reference the PCA is fit on), so the realized "
        "out-of-sample false-alarm rate can run slightly higher. These are empirical quantiles "
        "rather than textbook F / χ² limits because the score distributions are visibly "
        "non-Gaussian."
    )


def _render_drift_scalar(payload: dict) -> None:
    stats = gate_drift_stats(payload, gate=_GATE, track="extrapolation", stats=_DRIFT_STATS)
    if stats.empty:
        st.info("No frozen PCA diagnostics in benchmark JSON. Re-run `python -m secom.benchmark`.")
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
    render_caveat(
        "This measures **sensor/process distribution drift, not yield-prediction skill** - a high KS "
        "does not imply the gate catches fails. With n this large the p-value is tiny even for small "
        "shifts, so report the effect size (KS / AUC). On SECOM most forward-window signal lands in "
        "Q, not T²."
    )


def _render_factor_space(factor: dict, ranking) -> None:
    if not factor or ranking is None or ranking.empty:
        st.info("No frozen PCA component artifacts in benchmark JSON. Re-run `python -m secom.benchmark`.")
        return
    top = ranking["factor_idx"].tolist()
    dx = int(top[0])
    dy = int(top[1]) if len(top) > 1 else (dx + 1)
    st.plotly_chart(
        fig_pca_component_space(
            np.asarray(factor.get("reference_scores", []), dtype=float),
            np.asarray(factor.get("holdout_scores", []), dtype=float),
            np.asarray(factor.get("y_true", []), dtype=int),
            np.asarray(factor.get("flagged", []), dtype=bool),
            dims=(dx, dy),
            score_mean=np.asarray(factor.get("score_mean", []), dtype=float) if factor.get("score_mean") else None,
            score_cov=np.asarray(factor.get("score_cov", []), dtype=float) if factor.get("score_cov") else None,
            t2_alpha=float(factor.get("t2_alpha", 0.03)),
            factor_labels=(f"Component {dx + 1}", f"Component {dy + 1}"),
        ),
        width="stretch",
        theme="streamlit",
        key="p52_factor_space",
    )
    st.caption(
        "The two top-drifting PCA components. The ellipse is the bivariate χ² (Hotelling) control "
        "region at α; faint points are the passing-train reference, colored points are the temporal "
        "holdout split into pass, flagged pass, caught fail, and missed fail - so you can see which "
        "fails the gate's OOC region actually catches. Drift = the cloud sliding outside the region."
    )
    render_caveat(
        "Standard PCA, **single fit**; component axes are sign-ambiguous and not comparable across "
        "runs or to the BGM gate. The 2-D ellipse approximates the full-k T² limit - it is a "
        "qualitative geometry view, not the gate's actual decision boundary."
    )


def main() -> None:
    st.title("Hotelling T² gate (PCA, fab standard)")
    st.caption(
        "Standard PCA-MSPC on the raw post-cluster sensors → Hotelling T² (in-subspace excursions) "
        "+ Q/SPE residual (structural breaks). Feature pruning and scaling are fit on the full "
        "training set; PCA and control limits are fit on passing-train wafers. Abstain when either "
        "statistic exceeds its upper control limit. This is the fab-standard baseline for Gate comparison."
    )

    try:
        payload = load_payload()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    render_blue_note(
        "**Fab-standard drift monitor.** T² captures drift **along** the learned PCA component "
        "directions; Q/SPE captures novelty **orthogonal** to them. The control statistics below are "
        "computed over **all** holdout wafers (not just the ~17-20 fails), so the distribution shift "
        "and out-of-control rate are statistically solid - the trustworthy evidence the forward "
        "window has drifted out of control."
    )
    render_blue_note(PCA_VS_SBFA_ONE_LINER)

    statistic = st.radio(
        "Control statistic",
        list(_STATISTIC_SPECS),
        horizontal=True,
        key="p52_statistic",
    )
    stat_key, limit_key, limit_side, ooc_key = _STATISTIC_SPECS[statistic]

    st.subheader("📊 1. Control-statistic distribution shift")
    _render_distribution_shift(payload, stat_key=stat_key, limit_key=limit_key, limit_side=limit_side)

    st.subheader("📈 2. MSPC control chart (temporal holdout)")
    _render_control_chart(
        payload, stat_key=stat_key, limit_key=limit_key, limit_side=limit_side, ooc_key=ooc_key
    )

    st.divider()
    st.subheader("🔢 3. A real drift number")
    _render_drift_scalar(payload)

    st.divider()
    st.markdown("### The PCA latent space")
    factor = pca_component_diagnostics(payload, track="extrapolation")
    ranking = factor_drift_ranking(factor)

    st.subheader("🗺️ 4. PCA component space")
    _render_factor_space(factor, ranking)

    render_blue_note(
        "**This is the fab-standard sensor-space drift monitor.** Trust the population drift evidence "
        "(sections 1-3); treat the component-space view (section 4) as an interpretable diagnostic "
        "from a single PCA fit. The PCA gate uses 10 components as a comparable-dimensional baseline "
        "beside the 8-factor sBFA gate, not as a variance-explained-tuned optimum. For the sensor-level root cause use the custom gate's sparse loadings "
        "on **sBFA → BGM gate**; conditional yield lift and the head-to-head verdict live on **Gate comparison**."
    )


main()
