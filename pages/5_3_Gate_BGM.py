"""Gates 5.3 - sBFA → BGM gate with Tier-1 drift monitor (distribution shift + control chart)."""
from __future__ import annotations

import numpy as np
import streamlit as st

from secom.dashboard import render_blue_note
from secom.dashboard.charts import (
    fig_factor_drift,
    fig_gate_control_chart,
    fig_gate_statistic_distributions,
    fig_sbfa_factor_space,
    fig_sbfa_loadings_heatmap,
)
from secom.dashboard.data import (
    factor_drift_ranking,
    gate_diagnostics,
    gate_drift_stats,
    sbfa_diagnostics,
)
from secom.dashboard.model_views import EFA_VS_SBFA_ONE_LINER, load_payload

_GATE = "bayes"

# Statistic radio label -> (array key, limit key, limit side, OOC-mask key).
_STATISTIC_SPECS = {
    "Log-density": ("density", "density_lcl", "lower", "density_ooc"),
    "Q / SPE": ("q", "q_ucl", "upper", "q_ooc"),
}


def _arr(block: dict, key: str) -> np.ndarray:
    return np.asarray((block or {}).get(key, []), dtype=float)


def _render_distribution_shift(payload: dict, *, stat_key: str, limit_key: str, limit_side: str) -> None:
    interp = gate_diagnostics(payload, "interpolation", _GATE)
    extrap = gate_diagnostics(payload, "extrapolation", _GATE)
    if not interp and not extrap:
        st.info(
            "No frozen BGM diagnostics in benchmark JSON. Re-run `python -m secom.benchmark`."
        )
        return

    left, right = st.columns(2)
    with left:
        ref = (interp.get("reference") or {})
        ho = (interp.get("holdout") or {})
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
            key=f"p53_dist_interp_{stat_key}",
        )
    with right:
        ref = (extrap.get("reference") or {})
        ho = (extrap.get("holdout") or {})
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
            key=f"p53_dist_extrap_{stat_key}",
        )
    st.caption(
        "Each gate compared to its own passing-train reference (BGM log-density is only comparable "
        "within one fit). The in-distribution holdout (left) hugs the reference; under drift the "
        "temporal holdout (right) pushes past the control limit - the well-sampled drift signal."
    )
    st.warning(
        "BGM log-density is only comparable **within one fitted gate** - do not read the absolute "
        "density value across the interpolation and extrapolation panels; compare each holdout only "
        "against its own reference."
    )


def _render_control_chart(payload: dict, *, stat_key: str, limit_key: str, limit_side: str, ooc_key: str) -> None:
    extrap = gate_diagnostics(payload, "extrapolation", _GATE)
    ho = (extrap.get("holdout") or {})
    values = _arr(ho, stat_key)
    if values.size == 0:
        st.info("No temporal BGM diagnostics in benchmark JSON. Re-run `python -m secom.benchmark`.")
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
        key=f"p53_control_{stat_key}",
    )
    st.caption(
        "Wafers in measurement-time order on the temporal holdout. Points beyond the limit are "
        "out-of-control; the solid line is the trailing rolling OOC rate (the faint dotted line is "
        "the cumulative rate). A rolling rate climbing above its baseline into the latest era is "
        "the gate's drift / retraining trigger - exactly how a fab uses MSPC."
    )
    st.warning(
        "The limit is fixed from passing-train quantiles (α = density 0.03 / Q 0.005), so a ~3% "
        "density / ~0.5% Q in-control false-alarm rate is expected **by construction** - read the "
        "trend and the excess over that baseline, not the raw out-of-control count."
    )


def _render_drift_scalar(payload: dict) -> None:
    stats = gate_drift_stats(payload, gate=_GATE, track="extrapolation")
    if stats.empty:
        st.info("No frozen BGM diagnostics in benchmark JSON. Re-run `python -m secom.benchmark`.")
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
        "shifts, so report the effect size (KS / AUC), not just significance."
    )


def _render_factor_space(payload: dict, sbfa: dict, ranking) -> None:
    if not sbfa or ranking is None or ranking.empty:
        st.info("No frozen sBFA artifacts in benchmark JSON. Re-run `python -m secom.benchmark`.")
        return
    top = ranking["factor_idx"].tolist()
    dx = int(top[0])
    dy = int(top[1]) if len(top) > 1 else (dx + 1)
    bgm = sbfa.get("bgm") or {}
    st.plotly_chart(
        fig_sbfa_factor_space(
            np.asarray(sbfa.get("reference_scores", []), dtype=float),
            np.asarray(sbfa.get("holdout_scores", []), dtype=float),
            np.asarray(sbfa.get("y_true", []), dtype=int),
            np.asarray(sbfa.get("flagged", []), dtype=bool),
            dims=(dx, dy),
            bgm_means=np.asarray(bgm.get("means", []), dtype=float) if bgm.get("means") else None,
            bgm_covariances=np.asarray(bgm.get("covariances", []), dtype=float) if bgm.get("covariances") else None,
            factor_labels=(f"Factor {dx + 1}", f"Factor {dy + 1}"),
        ),
        width="stretch",
        theme="streamlit",
        key="p53_factor_space",
    )
    st.caption(
        "The two top-drifting sBFA factors. Ellipses are the BGM components' 2σ envelope (the "
        "learned 'normal operating' region), faint points are the passing-train reference, and "
        "colored points are the temporal holdout split into pass, flagged pass, caught fail, and "
        "missed fail - so you can see which fails the gate's OOC region actually catches. Drift = "
        "the temporal cloud sliding off the envelope."
    )
    st.warning(
        "This is **one representative seed** (`members_[0]`); factor axes are rotation- and "
        "sign-ambiguous and not comparable across runs or to the EFA gate. It is a qualitative "
        "geometry view - the gate's actual decision uses the 8-D ensemble density + Q, not this 2-D picture."
    )


def _render_root_cause(sbfa: dict, ranking) -> None:
    if not sbfa or ranking is None or ranking.empty:
        st.info("No frozen sBFA artifacts in benchmark JSON. Re-run `python -m secom.benchmark`.")
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
            key="p53_factor_drift",
        )
    with right:
        st.plotly_chart(
            fig_sbfa_loadings_heatmap(
                np.asarray(sbfa.get("loadings", []), dtype=float),
                list(sbfa.get("feature_names", [])),
                factor_order=ranking["factor_idx"].astype(int).tolist(),
                highlight_factor=int(top["factor_idx"]),
            ),
            width="stretch",
            theme="streamlit",
            key="p53_loadings",
        )
    st.caption(
        "Factors ranked by in-distribution→temporal KS on their scores; the heatmap shows the "
        "Laplace-sparse loadings (sensor × factor, strongest sensors only). The heavy-loading "
        "sensors of the top-drifting factor are the candidate drifting subsystem."
    )
    st.warning(
        "Loadings are correlational and Laplace-sparsified, **not causal** - read this as 'where to "
        "look first', not a proven root cause. Sign is arbitrary, so the magnitude (|loading|) is "
        "what matters."
    )


def main() -> None:
    st.title("5.3 sBFA → BGM gate")
    st.caption(
        "Sparse Bayesian factor analysis (NumPyro, ADVI) → Bayesian Gaussian mixture density + "
        "Q/SPE residual, in the raw post-cluster sensor space, fit on passing-train wafers. Low "
        "BGM log-density or high Q trips the gate."
    )

    try:
        payload = load_payload()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    render_blue_note(
        "**Bayesian drift monitor.** The control statistics below are computed over **all** "
        "holdout wafers (not just the ~17-20 fails), so the distribution shift and out-of-control "
        "rate are statistically solid - unlike the noisy conditional-AUC curve. This is the gate "
        "demonstrating, in sensor space, that the forward window has drifted out of control."
    )
    render_blue_note(EFA_VS_SBFA_ONE_LINER)

    statistic = st.radio(
        "Control statistic",
        list(_STATISTIC_SPECS),
        horizontal=True,
        key="p53_statistic",
    )
    stat_key, limit_key, limit_side, ooc_key = _STATISTIC_SPECS[statistic]

    st.subheader("1. Control-statistic distribution shift")
    _render_distribution_shift(
        payload, stat_key=stat_key, limit_key=limit_key, limit_side=limit_side
    )

    st.subheader("2. MSPC control chart (temporal holdout)")
    _render_control_chart(
        payload, stat_key=stat_key, limit_key=limit_key, limit_side=limit_side, ooc_key=ooc_key
    )

    st.divider()
    st.subheader("3. A real drift number")
    _render_drift_scalar(payload)

    st.divider()
    st.markdown("### Bayesian magic: the sBFA latent space")
    sbfa = sbfa_diagnostics(payload, track="extrapolation", gate=_GATE)
    ranking = factor_drift_ranking(sbfa)

    st.subheader("4. sBFA latent factor space")
    _render_factor_space(payload, sbfa, ranking)

    st.subheader("5. Which subsystem is drifting (root cause)")
    _render_root_cause(sbfa, ranking)

    render_blue_note(
        "**5.3 is a Bayesian sensor-space drift monitor.** Trust the population drift evidence "
        "(sections 1-3); treat the latent-space and loadings views (sections 4-5) as interpretable "
        "diagnostics from a single representative fit. Conditional yield lift lives on **5.4 Gate "
        "comparison**, where the confidence intervals are wide (~17-20 holdout fails)."
    )


main()
