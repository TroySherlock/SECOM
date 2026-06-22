"""Shared render helpers for the per-track model pages (Interpolation / Extrapolation)
and the Gates page. Each function takes the benchmark payload and a fixed track so the
pages stay thin wrappers with no track/protocol radios.
"""
from __future__ import annotations

import streamlit as st

from secom.costs import PROFILE_IDS, THRESHOLD_PROFILES
from secom.dashboard import render_blue_note
from secom.dashboard.charts import (
    C_PURPLE,
    fig_benchmark_leaderboard,
    fig_cv_vs_holdout_validation,
    fig_delta_bar,
    fig_holdout_confusion,
    fig_pr_curve_cv_holdout,
    fig_risk_coverage,
)
from secom.dashboard.data import (
    DELTA_METRIC_COLS,
    benchmark_has_multi_profile_thresholds,
    cv_leaderboard_blocked_df,
    cv_leaderboard_df,
    gate_conditional_df,
    gate_config,
    gate_lift_df,
    gate_risk_coverage_df,
    gate_vs_gate_df,
    holdout_auc_summary_df,
    holdout_confusion_by_profile,
    holdout_df,
    list_model_ids,
    load_benchmark_results,
    model_info,
    time_decay_meta,
)

# Gate identity per (track, gate) -> display label + flagged-count column stem.
GATE_LABELS = {
    "efa": "Regularized EFA → Hotelling T²",
    "bayes": "sBFA → BGM density",
}
GATE_FLAGGED = {"efa": "t2", "bayes": "density"}
from secom.dashboard.pr_curves import load_pr_curves

# label -> (mean_col, std_col, chart_title)
_CV_METRIC_SPECS_STRATIFIED: dict[str, tuple[str, str, str]] = {
    "PR-AUC": ("mean_pr_auc", "std_pr_auc", "Mean PR AUC (5×2 repeated stratified CV)"),
    "ROC-AUC": ("mean_roc_auc", "std_roc_auc", "Mean ROC AUC (5×2 repeated stratified CV)"),
}

_CV_METRIC_SPECS_BLOCKED: dict[str, tuple[str, str, str]] = {
    "PR-AUC": ("mean_pr_auc", "std_pr_auc", "Mean PR AUC (blocked time CV)"),
    "ROC-AUC": ("mean_roc_auc", "std_roc_auc", "Mean ROC AUC (blocked time CV)"),
}

_PROFILE_RADIO_LABELS = {pid: THRESHOLD_PROFILES[pid].display_name for pid in PROFILE_IDS}


@st.cache_data(show_spinner=False)
def load_payload() -> dict:
    return load_benchmark_results()


def _format_params(params: dict) -> str:
    if not params:
        return "_No tuned parameters recorded._"
    lines = [f"- `{k}`: `{v}`" for k, v in params.items()]
    return "\n".join(lines)


def _format_holdout_split_caption(split: dict) -> str | None:
    if not split:
        return None
    mode = split.get("split_mode")
    label = {
        "temporal": "Temporal holdout (latest 20% by time)",
        "random": "Random stratified holdout (in-distribution)",
    }.get(mode, "Holdout")
    train_fr = split.get("train_fail_rate")
    ho_fr = split.get("holdout_fail_rate")
    parts = [
        f"{label}: train {split.get('train_rows', '?')} rows "
        f"({split.get('train_ts_min', '?')} → {split.get('train_ts_max', '?')})",
        f"holdout {split.get('test_rows', '?')} rows "
        f"({split.get('holdout_ts_min', '?')} → {split.get('holdout_ts_max', '?')})",
    ]
    if train_fr is not None and ho_fr is not None:
        parts.append(
            f"fail rate train {100 * float(train_fr):.1f}% vs holdout {100 * float(ho_fr):.1f}%"
        )
    return " · ".join(parts)


def render_cv_leaderboard(payload: dict, *, blocked: bool) -> None:
    """Cross-validation leaderboard for one protocol (no protocol radio)."""
    suffix = "blocked" if blocked else "strat"
    protocol_cv_df = cv_leaderboard_blocked_df(payload) if blocked else cv_leaderboard_df(payload)
    metric_specs = _CV_METRIC_SPECS_BLOCKED if blocked else _CV_METRIC_SPECS_STRATIFIED

    st.subheader("Cross-validation leaderboard")
    if protocol_cv_df.empty:
        st.warning(
            "No blocked CV leaderboard rows in benchmark JSON. Re-run "
            "`python -m secom.cli.benchmark`."
            if blocked
            else "No stratified CV leaderboard rows in benchmark JSON."
        )
        return

    cv_metric = st.selectbox(
        "Evaluation Metric",
        list(metric_specs),
        key=f"cv_metric_{suffix}",
    )
    mean_col, std_col, chart_title = metric_specs[cv_metric]
    st.plotly_chart(
        fig_benchmark_leaderboard(
            protocol_cv_df,
            metric_col=mean_col,
            error_col=std_col,
            title=chart_title,
            marker_color=C_PURPLE,
        ),
        width="stretch",
        theme="streamlit",
        key=f"cv_leaderboard_{suffix}",
    )
    cols_to_show = ["pipeline", mean_col, std_col]
    display_cv = protocol_cv_df[
        [c for c in cols_to_show if c in protocol_cv_df.columns]
    ].copy()
    for col in display_cv.columns:
        if display_cv[col].dtype.kind == "f":
            display_cv[col] = display_cv[col].round(3)
    st.dataframe(display_cv, width="stretch", hide_index=True)
    st.caption(
        f"Rankings use mean {cv_metric} across all CV folds; error bars show ±1 SD."
    )


def render_holdout_validation(payload: dict, *, blocked: bool) -> None:
    """CV-vs-holdout validation chart + AUC summary for one protocol (no view radio)."""
    suffix = "blocked" if blocked else "strat"
    view_key = "holdout" if blocked else "holdout_random"
    split_key = "holdout_split" if blocked else "holdout_split_random"
    view_cv_df = cv_leaderboard_blocked_df(payload) if blocked else cv_leaderboard_df(payload)
    view_ho_df = holdout_df(payload, key=view_key)

    st.subheader("Holdout evaluation (reporting only)")
    split_caption = _format_holdout_split_caption(payload.get(split_key) or {})
    if split_caption:
        st.caption(split_caption)

    if view_ho_df.empty:
        st.warning(f"No `{view_key}` rows in benchmark JSON. Re-run the benchmark.")
        return
    if view_cv_df.empty:
        st.warning(
            "No blocked CV leaderboard in benchmark JSON. Re-run "
            "`python -m secom.cli.run_tuning` and `python -m secom.cli.benchmark`."
            if blocked
            else "No CV leaderboard rows in benchmark JSON."
        )
        return

    ctrl_col1, ctrl_col2 = st.columns([1, 1])
    with ctrl_col1:
        selected_metric = st.selectbox(
            "Evaluation Metric",
            ["PR-AUC", "ROC-AUC"],
            key=f"ho_metric_{suffix}",
        )
    with ctrl_col2:
        st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
        toggle_ci = st.checkbox(
            "Show Holdout 95% Bootstrap CIs",
            value=True,
            key=f"ho_ci_{suffix}",
        )

    st.plotly_chart(
        fig_cv_vs_holdout_validation(
            cv_df=view_cv_df,
            ho_df=view_ho_df,
            metric_type=selected_metric,
            show_ci=toggle_ci,
        ),
        width="stretch",
        theme="streamlit",
        key=f"validation_leaderboard_{suffix}",
    )
    cv_label = "blocked time CV folds" if blocked else "5×2 stratified folds"
    st.caption(
        f"Purple markers show CV mean ± 1 SD across {cv_label}; yellow diamonds are "
        "holdout point estimates; pale yellow bands are stratified bootstrap 95% CIs "
        "for PR-AUC and ROC-AUC."
    )
    if blocked:
        decay_meta = time_decay_meta(payload)
        lam_bits = [
            f"{mid}={float(m.get('decay_lambda', 0.0)):.2f}"
            for mid, m in decay_meta.items()
            if m.get("weight_capable")
        ]
        lam_txt = ", ".join(lam_bits) if lam_bits else "none"
        st.caption(
            "Forward view: blocked-tuned hyperparameters with exponential time-decay "
            f"sample weighting (tuned `decay_lambda`: {lam_txt})."
        )

    st.dataframe(holdout_auc_summary_df(view_ho_df), width="stretch", hide_index=True)


def render_model_deepdive(payload: dict, *, track: str) -> None:
    """Per-model architecture, tuned params, PR curve, threshold profiles, confusion."""
    is_extrap = track == "extrapolation"
    model_ids = list_model_ids(payload)
    tuned = payload.get("tuned_hyperparameters") or {}
    tuned_blocked = payload.get("tuned_hyperparameters_blocked") or {}

    st.subheader("Pipeline architecture & tuning")
    selected_id = st.selectbox(
        "Select pipeline",
        model_ids,
        format_func=lambda mid: model_info(mid).display_name,
        key=f"dd_model_{track}",
    )
    info = model_info(selected_id)
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown(f"### {info.display_name}")
        st.markdown(f"**Family:** {info.family}")
        st.markdown(f"**Classifier:** {info.classifier}")
        st.markdown(f"**Feature path:** {info.feature_path}")
        st.markdown(info.description)
        st.markdown(f"**Tuning notebook:** `{info.tuning_notebook}`")
    with right:
        if is_extrap:
            st.markdown("**Tuned hyperparameters (extrapolation / blocked CV)**")
            st.markdown(_format_params(tuned_blocked.get(selected_id, {})))
            decay_meta = time_decay_meta(payload).get(selected_id, {})
            if decay_meta.get("weight_capable"):
                st.markdown(
                    f"- `decay_lambda`: `{float(decay_meta.get('decay_lambda', 0.0)):.2f}` "
                    "(time-decay weighting)"
                )
        else:
            st.markdown("**Tuned hyperparameters (in-distribution / stratified CV)**")
            st.markdown(_format_params(tuned.get(selected_id, {})))

    st.markdown("#### Threshold profiles")
    st.markdown(
        "**F-beta tuning** (edit `F0_5_BETA` / `F2_BETA` / `F4_BETA` in "
        "`src/secom/costs.py`, then re-tune). **BER** minimises balanced error on the "
        "same threshold grid."
    )
    for _pid, prof in THRESHOLD_PROFILES.items():
        st.caption(f"**{prof.display_name}:** {prof.description}")

    if not benchmark_has_multi_profile_thresholds(payload):
        st.warning(
            "Tuned JSONs lack f0_5/f2/f4/ber `threshold_profiles`. Re-run Stage 2 tuning and "
            "`python -m secom.cli.benchmark` to populate holdout confusion matrices."
        )

    try:
        cv_curve, ho_curve, ber_point = load_pr_curves(selected_id, track)
    except FileNotFoundError as exc:
        st.info(str(exc))
    else:
        st.plotly_chart(
            fig_pr_curve_cv_holdout(
                cv_curve,
                ho_curve,
                ber_point=ber_point,
                title=f"Precision–recall — {info.display_name}",
            ),
            width="stretch",
            theme="streamlit",
            key=f"pr_curve_{track}_{selected_id}",
        )
    if is_extrap:
        st.caption(
            "Purple: blocked time CV OOF PR curve on validation blocks only (earliest "
            "train block has no OOF score under the expanding window). Yellow: temporal "
            "holdout PR curve with time-decay weighting. Blue dashed: random baseline "
            "(positive-class prevalence). Green diamond: BER-min threshold operating "
            "point on holdout."
        )
    else:
        st.caption(
            "Purple: 5×2 stratified CV out-of-fold PR curve (in-distribution). Yellow: "
            "random stratified holdout PR curve. Blue dashed: random baseline "
            "(positive-class prevalence). Green diamond: BER-min threshold operating "
            "point on holdout."
        )

    profile_choice = st.radio(
        "Threshold profile (holdout confusion matrix)",
        options=list(PROFILE_IDS),
        format_func=lambda pid: _PROFILE_RADIO_LABELS[pid],
        horizontal=True,
        key=f"profile_radio_{track}",
    )

    deepdive_ho_df = holdout_df(
        payload, key="holdout" if is_extrap else "holdout_random"
    )
    cms = holdout_confusion_by_profile(deepdive_ho_df, selected_id)
    cm = cms.get(profile_choice)
    ho_row = (
        deepdive_ho_df.loc[deepdive_ho_df["pipeline"] == selected_id].iloc[0]
        if not deepdive_ho_df.empty
        and selected_id in deepdive_ho_df["pipeline"].values
        else None
    )
    thr_val = None
    if ho_row is not None:
        thr_col = f"{profile_choice}_threshold"
        if thr_col in ho_row.index:
            thr_val = ho_row[thr_col]
    prof = THRESHOLD_PROFILES[profile_choice]
    thr_txt = f"threshold = {float(thr_val):.4f}" if thr_val is not None else ""
    st.markdown(f"**{prof.display_name}** — {thr_txt}")

    if cm is not None:
        st.plotly_chart(
            fig_holdout_confusion(cm, prof.display_name, height=320),
            width="stretch",
            theme="streamlit",
            key=f"cm_{track}_{selected_id}_{profile_choice}",
        )
    else:
        st.caption("Re-run `python -m secom.cli.benchmark` after tuning.")

    render_blue_note(
        "**F0.5 (conservative) thresholds** lean on well-ranked scores near the high-precision "
        "tail; all heads here are isotonic-calibrated, but the **elastic-net** and **Bayesian** "
        "linear heads can still miss more true fails (higher BER) under a stricter fail-class "
        "cutoff than the **Random Forest** head. **F2** is the default deploy profile; "
        "**BER** picks the symmetric misclassification minimum on the threshold grid."
    )


def render_gate_lift(payload: dict, *, track: str, metric: str) -> None:
    """Diverging delta bars: each gate's conditional-minus-global lift, plus EFA-vs-Bayes."""
    metric_col = DELTA_METRIC_COLS[metric]
    efa_lift = gate_lift_df(payload, track, "efa", metric_col)
    bayes_lift = gate_lift_df(payload, track, "bayes", metric_col)
    vs_df = gate_vs_gate_df(payload, track, metric_col)

    if efa_lift.empty and bayes_lift.empty:
        st.info(
            "No gate conditional metrics for this track in benchmark JSON. Re-run "
            "`python -m secom.cli.benchmark`."
        )
        return

    st.markdown("**Gate lift vs no gate** — conditional (kept wafers) minus global (all wafers)")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            fig_delta_bar(
                efa_lift,
                title=f"EFA → T²+Q gate lift ({metric})",
                value_label=f"conditional − global {metric}",
                positive_is_good=True,
            ),
            width="stretch",
            theme="streamlit",
            key=f"gate_lift_efa_{track}",
        )
    with right:
        st.plotly_chart(
            fig_delta_bar(
                bayes_lift,
                title=f"sBFA → BGM+Q gate lift ({metric})",
                value_label=f"conditional − global {metric}",
                positive_is_good=True,
            ),
            width="stretch",
            theme="streamlit",
            key=f"gate_lift_bayes_{track}",
        )
    st.caption(
        "Positive (green) = abstaining lifts conditional performance on the kept wafers. "
        "Bars sit inside wide CIs at ~17-20 fails; treat direction, not magnitude, as the signal."
    )

    if not vs_df.empty:
        st.markdown("**Gate vs gate** — EFA minus Bayes conditional metric")
        st.plotly_chart(
            fig_delta_bar(
                vs_df,
                title=f"EFA − Bayes conditional {metric}",
                value_label=f"EFA − Bayes {metric}",
                positive_is_good=True,
            ),
            width="stretch",
            theme="streamlit",
            key=f"gate_vs_gate_{track}",
        )
        st.caption(
            "Green = the EFA (T²) gate keeps a better-scoring set than the Bayes (BGM) gate on "
            "that model; red favours Bayes. This is which abstention rule wins, not whether either helps."
        )


def render_risk_coverage(payload: dict, *, track: str, gate: str, metric: str) -> None:
    """Risk-coverage severity sweep for one gate on one track."""
    rc_df = gate_risk_coverage_df(payload, track, gate)
    gate_title = GATE_LABELS.get(gate, gate)
    if rc_df is None or rc_df.empty:
        st.caption(f"No risk-coverage rows for the {gate_title} gate on this track.")
        return
    metric_key = DELTA_METRIC_COLS[metric]
    cond_df = gate_conditional_df(payload, track, gate)
    op_cov = None
    if not cond_df.empty and "coverage" in cond_df.columns and not cond_df["coverage"].dropna().empty:
        op_cov = float(cond_df["coverage"].dropna().iloc[0])
    st.plotly_chart(
        fig_risk_coverage(
            rc_df,
            metric=metric_key,
            operating_coverage=op_cov,
            title=f"{gate_title} risk–coverage ({metric})",
        ),
        width="stretch",
        theme="streamlit",
        key=f"risk_coverage_{track}_{gate}",
    )


def render_gate_section(payload: dict, *, track: str, gate: str) -> None:
    """Conditional-metrics table + config caption for one gate/track, inside an expander."""
    cond_df = gate_conditional_df(payload, track, gate)
    gate_meta = gate_config(payload, track, gate)
    gate_title = GATE_LABELS.get(gate, gate)
    flagged_label = GATE_FLAGGED.get(gate, "ooc")
    if cond_df.empty or not gate_meta:
        return

    logic_txt = str(gate_meta.get("logic", "or")).upper()
    with st.expander(f"{gate_title} — conditional metrics (CI detail)", expanded=False):
        q_ucl = gate_meta.get("q_ucl")
        q_txt = f"{float(q_ucl):.1f}" if q_ucl is not None else "?"
        base_caption = (
            f"Raw post-cluster sensors (smart_corr threshold "
            f"{gate_meta.get('gate_corr_threshold', '?')}); "
            f"fit on {gate_meta.get('n_reference_wafers', '?')} passing train wafers, "
            f"{gate_meta.get('n_features', '?')} features → "
            f"{gate_meta.get('n_factors', '?')} factors. "
        )
        if gate == "bayes":
            d_lcl = gate_meta.get("density_lcl")
            d_txt = f"{float(d_lcl):.2f}" if d_lcl is not None else "?"
            st.caption(
                base_caption
                + f"{gate_meta.get('n_mixture_components', '?')}-component BGM "
                f"(seed-ensemble n={gate_meta.get('n_seeds', '?')}). "
                f"Abstain when BGM log-density < {d_txt} "
                f"(α={gate_meta.get('density_alpha', '?')}) "
                f"**{logic_txt}** Q/SPE > {q_txt} (α={gate_meta.get('q_alpha', '?')})."
            )
        else:
            t2_ucl = gate_meta.get("t2_ucl")
            t2_txt = f"{float(t2_ucl):.1f}" if t2_ucl is not None else "?"
            st.caption(
                base_caption
                + f"Abstain when Hotelling T² > {t2_txt} "
                f"(α={gate_meta.get('t2_alpha', '?')}) "
                f"**{logic_txt}** Q/SPE > {q_txt} (α={gate_meta.get('q_alpha', '?')})."
            )
        gate_cols = [
            c
            for c in [
                "pipeline",
                "coverage",
                "conditional_pr_auc",
                "conditional_pr_auc_ci_low",
                "conditional_pr_auc_ci_high",
                "global_pr_auc",
                "conditional_roc_auc",
                "global_roc_auc",
                "n_in_control",
                f"n_flagged_{flagged_label}",
                "n_flagged_q",
                "n_flagged_both",
                "n_fails_in_control",
                "n_fails_flagged_ooc",
                f"n_fails_flagged_{flagged_label}",
                "n_fails_flagged_q",
            ]
            if c in cond_df.columns
        ]
        gate_display = cond_df[gate_cols].copy()
        for col in gate_display.select_dtypes(include="float").columns:
            gate_display[col] = gate_display[col].round(3)
        st.dataframe(gate_display, width="stretch", hide_index=True)
        st.caption(
            "`coverage` = fraction of holdout wafers in control (scored by the model). "
            f"{logic_txt} logic: flagged if either statistic trips per the rule. "
            "`conditional_pr_auc` is on in-control wafers only (None when "
            "fewer than 5 in-control fails). Read beside `n_flagged_*` and "
            "`n_fails_flagged_*` — gains can come from dropping easy negatives."
        )
