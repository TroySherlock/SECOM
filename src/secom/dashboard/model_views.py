"""Shared render helpers for the per-track model pages (Interpolation / Extrapolation)
and the Gates page. Each function takes the benchmark payload and a fixed track so the
pages stay thin wrappers with no track/protocol radios.
"""
from __future__ import annotations

import streamlit as st

from secom.costs import (
    BER_BAND_TOLERANCE,
    ESCAPE_OVERKILL_COST_RATIO,
    PROFILE_IDS,
    THRESHOLD_PROFILES,
    catch_overkill_from_confusion,
)
from secom.dashboard import render_blue_note
from secom.dashboard.charts import (
    C_PURPLE,
    fig_benchmark_leaderboard,
    fig_calibration,
    fig_catch_overkill_curve,
    fig_cv_vs_holdout_validation,
    fig_delta_bar,
    fig_expected_cost_curve,
    fig_holdout_confusion,
    fig_pr_curve_clean,
    fig_risk_coverage,
)
from secom.dashboard.data import (
    DELTA_METRIC_COLS,
    benchmark_has_multi_profile_thresholds,
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
    operating_table_df,
    resolved_threshold_profile_config,
)
from secom.dashboard.pr_curves import load_model_scores, load_pr_curves

# Gate identity per (track, gate) -> display label + flagged-count column stem.
GATE_LABELS = {
    "efa": "Regularized EFA → Hotelling T²",
    "bayes": "sBFA → BGM density",
}
GATE_FLAGGED = {"efa": "t2", "bayes": "density"}

# label -> (mean_col, std_col, chart_title)
_CV_METRIC_SPECS_STRATIFIED: dict[str, tuple[str, str, str]] = {
    "PR-AUC": ("mean_pr_auc", "std_pr_auc", "Mean PR AUC (5×2 repeated stratified CV)"),
    "ROC-AUC": ("mean_roc_auc", "std_roc_auc", "Mean ROC AUC (5×2 repeated stratified CV)"),
}

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


def render_cv_leaderboard(payload: dict) -> None:
    """In-distribution 5×2 stratified CV leaderboard (the single tuning protocol)."""
    protocol_cv_df = cv_leaderboard_df(payload)
    metric_specs = _CV_METRIC_SPECS_STRATIFIED

    st.subheader("Cross-validation leaderboard")
    if protocol_cv_df.empty:
        st.warning("No stratified CV leaderboard rows in benchmark JSON.")
        return

    cv_metric = st.selectbox(
        "Evaluation Metric",
        list(metric_specs),
        key="cv_metric_strat",
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
        key="cv_leaderboard_strat",
    )
    cols_to_show = ["pipeline", mean_col, std_col]
    display_cv = protocol_cv_df[
        [c for c in cols_to_show if c in protocol_cv_df.columns]
    ].copy()
    for col in display_cv.columns:
        if display_cv[col].dtype.kind == "f":
            display_cv[col] = display_cv[col].round(3)
    with st.expander("CV leaderboard table", expanded=False):
        st.dataframe(display_cv, width="stretch", hide_index=True)
        st.caption(
            f"Rankings use mean {cv_metric} across all CV folds; error bars show ±1 SD."
        )


def render_holdout_validation(payload: dict, *, track: str) -> None:
    """In-distribution CV reference vs this track's holdout (no view radio).

    Both tracks reference the single 5×2 stratified CV (the one tuning protocol);
    interpolation compares it to the random holdout, extrapolation to the temporal
    forward holdout, so the purple-to-yellow gap reads as the drift cost.
    """
    is_extrap = track == "extrapolation"
    suffix = "extrap" if is_extrap else "interp"
    view_key = "holdout" if is_extrap else "holdout_random"
    split_key = "holdout_split" if is_extrap else "holdout_split_random"
    view_cv_df = cv_leaderboard_df(payload)
    view_ho_df = holdout_df(payload, key=view_key)

    st.subheader("Holdout evaluation (reporting only)")
    split_caption = _format_holdout_split_caption(payload.get(split_key) or {})
    if split_caption:
        st.caption(split_caption)

    if view_ho_df.empty:
        st.warning(f"No `{view_key}` rows in benchmark JSON. Re-run the benchmark.")
        return
    if view_cv_df.empty:
        st.warning("No CV leaderboard rows in benchmark JSON.")
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
    st.caption(
        "Purple markers show in-distribution 5×2 stratified CV mean ± 1 SD; yellow "
        "diamonds are holdout point estimates; pale yellow bands are stratified "
        "bootstrap 95% CIs for PR-AUC and ROC-AUC."
    )
    if is_extrap:
        st.caption(
            "Forward view: the same in-distribution-tuned hyperparameters, refit on the "
            "temporal train and scored unweighted on the later holdout. The purple-to-yellow "
            "gap is the drift cost; recency weighting is explored in the Time-decay sweep tab."
        )

    with st.expander("Holdout metrics table (PR-AUC / ROC-AUC / BER, with 95% CIs)", expanded=False):
        st.dataframe(holdout_auc_summary_df(view_ho_df), width="stretch", hide_index=True)
        st.caption(
            "Point estimates with stratified bootstrap 95% CIs. BER is the balanced error rate at "
            "the default BER-min operating threshold; lower is better."
        )


def render_model_deepdive(payload: dict, *, track: str) -> None:
    """Per-model architecture, tuned params, PR curve, threshold profiles, confusion."""
    is_extrap = track == "extrapolation"
    model_ids = list_model_ids(payload)
    tuned = payload.get("tuned_hyperparameters") or {}

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
        st.markdown("**Tuned hyperparameters (in-distribution / stratified CV)**")
        st.markdown(_format_params(tuned.get(selected_id, {})))
        if is_extrap:
            st.caption(
                "Same in-distribution-tuned params reused for the temporal forward "
                "holdout (no separate temporal tuning)."
            )

    deepdive_ho_df = holdout_df(
        payload, key="holdout" if is_extrap else "holdout_random"
    )
    ho_row = (
        deepdive_ho_df.loc[deepdive_ho_df["pipeline"] == selected_id].iloc[0]
        if not deepdive_ho_df.empty
        and selected_id in deepdive_ho_df["pipeline"].values
        else None
    )

    def _row_val(col: str) -> float | None:
        if ho_row is None or col not in ho_row.index:
            return None
        val = ho_row[col]
        return None if val is None else float(val)

    scores = load_model_scores(selected_id, track)
    ho_scores = scores.get("holdout") or {}
    ho_y = ho_scores.get("y_true")
    ho_s = ho_scores.get("y_score")
    cms = holdout_confusion_by_profile(deepdive_ho_df, selected_id)

    tab_dive, tab_thresh = st.tabs(["Deep-dive", "Thresholding (cost system)"])

    with tab_dive:
        _render_deepdive_tab(
            info,
            track=track,
            is_extrap=is_extrap,
            selected_id=selected_id,
            ap=_row_val("pr_auc"),
            ap_ci=(_row_val("pr_auc_ci_low"), _row_val("pr_auc_ci_high")),
            ber_threshold=_row_val("ber_threshold"),
            ber_percent=_row_val("ber_ber_percent"),
            ber_tpr=_row_val("ber_true_positive_percent"),
            ber_tnr=_row_val("ber_true_negative_percent"),
            ber_cm=cms.get("ber"),
            cal_scores=scores.get("holdout" if is_extrap else "cv") or {},
        )

    with tab_thresh:
        _render_thresholding_tab(
            payload,
            info,
            track=track,
            selected_id=selected_id,
            deepdive_ho_df=deepdive_ho_df,
            ho_y=ho_y,
            ho_s=ho_s,
            profile_thresholds={
                THRESHOLD_PROFILES[pid].display_name: _row_val(f"{pid}_threshold")
                for pid in PROFILE_IDS
            },
            cms=cms,
        )


def _render_deepdive_tab(
    info,
    *,
    track: str,
    is_extrap: bool,
    selected_id: str,
    ap: float | None,
    ap_ci: tuple[float | None, float | None],
    ber_threshold: float | None,
    ber_percent: float | None,
    ber_tpr: float | None,
    ber_tnr: float | None,
    ber_cm,
    cal_scores: dict,
) -> None:
    """Deep-dive tab: clean PR curve, calibration, and the BER-min operating point."""
    st.markdown("#### Precision–recall")
    try:
        cv_curve, ho_curve, ber_point = load_pr_curves(selected_id, track)
    except FileNotFoundError as exc:
        st.info(str(exc))
        cv_curve = ho_curve = ber_point = None
    line_curve = ho_curve if is_extrap else cv_curve
    line_name = "Holdout (temporal)" if is_extrap else "CV out-of-fold (in-distribution)"
    baseline = ho_curve.baseline if ho_curve is not None else None
    st.plotly_chart(
        fig_pr_curve_clean(
            line_curve,
            ap=ap,
            ap_ci=ap_ci,
            ber_point=ber_point,
            baseline=baseline,
            draw_line=line_curve is not None,
            line_name=line_name,
            title=f"Precision–recall — {info.display_name}",
        ),
        width="stretch",
        theme="streamlit",
        key=f"pr_curve_{track}_{selected_id}",
    )
    if is_extrap:
        st.caption(
            "Yellow: the single temporal-forward holdout PR step-line (the only forward data; "
            "jagged at ~17-20 fails by nature). Blue dashed: prevalence baseline. Green diamond: "
            "BER-min operating point. Holdout AP with 95% bootstrap CI is annotated."
        )
    else:
        st.caption(
            "Purple: 5×2 stratified CV out-of-fold PR curve (the well-sampled, trustworthy shape). "
            "Blue dashed: prevalence baseline. Green diamond: BER-min operating point. The random "
            "holdout AP (with 95% bootstrap CI) is annotated rather than drawn as a second jagged line."
        )

    st.markdown("#### Calibration")
    cal_y = cal_scores.get("y_true")
    cal_s = cal_scores.get("y_score")
    if cal_y is not None and len(cal_y):
        st.plotly_chart(
            fig_calibration(cal_y, cal_s, title=f"Calibration — {info.display_name}"),
            width="stretch",
            theme="streamlit",
            key=f"calibration_{track}_{selected_id}",
        )
        source = (
            "the temporal forward holdout (so it shows calibration **under drift**)"
            if is_extrap
            else "the in-distribution CV out-of-fold scores (the well-sampled reliability check)"
        )
        st.caption(
            f"Reliability curve on {source}; quantile bins of predicted fail probability vs the "
            "observed fail fraction. On the diagonal = well-calibrated. Axes are zoomed to the "
            "predicted-probability range (imbalanced + isotonic-calibrated -> most probabilities "
            "are small); the faint histogram shows where that mass sits. Brier (lower = better) is in the title."
        )
    else:
        st.info(
            "Per-wafer scores are not in the report cache yet. Re-run "
            "`python -m secom.benchmark` to populate the calibration diagram."
        )

    st.markdown("#### Operating point (BER-min)")
    st.caption(
        "BER-min is the **default deploy** threshold (symmetric pass/fail cost). The fab "
        "trade-off lives in the **Thresholding** tab, where the cost-optimal *economic* point sits."
    )
    if ber_threshold is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Threshold", f"{ber_threshold:.4f}")
        m2.metric("BER", f"{ber_percent:.1f}%" if ber_percent is not None else "n/a")
        m3.metric("TPR (recall)", f"{ber_tpr:.1f}%" if ber_tpr is not None else "n/a")
        m4.metric("TNR", f"{ber_tnr:.1f}%" if ber_tnr is not None else "n/a")
    if ber_cm is not None:
        st.plotly_chart(
            fig_holdout_confusion(ber_cm, "BER-min", height=300),
            width="stretch",
            theme="streamlit",
            key=f"cm_bermin_{track}_{selected_id}",
        )
        fab = catch_overkill_from_confusion(ber_cm)
        if fab:
            st.caption(
                f"In fab terms: **catch {100 * fab['catch_rate']:.0f}%** of fails "
                f"({fab['fails_caught']}/{fab['fails_total']}) for "
                f"**{100 * fab['overkill_rate']:.0f}% overkill** "
                f"({fab['good_flagged']}/{fab['good_total']} good wafers flagged). "
                f"Precision is **{100 * fab['precision']:.0f}%** - low by design at this prevalence, "
                "so read flags as risk triage, not a precise gate."
            )
    else:
        st.caption("Re-run `python -m secom.benchmark` after tuning to populate the confusion matrix.")


def _render_thresholding_tab(
    payload: dict,
    info,
    *,
    track: str,
    selected_id: str,
    deepdive_ho_df,
    ho_y,
    ho_s,
    profile_thresholds: dict,
    cms: dict,
) -> None:
    """Thresholding tab: fab-economics operating points (catch vs overkill, cost-optimal)."""
    cfg = resolved_threshold_profile_config(payload) or {}
    tol = float(cfg.get("ber_band_tolerance", BER_BAND_TOLERANCE))
    cost_ratio = float(cfg.get("escape_overkill_cost_ratio", ESCAPE_OVERKILL_COST_RATIO))

    st.markdown(
        "A fab does not deploy on BER. It weighs two outcomes with very different price tags: "
        f"an **escape** (a failing wafer we pass -> ships a bad die, ~**{cost_ratio:g}x** the cost) "
        "versus an **overkill** (a good wafer we flag -> a re-test / hold). So the numbers that "
        "matter are **catch rate** (fraction of real fails flagged = recall/TPR) and **overkill "
        "rate** (fraction of good wafers flagged = 1 - TNR). We ship four operating points:"
    )
    st.markdown(
        "- **Conservative** (high threshold): fewest flags -> lowest overkill, lowest catch.\n"
        "- **BER-min** (balanced): symmetric pass/fail optimum; the default deploy threshold.\n"
        "- **Aggressive** (low threshold): recall-leaning -> higher catch, more overkill.\n"
        f"- **Economic** (cost-optimal): minimises expected cost at an explicit "
        f"**escape:overkill = {cost_ratio:g}:1** ratio. Because escapes dominate, it lands "
        "left of BER-min (higher catch, more overkill) - the point a fab would actually run."
    )
    render_blue_note(
        f"**Robustness, not a knife-edge:** conservative / BER-min / aggressive all sit within "
        f"{tol:g} BER points of optimal, so performance is **flat across this whole threshold "
        "range** - a plateau, not a spike. Small threshold drift does not blow up the model. "
        "All thresholds are chosen on the leakage-safe **CV** curve and applied unchanged to the holdout."
    )

    if not benchmark_has_multi_profile_thresholds(payload):
        st.warning(
            "Tuned JSONs lack the full `threshold_profiles` (conservative/ber/aggressive/economic). "
            "Re-run Stage 2 tuning (`python -m secom.cli.run_tuning --threshold-only`) and "
            "`python -m secom.benchmark`."
        )

    st.markdown("#### Catch rate vs overkill")
    if ho_y is not None and len(ho_y):
        st.plotly_chart(
            fig_catch_overkill_curve(
                ho_y,
                ho_s,
                profile_thresholds=profile_thresholds,
                title=f"Catch vs overkill — {info.display_name}",
            ),
            width="stretch",
            theme="streamlit",
            key=f"catch_overkill_{track}_{selected_id}",
        )
        st.caption(
            "The fab-vocabulary operating curve (an ROC relabelled): every point is one threshold. "
            "Up = catch more real fails; right = flag more good wafers. The four markers are the "
            "tuned operating points - economic sits up-and-right of BER-min (more catch, more overkill)."
        )
    else:
        st.info(
            "Per-wafer holdout scores are not in the report cache yet. Re-run "
            "`python -m secom.benchmark` to populate the operating curves."
        )

    st.markdown("#### Operating points")
    op_df = operating_table_df(deepdive_ho_df, selected_id)
    if not op_df.empty:
        st.dataframe(op_df, width="stretch", hide_index=True)
        st.caption(
            f"Holdout operating point per profile in fab terms: catch rate, overkill rate, precision, "
            f"and the raw fails-caught / good-flagged counts. **Economic** is the recall-leaning "
            f"cost-optimal point at {cost_ratio:g}:1. **Precision is low by design** here "
            "(~7% prevalence, PR-AUC ~0.2): most flags are good wafers, so treat a flag as **risk "
            "triage** - an enriched pool to inspect - not a precise gate. BER (right) is the "
            "symmetric, prevalence-free summary kept for model comparison."
        )

    st.markdown("#### Expected cost vs threshold")
    if ho_y is not None and len(ho_y):
        st.plotly_chart(
            fig_expected_cost_curve(
                ho_y,
                ho_s,
                cost_ratio=cost_ratio,
                profile_thresholds=profile_thresholds,
                title=f"Expected cost ({cost_ratio:g}:1) — {info.display_name}",
            ),
            width="stretch",
            theme="streamlit",
            key=f"expected_cost_{track}_{selected_id}",
        )
        st.caption(
            f"Expected per-wafer cost at escape:overkill = {cost_ratio:g}:1 "
            "(overkill-units: cost_ratio x prevalence x miss-rate + (1 - prevalence) x overkill-rate). "
            "**Log x-axis** (zoomed to the operating region); each **colour-coded dotted line is one "
            "operating point** (see legend, colours match the catch-vs-overkill chart), and the green "
            "diamond is the empirical cost minimum. A **shallow basin** = the cost-optimal point is "
            "robust to threshold drift; the curve shape is illustrative on this holdout."
        )

    st.markdown("#### Confusion matrix per operating point")
    cm_cols = st.columns(len(PROFILE_IDS))
    for col, pid in zip(cm_cols, PROFILE_IDS):
        with col:
            cm = cms.get(pid)
            if cm is not None:
                st.plotly_chart(
                    fig_holdout_confusion(cm, THRESHOLD_PROFILES[pid].display_name, height=260),
                    width="stretch",
                    theme="streamlit",
                    key=f"cm_{track}_{selected_id}_{pid}",
                )
            else:
                st.caption(f"{THRESHOLD_PROFILES[pid].display_name}: n/a")


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
