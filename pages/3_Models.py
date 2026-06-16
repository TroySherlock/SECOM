"""Models benchmark page: CV leaderboard and holdout reporting."""
from __future__ import annotations

import streamlit as st

from secom.dashboard import render_blue_note
from secom.dashboard.data import (
    benchmark_has_multi_profile_thresholds,
    cv_leaderboard_blocked_df,
    cv_leaderboard_df,
    holdout_comparison_df,
    holdout_conditional_df,
    holdout_confusion_by_profile,
    holdout_auc_summary_df,
    holdout_df,
    list_model_ids,
    load_benchmark_results,
    model_info,
    process_gate_meta,
    time_decay_meta,
)
from secom.dashboard.charts import (
    C_PURPLE,
    fig_benchmark_leaderboard,
    fig_cv_vs_holdout_validation,
    fig_holdout_confusion,
    fig_pr_curve_cv_holdout,
)
from secom.dashboard.pr_curves import load_pr_curves
from secom.costs import PROFILE_IDS, THRESHOLD_PROFILES

_CV_METRIC_SPECS_STRATIFIED: dict[str, tuple[str, str, str]] = {
    "PR-AUC": ("mean_pr_auc", "std_pr_auc", "Mean PR AUC (5×2 repeated stratified CV)"),
    "ROC-AUC": ("mean_roc_auc", "std_roc_auc", "Mean ROC AUC (5×2 repeated stratified CV)"),
}

_CV_METRIC_SPECS_BLOCKED: dict[str, tuple[str, str, str]] = {
    "PR-AUC": ("mean_pr_auc", "std_pr_auc", "Mean PR AUC (blocked time CV)"),
    "ROC-AUC": ("mean_roc_auc", "std_roc_auc", "Mean ROC AUC (blocked time CV)"),
}

_PROFILE_RADIO_LABELS = {pid: THRESHOLD_PROFILES[pid].display_name for pid in PROFILE_IDS}

# Holdout evaluation views: label -> (holdout key, split-meta key, use_blocked_cv)
_HOLDOUT_VIEWS: dict[str, tuple[str, str | None, bool]] = {
    "Forward (temporal)": ("holdout", "holdout_split", True),
    "In-distribution (random)": ("holdout_random", "holdout_split_random", False),
}


@st.cache_data(show_spinner=False)
def _load_payload() -> dict:
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


def main() -> None:
    st.title("Models & benchmark results")
    st.caption(
        "Four tuned pipelines compared on repeated CV and a held-out test split, "
        "with F0.5 / F2 / F4 F-beta thresholds and a BER-minimizing cutoff."
    )

    try:
        payload = _load_payload()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    cv_df = cv_leaderboard_df(payload)
    cv_blocked_df = cv_leaderboard_blocked_df(payload)
    ho_df = holdout_df(payload)
    model_ids = list_model_ids(payload)
    tuned = payload.get("tuned_hyperparameters") or {}
    tuned_blocked = payload.get("tuned_hyperparameters_blocked") or {}

    render_blue_note(
        "**In-distribution** evaluation uses **5×2 stratified CV** and a **random stratified "
        "holdout** (interpolation upper bound). **Extrapolation** uses **blocked time CV** with "
        "local stratification, **exponential time-decay sample weighting** (recent wafers "
        "weighted more, `decay_lambda` tuned on blocked CV), and a **temporal forward holdout** "
        "(latest 20% by time). A **process gate** (post-cluster Hotelling T² **OR** "
        "Isolation Forest on passing train wafers) abstains on out-of-control wafers; "
        "we report **conditional PR-AUC** and **coverage** on the wafers it scores. "
        "All holdout metrics are reporting-only "
        f"(`holdout_is_reporting_only={payload.get('holdout_is_reporting_only', True)}`). "
        "The random-minus-temporal gap is the cost of extrapolation."
    )

    tab_cv, tab_holdout, tab_model = st.tabs(
        [
            "CV leaderboard (5×2)",
            "Holdout reporting",
            "Model deep-dive",
        ]
    )

    with tab_cv:
        st.subheader("Cross-validation leaderboard")
        if cv_df.empty:
            st.warning("No CV leaderboard rows in benchmark JSON.")
        else:
            cv_metric = st.selectbox(
                "Evaluation Metric",
                list(_CV_METRIC_SPECS_STRATIFIED),
                key="p3_cv_metric_select",
            )
            mean_col, std_col, chart_title = _CV_METRIC_SPECS_STRATIFIED[cv_metric]
            st.plotly_chart(
                fig_benchmark_leaderboard(
                    cv_df,
                    metric_col=mean_col,
                    error_col=std_col,
                    title=chart_title,
                    marker_color=C_PURPLE,
                ),
                width="stretch",
                theme="streamlit",
                key="p3_cv_leaderboard",
            )
            cols_to_show = ["pipeline", mean_col, std_col]
            display_cv = cv_df[[c for c in cols_to_show if c in cv_df.columns]].copy()
            for col in display_cv.columns:
                if col.startswith("mean_") or col.startswith("std_"):
                    if display_cv[col].dtype.kind == "f":
                        display_cv[col] = display_cv[col].round(3)
            st.dataframe(display_cv, width="stretch", hide_index=True)
            st.caption(
                f"Rankings use mean {cv_metric} across all CV folds; error bars show ±1 SD."
            )

    with tab_holdout:
        st.subheader("Holdout evaluation (reporting only)")

        comparison_df = holdout_comparison_df(payload)
        if not comparison_df.empty:
            st.markdown("**Interpolation vs extrapolation (PR-AUC)**")
            st.dataframe(comparison_df, width="stretch", hide_index=True)
            st.caption(
                "`cv_pr_auc` = 5×2 stratified CV (in-distribution protocol); "
                "`cv_blocked_pr_auc` = blocked time CV (extrapolation protocol); "
                "`random_pr_auc` is an optimistic interpolation upper bound; "
                "`temporal_pr_auc` is the forward test with time-decay weighting; "
                "`interpolation_gap` = random − temporal."
            )

        view_label = st.radio(
            "Evaluation view",
            options=list(_HOLDOUT_VIEWS),
            horizontal=True,
            key="p3_ho_view",
        )
        view_key, split_key, use_blocked_cv = _HOLDOUT_VIEWS[view_label]
        view_cv_df = cv_blocked_df if use_blocked_cv else cv_df
        view_ho_df = holdout_df(payload, key=view_key)
        split_meta = payload.get(split_key) if split_key else None
        split_caption = _format_holdout_split_caption(split_meta or {})
        if split_caption:
            st.caption(split_caption)

        if view_ho_df.empty:
            st.warning(f"No `{view_key}` rows in benchmark JSON. Re-run the benchmark.")
        elif view_cv_df.empty:
            st.warning(
                "No blocked CV leaderboard in benchmark JSON. Re-run "
                "`python -m secom.cli.run_tuning` and `python -m secom.cli.benchmark`."
                if use_blocked_cv
                else "No CV leaderboard rows in benchmark JSON."
            )
        else:
            ctrl_col1, ctrl_col2 = st.columns([1, 1])
            with ctrl_col1:
                selected_metric = st.selectbox(
                    "Evaluation Metric",
                    ["PR-AUC", "ROC-AUC"],
                    key="p3_ho_metric_select",
                )
            with ctrl_col2:
                st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
                toggle_ci = st.checkbox(
                    "Show Holdout 95% Bootstrap CIs",
                    value=True,
                    key="p3_ho_ci_toggle",
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
                key="p3_validation_leaderboard",
            )
            cv_label = (
                "blocked time CV folds"
                if use_blocked_cv
                else "5×2 stratified folds"
            )
            st.caption(
                f"Purple markers show CV mean ± 1 SD across {cv_label}; yellow diamonds are "
                "holdout point estimates; pale yellow bands are stratified bootstrap 95% CIs "
                "for PR-AUC and ROC-AUC."
            )
            if use_blocked_cv:
                decay_meta = time_decay_meta(payload)
                lam_bits = [
                    f"{mid}={float(m.get('decay_lambda', 0.0)):.2f}"
                    for mid, m in decay_meta.items()
                    if m.get("weight_capable")
                ]
                lam_txt = ", ".join(lam_bits) if lam_bits else "none"
                st.caption(
                    "Forward view: blocked-tuned hyperparameters with exponential time-decay "
                    f"sample weighting (tuned `decay_lambda`: {lam_txt}; k-NN unweighted)."
                )

            st.dataframe(holdout_auc_summary_df(view_ho_df), width="stretch", hide_index=True)

            cond_key = "holdout_conditional" if use_blocked_cv else "holdout_random_conditional"
            cond_df = holdout_conditional_df(payload, key=cond_key)
            gate_meta = process_gate_meta(payload)
            if not cond_df.empty and gate_meta:
                st.markdown("**Process gate (T² OR Isolation Forest) — conditional metrics**")
                ucl = gate_meta.get("ucl")
                if_thr = gate_meta.get("if_threshold")
                ucl_txt = f"{float(ucl):.1f}" if ucl is not None else "?"
                if_txt = f"{float(if_thr):.3f}" if if_thr is not None else "?"
                st.caption(
                    f"Post-cluster features (smart_corr threshold "
                    f"{gate_meta.get('gate_corr_threshold', '?')}); "
                    f"fit on {gate_meta.get('n_reference_wafers', '?')} passing train wafers, "
                    f"{gate_meta.get('n_features', '?')} features. "
                    f"Abstain when T² > {ucl_txt} (α={gate_meta.get('t2_alpha', '?')}) "
                    f"**OR** IF score < {if_txt} (α={gate_meta.get('if_alpha', '?')}). "
                    f"IF n_estimators={gate_meta.get('if_n_estimators', '?')}."
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
                        "n_in_control",
                        "n_flagged_t2",
                        "n_flagged_if",
                        "n_flagged_both",
                        "n_fails_in_control",
                        "n_fails_flagged_ooc",
                        "n_fails_flagged_t2",
                        "n_fails_flagged_if",
                    ]
                    if c in cond_df.columns
                ]
                gate_display = cond_df[gate_cols].copy()
                for col in gate_display.select_dtypes(include="float").columns:
                    gate_display[col] = gate_display[col].round(3)
                st.dataframe(gate_display, width="stretch", hide_index=True)
                st.caption(
                    "`coverage` = fraction of holdout wafers in control (scored by the model). "
                    "OR logic: flagged if **either** T² or IF trips. "
                    "`conditional_pr_auc` is on in-control wafers only (None when "
                    "fewer than 5 in-control fails). Read beside `n_flagged_*` and "
                    "`n_fails_flagged_*` — gains can come from dropping easy negatives."
                )

    with tab_model:
        st.subheader("Pipeline architecture & tuning")
        selected_id = st.selectbox(
            "Select pipeline",
            model_ids,
            format_func=lambda mid: model_info(mid).display_name,
            key="p3_model_select",
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
            st.markdown("**Tuned hyperparameters (extrapolation / blocked CV)**")
            st.markdown(_format_params(tuned_blocked.get(selected_id, {})))
            decay_meta = time_decay_meta(payload).get(selected_id, {})
            if decay_meta.get("weight_capable"):
                st.markdown(
                    f"- `decay_lambda`: `{float(decay_meta.get('decay_lambda', 0.0)):.2f}` "
                    "(time-decay weighting)"
                )
            else:
                st.markdown("- `decay_lambda`: _unweighted (k-NN)_")
            st.markdown("**Tuned hyperparameters (in-distribution / stratified CV)**")
            st.markdown(_format_params(tuned.get(selected_id, {})))

        st.markdown("#### Threshold profiles")
        st.markdown(
            "**F-beta tuning** (edit `F0_5_BETA` / `F2_BETA` / `F4_BETA` in "
            "`src/secom/costs.py`, then re-tune). **BER** minimises balanced error on the "
            "same threshold grid."
        )
        for pid, prof in THRESHOLD_PROFILES.items():
            st.caption(f"**{prof.display_name}:** {prof.description}")

        if not benchmark_has_multi_profile_thresholds(payload):
            st.warning(
                "Tuned JSONs lack f0_5/f2/f4/ber `threshold_profiles`. Re-run Stage 2 tuning and "
                "`python -m secom.cli.benchmark` to populate holdout confusion matrices."
            )

        cv_curve, ho_curve, ber_point = load_pr_curves(selected_id)
        st.plotly_chart(
            fig_pr_curve_cv_holdout(
                cv_curve,
                ho_curve,
                ber_point=ber_point,
                title=f"Precision–recall — {info.display_name}",
            ),
            width="stretch",
            theme="streamlit",
            key=f"p3_pr_curve_{selected_id}",
        )
        st.caption(
            "Purple: blocked time CV OOF PR curve on validation blocks only (earliest train "
            "block has no OOF score under the expanding window). Yellow: temporal holdout PR "
            "curve with time-decay weighting. Blue dashed: random baseline "
            "(positive-class prevalence). Green diamond: BER-min threshold operating point "
            "on holdout."
        )

        profile_choice = st.radio(
            "Threshold profile (holdout confusion matrix)",
            options=list(PROFILE_IDS),
            format_func=lambda pid: _PROFILE_RADIO_LABELS[pid],
            horizontal=True,
            key="p3_profile_radio",
        )

        cms = holdout_confusion_by_profile(ho_df, selected_id)
        cm = cms.get(profile_choice)
        ho_row = (
            ho_df.loc[ho_df["pipeline"] == selected_id].iloc[0]
            if not ho_df.empty and selected_id in ho_df["pipeline"].values
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
                key=f"p3_cm_{selected_id}_{profile_choice}",
            )
        else:
            st.caption("Re-run `python -m secom.cli.benchmark` after tuning.")

        render_blue_note(
            "**F0.5 (conservative) thresholds** often hurt **Linear LR** and **k-NN**: "
            "their scores are less well-calibrated than tree models, so a stricter fail-class "
            "threshold misses more true fails (higher BER) while **Random Forest** and **XGBoost** "
            "retain ranking under conservative cutoffs. **F2** is the default deploy profile; "
            "**BER** picks the symmetric misclassification minimum on the threshold grid."
        )


main()
