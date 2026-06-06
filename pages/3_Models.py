"""Models benchmark page: CV leaderboard and holdout reporting."""
from __future__ import annotations

import json

import streamlit as st

from secom.dashboard import render_blue_note
from secom.dashboard.data import (
    benchmark_has_multi_profile_thresholds,
    cv_leaderboard_df,
    holdout_by_profile_df,
    holdout_confusion_by_profile,
    holdout_df,
    list_model_ids,
    load_benchmark_results,
    load_threshold_curves,
    merged_comparison_df,
    model_info,
    profile_threshold_summary_table,
    resolved_threshold_profile_config,
)
from secom.utils import load_tuned_params
from secom.dashboard.charts import (
    C_PURPLE,
    fig_benchmark_leaderboard,
    fig_ber_cv_vs_holdout,
    fig_cv_vs_holdout_scatter,
    fig_holdout_by_profile,
    fig_holdout_confusion,
    fig_threshold_objective_curves,
)
from secom.costs import PROFILE_IDS, THRESHOLD_PROFILES
from secom.pipelines import N_REPEATS, N_SPLITS, PRIMARY_TUNING_METRIC



def _metric_with_ci(
    point: float,
    ci_low: float | None,
    ci_high: float | None,
    *,
    fmt: str = ".3f",
    suffix: str = "",
) -> tuple[str, str | None]:
    """Format point estimate with optional bootstrap CI for st.metric."""
    label = f"{point:{fmt}}{suffix}"
    if ci_low is None or ci_high is None:
        return label, None
    help_text = f"95% bootstrap CI: [{ci_low:{fmt}}, {ci_high:{fmt}}]{suffix}"
    return label, help_text


@st.cache_data(show_spinner=False)
def _load_payload() -> dict:
    return load_benchmark_results()


def _format_params(params: dict) -> str:
    if not params:
        return "_No tuned parameters recorded._"
    lines = [f"- `{k}`: `{v}`" for k, v in params.items()]
    return "\n".join(lines)


def main() -> None:
    st.title("Models & benchmark results")
    st.caption(
        "Four tuned pipelines compared on repeated CV and a held-out test split, "
        "each with F1 (conservative), F2 (neutral), and F3 (aggressive) F-beta thresholds."
    )

    try:
        payload = _load_payload()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    cv_df = cv_leaderboard_df(payload)
    ho_df = holdout_df(payload)
    merged = merged_comparison_df(payload)
    model_ids = list_model_ids(payload)
    holdout_split = payload.get("holdout_split") or {}
    tuned = payload.get("tuned_hyperparameters") or {}

    render_blue_note(
        "**5×5 CV benchmark** ranks models using mean metrics from repeated stratified folds "
        "(used for tuning and comparison). **Holdout** is a single 20% test split, "
        f"reporting only (`holdout_is_reporting_only={payload.get('holdout_is_reporting_only', True)}`) — "
        "not used to select hyperparameters."
    )

    tab_cv, tab_holdout, tab_cost, tab_model, tab_compare = st.tabs(
        [
            "CV leaderboard (5×5)",
            "Holdout reporting",
            "Threshold profiles (F1/F2/F3)",
            "Model deep-dive",
            "CV vs holdout",
        ]
    )

    with tab_cv:
        st.subheader("Cross-validation leaderboard")
        if cv_df.empty:
            st.warning("No CV leaderboard rows in benchmark JSON.")
        else:
            st.plotly_chart(
                fig_benchmark_leaderboard(
                    cv_df,
                    metric_col="mean_pr_auc",
                    error_col="std_pr_auc",
                    title="Mean PR AUC (5×5 repeated stratified CV)",
                    marker_color=C_PURPLE,
                ),
                width="stretch",
                theme="streamlit",
                key="p3_cv_leaderboard",
            )
            display_cv = cv_df.copy()
            for col in display_cv.columns:
                if col.startswith("mean_") or col.startswith("std_"):
                    if display_cv[col].dtype.kind == "f":
                        display_cv[col] = display_cv[col].round(3)
            st.dataframe(display_cv, width="stretch", hide_index=True)
            st.caption("Rankings use mean PR AUC across all CV folds.")

    with tab_holdout:
        st.subheader("Holdout evaluation (reporting only)")
        if ho_df.empty:
            st.warning("No holdout rows in benchmark JSON.")
        else:
            st.plotly_chart(
                fig_benchmark_leaderboard(
                    ho_df,
                    metric_col="pr_auc",
                    title="Holdout PR AUC (20% test split)",
                ),
                width="stretch",
                theme="streamlit",
                key="p3_holdout_leaderboard",
            )
            cm_cols = [c for c in ho_df.columns if c.endswith("_confusion_matrix") or c == "confusion_matrix"]
            ho_display = ho_df.drop(columns=cm_cols, errors="ignore").copy()
            for col in ho_display.select_dtypes(include="float").columns:
                ho_display[col] = ho_display[col].round(3)
            st.dataframe(ho_display, width="stretch", hide_index=True)
            if "pr_auc_ci_low" in ho_df.columns:
                st.caption(
                    "Holdout PR AUC / BER include stratified bootstrap 95% CIs "
                    "(median + ci_low/ci_high columns; no refit per draw)."
                )

            if "confusion_matrix" in ho_df.columns:
                st.markdown("---")
                st.subheader("Confusion matrix")
                selected = st.selectbox(
                    "Pipeline",
                    ho_df["pipeline"].tolist(),
                    key="p3_holdout_cm_select",
                )
                row = ho_df.loc[ho_df["pipeline"] == selected].iloc[0]
                cm = row["confusion_matrix"]
                if isinstance(cm, str):
                    cm = json.loads(cm)
                st.plotly_chart(
                    fig_holdout_confusion(cm, selected),
                    width="stretch",
                    theme="streamlit",
                    key="p3_holdout_confusion",
                )

    with tab_cost:
        st.subheader("Threshold profiles (F-beta)")
        profile_cfg = resolved_threshold_profile_config(payload)
        st.markdown(
            "**F-beta tuning** (edit `F1_BETA` / `F2_BETA` / `F3_BETA` in "
            "`scripts/secom_costs.py`, then re-tune). Custom β sliders may be added later."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("F1 β (conservative)", f"{profile_cfg.get('f1_beta', 1)}", border=True)
        c2.metric("F2 β (neutral, default)", f"{profile_cfg.get('f2_beta', 2)}", border=True)
        c3.metric("F3 β (aggressive)", f"{profile_cfg.get('f3_beta', 3)}", border=True)
        c4.metric("Deploy profile", str(profile_cfg.get("default_profile", "f2")), border=True)

        for pid, prof in THRESHOLD_PROFILES.items():
            st.caption(f"**{prof.display_name}:** {prof.description}")

        if not benchmark_has_multi_profile_thresholds(payload):
            st.warning(
                "Tuned JSONs lack f1/f2/f3 `threshold_profiles`. Re-run Stage 2 tuning and "
                "`python -m secom.cli.benchmark` to populate F-score curves and holdout metrics."
            )

        curve_model = st.selectbox(
            "Model for objective curves",
            model_ids,
            format_func=lambda mid: model_info(mid).display_name,
            key="p3_cost_curve_model",
        )
        curves_df = load_threshold_curves(curve_model)
        if curves_df.empty:
            render_blue_note("No `objective_curves` in tuned JSON for this model yet.")
        else:
            try:
                tuned_full = load_tuned_params(curve_model)
                profiles = tuned_full.get("threshold_profiles") or {}
                best_thresholds = {
                    pid: float(profiles[pid]["best_threshold"])
                    for pid in profiles
                    if pid in profiles and "best_threshold" in profiles[pid]
                }
            except FileNotFoundError:
                best_thresholds = {}
            st.plotly_chart(
                fig_threshold_objective_curves(
                    curves_df,
                    best_thresholds,
                    title=f"CV mean F-score vs threshold — {curve_model}",
                ),
                width="stretch",
                theme="streamlit",
                key="p3_threshold_curves",
            )

        st.markdown("---")
        st.subheader("Holdout confusion matrices")
        cms = holdout_confusion_by_profile(ho_df, curve_model)
        ho_row_cm = (
            ho_df.loc[ho_df["pipeline"] == curve_model].iloc[0]
            if not ho_df.empty and curve_model in ho_df["pipeline"].values
            else None
        )
        cm_cols = st.columns(3)
        for col_widget, pid in zip(cm_cols, PROFILE_IDS, strict=True):
            with col_widget:
                prof = THRESHOLD_PROFILES[pid]
                thr_val = None
                if ho_row_cm is not None:
                    thr_col = f"{pid}_threshold"
                    if thr_col in ho_row_cm.index:
                        thr_val = ho_row_cm[thr_col]
                thr_txt = f"threshold = {float(thr_val):.4f}" if thr_val is not None else ""
                st.markdown(f"**{prof.display_name}**  \n{thr_txt}")
                cm = cms.get(pid)
                if cm is not None:
                    st.plotly_chart(
                        fig_holdout_confusion(
                            cm,
                            prof.display_name,
                            height=280,
                        ),
                        width="stretch",
                        theme="streamlit",
                        key=f"p3_cm_{curve_model}_{pid}",
                    )
                else:
                    st.caption("Re-run `python -m secom.cli.benchmark` after tuning.")

        ho_long = holdout_by_profile_df(payload)
        if not ho_long.empty:
            st.markdown("---")
            st.subheader("Holdout comparison by profile")
            metric_choice = st.radio(
                "Holdout comparison metric",
                ["F-beta score", "BER %"],
                index=1,
                horizontal=True,
                key="p3_cost_metric_choice",
            )
            metric_col = "fbeta" if metric_choice == "F-beta score" else "ber_percent"
            if metric_col == "fbeta" and (
                "fbeta" not in ho_long.columns or ho_long["fbeta"].isna().all()
            ):
                render_blue_note(
                    "F-beta columns appear after re-running the benchmark with f1/f2/f3 profiles."
                )
            else:
                st.plotly_chart(
                    fig_holdout_by_profile(
                        ho_long.dropna(subset=[metric_col]),
                        metric_col=metric_col,
                        title=f"Holdout {metric_choice} by threshold profile",
                    ),
                    width="stretch",
                    theme="streamlit",
                    key="p3_holdout_by_profile",
                )

        summary = profile_threshold_summary_table(payload)
        if not summary.empty:
            st.markdown("**CV vs holdout by profile**")
            display_summary = summary.copy()
            for col in display_summary.select_dtypes(include="float").columns:
                display_summary[col] = display_summary[col].round(4)
            st.dataframe(display_summary, width="stretch", hide_index=True)

        render_blue_note(
            "**F1 (conservative) thresholds** often hurt **Linear LR** and **k-NN**: "
            "their scores are less well-calibrated than tree models, so a stricter fail-class "
            "threshold misses more true fails (higher BER) while **Random Forest** and **XGBoost** "
            "retain ranking under conservative cutoffs. **F2** is the default deploy profile."
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
            st.markdown("**Tuned hyperparameters**")
            st.markdown(_format_params(tuned.get(selected_id, {})))

        if not merged.empty and selected_id in merged["pipeline"].values:
            row = merged.loc[merged["pipeline"] == selected_id].iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("PR AUC (CV)", f"{row['pr_auc_cv']:.3f}", border=True)
            pr_label, pr_help = _metric_with_ci(
                float(row["pr_auc_holdout"]),
                row.get("pr_auc_holdout_ci_low"),
                row.get("pr_auc_holdout_ci_high"),
            )
            m2.metric("PR AUC (holdout)", pr_label, help=pr_help, border=True)
            m3.metric("BER (CV)", f"{row['ber_cv']:.1f}%", border=True)
            ber_label, ber_help = _metric_with_ci(
                float(row["ber_holdout"]),
                row.get("ber_holdout_ci_low"),
                row.get("ber_holdout_ci_high"),
                fmt=".1f",
                suffix="%",
            )
            m4.metric("BER (holdout)", ber_label, help=ber_help, border=True)
            st.plotly_chart(
                fig_ber_cv_vs_holdout(merged, selected_id),
                width="stretch",
                theme="streamlit",
                key="p3_model_ber_compare",
            )

    with tab_compare:
        st.subheader("CV vs holdout comparison")
        if merged.empty:
            st.warning("Need both CV and holdout sections in benchmark JSON.")
        else:
            st.plotly_chart(
                fig_cv_vs_holdout_scatter(merged),
                width="stretch",
                theme="streamlit",
                key="p3_cv_holdout_scatter",
            )
            compare_cols = [
                "pipeline",
                "pr_auc_cv",
                "pr_auc_holdout",
                "cv_rank",
                "holdout_rank",
                "ber_cv",
                "ber_holdout",
            ]
            for extra in (
                "pr_auc_holdout_ci_low",
                "pr_auc_holdout_ci_high",
                "ber_holdout_ci_low",
                "ber_holdout_ci_high",
            ):
                if extra in merged.columns:
                    compare_cols.append(extra)
            compare_display = merged[[c for c in compare_cols if c in merged.columns]].copy()
            for col in compare_display.select_dtypes(include="float").columns:
                compare_display[col] = compare_display[col].round(3)
            st.dataframe(compare_display, width="stretch", hide_index=True)
            st.caption(
                "Points above the dashed line outperform on holdout relative to CV mean; "
                "rank shifts highlight generalization gaps."
            )


main()
