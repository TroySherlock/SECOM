"""Models benchmark page: CV leaderboard and holdout reporting."""
from __future__ import annotations

import json

import streamlit as st

from scripts.dashboard_app import ensure_repo_on_path
from scripts.dashboard_benchmark import (
    cv_leaderboard_df,
    holdout_df,
    list_model_ids,
    load_benchmark_results,
    merged_comparison_df,
    model_info,
)
from scripts.dashboard_charts import (
    fig_benchmark_leaderboard,
    fig_ber_cv_vs_holdout,
    fig_cv_vs_holdout_scatter,
    fig_holdout_confusion,
)
from scripts.dashboard_theme import plotly_chart
from scripts.secom_pipelines import N_REPEATS, N_SPLITS, PRIMARY_TUNING_METRIC

ensure_repo_on_path()


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
    st.caption("Six tuned pipelines compared on repeated CV and a held-out test split.")

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

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("CV protocol", f"{N_SPLITS}×{N_REPEATS} stratified")
    k2.metric("Primary metric", PRIMARY_TUNING_METRIC.upper())
    k3.metric("Train rows", f"{holdout_split.get('train_rows', '—'):,}")
    k4.metric("Holdout rows", f"{holdout_split.get('test_rows', '—'):,}")

    st.info(
        "**5×5 CV benchmark** ranks models using mean metrics from repeated stratified folds "
        "(used for tuning and comparison). **Holdout** is a single 20% test split, "
        f"reporting only (`holdout_is_reporting_only={payload.get('holdout_is_reporting_only', True)}`) — "
        "not used to select hyperparameters."
    )

    tab_cv, tab_holdout, tab_model, tab_compare = st.tabs(
        [
            "CV leaderboard (5×5)",
            "Holdout reporting",
            "Model deep-dive",
            "CV vs holdout",
        ]
    )

    with tab_cv:
        st.subheader("Cross-validation leaderboard")
        if cv_df.empty:
            st.warning("No CV leaderboard rows in benchmark JSON.")
        else:
            plotly_chart(
                fig_benchmark_leaderboard(
                    cv_df,
                    metric_col="mean_pr_auc",
                    error_col="std_pr_auc",
                    title="Mean PR AUC (5×5 repeated stratified CV)",
                ),
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
            plotly_chart(
                fig_benchmark_leaderboard(
                    ho_df,
                    metric_col="pr_auc",
                    title="Holdout PR AUC (20% test split)",
                ),
                key="p3_holdout_leaderboard",
            )
            ho_display = ho_df.drop(columns=["confusion_matrix"], errors="ignore").copy()
            for col in ho_display.select_dtypes(include="float").columns:
                ho_display[col] = ho_display[col].round(3)
            st.dataframe(ho_display, width="stretch", hide_index=True)

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
                plotly_chart(
                    fig_holdout_confusion(cm, selected),
                    key="p3_holdout_confusion",
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
            m1.metric("PR AUC (CV)", f"{row['pr_auc_cv']:.3f}")
            m2.metric("PR AUC (holdout)", f"{row['pr_auc_holdout']:.3f}")
            m3.metric("BER (CV)", f"{row['ber_cv']:.1f}%")
            m4.metric("BER (holdout)", f"{row['ber_holdout']:.1f}%")
            plotly_chart(
                fig_ber_cv_vs_holdout(merged, selected_id),
                key="p3_model_ber_compare",
            )

    with tab_compare:
        st.subheader("CV vs holdout comparison")
        if merged.empty:
            st.warning("Need both CV and holdout sections in benchmark JSON.")
        else:
            plotly_chart(fig_cv_vs_holdout_scatter(merged), key="p3_cv_holdout_scatter")
            compare_display = merged[
                [
                    "pipeline",
                    "pr_auc_cv",
                    "pr_auc_holdout",
                    "cv_rank",
                    "holdout_rank",
                    "ber_cv",
                    "ber_holdout",
                ]
            ].copy()
            for col in compare_display.select_dtypes(include="float").columns:
                compare_display[col] = compare_display[col].round(3)
            st.dataframe(compare_display, width="stretch", hide_index=True)
            st.caption(
                "Points above the dashed line outperform on holdout relative to CV mean; "
                "rank shifts highlight generalization gaps."
            )


main()
