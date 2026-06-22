"""Extrapolation track: forward-in-time test (blocked CV + temporal holdout)."""
from __future__ import annotations

import streamlit as st

from secom.dashboard import render_blue_note
from secom.dashboard.charts import fig_delta_bar
from secom.dashboard.data import (
    DELTA_METRIC_COLS,
    holdout_comparison_df,
    holdout_delta_df,
)
from secom.dashboard.model_views import (
    load_payload,
    render_cv_leaderboard,
    render_holdout_validation,
    render_model_deepdive,
)


def main() -> None:
    st.title("Extrapolation — forward-in-time test")
    st.caption(
        "The hard, honest case: blocked time CV with local stratification and a temporal forward "
        "holdout (latest 20% by measurement time), with exponential time-decay sample weighting. "
        "Models trained on the past are scored on a later, drifted regime — the realistic fab "
        "deployment setting."
    )

    try:
        payload = load_payload()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    render_blue_note(
        "**Forward view.** Blocked time CV (expanding window with a warm-up minimum train size) "
        "for tuning, temporal forward holdout for reporting, optional exponential time-decay "
        "weighting. The drop from the Interpolation page is the cost of drift; selection-based "
        "models that lock onto era-specific sensors degrade most here. All holdout metrics are "
        f"reporting-only (`holdout_is_reporting_only={payload.get('holdout_is_reporting_only', True)}`)."
    )

    st.subheader("Cost of drift — interpolation minus extrapolation holdout")
    drift_metric = st.radio(
        "Metric",
        list(DELTA_METRIC_COLS),
        horizontal=True,
        key="drift_delta_metric",
    )
    metric_col = DELTA_METRIC_COLS[drift_metric]
    delta_df = holdout_delta_df(payload, metric_col)
    if delta_df.empty:
        st.warning("No holdout rows for both tracks in benchmark JSON. Re-run the benchmark.")
    else:
        st.plotly_chart(
            fig_delta_bar(
                delta_df,
                value_col="delta",
                title=f"Drift cost per model ({drift_metric}: interpolation − extrapolation)",
                value_label=f"{drift_metric} lost to drift",
                positive_is_good=False,
            ),
            width="stretch",
            theme="streamlit",
            key="drift_delta_bar",
        )
        st.caption(
            "Each bar is one model's in-distribution holdout minus its temporal-forward holdout "
            f"{drift_metric}. Longer red bars = more {drift_metric} surrendered to drift; bars near "
            "zero (or green) are the drift-robust models. Selection-based heads that lock onto "
            "era-specific sensors tend to lose the most."
        )

    comparison_df = holdout_comparison_df(payload)
    if not comparison_df.empty:
        with st.expander("Per-model detail — CV vs holdout PR-AUC (both tracks)", expanded=False):
            st.dataframe(comparison_df, width="stretch", hide_index=True)
            st.caption(
                "Each model under its own track. `cv_pr_auc` is that track's CV (stratified for "
                "interpolation, blocked time CV for extrapolation); `holdout_pr_auc` is the matching "
                "holdout (random vs temporal forward). The interpolation-minus-extrapolation gap is "
                "the drift penalty."
            )

    tab_cv, tab_holdout, tab_model = st.tabs(
        ["CV leaderboard (blocked)", "Holdout", "Deep-dive"]
    )
    with tab_cv:
        render_cv_leaderboard(payload, blocked=True)
    with tab_holdout:
        render_holdout_validation(payload, blocked=True)
    with tab_model:
        render_model_deepdive(payload, track="extrapolation")


main()
