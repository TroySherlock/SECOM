"""Extrapolation (1/2): holdout evaluation — forward-in-time test."""
from __future__ import annotations

import streamlit as st

from secom.dashboard import render_blue_note
from secom.dashboard.charts import fig_delta_bar
from secom.dashboard.components import load_payload, render_holdout_validation
from secom.dashboard.data import (
    DELTA_METRIC_COLS,
    holdout_comparison_df,
    holdout_delta_df,
)


def _render_drift_cost(payload: dict) -> None:
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
        st.dataframe(
            comparison_df,
            width="stretch",
            hide_index=True,
            column_config={
                "pipeline": st.column_config.TextColumn("Model"),
                "track": st.column_config.TextColumn("Track"),
                "cv_pr_auc": st.column_config.NumberColumn("CV PR-AUC", format="%.3f"),
                "holdout_pr_auc": st.column_config.NumberColumn("Holdout PR-AUC", format="%.3f"),
            },
        )
        st.caption(
            "Each model under its own track. **CV PR-AUC** is the shared in-distribution 5×2 "
            "stratified CV reference; **Holdout PR-AUC** is the matching holdout (random for "
            "interpolation, temporal forward for extrapolation). The "
            "interpolation-minus-extrapolation gap is the drift penalty."
        )


def main() -> None:
    st.title("Extrapolation — forward-in-time test")
    st.caption(
        "The hard, honest case: every model is tuned once in-distribution, then refit on the "
        "temporal train and scored on a forward holdout (latest 20% by measurement time). "
        "Models trained on the past meet a later, drifted regime — the realistic fab "
        "deployment setting."
    )

    try:
        payload = load_payload()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    render_blue_note(
        "**Forward view.** Hyperparameters come from the single in-distribution 5×2 stratified "
        "CV (no separate temporal tuning — blocked CV collapsed to one noisy fold, so it was "
        "dropped). The headline holdout is unweighted: recency weighting needs a temporal-CV "
        "validation signal we do not have, so we do not sweep λ on a single forward holdout. The "
        "drop from the Interpolation page is the cost of drift; selection-based models that lock "
        "onto era-specific sensors degrade most here. All holdout metrics are reporting-only "
        f"(`holdout_is_reporting_only={payload.get('holdout_is_reporting_only', True)}`)."
    )

    render_holdout_validation(payload, track="extrapolation")

    st.divider()
    _render_drift_cost(payload)

    render_blue_note(
        "**Why the ranking flips under drift.** `hsic_rf` is the interpolation champion "
        "(PR-AUC ≈ 0.34) but collapses to the **worst** model here (≈ 0.09): selection-based heads "
        "lock onto a handful of era-specific sensors, so when those sensors drift away the signal "
        "evaporates on the later regime. The PLS heads do the opposite — by **projecting every "
        "clustered sensor into a few supervised latent components** they average over hundreds of "
        "weak, redundant channels, and that aggregate signal survives drift. The **linear** PLS "
        "heads extrapolate best: `pls_enet` rises ≈ 0.10 → 0.18 and `pls_bayes` ≈ 0.11 → 0.20 to "
        "become the **extrapolation champion** — a genuine *boost* on the forward holdout, not just "
        "a smaller drop. (Read direction over exact magnitude: the temporal holdout carries only "
        "~17–20 fails.)"
    )


main()
