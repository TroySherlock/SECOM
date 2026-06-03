"""Model explainability: global importance and holdout wafer inspector."""
from __future__ import annotations

import streamlit as st

from scripts.dashboard_app import ensure_repo_on_path, render_blue_note
from scripts.dashboard_benchmark import model_info
from scripts.secom_pipelines import BENCHMARK_MODEL_IDS
from scripts.dashboard_charts import (
    fig_coef_signed_bar,
    fig_local_contributions,
    fig_top_features_bar,
)
from scripts.dashboard_explainability import (
    cached_global_importance,
    cached_wafer_explanation,
    holdout_wafer_ids,
)
from scripts.secom_pipelines import TUNED_PARAMS_DIR

ensure_repo_on_path()


def main() -> None:
    st.title("Model explainability")
    st.caption(
        "Global drivers (top 15 features) and per-wafer breakdowns on the 20% holdout split. "
        "Fit uses tuned hyperparameters from `data/processed/tuned/`."
    )

    try:
        wafer_ids = holdout_wafer_ids()
    except FileNotFoundError as exc:
        st.error(f"{exc}\n\nRun tuning and `python -m scripts.benchmark_models` first.")
        return

    model_ids = list(BENCHMARK_MODEL_IDS)

    st.subheader("Global view")
    global_model = st.selectbox(
        "Model",
        model_ids,
        format_func=lambda mid: model_info(mid).display_name,
        key="p4_global_model",
    )

    try:
        top_df, signed_df, caption = cached_global_importance(global_model)
    except FileNotFoundError as exc:
        st.error(
            f"Missing tuned params: {exc}\n\n"
            f"Expected JSON under `{TUNED_PARAMS_DIR}/`."
        )
        return
    except Exception as exc:
        st.exception(exc)
        return

    st.caption(caption)
    if top_df.empty:
        st.warning(
            "Could not compute global feature importance for this model "
            "(no valid permutation scores)."
        )
    else:
        st.plotly_chart(
            fig_top_features_bar(
                top_df,
                title=f"Top {len(top_df)} features — {model_info(global_model).display_name}",
            ),
            width="stretch",
            theme="streamlit",
            key="p4_global_bar",
        )
    if signed_df is not None and not signed_df.empty:
        st.plotly_chart(
            fig_coef_signed_bar(
                signed_df,
                title="Positive vs negative coefficients (linear)",
            ),
            width="stretch",
            theme="streamlit",
            key="p4_coef_signed",
        )

    st.divider()
    st.subheader("Wafer inspector")

    c1, c2 = st.columns(2)
    with c1:
        inspect_model = st.selectbox(
            "Model",
            model_ids,
            index=model_ids.index(global_model),
            format_func=lambda mid: model_info(mid).display_name,
            key="p4_wafer_model",
        )
    with c2:
        wafer_id = st.selectbox(
            "Wafer ID (holdout)",
            wafer_ids,
            format_func=str,
            key="p4_wafer_id",
        )

    try:
        result = cached_wafer_explanation(inspect_model, wafer_id)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.exception(exc)
        return

    if result is None:
        st.warning("Wafer not found in holdout split.")
        return

    actual = "Fail" if result.actual_label == 1 else "Pass"
    predicted = "Fail" if result.predicted_label == 1 else "Pass"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Actual", actual, border=True)
    m2.metric("Predicted (F2 threshold)", predicted, border=True)
    m3.metric("P(fail)", f"{100 * result.fail_probability:.1f}%", border=True)
    m4.metric("Deploy threshold", f"{result.threshold:.4f}", border=True)

    st.caption(result.method)
    st.plotly_chart(
        fig_local_contributions(
            result.local_df,
            title=f"Top local contributors — wafer {wafer_id}",
        ),
        width="stretch",
        theme="streamlit",
        key="p4_local_contrib",
    )

    with st.expander("Local contribution table"):
        st.dataframe(
            result.local_df.drop(columns=["abs_contribution"], errors="ignore"),
            width="stretch",
            hide_index=True,
        )

    if inspect_model == "topk_knn":
        render_blue_note(
            "k-NN has no standard SHAP waterfall; local view highlights features that differ "
            "most from the training neighbors' centroid in scaled space, with neighbor fail-rate "
            "as context."
        )


main()
