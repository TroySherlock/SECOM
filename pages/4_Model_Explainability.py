"""Model explainability: global importance and holdout wafer inspector."""
from __future__ import annotations

import streamlit as st

from secom.dashboard import render_blue_note
from secom.dashboard.data import model_info
from secom.pipelines import MODEL_IDS, TUNED_BLOCKED_PARAMS_DIR, TUNED_PARAMS_DIR
from secom.dashboard.charts import (
    fig_coef_signed_bar,
    fig_local_contributions,
    fig_top_features_bar,
)
from secom.dashboard.explainability import (
    cached_global_importance,
    cached_wafer_explanation,
    holdout_wafer_ids,
)
from secom.dashboard.narrator import NARRATIVE_MODEL_ID, get_wafer_narrative, load_narratives

_TRACK_LABELS = {
    "Extrapolation (temporal)": "extrapolation",
    "Interpolation (random)": "interpolation",
}



@st.cache_data(show_spinner=False)
def _load_frozen_narratives() -> dict | None:
    try:
        return load_narratives()
    except FileNotFoundError:
        return None
    except (ValueError, OSError):
        return None


def main() -> None:
    st.title("Model explainability")
    st.caption(
        "Global drivers (top 15 features) and per-wafer breakdowns on the chosen track's "
        "holdout — extrapolation uses the **latest 20%** of wafers by measurement time "
        "(blocked-tuned hyperparameters + time-decay weighting); interpolation uses the "
        "random stratified holdout (in-distribution)."
    )

    track_label = st.radio(
        "Track",
        options=list(_TRACK_LABELS),
        horizontal=True,
        key="p4_track",
    )
    track = _TRACK_LABELS[track_label]
    tuned_dir = TUNED_BLOCKED_PARAMS_DIR if track == "extrapolation" else TUNED_PARAMS_DIR

    try:
        wafer_ids = holdout_wafer_ids(track)
    except FileNotFoundError as exc:
        st.error(f"{exc}\n\nRun tuning and `python -m secom.cli.benchmark` first.")
        return

    model_ids = list(MODEL_IDS)

    st.subheader("Global view")
    global_model = st.selectbox(
        "Model",
        model_ids,
        format_func=lambda mid: model_info(mid).display_name,
        key="p4_global_model",
    )

    try:
        top_df, signed_df, caption = cached_global_importance(global_model, track)
    except FileNotFoundError as exc:
        st.error(
            f"Missing tuned params: {exc}\n\n"
            f"Expected JSON under `{tuned_dir}/`."
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
        result = cached_wafer_explanation(inspect_model, wafer_id, track)
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

    if inspect_model != NARRATIVE_MODEL_ID:
        st.caption(
            "Plain-English summary is available only for "
            f"**{model_info(NARRATIVE_MODEL_ID).display_name}** (the narrative model)."
        )

    if inspect_model == NARRATIVE_MODEL_ID:
        st.divider()
        st.subheader("Plain-English summary")
        render_blue_note(
            "Statistical interpreter only — summarizes model inputs and attributions. "
            "Not a fab diagnosis or root-cause analysis."
        )
        narratives_payload = _load_frozen_narratives()
        if narratives_payload is None:
            st.warning(
                "No frozen narratives file found. Run: "
                "`python -m secom.cli.build_narratives`"
            )
        else:
            narrative = get_wafer_narrative(wafer_id, narratives_payload)
            if narrative is None:
                st.warning(
                    f"No frozen narrative for wafer {wafer_id}. Run: "
                    "`python -m secom.cli.build_narratives`"
                )
            else:
                st.markdown(narrative)
                st.caption("Pre-generated summary (local Gemma)")


main()
