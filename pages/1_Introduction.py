"""Introduction page: SECOM stg_secom macro health snapshot."""
from __future__ import annotations

import streamlit as st

from secom.dashboard import render_blue_note
from secom.dashboard.charts import (
    fig_class_donut,
    fig_fails_over_time,
    fig_missing_rate_distribution,
    fig_missingness_structure,
    fig_sensor_drift_heatmap,
    fig_sensor_histogram,
    fig_sensor_multicollinearity,
)
from secom.dashboard.pipeline import render_pipeline_flowchart
from secom.dashboard.stg import (
    STG_RELATION,
    StgSnapshot,
    build_stg_snapshot,
    slice_stg_for_display,
    stg_available,
)
from secom.pipelines import DB_PATH, TARGET_COL, TEST_SIZE, TIMESTAMP_COL


MAX_TABLE_ROWS = 50


@st.cache_data(show_spinner="Loading stg_secom...")
def load_stg_snapshot() -> StgSnapshot:
    return build_stg_snapshot()


def _stg_data_dictionary_md() -> str:
    return f"""
| Field | Role |
|-------|------|
| `{TIMESTAMP_COL}` | Parsed measurement time |
| `{TARGET_COL}` | **0 = pass**, **1 = fail** |
| `c_0` … `c_590` | 591 sensor readings; `NaN` = missing |

**1,567 rows × 591 sensors.** Missingness clusters by sensor and time window.
"""


def _mart_pipeline_md() -> str:
    return """
`stg_secom` → `int_secom_features` → `int_secom_column_metadata` → `mart_secom_features`

Training and benchmarking read **`public.mart_secom_features`**.
"""


def main() -> None:
    st.title("SECOM defect detection")
    st.caption(
        "Semiconductor process monitoring: multivariate sensor snapshots labeled pass/fail. "
        "Rare failures (~7%) require combining hundreds of weak signals."
    )

    if not DB_PATH.exists() or not stg_available():
        st.error(f"Build staging first: `dbt run -s stg_secom` (needs `{STG_RELATION}` in DuckDB).")
        return

    try:
        snapshot = load_stg_snapshot()
    except Exception as exc:
        st.error(f"Could not load `{STG_RELATION}`.\n\n**Error:** {exc}")
        return

    df = snapshot.df
    sensor_cols = snapshot.sensor_cols
    stats = snapshot.stats
    pass_rate = 100 * (1 - stats["fail_rate"])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total wafers processed", f"{stats['n_obs']:,}", border=True)
    k2.metric("Detected failures", f"{stats['n_fail']:,}", border=True)
    k3.metric("Baseline sensors", f"{stats['n_sensors']:,}", border=True)
    k4.metric("Yield pass rate", f"{pass_rate:.1f}%", border=True)

    st.divider()
    st.subheader("Project pipeline")
    render_pipeline_flowchart()
    render_blue_note(_mart_pipeline_md())

    st.divider()

    tab_drift, tab_sensor, tab_data = st.tabs(
        [
            "Factory drift",
            "Sensor explorer",
            "Staged data inventory",
        ]
    )

    with tab_drift:
        st.subheader("Multivariate process signals")
        show_weekly = st.checkbox("Show weekly fail rate overlay", value=False, key="p1_weekly")
        drift_left, drift_right = st.columns(2, gap="large")
        with drift_left:
            st.plotly_chart(
                fig_fails_over_time(
                    df,
                    timestamp_col=TIMESTAMP_COL,
                    target_col=TARGET_COL,
                    show_weekly=show_weekly,
                ),
                width="stretch",
                theme="streamlit",
                key="p1_fails_time",
            )
        with drift_right:
            st.plotly_chart(
                fig_class_donut(df, TARGET_COL),
                width="stretch",
                theme="streamlit",
                key="p1_class_donut",
            )

        st.markdown("---")
        st.subheader("Sensor drift — what extrapolation works against")
        render_blue_note(
            "Each row is one of the most drift-prone sensors; each column is a time window. "
            "Colour is the sensor's mean **standardized against the training era** (first 80% by "
            "time), so blue/red cells show how far later wafers drift from what the models were "
            "fit on. The dashed line marks the temporal holdout (latest 20%): the extrapolation "
            "track must predict on this drifted regime, which is why selection-based models that "
            "lock onto era-specific sensors degrade there while aggregation (sPLS) holds up."
        )
        st.plotly_chart(
            fig_sensor_drift_heatmap(
                df,
                timestamp_col=TIMESTAMP_COL,
                sensor_cols=sensor_cols,
                test_size=TEST_SIZE,
            ),
            width="stretch",
            theme="streamlit",
            key="p1_sensor_drift",
        )

        st.markdown("---")
        st.subheader("Missingness and redundancy")
        render_blue_note(
            "We can see that missing values are structured in our dataset and not completely random."
            "We can also see that there are some sensors that are highly correlated with each other."
        )
        miss_col, corr_col = st.columns(2, gap="large")
        with miss_col:
            st.plotly_chart(
                fig_missingness_structure(
                    df,
                    timestamp_col=TIMESTAMP_COL,
                    sensor_cols=sensor_cols,
                ),
                width="stretch",
                theme="streamlit",
                key="p1_missingness",
            )
        with corr_col:
            if sensor_cols:
                fig_corr, corr_stats = fig_sensor_multicollinearity(df, sensor_cols)
                st.plotly_chart(
                    fig_corr,
                    width="stretch",
                    theme="streamlit",
                    key="p1_multicollinearity",
                )
                st.caption(
                    f"Top {corr_stats['n_used']} variance sensors; "
                    f"{corr_stats['high_corr_pairs']} pairs with |r| ≥ 0.90."
                )
        st.plotly_chart(
            fig_missing_rate_distribution(df, sensor_cols),
            width="stretch",
            theme="streamlit",
            key="p1_missing_rate",
        )
        render_blue_note(
            "I chose to drop sensors with >10% missing rate and zero variance."
        )
        

    with tab_sensor:
        st.subheader("Individual channel distributions")
        if not sensor_cols:
            st.warning("No sensor columns found.")
        else:
            ctrl_col, chart_col = st.columns([1, 4], gap="medium")
            with ctrl_col:
                default_idx = (
                    sensor_cols.index(snapshot.default_sensor)
                    if snapshot.default_sensor in sensor_cols
                    else 0
                )
                sensor = st.selectbox(
                    "Sensor",
                    sensor_cols,
                    index=default_idx,
                    key="p1_sensor_select",
                )
                log_scale = st.checkbox("Log scale", value=False, key="p1_log_scale")
            with chart_col:
                st.plotly_chart(
                    fig_sensor_histogram(df, sensor, target_col=TARGET_COL, log_scale=log_scale),
                    width="stretch",
                    theme="streamlit",
                    key="p1_sensor_hist",
                )

    with tab_data:
        st.subheader("Cleaned telemetry matrix (`stg_secom`)")
        dict_col, table_col = st.columns([1, 2.5], gap="large")
        with dict_col:
            with st.container(border=True):
                st.markdown(_stg_data_dictionary_md())
        with table_col:
            filt_col, ncol_col = st.columns(2)
            with filt_col:
                target_filter = st.selectbox(
                    "Filter by label",
                    ["All", "Pass (0)", "Fail (1)"],
                    key="p1_target_filter",
                )
            with ncol_col:
                n_cols = st.selectbox(
                    "Sensor columns shown",
                    [50, 100, 200, 591],
                    index=0,
                    key="p1_n_cols",
                )
            filtered = slice_stg_for_display(
                df,
                n_sensor_cols=n_cols,
                target_filter=target_filter,
            )
            display_df = filtered.head(MAX_TABLE_ROWS)
            st.dataframe(display_df, height=400, width="stretch", hide_index=True)
            st.caption(
                f"Showing {len(display_df):,} of {len(filtered):,} filtered rows "
                f"× {len(display_df.columns)} columns."
            )


main()
