"""Introduction page: SECOM stg_secom macro health snapshot."""
from __future__ import annotations

import streamlit as st

from scripts.dashboard_app import ensure_repo_on_path
from scripts.dashboard_charts import (
    best_pair_sensors,
    fig_class_donut,
    fig_fails_over_time,
    fig_missing_rate_distribution,
    fig_missingness_structure,
    fig_sensor_histogram,
    fig_sensor_multicollinearity,
)
from scripts.dashboard_pipeline import render_pipeline_flowchart
from scripts.dashboard_stg import (
    STG_RELATION,
    cohens_d,
    load_stg_secom,
    slice_stg_for_display,
    stg_available,
    stg_sensor_columns,
    stg_summary_stats,
)
from scripts.dashboard_theme import (
    STG_TABLE_MAX_ROWS,
    display_stg_dataframe,
    plotly_chart,
    render_callout,
)
from scripts.secom_pipelines import DB_PATH, TARGET_COL, TIMESTAMP_COL

ensure_repo_on_path()


@st.cache_data(show_spinner="Loading stg_secom...")
def load_stg_df():
    return load_stg_secom()


@st.cache_data(show_spinner=False)
def cached_stg_slice(target_filter: str, n_sensor_cols: int) -> tuple:
    full = slice_stg_for_display(
        load_stg_df(),
        n_sensor_cols=n_sensor_cols,
        target_filter=target_filter,
    )
    total_filtered = len(full)
    display = full.head(STG_TABLE_MAX_ROWS)
    return display, total_filtered


def _default_sensor(df, sensor_cols: list[str]) -> str:
    if not sensor_cols:
        return ""
    variances = df[sensor_cols].var(numeric_only=True).sort_values(ascending=False)
    return str(variances.index[0])


def _stg_data_dictionary_md() -> str:
    return f"""
| Field | Role |
|-------|------|
| `{TIMESTAMP_COL}` | Parsed measurement time |
| `{TARGET_COL}` | **0 = pass**, **1 = fail** |
| `c_0` … `c_590` | 591 sensor readings; `NaN` = missing |

**1,567 rows × 591 sensors.** Missingness clusters by sensor and time window.
"""


def _cohens_d_explanation_md(sensor_x: str, sensor_y: str, d_x: float, d_y: float) -> str:
    return f"""**Cohen's d** — standardized pass vs fail separation.

**Top sensors:** `{sensor_x}` (d = {d_x:.2f}), `{sensor_y}` (d = {d_y:.2f}). Defects are multivariate; collinearity clusters matter as much as single-sensor effect size.
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
        df = load_stg_df()
    except Exception as exc:
        st.error(f"Could not load `{STG_RELATION}`.\n\n**Error:** {exc}")
        return

    sensor_cols = stg_sensor_columns(df)
    stats = stg_summary_stats(df)
    pass_rate = 100 * (1 - stats["fail_rate"])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total wafers processed", f"{stats['n_obs']:,}")
    k2.metric("Detected failures", f"{stats['n_fail']:,}")
    k3.metric("Baseline sensors", f"{stats['n_sensors']:,}")
    k4.metric("Yield pass rate", f"{pass_rate:.1f}%")

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
        drift_left, drift_right = st.columns([1.4, 1], gap="large")
        with drift_left:
            show_weekly = st.checkbox("Show weekly fail rate overlay", value=False, key="p1_weekly")
            plotly_chart(
                fig_fails_over_time(
                    df,
                    timestamp_col=TIMESTAMP_COL,
                    target_col=TARGET_COL,
                    show_weekly=show_weekly,
                ),
                key="p1_fails_time",
            )
        with drift_right:
            plotly_chart(fig_class_donut(df, TARGET_COL), key="p1_class_donut")

        st.markdown("---")
        st.subheader("Missingness and redundancy")
        miss_col, corr_col = st.columns(2, gap="large")
        with miss_col:
            st.info("Missing values cluster by sensor and time—not as independent random gaps.")
            plotly_chart(
                fig_missingness_structure(
                    df,
                    timestamp_col=TIMESTAMP_COL,
                    sensor_cols=sensor_cols,
                ),
                key="p1_missingness",
            )
            plotly_chart(
                fig_missing_rate_distribution(df, sensor_cols),
                key="p1_missing_rate",
            )
        with corr_col:
            if sensor_cols:
                auto_x, auto_y = best_pair_sensors(df, sensor_cols, TARGET_COL)
                d_x = cohens_d(df, auto_x, TARGET_COL)
                d_y = cohens_d(df, auto_y, TARGET_COL)
                render_callout(
                    "edu",
                    _cohens_d_explanation_md(auto_x, auto_y, d_x, d_y),
                    key_suffix="cohens",
                )
                fig_corr, corr_stats = fig_sensor_multicollinearity(df, sensor_cols)
                st.caption(
                    f"Top {corr_stats['n_used']} variance sensors; "
                    f"{corr_stats['high_corr_pairs']} pairs with |r| ≥ 0.90."
                )
                plotly_chart(fig_corr, key="p1_multicollinearity")

    with tab_sensor:
        st.subheader("Individual channel distributions")
        if not sensor_cols:
            st.warning("No sensor columns found.")
        else:
            ctrl_col, chart_col = st.columns([1, 4], gap="medium")
            with ctrl_col:
                default_sensor = _default_sensor(df, sensor_cols)
                sensor = st.selectbox(
                    "Sensor",
                    sensor_cols,
                    index=sensor_cols.index(default_sensor) if default_sensor in sensor_cols else 0,
                    key="p1_sensor_select",
                )
                log_scale = st.checkbox("Log scale", value=False, key="p1_log_scale")
            with chart_col:
                plotly_chart(
                    fig_sensor_histogram(df, sensor, target_col=TARGET_COL, log_scale=log_scale),
                    key="p1_sensor_hist",
                )

    with tab_data:
        st.subheader("Cleaned telemetry matrix (`stg_secom`)")
        dict_col, table_col = st.columns([1, 2.5], gap="large")
        with dict_col:
            render_callout("edu", _stg_data_dictionary_md(), key_suffix="stg_dict")
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
            display_df, total_filtered = cached_stg_slice(target_filter, n_cols)
            display_stg_dataframe(display_df, height=400)
            st.caption(
                f"Showing {len(display_df):,} of {total_filtered:,} filtered rows "
                f"× {len(display_df.columns)} columns."
            )

    st.divider()
    st.subheader("Project pipeline")
    flow_col, mart_col = st.columns([1.2, 1], gap="large")
    with flow_col:
        render_pipeline_flowchart()
    with mart_col:
        render_callout("pipeline", _mart_pipeline_md(), key_suffix="mart")


main()
