"""Overview page: the SECOM problem (rarity + drift + structured missingness)."""
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
    fig_two_track_schematic,
)
from secom.dashboard.stg import (
    STG_RELATION,
    StgSnapshot,
    build_stg_snapshot,
    era_drift_summary,
    slice_stg_for_display,
    stg_available,
)
from secom.pipelines import DB_PATH, TARGET_COL, TEST_SIZE, TIMESTAMP_COL

MAX_TABLE_ROWS = 50


@st.cache_data(show_spinner="Loading stg_secom...")
def load_stg_snapshot() -> StgSnapshot:
    return build_stg_snapshot()


def _stg_data_dictionary_md(stats: dict) -> str:
    n_obs = stats.get("n_obs", 0)
    n_sensors = stats.get("n_sensors", 0)
    return f"""
| Field | Role |
|-------|------|
| `{TIMESTAMP_COL}` | Parsed measurement time |
| `{TARGET_COL}` | **0 = pass**, **1 = fail** |
| `c_*` | {n_sensors:,} sensor readings; `NaN` = missing |

**{n_obs:,} rows × {n_sensors:,} sensors.** Missingness clusters by sensor and time window.
"""


def _mart_pipeline_md() -> str:
    return """
`stg_secom` → `int_secom_features` → `int_secom_column_metadata` → `mart_secom_features`

Training and benchmarking read **`public.mart_secom_features`**.
"""


def main() -> None:
    st.title("SECOM defect detection — the problem")
    st.caption(
        "What we are up against: semiconductor process monitoring where ~7% of wafers fail, "
        "the signal is spread across hundreds of weak, redundant sensors with structured "
        "missingness, and the process drifts over time — so a model trained on the past must "
        "still hold up on a later, shifted regime."
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
    fail_pct = 100 * stats["fail_rate"]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Wafers", f"{stats['n_obs']:,}", border=True)
    k2.metric("Fails", f"{stats['n_fail']:,}", border=True)
    k3.metric("Fail prevalence", f"{fail_pct:.1f}%", border=True)
    k4.metric("Sensor channels", f"{stats['n_sensors']:,}", border=True)
    k5.metric("Campaign window", stats["date_range"], border=True)
    st.caption(
        f"Positive class = **fail** (engineering convention); fails are the rare class at "
        f"{fail_pct:.1f}% prevalence. These are labelled outcomes of a curated benchmark set "
        f"(class balance), not a fab line-yield figure. The cleaned `stg_secom` view exposes "
        f"{stats['n_sensors']:,} sensor channels (the classic SECOM feature count); the dbt "
        "feature funnel on the Pipeline page counts 591 raw staged columns before cleaning."
    )

    st.divider()
    st.subheader("Two evaluation tracks")
    render_blue_note(
        "Every model is scored two ways. **Interpolation** uses a random train/test split "
        "(can the model fit the process as sampled?). **Extrapolation** uses a temporal forward "
        "holdout — train on the earliest 80% by time, test on the latest 20% — which is the "
        "realistic deployment question: does a model fit on the past still hold up on a later, "
        "drifted regime? Pages 3 and 4 report these two tracks."
    )
    st.plotly_chart(
        fig_two_track_schematic(
            df,
            timestamp_col=TIMESTAMP_COL,
            target_col=TARGET_COL,
            test_size=TEST_SIZE,
        ),
        width="stretch",
        theme="streamlit",
        key="p1_two_track",
    )

    st.divider()

    tab_balance, tab_drift, tab_quality, tab_sensor, tab_data = st.tabs(
        [
            "Class balance & time",
            "Drift",
            "Data quality",
            "Sensor explorer",
            "Staged data inventory",
        ]
    )

    with tab_balance:
        st.subheader("Rare fails across the campaign")
        show_weekly = st.checkbox("Show weekly fail rate overlay", value=False, key="p1_weekly")
        balance_left, balance_right = st.columns(2, gap="large")
        with balance_left:
            st.plotly_chart(
                fig_fails_over_time(
                    df,
                    timestamp_col=TIMESTAMP_COL,
                    target_col=TARGET_COL,
                    show_weekly=show_weekly,
                    prevalence=stats["fail_rate"],
                ),
                width="stretch",
                theme="streamlit",
                key="p1_fails_time",
            )
        with balance_right:
            st.plotly_chart(
                fig_class_donut(df, TARGET_COL),
                width="stretch",
                theme="streamlit",
                key="p1_class_donut",
            )
        st.caption(
            f"Fails are the rare positive class at {100 * stats['fail_rate']:.1f}% prevalence — "
            "this is the no-skill baseline the precision–recall curves on pages 3 and 4 are scored "
            "against (enable the weekly overlay to see it as a dotted line)."
        )

    with tab_drift:
        st.subheader("Sensor drift — what extrapolation works against")
        drift = era_drift_summary(
            df,
            timestamp_col=TIMESTAMP_COL,
            sensor_cols=sensor_cols,
            test_size=TEST_SIZE,
        )
        d1, d2, d3 = st.columns(3)
        d1.metric(
            f"Sensors drifting >{drift['z_threshold']:.0f}σ",
            f"{drift['pct_drifted']:.0f}%",
            border=True,
        )
        d2.metric(
            "Drifting / evaluated",
            f"{drift['n_drifted']:,} / {drift['n_evaluated']:,}",
            border=True,
        )
        d3.metric(
            "Median |shift| (holdout)",
            f"{drift['median_abs_shift']:.2f}σ",
            border=True,
        )
        render_blue_note(
            "Each row is one of the most drift-prone sensors; each column is a time window. "
            "Colour is the sensor's mean **standardized against the training era** (first 80% by "
            "time), so blue/red cells show how far later wafers drift from what the models were "
            "fit on. The dashed line marks the temporal holdout (latest 20%): the extrapolation "
            "track must predict on this drifted regime, which is why selection-based models that "
            "lock onto era-specific sensors degrade there while aggregation (PLS) holds up. The "
            "scalars above quantify the heatmap: the share of sensors whose holdout-era mean lands "
            f"beyond {drift['z_threshold']:.0f} SD of their training-era baseline."
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

    with tab_quality:
        st.subheader("Missingness and redundancy")
        render_blue_note(
            "Missingness is structured, not missing-at-random: NaNs cluster by sensor and by time "
            "window (recipe / tool-state changes), and many sensors are near-duplicates with "
            "pairwise |r| above 0.9. Both properties motivate the imputation and correlated-"
            "selection steps on the Pipeline page."
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
            "Sensors with a >10% missing rate or zero variance are dropped upstream in dbt "
            f"before the mart: {stats['n_sensors']:,} staged channels reduce to the mart feature "
            "set used for training (see the reduction funnel on the Pipeline page)."
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
            with st.container(border=True, key="card_p1_data_dict"):
                st.markdown(_stg_data_dictionary_md(stats))
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
