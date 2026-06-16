"""Plotly chart builders for the SECOM Streamlit dashboard."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from secom.dashboard.stg import ROW_INDEX_COL, sensor_columns
from secom.metrics import PRCurve

# SECOM chart palette — keep in sync with .streamlit/config.toml chartCategoricalColors
C_RED = "#ea6962"
C_ORANGE = "#e78a4e"
C_YELLOW = "#d8a657"
C_GREEN = "#a9b665"
C_AQUA = "#89b482"
C_BLUE = "#7daea3"
C_PURPLE = "#d3869b"

C = [C_RED, C_ORANGE, C_YELLOW, C_GREEN, C_AQUA, C_BLUE, C_PURPLE]

# Validation chart overlays (Gruvbox yellow holdout, purple CV)
CI_YELLOW_RGBA = "rgba(216, 166, 87, 0.35)"
CV_ERROR_PURPLE_RGBA = "rgba(211, 134, 155, 0.6)"

CHART_BG = "#3c3836"

# Heatmaps: low → green, high → red (yellow mid-tone)
COLORSCALE_LOW_GREEN_HIGH_RED = [
    [0.0, C_GREEN],
    [0.5, C_YELLOW],
    [1.0, C_RED],
]
# Correlation r ∈ [-1, 1]: blue at -1, green at 0, red at +1
COLORSCALE_CORRELATION = [
    [0.0, C_BLUE],
    [0.5, C_GREEN],
    [1.0, C_RED],
]


def _sized(fig: go.Figure, *, height: int, **layout: Any) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        **layout,
    )
    return fig


def fig_class_donut(
    df: pd.DataFrame,
    target_col: str,
    *,
    pass_value: int = 0,
    fail_value: int = 1,
) -> go.Figure:
    counts = df[target_col].astype(int).value_counts()
    labels = ["Pass", "Fail"]
    values = [int(counts.get(pass_value, 0)), int(counts.get(fail_value, 0))]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                marker=dict(colors=[C_GREEN, C_RED]),
                textinfo="label+percent",
                hovertemplate="%{label}<br>%{value} rows<br>%{percent}<extra></extra>",
            )
        ]
    )
    total = sum(values)
    fig.update_layout(
        title=dict(text="Pass vs fail", x=0.5, xanchor="center"),
        annotations=[
            dict(
                text=f"{total:,}<br>rows",
                x=0.5,
                y=0.5,
                font=dict(size=16),
                showarrow=False,
            )
        ],
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, x=0.5, xanchor="center"),
    )
    return _sized(fig, height=360, legend=dict(orientation="h", yanchor="bottom", y=-0.12, x=0.5, xanchor="center"))


def fig_fails_over_time(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    target_col: str,
    row_index_col: str = ROW_INDEX_COL,
    show_weekly: bool,
) -> go.Figure:
    plot_df = df[[row_index_col, timestamp_col, target_col]].copy()
    plot_df[timestamp_col] = pd.to_datetime(plot_df[timestamp_col])
    plot_df["label"] = plot_df[target_col].map({0: "Pass", 1: "Fail"})
    plot_df["y_jitter"] = plot_df[target_col].astype(float) + np.random.default_rng(42).uniform(
        -0.08, 0.08, size=len(plot_df)
    )

    fig = go.Figure()

    for label, symbol, color in [("Pass", "circle", C_GREEN), ("Fail", "diamond", C_RED)]:
        subset = plot_df.loc[plot_df["label"] == label]
        fig.add_trace(
            go.Scatter(
                x=subset[timestamp_col],
                y=subset["y_jitter"],
                mode="markers",
                name=label,
                marker=dict(
                    color=color,
                    size=7 if label == "Fail" else 4,
                    opacity=0.85 if label == "Fail" else 0.35,
                    symbol=symbol,
                ),
                customdata=np.stack([subset[row_index_col], subset[target_col]], axis=-1),
                hovertemplate=(
                    f"{label}<br>"
                    "%{x|%Y-%m-%d %H:%M}<br>"
                    "row=%{customdata[0]}<extra></extra>"
                ),
            )
        )

    if show_weekly:
        weekly = (
            plot_df.set_index(timestamp_col)[target_col]
            .resample("W")
            .mean()
            .reset_index(name="fail_rate")
        )
        fig.add_trace(
            go.Scatter(
                x=weekly[timestamp_col],
                y=weekly["fail_rate"],
                mode="lines+markers",
                name="Weekly fail rate",
                yaxis="y2",
                line=dict(color=C_BLUE, width=2),
                marker=dict(size=5, color=C_BLUE),
                hovertemplate="Week of %{x|%Y-%m-%d}<br>Fail rate %{y:.1%}<extra></extra>",
            )
        )
        layout_y2 = dict(
            overlaying="y",
            side="right",
            range=[0, 1],
            tickformat=".0%",
            title=dict(text="Weekly fail rate"),
        )
    else:
        layout_y2 = None

    fig.update_layout(
        title=dict(text="Failures over time"),
        xaxis_title="Measurement time",
        yaxis=dict(
            title="Outcome (jittered)",
            tickvals=[0, 1],
            ticktext=["Pass", "Fail"],
            range=[-0.2, 1.2],
        ),
        yaxis2=layout_y2,
        showlegend=True,
    )
    return _sized(fig, height=400)


def fig_missingness_structure(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    sensor_cols: list[str] | None = None,
    column_stride: int = 10,
) -> go.Figure:
    sensor_cols = sensor_cols or sensor_columns(df)
    if not sensor_cols:
        return _sized(go.Figure(), height=360)

    plot_df = df[[timestamp_col] + sensor_cols].copy()
    plot_df["_sort_ts"] = pd.to_datetime(plot_df[timestamp_col], errors="coerce")
    plot_df = plot_df.sort_values("_sort_ts").drop(columns="_sort_ts").reset_index(drop=True)

    sampled = sensor_cols[::column_stride]
    missing = plot_df[sampled].isna().astype(int).to_numpy()
    sensor_labels = [c.replace("c_", "") for c in sampled]

    miss_rate = plot_df[sensor_cols].isna().mean()
    high_miss_pct = 100 * (miss_rate > 0.5).mean()

    fig = go.Figure(
        data=go.Heatmap(
            z=missing.T,
            x=list(range(len(plot_df))),
            y=sensor_labels,
            colorscale=[[0, C_GREEN], [1, C_RED]],
            showscale=False,
            hovertemplate="Row %{x}<br>Sensor c_%{y}<br>%{text}<extra></extra>",
            text=np.where(missing.T == 1, "Missing", "Present"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=9, color=C_GREEN),
            name="Present",
            showlegend=True,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=9, color=C_RED),
            name="Missing",
            showlegend=True,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=dict(text="Sensor missingness structure (every 10th sensor)"),
        xaxis_title="Observation index (time-ordered)",
        yaxis_title="Sensor id",
        annotations=[
            dict(
                text=f"{high_miss_pct:.0f}% of sensors are &gt;50% missing",
                xref="paper",
                yref="paper",
                x=0,
                y=1.12,
                showarrow=False,
                font=dict(size=11),
                align="left",
            )
        ],
        showlegend=True,
    )
    return _sized(fig, height=400, margin=dict(t=80))


def fig_missing_rate_distribution(
    df: pd.DataFrame,
    sensor_cols: list[str],
) -> go.Figure:
    """Histogram of per-sensor missing rates (fraction of rows with NaN)."""
    if not sensor_cols:
        return _sized(go.Figure(), height=220)

    n_sensors = len(sensor_cols)
    miss_pct = df[sensor_cols].isna().mean().mul(100)
    miss_pct = miss_pct[miss_pct > 0]
    if miss_pct.empty:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text="Per-sensor missing rate distribution (>0% only)"),
            xaxis_title="Missing rate (% of rows)",
            yaxis_title="Number of sensors",
            annotations=[
                dict(
                    text="All shown sensors have 0% missing rate.",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(),
                )
            ],
            showlegend=False,
        )
        return _sized(fig, height=220)

    rounded = miss_pct.round(1)
    bin_counts = rounded.value_counts()
    modal_rate = float(bin_counts.index[0])
    modal_n = int(bin_counts.iloc[0])
    exclude_modal = modal_n >= 100 or modal_n >= 0.15 * n_sensors

    if exclude_modal:
        plotted = miss_pct[rounded != modal_rate]
        exclude_note = (
            f"Excluded {modal_n} sensors at {modal_rate:.1f}% missing "
            "(dominant low-missing cluster)"
        )
    else:
        plotted = miss_pct
        exclude_note = None

    if plotted.empty:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text="Per-sensor missing rate distribution (>0% only)"),
            xaxis_title="Missing rate (% of rows)",
            yaxis_title="Number of sensors",
            annotations=[
                dict(
                    text=exclude_note or "No sensors remain after exclusions.",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(),
                )
            ],
            showlegend=False,
        )
        return _sized(fig, height=220)

    fig = go.Figure(
        data=[
            go.Histogram(
                x=plotted,
                nbinsx=20,
                marker_color=C_PURPLE,
                opacity=0.85,
                hovertemplate="Missing rate %{x:.1f}%<br>%{y} sensors<extra></extra>",
            )
        ]
    )
    layout_kw: dict[str, Any] = dict(
        title=dict(text="Per-sensor missing rate distribution (>0% only, log scale)"),
        xaxis_title="Missing rate (% of rows)",
        yaxis_title="Number of sensors (log)",
        yaxis_type="log",
        showlegend=False,
    )
    if exclude_note:
        layout_kw["annotations"] = [
            dict(
                text=exclude_note,
                xref="paper",
                yref="paper",
                x=0,
                y=1.14,
                showarrow=False,
                font=dict(size=11),
                align="left",
            )
        ]
    fig.update_layout(**layout_kw)
    return _sized(fig, height=240, margin=dict(t=72))


def fig_sensor_multicollinearity(
    df: pd.DataFrame,
    sensor_cols: list[str],
    *,
    top_n: int = 36,
    corr_threshold: float = 0.9,
    min_periods: int = 100,
) -> tuple[go.Figure, dict[str, Any]]:
    """Correlation heatmap on a capped sensor subset for multicollinearity review."""
    if not sensor_cols:
        return _sized(go.Figure(), height=420), {
            "n_candidates": 0,
            "n_used": 0,
            "high_corr_pairs": 0,
            "max_pair": None,
            "max_abs_corr": 0.0,
        }

    numeric = df[sensor_cols].apply(pd.to_numeric, errors="coerce")
    variances = numeric.var(numeric_only=True).replace([np.inf, -np.inf], np.nan).dropna()
    variances = variances[variances > 0].sort_values(ascending=False)
    selected = variances.head(top_n).index.tolist()
    if len(selected) < 2:
        return _sized(go.Figure(), height=420), {
            "n_candidates": len(sensor_cols),
            "n_used": len(selected),
            "high_corr_pairs": 0,
            "max_pair": None,
            "max_abs_corr": 0.0,
        }

    corr = numeric[selected].corr(min_periods=min_periods)
    corr = corr.fillna(0.0)
    order = corr.abs().mean().sort_values(ascending=False).index.tolist()
    corr = corr.loc[order, order]

    values = corr.to_numpy()
    tri_upper = np.triu(np.ones(values.shape, dtype=bool), k=1)
    abs_vals = np.abs(values[tri_upper])
    high_pairs = int((abs_vals >= corr_threshold).sum()) if abs_vals.size else 0
    if abs_vals.size:
        max_idx = np.unravel_index(np.argmax(np.abs(values * tri_upper)), values.shape)
        max_pair = (corr.index[max_idx[0]], corr.columns[max_idx[1]])
        max_abs = float(abs(values[max_idx]))
    else:
        max_pair = None
        max_abs = 0.0

    labels = corr.columns.tolist()
    fig = go.Figure(
        data=go.Heatmap(
            z=values,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale=COLORSCALE_CORRELATION,
            colorbar=dict(title="r"),
            xgap=1,
            ygap=1,
            hovertemplate="%{x} vs %{y}<br>corr=%{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=f"Sensor multicollinearity (top {len(corr.columns)} variance sensors)"),
        xaxis_title="Sensor",
        yaxis_title="Sensor",
    )
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=labels)
    fig.update_yaxes(type="category", categoryorder="array", categoryarray=labels, autorange="reversed")
    stats = {
        "n_candidates": len(sensor_cols),
        "n_used": len(corr.columns),
        "high_corr_pairs": high_pairs,
        "max_pair": max_pair,
        "max_abs_corr": max_abs,
    }
    return _sized(fig, height=440, margin=dict(l=70, r=40, t=72, b=60)), stats


def fig_reduction_impact(stage_counts: list[tuple[str, int]]) -> go.Figure:
    """Horizontal bar chart of retained feature counts by preprocessing stage."""
    if not stage_counts:
        return _sized(go.Figure(), height=320)

    labels = [label for label, _ in stage_counts]
    counts = [count for _, count in stage_counts]
    fig = go.Figure(
        data=[
            go.Bar(
                x=counts,
                y=labels,
                orientation="h",
                marker_color=C[2],
                text=[f"{n:,}" for n in counts],
                textposition="outside",
                hovertemplate="%{y}: %{x:,} sensors<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=dict(text="Dimensionality reduction impact (snapshot)"),
        xaxis_title="Sensors retained",
        yaxis_title="Stage",
        showlegend=False,
    )
    fig.update_yaxes(autorange="reversed")
    return _sized(fig, height=360, margin=dict(l=140, r=48, t=72, b=50))


def fig_hotelling_t2_intuition() -> go.Figure:
    """Simple ellipse-style intuition plot for multivariate distance."""
    rng = np.random.default_rng(42)
    base_x = rng.normal(0, 1.0, 220)
    base_y = 0.6 * base_x + rng.normal(0, 0.7, 220)
    drift_x = rng.normal(2.4, 0.45, 16)
    drift_y = rng.normal(2.0, 0.5, 16)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=base_x,
            y=base_y,
            mode="markers",
            name="In-control profile",
            marker=dict(color=C[3], size=6, opacity=0.45),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=drift_x,
            y=drift_y,
            mode="markers",
            name="Drifted profile",
            marker=dict(color=C[0], size=8, opacity=0.9, symbol="diamond"),
        )
    )
    theta = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(
        go.Scatter(
            x=2.0 * np.cos(theta),
            y=1.2 * np.sin(theta),
            mode="lines",
            name="T2 control region",
            line=dict(color=C[5], dash="dash"),
        )
    )
    fig.update_layout(
        title=dict(text="Hotelling T2 intuition: distance from multivariate center"),
        xaxis_title="Latent axis 1",
        yaxis_title="Latent axis 2",
    )
    return _sized(fig, height=340)


def fig_rf_topk_selection_example(top_k: int) -> go.Figure:
    """Illustrative feature-importance ranking with top-k cutoff."""
    names = [f"c_{i}" for i in range(20)]
    importances = np.linspace(0.19, 0.03, num=20)
    colors = [C[1] if i < top_k else C[5] for i in range(20)]
    fig = go.Figure(
        data=[
            go.Bar(
                x=importances,
                y=names,
                orientation="h",
                marker_color=colors,
                hovertemplate="%{y}: importance=%{x:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=dict(text=f"RF importance ranking (illustrative, top-{top_k} selected)"),
        xaxis_title="Relative importance",
        yaxis_title="Sensor",
        showlegend=False,
    )
    fig.update_yaxes(autorange="reversed")
    return _sized(fig, height=360, margin=dict(l=80, r=20, t=70, b=45))


def fig_spearman_cluster_example() -> go.Figure:
    """Illustrative Spearman-correlation cluster for c_340/c_204/c_67."""
    sensors = ["c_340", "c_204", "c_67"]
    corr = np.array(
        [
            [1.0, 0.93, 0.89],
            [0.93, 1.0, 0.91],
            [0.89, 0.91, 1.0],
        ]
    )
    return fig_spearman_cluster({"members": sensors, "correlations": corr.tolist()})


def fig_rf_topk_selection(model_artifact: dict) -> go.Figure:
    """RF importance ranking from holdout-fit SelectFromModel."""
    rf = model_artifact.get("rf_selection") or {}
    ranked = rf.get("importances") or []
    top_k = rf.get("top_k", len(rf.get("selected_features", [])))

    if not ranked:
        return fig_rf_topk_selection_example(top_k=top_k or 15)

    names = [r["feature"] for r in ranked]
    importances = [r["importance"] for r in ranked]
    selected_set = set(rf.get("selected_features") or [])
    colors = [C[1] if r["feature"] in selected_set else C[5] for r in ranked]
    fig = go.Figure(
        data=[
            go.Bar(
                x=importances,
                y=names,
                orientation="h",
                marker_color=colors,
                hovertemplate="%{y}: importance=%{x:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=dict(text=f"RF importance ranking (top-{top_k} selected, holdout fit)"),
        xaxis_title="Relative importance",
        yaxis_title="Sensor",
        showlegend=False,
    )
    fig.update_yaxes(autorange="reversed")
    return _sized(fig, height=360, margin=dict(l=80, r=20, t=70, b=45))


def fig_spearman_cluster(cluster_example: dict | None) -> go.Figure:
    """Spearman ρ heatmap from pipeline cluster example."""
    if not cluster_example:
        return fig_spearman_cluster_example()
    sensors = cluster_example.get("members") or []
    corr_raw = cluster_example.get("correlations")
    if not sensors or corr_raw is None:
        return fig_spearman_cluster_example()

    corr = np.asarray(corr_raw, dtype=float)
    title_members = ", ".join(sensors[:4])
    if len(sensors) > 4:
        title_members += ", …"

    fig = go.Figure(
        data=go.Heatmap(
            z=corr,
            x=sensors,
            y=sensors,
            zmin=0.0,
            zmax=1.0,
            colorscale=COLORSCALE_LOW_GREEN_HIGH_RED,
            colorbar=dict(title="Spearman ρ"),
            text=np.round(corr, 2),
            texttemplate="%{text:.2f}",
            hovertemplate="%{y} vs %{x}<br>ρ=%{z:.2f}<extra></extra>",
            xgap=2,
            ygap=2,
        )
    )
    fig.update_layout(
        title=dict(text=f"Spearman cluster: {title_members}"),
        xaxis_title="Sensor",
        yaxis_title="Sensor",
        showlegend=False,
    )
    return _sized(fig, height=340, margin=dict(l=70, r=40, t=78, b=60))


def fig_reduction_impact_from_stages(profile: dict[str, int]) -> go.Figure:
    """Sensor-count reduction from stg → dbt mart → sklearn clustering."""
    stg = profile.get("stg_sensors", 591)
    mart = profile.get("mart_sensors", profile.get("raw_sensors", 0))
    after_cluster = profile.get("after_cluster", profile.get("retained_for_selection", 0))
    classifier_input = profile.get("classifier_input", 0)

    stage_counts = [
        ("Raw SECOM sensors", stg),
        ("After dbt profiling", mart),
        ("After clustering drop", after_cluster),
    ]
    if classifier_input:
        stage_counts.append(
            ("Classifier input (MSPC ref + aux)", classifier_input),
        )
    return fig_reduction_impact(stage_counts)


def fig_benchmark_leaderboard(
    df: pd.DataFrame,
    *,
    metric_col: str,
    label_col: str = "pipeline",
    error_col: str | None = None,
    title: str = "Benchmark leaderboard",
    ascending: bool = False,
    marker_color: str | None = None,
) -> go.Figure:
    """Horizontal leaderboard: mean point estimates with ±1 SD error bars."""
    if df.empty:
        return _sized(go.Figure(), height=360)

    plot_df = df.sort_values(metric_col, ascending=ascending).copy().reset_index(drop=True)
    color = marker_color or C_PURPLE
    error_rgba = CV_ERROR_PURPLE_RGBA if color == C_PURPLE else "rgba(200, 200, 200, 0.6)"

    error_x = None
    if error_col and error_col in plot_df.columns:
        error_x = dict(
            type="data",
            array=plot_df[error_col].astype(float),
            visible=True,
            color=error_rgba,
            width=5,
            thickness=1.5,
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df[metric_col],
            y=plot_df[label_col],
            mode="markers",
            marker=dict(color=color, size=10, symbol="circle"),
            error_x=error_x,
            name="CV Fold Spread (Mean ± 1 SD)",
            hovertemplate="%{y}<br>mean=%{x:.3f}<extra></extra>",
        )
    )

    xaxis_title = metric_col.replace("_", " ").replace("mean ", "Mean ")
    xaxis_kwargs: dict = dict(
        gridcolor="rgba(200, 200, 200, 0.15)",
        zeroline=False,
    )
    if "auc" in metric_col.lower():
        xaxis_kwargs["range"] = [0, 1.0]

    fig.update_layout(
        title=dict(text=title),
        xaxis_title=xaxis_title,
        xaxis=xaxis_kwargs,
        yaxis=dict(autorange="reversed", gridcolor="rgba(200, 200, 200, 0.1)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        showlegend=error_x is not None,
    )
    return _sized(fig, height=380, margin=dict(l=100, r=48, t=72, b=48))


def fig_cv_vs_holdout_validation(
    cv_df: pd.DataFrame,
    ho_df: pd.DataFrame,
    *,
    metric_type: str = "PR-AUC",
    show_ci: bool = True,
) -> go.Figure:
    """Overlay CV fold spread, holdout point estimates, and bootstrap CIs per pipeline."""
    if cv_df.empty or ho_df.empty:
        return _sized(go.Figure(), height=360)

    df = pd.merge(cv_df, ho_df, on="pipeline", suffixes=("_cv", "_ho"))

    if metric_type == "PR-AUC":
        cv_mean, cv_std, ho_val = "mean_pr_auc", "std_pr_auc", "pr_auc"
        ci_low, ci_high = "pr_auc_ci_low", "pr_auc_ci_high"
    else:
        cv_mean, cv_std, ho_val = "mean_roc_auc", "std_roc_auc", "roc_auc"
        ci_low = "roc_auc_ci_low" if "roc_auc_ci_low" in df.columns else None
        ci_high = "roc_auc_ci_high" if "roc_auc_ci_high" in df.columns else None

    df = df.sort_values(ho_val, ascending=False).reset_index(drop=True)

    fig = go.Figure()
    ci_available = (
        show_ci
        and ci_low
        and ci_high
        and ci_low in df.columns
        and ci_high in df.columns
    )

    if ci_available:
        for idx, row in df.iterrows():
            low_val = row.get(ci_low)
            high_val = row.get(ci_high)
            if pd.notna(low_val) and pd.notna(high_val):
                fig.add_trace(
                    go.Scatter(
                        x=[low_val, high_val],
                        y=[row["pipeline"], row["pipeline"]],
                        mode="lines",
                        line=dict(color=CI_YELLOW_RGBA, width=10),
                        name="Holdout 95% Bootstrap CI",
                        showlegend=idx == 0,
                        hoverinfo="skip",
                    )
                )

    fig.add_trace(
        go.Scatter(
            x=df[cv_mean],
            y=df["pipeline"],
            mode="markers",
            marker=dict(color=C_PURPLE, size=10, symbol="circle"),
            error_x=dict(
                type="data",
                array=df[cv_std],
                visible=True,
                color=CV_ERROR_PURPLE_RGBA,
                width=5,
                thickness=1.5,
            ),
            name="CV Fold Spread (Mean ± 1 SD)",
            hovertemplate=(
                "%{y}<br>CV mean=%{x:.3f}<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df[ho_val],
            y=df["pipeline"],
            mode="markers",
            marker=dict(
                color=C_YELLOW,
                size=12,
                symbol="diamond",
                line=dict(color=C_ORANGE, width=1),
            ),
            name="Final Holdout Score",
            hovertemplate="%{y}<br>Holdout=%{x:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=f"CV vs Holdout Validation ({metric_type})"),
        xaxis_title=f"Metric Value ({metric_type})",
        xaxis=dict(range=[0, 1.0], gridcolor="rgba(200, 200, 200, 0.15)", zeroline=False),
        yaxis=dict(autorange="reversed", gridcolor="rgba(200, 200, 200, 0.1)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
    )
    return _sized(fig, height=380, margin=dict(l=100, r=48, t=72, b=48))


def fig_holdout_confusion(
    confusion_matrix: list[list[int]],
    pipeline_name: str,
    *,
    height: int = 320,
) -> go.Figure:
    """Confusion matrix heatmap for holdout evaluation."""
    cm = np.asarray(confusion_matrix)
    labels = [["TN", "FP"], ["FN", "TP"]]
    text = [[f"{labels[i][j]}<br>{cm[i, j]}" for j in range(2)] for i in range(2)]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Pred pass (0)", "Pred fail (1)"],
            y=["Actual pass (0)", "Actual fail (1)"],
            colorscale=COLORSCALE_LOW_GREEN_HIGH_RED,
            text=text,
            texttemplate="%{text}",
            hovertemplate="%{y}, %{x}<br>count=%{z}<extra></extra>",
            showscale=False,
        )
    )
    fig.update_layout(
        title=dict(text=f"Holdout confusion matrix — {pipeline_name}"),
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    fig.update_yaxes(autorange="reversed")
    return _sized(fig, height=height)


def _pr_axis_limits(
    cv_curve: PRCurve | None,
    ho_curve: PRCurve | None,
    ber_point: tuple[float, float] | None,
) -> tuple[float, float]:
    recall_vals: list[float] = []
    precision_vals: list[float] = []
    for curve in (cv_curve, ho_curve):
        if curve is not None and curve.recall.size:
            recall_vals.append(float(np.max(curve.recall)))
            precision_vals.append(float(np.max(curve.precision)))
    if ber_point is not None:
        recall_vals.append(float(ber_point[0]))
        precision_vals.append(float(ber_point[1]))
    recall_hi = min(1.0, (max(recall_vals) if recall_vals else 0.5) * 1.10 + 0.05)
    precision_hi = min(1.0, (max(precision_vals) if precision_vals else 0.5) * 1.15 + 0.02)
    return recall_hi, precision_hi


def fig_pr_curve_cv_holdout(
    cv_curve: PRCurve | None,
    ho_curve: PRCurve | None,
    *,
    ber_point: tuple[float, float] | None = None,
    title: str = "Precision–recall curve",
) -> go.Figure:
    """CV (purple) and holdout (yellow) PR curves with BER operating point and baseline."""
    fig = go.Figure()
    if cv_curve is not None:
        fig.add_trace(
            go.Scatter(
                x=cv_curve.recall,
                y=cv_curve.precision,
                mode="lines",
                name="CV (5-fold OOF)",
                line=dict(color=C_PURPLE, width=2.5),
                hovertemplate="Recall=%{x:.3f}<br>Precision=%{y:.3f}<extra></extra>",
            )
        )
    if ho_curve is not None:
        fig.add_trace(
            go.Scatter(
                x=ho_curve.recall,
                y=ho_curve.precision,
                mode="lines",
                name="Holdout",
                line=dict(color=C_YELLOW, width=2.5),
                hovertemplate="Recall=%{x:.3f}<br>Precision=%{y:.3f}<extra></extra>",
            )
        )

    baseline = None
    if cv_curve is not None:
        baseline = cv_curve.baseline
    elif ho_curve is not None:
        baseline = ho_curve.baseline
    if baseline is not None:
        fig.add_hline(
            y=baseline,
            line_dash="dash",
            line_color=C_BLUE,
            annotation_text="Random baseline",
            annotation_position="right",
        )

    if ber_point is not None:
        fig.add_trace(
            go.Scatter(
                x=[ber_point[0]],
                y=[ber_point[1]],
                mode="markers",
                name="BER min (holdout)",
                marker=dict(color=C_GREEN, size=12, symbol="diamond"),
                hovertemplate=(
                    "BER threshold<br>Recall=%{x:.3f}<br>Precision=%{y:.3f}<extra></extra>"
                ),
            )
        )

    recall_hi, precision_hi = _pr_axis_limits(cv_curve, ho_curve, ber_point)
    fig.update_layout(
        title=dict(text=title),
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, recall_hi], gridcolor="rgba(200, 200, 200, 0.15)"),
        yaxis=dict(range=[0, precision_hi], gridcolor="rgba(200, 200, 200, 0.15)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return _sized(fig, height=440, margin=dict(l=56, r=48, t=72, b=48))


def fig_sensor_histogram(
    df: pd.DataFrame,
    sensor: str,
    *,
    target_col: str,
    log_scale: bool,
) -> go.Figure:
    plot_df = df[[sensor, target_col]].dropna().copy()
    plot_df["label"] = plot_df[target_col].map({0: "Pass", 1: "Fail"})

    fig = go.Figure()
    for label, color in (("Pass", C_GREEN), ("Fail", C_RED)):
        values = plot_df.loc[plot_df["label"] == label, sensor]
        fig.add_trace(
            go.Histogram(
                x=values,
                name=label,
                marker_color=color,
                opacity=0.55,
                nbinsx=40,
            )
        )

    fig.update_layout(
        barmode="overlay",
        title=dict(text=f"Sensor {sensor}: pass vs fail"),
        xaxis_title=sensor,
        yaxis=dict(title="Count", type="log" if log_scale else "linear"),
        showlegend=True,
    )
    return _sized(fig, height=380)


def fig_top_features_bar(
    df: pd.DataFrame,
    *,
    value_col: str = "importance",
    feature_col: str = "feature",
    title: str = "Top features",
    height: int = 420,
) -> go.Figure:
    """Horizontal bar chart of ranked feature importance."""
    if df.empty:
        return _sized(go.Figure(), height=height)
    plot_df = df.sort_values(value_col, ascending=True)
    fig = go.Figure(
        data=[
            go.Bar(
                x=plot_df[value_col],
                y=plot_df[feature_col],
                orientation="h",
                marker_color=C[2],
                text=[f"{v:.4g}" for v in plot_df[value_col]],
                textposition="outside",
                hovertemplate="%{y}<br>%{x:.4g}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=dict(text=title),
        xaxis_title=value_col.replace("_", " "),
        yaxis_title="Feature",
        showlegend=False,
    )
    return _sized(fig, height=height, margin=dict(l=140, r=48, t=72, b=48))


def fig_coef_signed_bar(
    df: pd.DataFrame,
    *,
    title: str = "Largest positive and negative coefficients",
    height: int = 400,
) -> go.Figure:
    """Diverging-style bars: green positive, red negative coefficients."""
    if df.empty:
        return _sized(go.Figure(), height=height)
    plot_df = df.sort_values("coefficient", ascending=True)
    colors = [C_GREEN if c >= 0 else C_RED for c in plot_df["coefficient"]]
    fig = go.Figure(
        data=[
            go.Bar(
                x=plot_df["coefficient"],
                y=plot_df["feature"],
                orientation="h",
                marker_color=colors,
                hovertemplate="%{y}<br>coef=%{x:.4g}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=dict(text=title),
        xaxis_title="Coefficient (scaled features)",
        yaxis_title="Feature",
        showlegend=False,
    )
    return _sized(fig, height=height, margin=dict(l=140, r=48, t=72, b=48))


def fig_local_contributions(
    df: pd.DataFrame,
    *,
    title: str = "Top local contributions",
    height: int = 320,
) -> go.Figure:
    """Wafer-level feature contributions (signed or unsigned)."""
    if df.empty:
        return _sized(fig, height=height)
    plot_df = df.sort_values("contribution", key=lambda s: s.abs(), ascending=True)
    colors = [C_RED if v < 0 else C_GREEN for v in plot_df["contribution"]]
    fig = go.Figure(
        data=[
            go.Bar(
                x=plot_df["contribution"],
                y=plot_df["feature"],
                orientation="h",
                marker_color=colors,
                hovertemplate="%{y}<br>contrib=%{x:.4g}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=dict(text=title),
        xaxis_title="Contribution",
        yaxis_title="Feature",
        showlegend=False,
    )
    return _sized(fig, height=height, margin=dict(l=120, r=40, t=64, b=40))
