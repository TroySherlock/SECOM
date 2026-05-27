"""Plotly chart builders for the SECOM Streamlit dashboard."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scripts.dashboard_stg import ROW_INDEX_COL, cohens_d
from scripts.dashboard_theme import apply_plotly_theme
from theme.gruvbox_material import GRUVBOX

_SENSOR_PATTERN = re.compile(r"^c_\d+$")


def _sensor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if _SENSOR_PATTERN.fullmatch(str(c))]


def best_pair_sensors(
    df: pd.DataFrame,
    sensor_cols: list[str],
    target_col: str,
) -> tuple[str, str]:
    """Return two sensors with largest |Cohen's d| for pass vs fail."""
    ranked = [(col, cohens_d(df, col, target_col)) for col in sensor_cols]
    ranked.sort(key=lambda x: x[1], reverse=True)
    first = ranked[0][0] if ranked else sensor_cols[0]
    second = next((s for s, _ in ranked if s != first), ranked[1][0] if len(ranked) > 1 else first)
    return first, second


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
    colors = [GRUVBOX["pass"], GRUVBOX["fail"]]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                marker=dict(colors=colors, line=dict(color=GRUVBOX["bg"], width=2)),
                textinfo="label+percent",
                textfont=dict(color=GRUVBOX["fg"]),
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
                font=dict(size=16, color=GRUVBOX["fg"]),
                showarrow=False,
            )
        ],
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, x=0.5, xanchor="center"),
    )
    return apply_plotly_theme(fig, height=360, legend=dict(orientation="h", yanchor="bottom", y=-0.12, x=0.5, xanchor="center"))


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

    for label, color, symbol in [("Pass", GRUVBOX["pass"], "circle"), ("Fail", GRUVBOX["fail"], "diamond")]:
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
                line=dict(color=GRUVBOX["accent"], width=2),
                marker=dict(size=5),
                hovertemplate="Week of %{x|%Y-%m-%d}<br>Fail rate %{y:.1%}<extra></extra>",
            )
        )
        layout_y2 = dict(
            overlaying="y",
            side="right",
            range=[0, 1],
            tickformat=".0%",
            gridcolor=GRUVBOX["border"],
            tickfont=dict(color=GRUVBOX["fg_muted"]),
            title=dict(text="Weekly fail rate", font=dict(color=GRUVBOX["fg"])),
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
    return apply_plotly_theme(fig, height=400)


def fig_missingness_structure(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    sensor_cols: list[str] | None = None,
    column_stride: int = 10,
) -> go.Figure:
    sensor_cols = sensor_cols or _sensor_columns(df)
    if not sensor_cols:
        return apply_plotly_theme(go.Figure(), height=360)

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
            colorscale=[[0, GRUVBOX["pass"]], [1, GRUVBOX["fail"]]],
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
            marker=dict(size=9, color=GRUVBOX["pass"]),
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
            marker=dict(size=9, color=GRUVBOX["fail"]),
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
                font=dict(size=11, color=GRUVBOX["fg_muted"]),
                align="left",
            )
        ],
        showlegend=True,
    )
    return apply_plotly_theme(fig, height=400, margin=dict(t=80))


def fig_missing_rate_distribution(
    df: pd.DataFrame,
    sensor_cols: list[str],
) -> go.Figure:
    """Histogram of per-sensor missing rates (fraction of rows with NaN)."""
    if not sensor_cols:
        return apply_plotly_theme(go.Figure(), height=220)

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
                    font=dict(color=GRUVBOX["fg_muted"]),
                )
            ],
            showlegend=False,
        )
        return apply_plotly_theme(fig, height=220)

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
                    font=dict(color=GRUVBOX["fg_muted"]),
                )
            ],
            showlegend=False,
        )
        return apply_plotly_theme(fig, height=220)

    fig = go.Figure(
        data=[
            go.Histogram(
                x=plotted,
                nbinsx=20,
                marker_color=GRUVBOX["yellow"],
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
                font=dict(size=11, color=GRUVBOX["fg_muted"]),
                align="left",
            )
        ]
    fig.update_layout(**layout_kw)
    return apply_plotly_theme(fig, height=240, margin=dict(t=72))


def fig_sensor_scatter(
    df: pd.DataFrame,
    *,
    sensor_x: str,
    sensor_y: str,
    target_col: str,
    row_index_col: str = ROW_INDEX_COL,
) -> go.Figure:
    plot_df = df[[sensor_x, sensor_y, target_col, row_index_col]].dropna(subset=[sensor_x, sensor_y]).copy()
    plot_df["label"] = plot_df[target_col].map({0: "Pass", 1: "Fail"})

    fig = go.Figure()
    for label, color in [("Pass", GRUVBOX["pass"]), ("Fail", GRUVBOX["fail"])]:
        subset = plot_df.loc[plot_df["label"] == label]
        fig.add_trace(
            go.Scatter(
                x=subset[sensor_x],
                y=subset[sensor_y],
                mode="markers",
                name=label,
                marker=dict(color=color, size=6, opacity=0.7 if label == "Fail" else 0.4),
                customdata=subset[row_index_col],
                hovertemplate=(
                    f"{label}<br>"
                    f"{sensor_x}=%{{x:.4g}}<br>"
                    f"{sensor_y}=%{{y:.4g}}<br>"
                    "row=%{customdata}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(text=f"Scatter: {sensor_x} vs {sensor_y}"),
        xaxis_title=sensor_x,
        yaxis_title=sensor_y,
        showlegend=True,
    )
    return apply_plotly_theme(fig, height=400)


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
        return apply_plotly_theme(go.Figure(), height=420), {
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
        return apply_plotly_theme(go.Figure(), height=420), {
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
            colorscale=[
                [0.0, GRUVBOX["accent"]],
                [0.5, GRUVBOX["bg_soft"]],
                [1.0, GRUVBOX["fail"]],
            ],
            colorbar=dict(title="r", tickcolor=GRUVBOX["fg_muted"]),
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
    return apply_plotly_theme(fig, height=440, margin=dict(l=70, r=40, t=72, b=60)), stats


def fig_reduction_impact(stage_counts: list[tuple[str, int]]) -> go.Figure:
    """Horizontal bar chart of retained feature counts by preprocessing stage."""
    if not stage_counts:
        return apply_plotly_theme(go.Figure(), height=320)

    labels = [label for label, _ in stage_counts]
    counts = [count for _, count in stage_counts]
    fig = go.Figure(
        data=[
            go.Bar(
                x=counts,
                y=labels,
                orientation="h",
                marker_color=GRUVBOX["yellow"],
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
    return apply_plotly_theme(fig, height=360, margin=dict(l=140, r=48, t=72, b=50))


def fig_pipeline_flow_order() -> go.Figure:
    """Sankey-like flow for shared steps then split methods."""
    labels = [
        "Raw sensors",
        "Median imputation",
        "Redundancy clustering",
        "Hotelling T2 features",
        "PLS + Q branch",
        "RF top-k branch",
    ]
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    label=labels,
                    pad=18,
                    thickness=18,
                    color=[
                        GRUVBOX["bg_soft"],
                        GRUVBOX["yellow"],
                        GRUVBOX["accent"],
                        GRUVBOX["orange"],
                        GRUVBOX["pass"],
                        GRUVBOX["fail"],
                    ],
                    line=dict(color=GRUVBOX["border"], width=1),
                ),
                link=dict(
                    source=[0, 1, 2, 3, 3],
                    target=[1, 2, 3, 4, 5],
                    value=[591, 591, 250, 40, 15],
                    color=[
                        "rgba(216,166,87,0.35)",
                        "rgba(78,154,204,0.35)",
                        "rgba(231,138,78,0.35)",
                        "rgba(80,161,79,0.35)",
                        "rgba(234,105,98,0.35)",
                    ],
                ),
            )
        ]
    )
    fig.update_layout(title=dict(text="Pipeline operation order (shared steps then split)"))
    return apply_plotly_theme(fig, height=320, margin=dict(l=20, r=20, t=70, b=20))


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
            marker=dict(color=GRUVBOX["accent"], size=6, opacity=0.45),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=drift_x,
            y=drift_y,
            mode="markers",
            name="Drifted profile",
            marker=dict(color=GRUVBOX["fail"], size=8, opacity=0.9, symbol="diamond"),
        )
    )
    theta = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(
        go.Scatter(
            x=2.0 * np.cos(theta),
            y=1.2 * np.sin(theta),
            mode="lines",
            name="T2 control region",
            line=dict(color=GRUVBOX["fg_muted"], dash="dash"),
        )
    )
    fig.update_layout(
        title=dict(text="Hotelling T2 intuition: distance from multivariate center"),
        xaxis_title="Latent axis 1",
        yaxis_title="Latent axis 2",
    )
    return apply_plotly_theme(fig, height=340)


def fig_pls_compression_example() -> go.Figure:
    """Illustrative bar chart for compression from sensors to latent outputs."""
    labels = ["Input sensors", "After shared clustering", "PLS components", "Q statistic", "Total PLS path features"]
    values = [591, 250, 15, 1, 16]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=[GRUVBOX["yellow"], GRUVBOX["accent"], GRUVBOX["pass"], GRUVBOX["orange"], GRUVBOX["pass"]],
                text=[f"{v:,}" for v in values],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title=dict(text="PLS + Q compression (illustrative feature counts)"),
        xaxis_title="Stage",
        yaxis_title="Feature count",
        showlegend=False,
    )
    return apply_plotly_theme(fig, height=330)


def fig_rf_topk_selection_example(top_k: int) -> go.Figure:
    """Illustrative feature-importance ranking with top-k cutoff."""
    names = [f"c_{i}" for i in range(20)]
    importances = np.linspace(0.19, 0.03, num=20)
    colors = [GRUVBOX["fail"] if i < top_k else GRUVBOX["fg_muted"] for i in range(20)]
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
    return apply_plotly_theme(fig, height=360, margin=dict(l=80, r=20, t=70, b=45))


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
    fig = go.Figure(
        data=go.Heatmap(
            z=corr,
            x=sensors,
            y=sensors,
            zmin=0.0,
            zmax=1.0,
            colorscale=[
                [0.0, GRUVBOX["bg_soft"]],
                [0.55, GRUVBOX["yellow"]],
                [1.0, GRUVBOX["fail"]],
            ],
            colorbar=dict(title="Spearman ρ"),
            text=np.round(corr, 2),
            texttemplate="%{text:.2f}",
            hovertemplate="%{y} vs %{x}<br>ρ=%{z:.2f}<extra></extra>",
            xgap=2,
            ygap=2,
        )
    )
    fig.update_layout(
        title=dict(text="Spearman cluster example: c_340, c_204, c_67"),
        xaxis_title="Sensor",
        yaxis_title="Sensor",
        showlegend=False,
    )
    return apply_plotly_theme(fig, height=340, margin=dict(l=70, r=40, t=78, b=60))


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
    for label, color in [("Pass", GRUVBOX["pass"]), ("Fail", GRUVBOX["fail"])]:
        values = plot_df.loc[plot_df["label"] == label, sensor]
        fig.add_trace(
            go.Histogram(
                x=values,
                name=label,
                opacity=0.55,
                marker_color=color,
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
    return apply_plotly_theme(fig, height=380)
