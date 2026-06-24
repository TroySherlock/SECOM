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
# Standardized drift z ∈ [-, +]: blue below baseline, ~background at 0, red above.
COLORSCALE_DRIFT_DIVERGING = [
    [0.0, C_BLUE],
    [0.5, CHART_BG],
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
    prevalence: float | None = None,
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
        if prevalence is not None:
            fig.add_trace(
                go.Scatter(
                    x=[weekly[timestamp_col].min(), weekly[timestamp_col].max()],
                    y=[prevalence, prevalence],
                    mode="lines",
                    name=f"Overall prevalence ({prevalence:.1%})",
                    yaxis="y2",
                    line=dict(color=C_PURPLE, width=1.5, dash="dot"),
                    hovertemplate=f"Prevalence baseline {prevalence:.1%}<extra></extra>",
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


def fig_two_track_schematic(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    target_col: str,
    test_size: float = 0.20,
) -> go.Figure:
    """Two-lane evaluation schematic: interpolation (random split) vs extrapolation
    (temporal forward holdout), sharing one time axis with the 80/20 cut marked.

    Top lane shows the random holdout sampled across the whole campaign; bottom lane
    shows the temporal holdout (latest ``test_size`` by time) shaded. Frames pages 3-4.
    """
    work = df[[timestamp_col, target_col]].copy()
    work["_ts"] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work = work.sort_values("_ts").reset_index(drop=True)
    n_rows = len(work)
    if n_rows < 4:
        return _sized(go.Figure(), height=300)

    ts = work["_ts"]
    train_n = int(round(n_rows * (1.0 - test_size)))
    cut_ts = ts.iloc[min(train_n, n_rows - 1)]
    rng = np.random.default_rng(42)
    # Random holdout: ~test_size of rows sampled uniformly across the whole span.
    rand_holdout = rng.random(n_rows) < test_size
    is_fail = work[target_col].astype(int).to_numpy() == 1

    fig = go.Figure()

    def _lane(y: float, mask_holdout: np.ndarray, lane: str) -> None:
        for held, name, color, op in [
            (False, "Train", C_BLUE, 0.35),
            (True, "Holdout", C_YELLOW, 0.9),
        ]:
            sel = (mask_holdout == held)
            fig.add_trace(
                go.Scatter(
                    x=ts[sel],
                    y=np.full(int(sel.sum()), y) + rng.uniform(-0.04, 0.04, int(sel.sum())),
                    mode="markers",
                    name=f"{name}",
                    legendgroup=name,
                    showlegend=(lane == "interpolation"),
                    marker=dict(color=color, size=5, opacity=op,
                                symbol="diamond" if held else "circle"),
                    hovertemplate=f"{lane} · {name}<br>%{{x|%Y-%m-%d}}<extra></extra>",
                )
            )

    _lane(1.0, rand_holdout, "interpolation")
    _lane(0.0, ts.index.to_numpy() >= train_n, "extrapolation")

    # Temporal cut marker (only meaningful for the extrapolation lane).
    fig.add_vline(x=cut_ts, line=dict(color=C_YELLOW, width=2, dash="dash"))
    fig.add_annotation(
        x=cut_ts, y=1.12, yref="paper", showarrow=False,
        text=f"80/20 temporal cut ({cut_ts:%Y-%m-%d})",
        font=dict(size=11, color=C_YELLOW),
    )
    fig.update_layout(
        title=dict(text="Two evaluation tracks on one campaign timeline"),
        xaxis_title="Measurement time",
        yaxis=dict(
            tickvals=[0.0, 1.0],
            ticktext=["Extrapolation<br>(temporal holdout)", "Interpolation<br>(random split)"],
            range=[-0.4, 1.4],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return _sized(fig, height=320, margin=dict(l=140, r=40, t=86, b=50))


def fig_sensor_drift_heatmap(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    sensor_cols: list[str] | None = None,
    n_sensors: int = 30,
    n_time_bins: int = 40,
    test_size: float = 0.20,
    z_clip: float = 4.0,
) -> go.Figure:
    """Heatmap of the most drift-prone sensors over time, standardized against the
    training-era baseline (first ``1 - test_size`` of the timeline).

    Each cell is a sensor's mean value in a time bin expressed in standard deviations
    away from what it looked like during the training era, so later eras light up as
    they drift from what the models were fit on. A vertical marker shows the temporal
    holdout boundary (latest ``test_size``).
    """
    sensor_cols = sensor_cols or sensor_columns(df)
    if not sensor_cols:
        return _sized(go.Figure(), height=460)

    plot_df = df[[timestamp_col] + sensor_cols].copy()
    plot_df["_sort_ts"] = pd.to_datetime(plot_df[timestamp_col], errors="coerce")
    plot_df = plot_df.sort_values("_sort_ts").reset_index(drop=True)
    ts = plot_df["_sort_ts"]

    n_rows = len(plot_df)
    train_n = int(round(n_rows * (1.0 - test_size)))
    if n_rows < 4 or train_n < 2 or train_n >= n_rows:
        return _sized(go.Figure(), height=460)

    # Median-impute, then z-score each sensor against its training-era mean/std.
    values = plot_df[sensor_cols].astype(float)
    values = values.fillna(values.median(numeric_only=True))
    base = values.iloc[:train_n]
    base_mean = base.mean()
    base_std = base.std(ddof=0)
    keep = base_std[base_std > 0].index.tolist()
    if not keep:
        return _sized(go.Figure(), height=460)
    z = (values[keep] - base_mean[keep]) / base_std[keep]

    # Rank sensors by how far the holdout era drifts from the training baseline.
    holdout_drift = z.iloc[train_n:].mean().abs().sort_values(ascending=False)
    top = holdout_drift.head(min(n_sensors, len(keep))).index.tolist()

    # Equal-count time bins (oldest -> newest), one mean z per sensor per bin.
    n_bins = min(n_time_bins, n_rows)
    bins = np.array_split(np.arange(n_rows), n_bins)
    z_top = z[top].to_numpy()
    matrix = np.vstack([z_top[idx].mean(axis=0) for idx in bins]).T  # (sensor, bin)
    matrix = np.clip(matrix, -z_clip, z_clip)

    bin_end_labels = [f"{ts.iloc[idx[-1]]:%Y-%m-%d}" for idx in bins]
    sensor_labels = list(top)

    # Holdout boundary: first bin whose rows cross into the latest test_size.
    boundary_bin = next(
        (b for b, idx in enumerate(bins) if idx[-1] >= train_n), n_bins - 1
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=list(range(n_bins)),
            y=sensor_labels,
            zmid=0.0,
            zmin=-z_clip,
            zmax=z_clip,
            colorscale=COLORSCALE_DRIFT_DIVERGING,
            colorbar=dict(title="σ vs train<br>baseline"),
            customdata=np.broadcast_to(
                np.array(bin_end_labels), matrix.shape
            ),
            hovertemplate=(
                "Sensor %{y}<br>through %{customdata}<br>"
                "%{z:.2f} σ vs train baseline<extra></extra>"
            ),
        )
    )
    fig.add_vline(
        x=boundary_bin - 0.5,
        line=dict(color=C_YELLOW, width=2, dash="dash"),
    )
    fig.add_annotation(
        x=boundary_bin - 0.5,
        y=1.02,
        yref="paper",
        text="← train · holdout (latest 20%) →",
        showarrow=False,
        font=dict(size=11, color=C_YELLOW),
    )

    tick_step = max(1, n_bins // 6)
    tickvals = list(range(0, n_bins, tick_step))
    fig.update_layout(
        title=dict(text=f"Sensor drift over time (top {len(top)} drifting sensors)"),
        xaxis=dict(
            title="Measurement time (binned, oldest → newest)",
            tickvals=tickvals,
            ticktext=[bin_end_labels[i] for i in tickvals],
            tickangle=-30,
        ),
        yaxis=dict(
            title="Sensor id",
            type="category",
            autorange="reversed",
            tickfont=dict(size=10),
        ),
    )
    return _sized(fig, height=520, margin=dict(t=80))


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


def fig_reduction_sankey(profile: dict[str, int]) -> go.Figure:
    """Sankey of the dbt sensor-drop: staged sensors split into the kept mart set and
    the high-missing / zero-variance sensors dropped before the mart.

    Kept to the sensor unit on purpose: this is the one stage where the counts conserve
    (stg = mart + dropped). The downstream sklearn steps change units (raw+rz doubling,
    hub/T2 additions, auxiliary features), so they are shown as a separate bar snapshot.
    """
    stg = int(profile.get("stg_sensors", 0))
    mart = int(profile.get("mart_sensors", 0))
    dbt_dropped = int(profile.get("dbt_dropped_sensors", max(0, stg - mart)))
    if not stg or not mart:
        return _sized(go.Figure(), height=320)

    labels = [
        f"Staged sensors ({stg})",                            # 0
        f"Kept to dbt mart ({mart})",                         # 1
        f"Dropped: >10% missing / zero-variance ({dbt_dropped})",  # 2
    ]
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=labels,
                color=[C_BLUE, C_GREEN, C_RED],
                pad=22,
                thickness=18,
                line=dict(color="rgba(0,0,0,0)", width=0),
            ),
            link=dict(
                source=[0, 0],
                target=[1, 2],
                value=[mart, dbt_dropped],
                color=["rgba(169,182,101,0.45)", "rgba(234,105,98,0.40)"],
            ),
        )
    )
    fig.update_layout(title=dict(text="dbt sensor profiling: kept vs dropped"))
    return _sized(fig, height=320, margin=dict(l=20, r=20, t=72, b=20))


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
    """Horizontal leaderboard of point estimates with ±1 SD error bars."""
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


def fig_pr_curve_clean(
    curve: PRCurve | None,
    *,
    ap: float | None = None,
    ap_ci: tuple[float, float] | None = None,
    ber_point: tuple[float, float] | None = None,
    baseline: float | None = None,
    draw_line: bool = True,
    line_name: str = "CV (out-of-fold)",
    title: str = "Precision–recall curve",
) -> go.Figure:
    """Decluttered PR view: one line max, one operating point, AP as a number.

    ``curve`` is the single smooth curve to draw (CV-OOF for interpolation);
    pass ``draw_line=False`` (extrapolation) to suppress the jagged small-sample
    staircase and show only the BER operating point against the prevalence
    baseline. Holdout AP (+ optional CI) is annotated as text, not a band.
    """
    fig = go.Figure()
    if draw_line and curve is not None and curve.recall.size:
        fig.add_trace(
            go.Scatter(
                x=curve.recall,
                y=curve.precision,
                mode="lines",
                name=line_name,
                line=dict(color=C_PURPLE, width=2.5, shape="hv"),
                hovertemplate="Recall=%{x:.3f}<br>Precision=%{y:.3f}<extra></extra>",
            )
        )

    if baseline is None and curve is not None:
        baseline = curve.baseline
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
                name="BER-min (holdout)",
                marker=dict(color=C_GREEN, size=13, symbol="diamond"),
                hovertemplate=(
                    "BER-min threshold<br>Recall=%{x:.3f}<br>Precision=%{y:.3f}<extra></extra>"
                ),
            )
        )

    if ap is not None:
        ap_txt = f"Holdout AP = {ap:.3f}"
        if ap_ci is not None and all(v is not None for v in ap_ci):
            ap_txt += f"  [{ap_ci[0]:.3f}, {ap_ci[1]:.3f}]"
        fig.add_annotation(
            x=0.98,
            y=0.98,
            xref="paper",
            yref="paper",
            text=ap_txt,
            showarrow=False,
            align="right",
            font=dict(size=13, color=C_YELLOW),
            bgcolor="rgba(0,0,0,0.25)",
            borderpad=4,
        )

    recall_hi, precision_hi = _pr_axis_limits(curve if draw_line else None, None, ber_point)
    fig.update_layout(
        title=dict(text=title),
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, recall_hi], gridcolor="rgba(200, 200, 200, 0.15)"),
        yaxis=dict(range=[0, precision_hi], gridcolor="rgba(200, 200, 200, 0.15)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return _sized(fig, height=440, margin=dict(l=56, r=48, t=72, b=48))


def _ber_curve(y_true: np.ndarray, y_score: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Balanced error rate (%) at each threshold (fail = positive)."""
    pos = y_true == 1
    neg = ~pos
    n_pos = max(int(pos.sum()), 1)
    n_neg = max(int(neg.sum()), 1)
    out = np.empty(thresholds.size, dtype=float)
    for i, t in enumerate(thresholds):
        pred = y_score >= t
        tpr = float(np.sum(pred & pos)) / n_pos
        tnr = float(np.sum(~pred & neg)) / n_neg
        out[i] = 100.0 * (1.0 - 0.5 * (tpr + tnr))
    return out


def fig_ber_threshold_sweep(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    profile_thresholds: dict[str, float] | None = None,
    title: str = "Balanced error rate vs threshold",
) -> go.Figure:
    """BER across the threshold grid with the tuned profile thresholds marked."""
    fig = go.Figure()
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.size == 0 or y_score.size == 0:
        return _sized(
            fig.update_layout(title=dict(text=title)),
            height=420,
            margin=dict(l=56, r=48, t=72, b=48),
        )

    grid = np.unique(np.clip(y_score, 0.0, 1.0))
    if grid.size > 400:
        grid = np.quantile(grid, np.linspace(0.0, 1.0, 400))
    ber = _ber_curve(y_true, y_score, grid)
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=ber,
            mode="lines",
            name="BER",
            line=dict(color=C_AQUA, width=2.5),
            hovertemplate="threshold=%{x:.3f}<br>BER=%{y:.1f}%<extra></extra>",
        )
    )
    # Empirical minimum on this holdout.
    imin = int(np.argmin(ber))
    fig.add_trace(
        go.Scatter(
            x=[grid[imin]],
            y=[ber[imin]],
            mode="markers",
            name="BER-min (empirical)",
            marker=dict(color=C_GREEN, size=12, symbol="diamond"),
            hovertemplate="BER-min<br>threshold=%{x:.3f}<br>BER=%{y:.1f}%<extra></extra>",
        )
    )

    for label, thr in (profile_thresholds or {}).items():
        if thr is None or not np.isfinite(thr):
            continue
        fig.add_vline(
            x=float(thr),
            line_dash="dot",
            line_color=C_ORANGE,
            annotation_text=label,
            annotation_position="top",
        )

    fig.update_layout(
        title=dict(text=title),
        xaxis_title="Decision threshold (fail-class probability)",
        yaxis_title="Balanced error rate (%)",
        xaxis=dict(gridcolor="rgba(200, 200, 200, 0.15)"),
        yaxis=dict(gridcolor="rgba(200, 200, 200, 0.15)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return _sized(fig, height=420, margin=dict(l=56, r=48, t=72, b=48))


def fig_calibration(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bins: int = 5,
    title: str = "Calibration (reliability) curve",
) -> go.Figure:
    """Reliability curve (binned observed vs predicted) + diagonal; Brier in title.

    Axes are zoomed to the predicted-probability range actually emitted (an
    isotonic-calibrated model on imbalanced data predicts mostly small
    probabilities, so a fixed [0,1] view looks empty). A faint predicted-score
    histogram on a secondary axis shows where the mass sits.
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    fig = go.Figure()
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.size == 0 or y_score.size == 0 or len(np.unique(y_true)) < 2:
        return _sized(
            fig.update_layout(title=dict(text=title)),
            height=420,
            margin=dict(l=56, r=48, t=72, b=48),
        )

    brier = float(brier_score_loss(y_true, y_score))
    try:
        prob_true, prob_pred = calibration_curve(
            y_true, y_score, n_bins=n_bins, strategy="quantile"
        )
    except ValueError:
        prob_true, prob_pred = np.array([]), np.array([])

    # Zoom to the data range so the curve fills the panel.
    hi = float(np.max(y_score)) if y_score.size else 1.0
    if prob_pred.size:
        hi = max(hi, float(np.max(prob_pred)), float(np.max(prob_true)))
    axis_hi = float(min(1.0, hi * 1.1 + 0.02))

    # Predicted-score histogram (secondary axis) to show where the mass sits.
    fig.add_trace(
        go.Histogram(
            x=y_score,
            nbinsx=40,
            name="Predicted scores",
            marker=dict(color=C_BLUE),
            opacity=0.25,
            yaxis="y2",
            hovertemplate="score=%{x:.3f}<br>count=%{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, axis_hi],
            y=[0, axis_hi],
            mode="lines",
            name="Perfect calibration",
            line=dict(color=C_BLUE, width=1.5, dash="dash"),
            hoverinfo="skip",
        )
    )
    if prob_pred.size:
        fig.add_trace(
            go.Scatter(
                x=prob_pred,
                y=prob_true,
                mode="lines+markers",
                name="Model",
                line=dict(color=C_PURPLE, width=2.5),
                marker=dict(size=8),
                hovertemplate="predicted=%{x:.3f}<br>observed=%{y:.3f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=dict(text=f"{title} — Brier = {brier:.3f}"),
        xaxis_title="Mean predicted fail probability",
        yaxis_title="Observed fail fraction",
        xaxis=dict(range=[0, axis_hi], gridcolor="rgba(200, 200, 200, 0.15)"),
        yaxis=dict(range=[0, axis_hi], gridcolor="rgba(200, 200, 200, 0.15)"),
        yaxis2=dict(
            overlaying="y",
            side="right",
            showgrid=False,
            title="Predicted-score count",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return _sized(fig, height=420, margin=dict(l=56, r=48, t=72, b=48))


def fig_risk_coverage(
    df: pd.DataFrame,
    *,
    metric: str = "pr_auc",
    operating_coverage: float | None = None,
    title: str = "Gate risk–coverage curve",
) -> go.Figure:
    """One line per pipeline of conditional AUC vs gate coverage.

    Coverage runs high→low left→right so abstention (dropping the most suspicious
    wafers) increases to the right. A dashed vertical line marks the gate's actual
    operating coverage. ``metric`` is ``"pr_auc"`` or ``"roc_auc"``.
    """
    fig = go.Figure()
    if df is None or df.empty or metric not in df.columns:
        return _sized(
            fig.update_layout(title=dict(text=title)),
            height=440,
            margin=dict(l=56, r=48, t=72, b=48),
        )

    metric_label = "PR-AUC" if metric == "pr_auc" else "ROC-AUC"
    pipelines = sorted(df["pipeline"].unique())
    for i, name in enumerate(pipelines):
        sub = df[df["pipeline"] == name].sort_values("coverage", ascending=False)
        fig.add_trace(
            go.Scatter(
                x=sub["coverage"],
                y=sub[metric],
                customdata=sub[["n_kept", "n_fails_kept"]].to_numpy(),
                mode="lines+markers",
                name=name,
                line=dict(color=C[i % len(C)], width=2.5),
                marker=dict(size=6),
                connectgaps=False,
                hovertemplate=(
                    f"{name}<br>Coverage=%{{x:.2f}}<br>{metric_label}=%{{y:.3f}}"
                    "<br>kept=%{customdata[0]} (fails=%{customdata[1]})<extra></extra>"
                ),
            )
        )

    if operating_coverage is not None:
        fig.add_vline(
            x=float(operating_coverage),
            line_dash="dash",
            line_color=C_BLUE,
            annotation_text=f"gate op. coverage {float(operating_coverage):.2f}",
            annotation_position="top",
        )

    fig.update_layout(
        title=dict(text=title),
        xaxis_title="Coverage (fraction of wafers kept)",
        yaxis_title=f"Conditional {metric_label}",
        xaxis=dict(autorange="reversed", gridcolor="rgba(200, 200, 200, 0.15)"),
        yaxis=dict(gridcolor="rgba(200, 200, 200, 0.15)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return _sized(fig, height=440, margin=dict(l=56, r=48, t=72, b=48))


# Gate-monitor populations: reference (train passing), in-dist, temporal holdout.
_GATE_POP_COLORS = {
    "reference": C_BLUE,
    "in_distribution": C_GREEN,
    "temporal": C_RED,
}
_GATE_POP_LABELS = {
    "reference": "Reference (passing train)",
    "in_distribution": "In-distribution holdout",
    "temporal": "Temporal holdout",
}


def fig_gate_statistic_distributions(
    populations: dict[str, np.ndarray],
    *,
    statistic: str,
    limit: float | None = None,
    limit_side: str = "upper",
    title: str = "Gate statistic distribution",
) -> go.Figure:
    """Overlaid violins of a BGM gate statistic across wafer populations.

    ``populations`` maps ``reference`` / ``in_distribution`` / ``temporal`` to
    per-wafer values (e.g. BGM log-density or Q/SPE). ``limit`` draws the control
    limit; ``limit_side`` is ``"upper"`` (Q UCL) or ``"lower"`` (density LCL).
    Drift shows as the temporal violin pushing past the limit.
    """
    fig = go.Figure()
    has_data = False
    for key in ("reference", "in_distribution", "temporal"):
        values = np.asarray(populations.get(key, []), dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        has_data = True
        fig.add_trace(
            go.Violin(
                y=values,
                name=_GATE_POP_LABELS[key],
                line_color=_GATE_POP_COLORS[key],
                box_visible=True,
                meanline_visible=True,
                points=False,
                hovertemplate=f"{_GATE_POP_LABELS[key]}<br>{statistic}=%{{y:.3f}}<extra></extra>",
            )
        )
    if not has_data:
        return _sized(
            fig.update_layout(title=dict(text=title)),
            height=440,
            margin=dict(l=56, r=48, t=72, b=48),
        )

    if limit is not None:
        side_label = "UCL" if limit_side == "upper" else "LCL"
        fig.add_hline(
            y=float(limit),
            line_dash="dash",
            line_color=C_YELLOW,
            annotation_text=f"{side_label} {float(limit):.2f}",
            annotation_position="right",
        )
    fig.update_layout(
        title=dict(text=title),
        yaxis_title=statistic,
        yaxis=dict(gridcolor="rgba(200, 200, 200, 0.15)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        violingap=0.25,
    )
    return _sized(fig, height=440, margin=dict(l=56, r=48, t=72, b=48))


def fig_gate_control_chart(
    values: np.ndarray,
    *,
    limit: float,
    ooc_mask: np.ndarray,
    x: np.ndarray | None = None,
    statistic: str = "Q / SPE",
    limit_side: str = "upper",
    title: str = "MSPC control chart",
) -> go.Figure:
    """Time-ordered control chart with the control limit, OOC points, rolling rate.

    Points are plotted in wafer order (``x`` = timestamps if given, else index);
    points beyond the control limit (``ooc_mask``) are highlighted, and a rolling
    out-of-control rate is overlaid on a secondary axis to show the OOC rate
    climbing into the drifted era.
    """
    vals = np.asarray(values, dtype=float)
    fig = go.Figure()
    if vals.size == 0:
        return _sized(
            fig.update_layout(title=dict(text=title)),
            height=460,
            margin=dict(l=56, r=64, t=72, b=48),
        )

    idx = np.arange(vals.size)
    xs = np.asarray(x) if x is not None and len(x) == vals.size else idx
    ooc = np.asarray(ooc_mask, dtype=bool)
    in_ctrl = ~ooc

    fig.add_trace(
        go.Scatter(
            x=xs[in_ctrl],
            y=vals[in_ctrl],
            mode="markers",
            name="In control",
            marker=dict(color=C_BLUE, size=6),
            hovertemplate=f"{statistic}=%{{y:.3f}}<extra></extra>",
        )
    )
    if ooc.any():
        fig.add_trace(
            go.Scatter(
                x=xs[ooc],
                y=vals[ooc],
                mode="markers",
                name="Out of control",
                marker=dict(color=C_RED, size=8, symbol="x"),
                hovertemplate=f"OOC<br>{statistic}=%{{y:.3f}}<extra></extra>",
            )
        )

    side_label = "UCL" if limit_side == "upper" else "LCL"
    fig.add_hline(
        y=float(limit),
        line_dash="dash",
        line_color=C_YELLOW,
        annotation_text=f"{side_label} {float(limit):.2f}",
        annotation_position="right",
    )

    # Trailing-window rolling OOC rate (headline trend) + faint cumulative reference,
    # both on a secondary axis. The cumulative average is sticky and hides a late-era
    # spike, so the rolling window is the trend a fab actually reads.
    ooc_f = ooc.astype(float)
    window = int(max(20, ooc_f.size // 10))
    window = min(window, ooc_f.size) or 1
    kernel = np.ones(window, dtype=float)
    counts = np.convolve(np.ones_like(ooc_f), kernel, mode="full")[: ooc_f.size]
    rolling_rate = np.convolve(ooc_f, kernel, mode="full")[: ooc_f.size] / np.maximum(counts, 1.0)
    cumulative_rate = np.cumsum(ooc_f) / (idx + 1.0)
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=cumulative_rate,
            mode="lines",
            name="Cumulative OOC rate",
            line=dict(color=C_ORANGE, width=1.2, dash="dot"),
            opacity=0.5,
            yaxis="y2",
            hovertemplate="cumulative=%{y:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=rolling_rate,
            mode="lines",
            name=f"Rolling OOC rate (window={window})",
            line=dict(color=C_ORANGE, width=2.5),
            yaxis="y2",
            hovertemplate="rolling=%{y:.1%}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title),
        xaxis_title="Wafer order (measurement time)" if x is not None else "Wafer order",
        yaxis=dict(title=statistic, gridcolor="rgba(200, 200, 200, 0.15)"),
        yaxis2=dict(
            title="OOC rate",
            overlaying="y",
            side="right",
            range=[0, 1],
            showgrid=False,
            tickformat=".0%",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return _sized(fig, height=460, margin=dict(l=56, r=64, t=72, b=48))


def _ellipse_path(mean2: np.ndarray, cov2: np.ndarray, *, n_sigma: float = 2.0, n: int = 60):
    """2-sigma ellipse (x, y) arrays for a 2-D Gaussian via eigen-decomposition."""
    try:
        vals, vecs = np.linalg.eigh(np.asarray(cov2, dtype=float))
    except np.linalg.LinAlgError:
        return None
    vals = np.clip(vals, 1e-12, None)
    t = np.linspace(0.0, 2.0 * np.pi, n)
    circle = np.stack([np.cos(t), np.sin(t)], axis=0)  # (2, n)
    ellipse = vecs @ (np.sqrt(vals)[:, None] * circle) * n_sigma  # (2, n)
    return mean2[0] + ellipse[0], mean2[1] + ellipse[1]


def fig_sbfa_factor_space(
    reference_scores: np.ndarray,
    holdout_scores: np.ndarray,
    y_true: np.ndarray,
    flagged: np.ndarray,
    *,
    dims: tuple[int, int] = (0, 1),
    bgm_means: np.ndarray | None = None,
    bgm_covariances: np.ndarray | None = None,
    factor_labels: tuple[str, str] | None = None,
    title: str = "sBFA latent factor space",
) -> go.Figure:
    """2-D scatter of two sBFA factors with the BGM 2-sigma envelope overlaid.

    Faint reference points = passing-train envelope; holdout points colored by
    Pass / Fail / Flagged. BGM components are drawn as 2-sigma covariance ellipses
    (the learned multimodal 'normal operating' region). Drift = the holdout cloud
    sliding off the envelope.
    """
    fig = go.Figure()
    ref = np.asarray(reference_scores, dtype=float)
    hold = np.asarray(holdout_scores, dtype=float)
    dx, dy = dims
    if ref.ndim != 2 or hold.ndim != 2 or ref.shape[1] <= max(dims):
        return _sized(fig.update_layout(title=dict(text=title)), height=520, margin=dict(l=56, r=48, t=72, b=48))

    # BGM envelope ellipses (under the points).
    if bgm_means is not None and bgm_covariances is not None:
        means = np.asarray(bgm_means, dtype=float)
        covs = np.asarray(bgm_covariances, dtype=float)
        for c in range(means.shape[0]):
            mean2 = means[c, [dx, dy]]
            cov2 = covs[c][np.ix_([dx, dy], [dx, dy])]
            path = _ellipse_path(mean2, cov2)
            if path is None:
                continue
            ex, ey = path
            fig.add_trace(
                go.Scatter(
                    x=ex,
                    y=ey,
                    mode="lines",
                    line=dict(color=C_YELLOW, width=1.5),
                    fill="toself",
                    fillcolor="rgba(216, 166, 87, 0.07)",
                    name="BGM 2σ envelope" if c == 0 else None,
                    showlegend=c == 0,
                    hoverinfo="skip",
                )
            )

    fig.add_trace(
        go.Scatter(
            x=ref[:, dx],
            y=ref[:, dy],
            mode="markers",
            name="Reference (passing train)",
            marker=dict(color="rgba(125, 174, 163, 0.28)", size=4),
            hoverinfo="skip",
        )
    )

    _add_factor_points(fig, hold, y_true, flagged, dx, dy)

    xlab, ylab = factor_labels or (f"Factor {dx + 1}", f"Factor {dy + 1}")
    fig.update_layout(
        title=dict(text=title),
        xaxis=dict(title=xlab, gridcolor="rgba(200, 200, 200, 0.15)", zeroline=False),
        yaxis=dict(title=ylab, gridcolor="rgba(200, 200, 200, 0.15)", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return _sized(fig, height=520, margin=dict(l=56, r=48, t=72, b=48))


def _add_factor_points(fig: go.Figure, hold: np.ndarray, y_true, flagged, dx: int, dy: int) -> None:
    """4-way holdout marker layers (Pass / Flagged pass / Caught fail / Missed fail).

    The split makes the gate's catch/miss behaviour explicit. "Flagged pass" is the
    neutral name (not "false alarm"): a flagged passing wafer can be a genuine
    process excursion that simply did not cause a yield fail.
    """
    y_arr = np.asarray(y_true, dtype=int)
    flag_arr = np.asarray(flagged, dtype=bool)
    fail = y_arr == 1
    groups = [
        ("Pass", ~fail & ~flag_arr, C_BLUE, "circle", 6),
        ("Flagged pass", ~fail & flag_arr, C_ORANGE, "diamond", 8),
        ("Caught fail", fail & flag_arr, C_RED, "x", 11),
        ("Missed fail", fail & ~flag_arr, C_PURPLE, "circle-open", 11),
    ]
    for label, mask, color, symbol, size in groups:
        if not mask.any():
            continue
        fig.add_trace(
            go.Scatter(
                x=hold[mask, dx],
                y=hold[mask, dy],
                mode="markers",
                name=f"Holdout - {label}",
                marker=dict(color=color, size=size, symbol=symbol, line=dict(width=1.0, color=color)),
                hovertemplate=f"{label}<extra></extra>",
            )
        )


def fig_efa_factor_space(
    reference_scores: np.ndarray,
    holdout_scores: np.ndarray,
    y_true: np.ndarray,
    flagged: np.ndarray,
    *,
    dims: tuple[int, int] = (0, 1),
    score_mean: np.ndarray | None = None,
    score_cov: np.ndarray | None = None,
    t2_alpha: float = 0.03,
    factor_labels: tuple[str, str] | None = None,
    title: str = "EFA latent factor space",
) -> go.Figure:
    """2-D scatter of two EFA factors with the Hotelling T² control ellipse overlaid.

    The frequentist analogue of ``fig_sbfa_factor_space``: faint reference points =
    passing-train envelope, holdout points colored Pass / Fail / Flagged, and a
    single bivariate χ² (Hotelling) control ellipse at ``t2_alpha`` for the two
    plotted factors. Drift = the holdout cloud sliding outside the control region.
    """
    from scipy.stats import chi2

    fig = go.Figure()
    ref = np.asarray(reference_scores, dtype=float)
    hold = np.asarray(holdout_scores, dtype=float)
    dx, dy = dims
    if ref.ndim != 2 or hold.ndim != 2 or ref.shape[1] <= max(dims):
        return _sized(fig.update_layout(title=dict(text=title)), height=520, margin=dict(l=56, r=48, t=72, b=48))

    if score_mean is not None and score_cov is not None:
        mean = np.asarray(score_mean, dtype=float)
        cov = np.asarray(score_cov, dtype=float)
        if mean.shape[0] > max(dims) and cov.shape[0] > max(dims):
            radius = float(np.sqrt(chi2.ppf(1.0 - t2_alpha, df=2)))
            path = _ellipse_path(mean[[dx, dy]], cov[np.ix_([dx, dy], [dx, dy])], n_sigma=radius)
            if path is not None:
                ex, ey = path
                fig.add_trace(
                    go.Scatter(
                        x=ex,
                        y=ey,
                        mode="lines",
                        line=dict(color=C_YELLOW, width=1.5),
                        fill="toself",
                        fillcolor="rgba(216, 166, 87, 0.07)",
                        name=f"Hotelling T² limit (α={t2_alpha:g})",
                        hoverinfo="skip",
                    )
                )

    fig.add_trace(
        go.Scatter(
            x=ref[:, dx],
            y=ref[:, dy],
            mode="markers",
            name="Reference (passing train)",
            marker=dict(color="rgba(125, 174, 163, 0.28)", size=4),
            hoverinfo="skip",
        )
    )
    _add_factor_points(fig, hold, y_true, flagged, dx, dy)

    xlab, ylab = factor_labels or (f"Factor {dx + 1}", f"Factor {dy + 1}")
    fig.update_layout(
        title=dict(text=title),
        xaxis=dict(title=xlab, gridcolor="rgba(200, 200, 200, 0.15)", zeroline=False),
        yaxis=dict(title=ylab, gridcolor="rgba(200, 200, 200, 0.15)", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return _sized(fig, height=520, margin=dict(l=56, r=48, t=72, b=48))


def fig_factor_drift(
    ranking: pd.DataFrame,
    *,
    title: str = "Per-factor drift (reference → holdout KS)",
) -> go.Figure:
    """Horizontal bars of per-factor KS drift; the top-drifting factor highlighted."""
    fig = go.Figure()
    if ranking is None or ranking.empty:
        return _sized(fig.update_layout(title=dict(text=title)), height=360, margin=dict(l=72, r=48, t=64, b=48))
    df = ranking.sort_values("ks_distance", ascending=True)
    colors = [C_RED if i == len(df) - 1 else C_BLUE for i in range(len(df))]
    fig.add_trace(
        go.Bar(
            x=df["ks_distance"],
            y=[f"Factor {int(f)}" for f in df["factor"]],
            orientation="h",
            marker_color=colors,
            customdata=df["top_sensors"],
            hovertemplate="KS=%{x:.3f}<br>top: %{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title),
        xaxis=dict(title="KS distance", gridcolor="rgba(200, 200, 200, 0.15)"),
        yaxis=dict(title=""),
    )
    return _sized(fig, height=max(280, 60 + 34 * len(df)), margin=dict(l=72, r=48, t=64, b=48))


def fig_sbfa_loadings_heatmap(
    loadings: np.ndarray,
    feature_names: list[str],
    *,
    factor_order: list[int] | None = None,
    highlight_factor: int | None = None,
    max_sensors: int = 25,
    title: str = "Sparse sBFA loadings (sensor × factor)",
) -> go.Figure:
    """Diverging sensor×factor loadings heatmap, factors drift-ordered.

    Only the strongest-loading sensors (by max |loading| across factors) are
    shown for legibility; the Laplace prior makes most loadings ~0.
    """
    fig = go.Figure()
    W = np.asarray(loadings, dtype=float)
    if W.ndim != 2 or W.size == 0 or not feature_names:
        return _sized(fig.update_layout(title=dict(text=title)), height=520, margin=dict(l=140, r=48, t=64, b=56))

    n_factors = W.shape[1]
    order = [f for f in (factor_order or list(range(n_factors))) if 0 <= f < n_factors]
    if not order:
        order = list(range(n_factors))

    sensor_strength = np.abs(W).max(axis=1)
    keep = np.argsort(sensor_strength)[::-1][:max_sensors]
    keep = keep[np.argsort(sensor_strength[keep])]  # ascending so strongest on top
    z = W[np.ix_(keep, order)]
    sensors = [str(feature_names[i]) for i in keep]
    col_labels = [
        f"F{f + 1}★" if (highlight_factor is not None and f == highlight_factor) else f"F{f + 1}"
        for f in order
    ]
    vmax = float(np.abs(z).max()) or 1.0
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=col_labels,
            y=sensors,
            colorscale=COLORSCALE_DRIFT_DIVERGING,
            zmid=0.0,
            zmin=-vmax,
            zmax=vmax,
            colorbar=dict(title="loading"),
            hovertemplate="%{y} · %{x}<br>loading=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title),
        xaxis=dict(title="latent factor (drift-ordered, ★ = top)", side="top"),
        yaxis=dict(title=""),
    )
    return _sized(fig, height=max(420, 60 + 22 * len(sensors)), margin=dict(l=140, r=48, t=72, b=48))


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


def fig_delta_bar(
    df: pd.DataFrame,
    *,
    value_col: str = "delta",
    label_col: str = "pipeline",
    title: str = "Delta",
    value_label: str = "delta",
    positive_is_good: bool = True,
    height: int = 380,
) -> go.Figure:
    """Diverging horizontal bars for signed deltas, sorted by magnitude.

    Green/red encode whether a bar is desirable: when ``positive_is_good`` is
    True, positive bars are green; flip it for loss-style deltas (e.g. the
    interpolation-minus-extrapolation drift cost) so larger losses read red.
    """
    if df.empty or value_col not in df:
        return _sized(go.Figure(), height=height, title=dict(text=title))
    plot_df = df.dropna(subset=[value_col]).copy()
    if plot_df.empty:
        return _sized(go.Figure(), height=height, title=dict(text=title))
    plot_df = plot_df.reindex(
        plot_df[value_col].abs().sort_values(ascending=True).index
    )
    good = C_GREEN if positive_is_good else C_RED
    bad = C_RED if positive_is_good else C_GREEN
    colors = [good if v >= 0 else bad for v in plot_df[value_col]]
    fig = go.Figure(
        data=[
            go.Bar(
                x=plot_df[value_col],
                y=plot_df[label_col],
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.3f}" for v in plot_df[value_col]],
                textposition="outside",
                hovertemplate="%{y}<br>" + value_label + "=%{x:+.4g}<extra></extra>",
            )
        ]
    )
    fig.add_vline(x=0.0, line_width=1, line_color="rgba(235,235,235,0.45)")
    fig.update_layout(
        title=dict(text=title),
        xaxis_title=value_label,
        yaxis_title="",
        showlegend=False,
    )
    return _sized(fig, height=height, margin=dict(l=150, r=64, t=72, b=48))


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
        empty = go.Figure()
        empty.update_layout(title=dict(text=title))
        return _sized(empty, height=height)
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
