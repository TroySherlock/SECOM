"""Overview / EDA charts (page 1): class balance, missingness, drift, collinearity."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from secom.dashboard.charts._base import (
    C_BLUE,
    C_GREEN,
    C_PURPLE,
    C_RED,
    C_YELLOW,
    COLORSCALE_CORRELATION,
    COLORSCALE_DRIFT_DIVERGING,
    _sized,
)
from secom.dashboard.stg import ROW_INDEX_COL, sensor_columns


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
        title=dict(text="Failures over time", y=0.97, yanchor="top"),
        xaxis_title="Measurement time",
        yaxis=dict(
            title="Outcome (jittered)",
            tickvals=[0, 1],
            ticktext=["Pass", "Fail"],
            range=[-0.2, 1.2],
        ),
        yaxis2=layout_y2,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        margin=dict(t=80),
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


def fig_wafer_drift_spikes(
    sensors: list[str],
    robust_z: np.ndarray,
    drift_shift: np.ndarray,
    *,
    z_band: float = 2.0,
    title: str = "Known-drifting sensors on this wafer (robust SPC z)",
) -> go.Figure:
    """One wafer's robust SPC z across the globally known-drifting sensors.

    Each bar is a sensor from the era-drift set (its training->holdout mean shift
    drifted past the global threshold); the bar height is *this* wafer's robust z
    against the in-control passing-train distribution. Bars outside the +/- ``z_band``
    reference region are the sensors actually spiking on this wafer - the in-control
    excursion that trips the density gate even though the wafer passed.
    """
    z = np.asarray(robust_z, dtype=float)
    if len(sensors) == 0 or z.size == 0:
        return _sized(go.Figure(), height=320)
    shift = np.asarray(drift_shift, dtype=float)
    spiking = np.abs(z) > z_band
    colors = [C_RED if s else C_BLUE for s in spiking]
    customdata = np.column_stack([shift])

    fig = go.Figure(
        data=go.Bar(
            x=z,
            y=list(sensors),
            orientation="h",
            marker=dict(color=colors),
            customdata=customdata,
            hovertemplate=(
                "Sensor %{y}<br>this wafer: %{x:.2f} σ (robust z vs in-control)"
                "<br>era drift: %{customdata[0]:+.2f} σ<extra></extra>"
            ),
        )
    )
    for edge in (z_band, -z_band):
        fig.add_vline(x=edge, line=dict(color=C_YELLOW, width=1, dash="dash"))
    fig.add_vrect(
        x0=-z_band,
        x1=z_band,
        fillcolor=C_YELLOW,
        opacity=0.08,
        line_width=0,
    )
    fig.add_vline(x=0.0, line=dict(color="rgba(128,128,128,0.6)", width=1))
    fig.update_layout(
        title=dict(text=title),
        xaxis=dict(title=f"Robust SPC z (|z| > {z_band:g} = spiking)"),
        yaxis=dict(title="Known-drifting sensor", type="category", autorange="reversed"),
        showlegend=False,
    )
    return _sized(fig, height=460, margin=dict(l=120, r=40, t=70, b=50))


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
