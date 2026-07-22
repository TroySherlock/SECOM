"""Explainability charts (page 6): feature importance, SPC, posterior forest."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from secom.dashboard.charts._base import (
    C_BLUE,
    C_GREEN,
    C_RED,
    C,
    _sized,
)


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


def fig_spc_distribution(
    train_values,
    wafer_value: float | None,
    *,
    sensor: str,
    spc_z: float | None = None,
    title: str | None = None,
    height: int = 300,
) -> go.Figure:
    """SPC view: in-control (passing-train) distribution with this wafer marked.

    A violin of the passing training values for one contributor sensor, plus a
    dashed line at this wafer's value, so a reviewer can see how far the wafer
    sits from the in-control population.
    """
    vals = np.asarray(train_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    fig = go.Figure()
    if vals.size:
        fig.add_trace(
            go.Violin(
                y=vals,
                name="Passing train",
                line_color=C_BLUE,
                fillcolor="rgba(125,174,163,0.25)",
                box_visible=True,
                meanline_visible=True,
                points=False,
                hovertemplate="passing train<br>%{y:.3g}<extra></extra>",
            )
        )
    if wafer_value is not None and np.isfinite(wafer_value):
        fig.add_hline(
            y=float(wafer_value),
            line_dash="dash",
            line_color=C_RED,
            annotation_text="this wafer",
            annotation_position="right",
        )
    sub = ""
    if spc_z is not None and np.isfinite(spc_z):
        sub = f"  (SPC z = {spc_z:+.1f})"
    fig.update_layout(
        title=dict(text=title or f"{sensor}: in-control vs this wafer{sub}"),
        yaxis=dict(title=sensor, gridcolor="rgba(200, 200, 200, 0.15)"),
        showlegend=False,
    )
    return _sized(fig, height=height, margin=dict(l=60, r=48, t=58, b=28))


def fig_posterior_forest(
    df: pd.DataFrame,
    *,
    title: str = "Sensor posterior (mean ± 95% credible interval)",
    xaxis_title: str = "Posterior coefficient",
    height: int = 440,
) -> go.Figure:
    """Forest plot: point = posterior mean, whiskers = 95% equal-tailed credible
    interval, dashed zero line.

    ``robust`` rows (interval excludes 0) are drawn solid and color-coded by sign;
    the rest are dimmed, so a reviewer sees at a glance which sensors the model
    is confident about.
    """
    if df is None or df.empty:
        return _sized(go.Figure().update_layout(title=dict(text=title)), height=height)
    plot_df = df.sort_values("mean").reset_index(drop=True)
    fig = go.Figure()
    for _, r in plot_df.iterrows():
        robust = bool(r.get("robust", False))
        mean, lo, hi = float(r["mean"]), float(r["hdi_low"]), float(r["hdi_high"])
        if robust:
            color = C_GREEN if mean >= 0 else C_RED
            bar_color = color
        else:
            color = "rgba(168, 182, 101, 0.40)"
            bar_color = "rgba(150, 150, 150, 0.40)"
        fig.add_trace(
            go.Scatter(
                x=[mean],
                y=[str(r["feature"])],
                mode="markers",
                marker=dict(
                    color=color,
                    size=10 if robust else 7,
                    symbol="diamond" if robust else "circle",
                ),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[hi - mean],
                    arrayminus=[mean - lo],
                    color=bar_color,
                    thickness=1.6,
                    width=4,
                ),
                hovertemplate=(
                    f"{r['feature']}<br>mean=%{{x:.4g}}<br>"
                    f"95% CrI=[{lo:.3g}, {hi:.3g}]"
                    f"{' · robust (clears 0)' if robust else ''}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    fig.add_vline(x=0.0, line_dash="dash", line_color="rgba(220, 220, 220, 0.5)")
    fig.update_layout(
        title=dict(text=title),
        xaxis_title=xaxis_title,
        yaxis_title="Sensor",
    )
    return _sized(fig, height=height, margin=dict(l=120, r=48, t=64, b=40))


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
