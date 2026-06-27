"""Shared palette, colorscales, and layout helpers for SECOM chart builders."""
from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

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
