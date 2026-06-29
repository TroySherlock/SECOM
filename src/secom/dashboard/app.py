"""Shared Streamlit app setup for SECOM dashboard pages."""
from __future__ import annotations

import streamlit as st

METRIC_BG = "#3c3836"
NOTE_BLUE = "#7daea3"
TEXT_COLOR = "#d4be98"
RADIUS = "0.75rem"


def inject_dashboard_styles() -> None:
    st.markdown(
        f"""
        <style>
        [data-testid="stPlotlyChart"],
        [data-testid="stPlotlyChart"] > div,
        [data-testid="stPlotlyChart"] iframe {{
            border-radius: {RADIUS};
            overflow: hidden;
        }}
        [data-testid="stMetric"],
        [data-testid="stMetricBorder"] {{
            background-color: {METRIC_BG} !important;
            border-radius: {RADIUS};
            overflow: hidden;
        }}
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: {METRIC_BG} !important;
            border-radius: {RADIUS};
            overflow: hidden;
        }}
        [data-testid="stIFrame"],
        [data-testid="stIFrame"] iframe {{
            border-radius: {RADIUS};
            overflow: hidden;
        }}
        [data-testid="stAlert"] {{
            background: transparent !important;
            padding: 0;
        }}
        [data-testid="stAlert"] div[data-baseweb="notification"] {{
            background-color: color-mix(in srgb, {NOTE_BLUE} 22%, {METRIC_BG}) !important;
            border: none !important;
            border-radius: {RADIUS} !important;
            overflow: hidden;
        }}
        [data-testid="stAlert"] div[data-baseweb="notification"] * {{
            color: {TEXT_COLOR} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_blue_note(message: str) -> None:
    st.info(message)


def configure_page(*, page_title: str = "SECOM", page_icon: str = "🔬") -> None:
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")
    inject_dashboard_styles()
