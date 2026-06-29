"""Shared Streamlit app setup for SECOM dashboard pages."""
from __future__ import annotations

from itertools import count

import streamlit as st

METRIC_BG = "#3c3836"
NOTE_BLUE = "#7daea3"
NOTE_GREEN = "#a9b665"
NOTE_AMBER = "#d8a657"
TEXT_COLOR = "#d4be98"
RADIUS = "0.75rem"

# Per-call counter so each note container gets a unique Streamlit key while the
# CSS still matches the shared tone via the `st-key-note_<tone>` substring.
_NOTE_SEQ = count()


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
            border: none !important;
            border-radius: {RADIUS};
            overflow: hidden;
        }}
        [class*="st-key-card_"] {{
            background-color: {METRIC_BG} !important;
            border: none !important;
            border-radius: {RADIUS} !important;
            padding: 1rem !important;
            overflow: hidden;
        }}
        [class*="st-key-note_"] {{
            border: none !important;
            border-radius: {RADIUS} !important;
            padding: 0.75rem 1rem !important;
            overflow: hidden;
        }}
        [class*="st-key-note_blue"] {{
            background-color: color-mix(in srgb, {NOTE_BLUE} 22%, {METRIC_BG}) !important;
        }}
        [class*="st-key-note_green"] {{
            background-color: color-mix(in srgb, {NOTE_GREEN} 22%, {METRIC_BG}) !important;
        }}
        [class*="st-key-note_amber"] {{
            background-color: color-mix(in srgb, {NOTE_AMBER} 22%, {METRIC_BG}) !important;
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


def _render_note(message: str, tone: str) -> None:
    with st.container(key=f"note_{tone}_{next(_NOTE_SEQ)}"):
        st.markdown(message)


def render_blue_note(message: str) -> None:
    """Neutral teaching note (blue) - the default callout used across pages."""
    _render_note(message, "blue")


def render_verdict(message: str) -> None:
    """Positive-emphasis note (green) - deploy verdicts, gates-agree conclusions."""
    _render_note(message, "green")


def render_caveat(message: str) -> None:
    """Caution-emphasis note (amber) - genuine interpretation caveats."""
    _render_note(message, "amber")


def configure_page(*, page_title: str = "SECOM", page_icon: str = "🔬") -> None:
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")
    inject_dashboard_styles()
