"""Shared Streamlit app setup for SECOM dashboard pages."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]

METRIC_BG = "#3c3836"
NOTE_BLUE = "#7daea3"


def ensure_repo_on_path() -> Path:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return REPO_ROOT


def inject_dashboard_styles() -> None:
    st.markdown(
        f"""
        <style>
        [data-testid="stMetric"],
        [data-testid="stMetricBorder"] {{
            background-color: {METRIC_BG} !important;
        }}
        [data-testid="stAlert"] {{
            background-color: color-mix(in srgb, {NOTE_BLUE} 22%, transparent) !important;
            border: none !important;
            color: #d4be98;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_blue_note(message: str) -> None:
    st.info(message)


def configure_page(*, page_title: str = "SECOM", page_icon: str = "🔬") -> None:
    ensure_repo_on_path()
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")
    inject_dashboard_styles()
