"""Shared Streamlit app setup for SECOM dashboard pages."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

from scripts.dashboard_theme import streamlit_css

REPO_ROOT = Path(__file__).resolve().parents[1]


def ensure_repo_on_path() -> Path:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return REPO_ROOT


def configure_page(*, page_title: str = "SECOM", page_icon: str = "🔬") -> None:
    ensure_repo_on_path()
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")
    st.markdown(streamlit_css(), unsafe_allow_html=True)
