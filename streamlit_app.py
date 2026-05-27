"""SECOM defect-detection dashboard entry point.

Run from repo root:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import streamlit as st

from scripts.dashboard_app import configure_page, ensure_repo_on_path

ensure_repo_on_path()

pages = [
    st.Page("pages/1_Introduction.py", title="Introduction", default=True),
    st.Page("pages/2_Pipeline.py", title="Pipeline"),
    st.Page("pages/3_Models.py", title="Models"),
    st.Page("pages/4_Wafer.py", title="Wafer inspector"),
    st.Page("pages/5_Explanations.py", title="Explanations"),
]

configure_page()
st.navigation(pages, position="sidebar").run()
