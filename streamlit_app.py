"""SECOM defect-detection dashboard entry point.

Run from repo root:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import streamlit as st

from secom.dashboard import configure_page

pages = [
    st.Page("pages/1_Overview.py", title="Overview", default=True),
    st.Page("pages/2_Pipeline.py", title="Pipeline"),
    st.Page("pages/3_Interpolation.py", title="Interpolation"),
    st.Page("pages/4_Extrapolation.py", title="Extrapolation"),
    st.Page("pages/5_Gates.py", title="Gates & risk-coverage"),
    st.Page("pages/6_Model_Explainability.py", title="Model explainability"),
]

configure_page()
st.navigation(pages, position="sidebar").run()
