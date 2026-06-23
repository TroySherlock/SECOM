"""SECOM defect-detection dashboard entry point.

Run from repo root:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import streamlit as st

from secom.dashboard import configure_page

pages = {
    "Overview": [
        st.Page("pages/1_Overview.py", title="Overview", default=True),
        st.Page("pages/2_Pipeline.py", title="Pipeline"),
    ],
    "Models": [
        st.Page("pages/3_Interpolation.py", title="Interpolation"),
        st.Page("pages/4_Extrapolation.py", title="Extrapolation"),
    ],
    "Gates": [
        st.Page("pages/5_1_Gates_Intro.py", title="5.1 Introduction"),
        st.Page("pages/5_2_Gate_T2.py", title="5.2 Hotelling T² gate"),
        st.Page("pages/5_3_Gate_BGM.py", title="5.3 sBFA → BGM gate"),
        st.Page("pages/5_4_Gate_Comparison.py", title="5.4 Gate comparison"),
    ],
    "Explainability": [
        st.Page("pages/6_Model_Explainability.py", title="Model explainability"),
    ],
}

configure_page()
st.navigation(pages, position="sidebar").run()
