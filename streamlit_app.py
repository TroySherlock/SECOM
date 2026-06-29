"""SECOM defect-detection dashboard entry point.

Run from repo root:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the src-layout ``secom`` package importable on hosts that only install
# requirements.txt (e.g. Streamlit Community Cloud), not the project itself.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st  # noqa: E402

from secom.dashboard import configure_page  # noqa: E402

pages = {
    "Overview": [
        st.Page("pages/1_Overview.py", title="Overview", default=True),
    ],
    "Pipeline": [
        st.Page("pages/2_1_Pipeline_Spine.py", title="Feature spine"),
        st.Page("pages/2_2_Pipeline_FrontEnds.py", title="Front-ends & journeys"),
    ],
    "Interpolation": [
        st.Page(
            "pages/3_1_Interpolation_Holdout.py",
            title="Holdout evaluation",
            url_path="interpolation-holdout",
        ),
        st.Page(
            "pages/3_2_Interpolation_Architecture.py",
            title="Pipeline architecture & tuning",
            url_path="interpolation-architecture",
        ),
    ],
    "Extrapolation": [
        st.Page(
            "pages/4_1_Extrapolation_Holdout.py",
            title="Holdout evaluation",
            url_path="extrapolation-holdout",
        ),
        st.Page(
            "pages/4_2_Extrapolation_Architecture.py",
            title="Pipeline architecture & tuning",
            url_path="extrapolation-architecture",
        ),
    ],
    "Gates": [
        st.Page("pages/5_1_Gates_Intro.py", title="Introduction"),
        st.Page("pages/5_2_Gate_T2.py", title="Hotelling T² gate"),
        st.Page("pages/5_3_Gate_BGM.py", title="sBFA → BGM gate"),
        st.Page("pages/5_4_Gate_Comparison.py", title="Gate comparison"),
    ],
    "Results": [
        st.Page("pages/6_Model_Explainability.py", title="Model explainability"),
        st.Page("pages/7_Conclusion.py", title="Conclusion"),
    ],
}

configure_page()
st.navigation(pages, position="sidebar").run()
