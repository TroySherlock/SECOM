"""SHAP / feature explanations (placeholder)."""
from __future__ import annotations

import streamlit as st

from scripts.dashboard_app import ensure_repo_on_path

ensure_repo_on_path()

st.title("Explanations")
st.info(
    "Not implemented yet. TreeSHAP or coefficient views for why a measurement "
    "was flagged fail, tied to the wafer inspector."
)
st.markdown(
    """
    **Planned content**

    - Global feature importance from benchmark models
    - Local SHAP waterfall for selected observation
    - MSPC vs sensor contribution breakdown
    """
)
