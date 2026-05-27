"""Per-wafer / observation inspector (placeholder)."""
from __future__ import annotations

import streamlit as st

from scripts.dashboard_app import ensure_repo_on_path

ensure_repo_on_path()

st.title("Wafer inspector")
st.info(
    "Not implemented yet. Pick an observation by ID or time range and inspect "
    "sensor traces, MSPC scores (T², Q), and model prediction."
)
st.markdown(
    """
    **Planned content**

    - `observation_id` or timestamp selector
    - Sensor profile vs cohort
  - Champion / benchmark model score and predicted class
    """
)
