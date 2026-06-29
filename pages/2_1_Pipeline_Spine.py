"""Pipeline (1/2): the shared feature spine — rz twins and shared preprocessing."""
from __future__ import annotations

import streamlit as st

from secom.dashboard.pipeline_views import render_feature_spine


def main() -> None:
    st.title("Pipeline — feature spine")
    st.caption(
        "How raw SECOM sensors become the shared feature spine every model starts from: the "
        "robust-z twin features and the impute → cluster → scale → calibrate steps shared across "
        "the whole 3×3 grid. The Front-ends & journeys page shows how each model diverges from here."
    )
    render_feature_spine()


main()
