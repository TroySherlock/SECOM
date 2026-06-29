"""Pipeline (2/2): the three front-ends and the two-champion feature journeys."""
from __future__ import annotations

import streamlit as st

from secom.dashboard.pipeline_views import render_front_ends_and_journeys


def main() -> None:
    st.title("Pipeline — front-ends & journeys")
    st.caption(
        "Where the nine pipelines diverge: the three feature front-ends (HSIC-Lasso, RF-selection, "
        "PLS) and the end-to-end feature count for the two champions. See the Interpolation / "
        "Extrapolation pages' architecture subpage for PR curves and operating points."
    )
    render_front_ends_and_journeys()


main()
