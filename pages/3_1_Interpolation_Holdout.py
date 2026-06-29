"""Interpolation (1/2): holdout evaluation — in-distribution ceiling."""
from __future__ import annotations

import streamlit as st

from secom.dashboard import render_blue_note
from secom.dashboard.model_views import (
    load_payload,
    render_cv_leaderboard,
    render_holdout_validation,
)


def main() -> None:
    st.title("Interpolation — in-distribution ceiling")
    st.caption(
        "The optimistic upper bound: every pipeline tuned with 5×2 repeated stratified CV and "
        "scored on a random stratified holdout. Train and test come from the same era, so this "
        "is what the models can do when there is no drift — the ceiling the forward (extrapolation) "
        "track is measured against."
    )

    try:
        payload = load_payload()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    render_blue_note(
        "**In-distribution view.** 5×2 repeated stratified CV for tuning, random stratified "
        "holdout for reporting. Treat these numbers as the best case — the gap between this page "
        "and the Extrapolation page is the cost of temporal drift. All holdout metrics are "
        f"reporting-only (`holdout_is_reporting_only={payload.get('holdout_is_reporting_only', True)}`)."
    )

    render_holdout_validation(payload, track="interpolation")

    with st.expander("Cross-validation leaderboard (5×2 stratified)", expanded=False):
        render_cv_leaderboard(payload)


main()
