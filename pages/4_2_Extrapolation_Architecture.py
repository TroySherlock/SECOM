"""Extrapolation (2/2): pipeline architecture & tuning per model."""
from __future__ import annotations

import streamlit as st

from secom.dashboard.model_views import load_payload, render_model_deepdive


def main() -> None:
    st.title("Extrapolation — pipeline architecture & tuning")
    st.caption(
        "Per-model deep-dive on the extrapolation (temporal forward holdout) track: the chosen "
        "pipeline architecture, the in-distribution-tuned hyperparameters reused for the forward "
        "refit, precision–recall and calibration, and the cost-system thresholding."
    )

    try:
        payload = load_payload()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    render_model_deepdive(payload, track="extrapolation")


main()
