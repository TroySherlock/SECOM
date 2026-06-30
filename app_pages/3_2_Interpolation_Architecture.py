"""Interpolation (2/2): pipeline architecture & tuning per model."""
from __future__ import annotations

import streamlit as st

from secom.dashboard.components import load_payload, render_model_deepdive


def main() -> None:
    st.title("Interpolation — pipeline architecture & tuning")
    st.caption(
        "Per-model deep-dive on the interpolation (random holdout) track: the chosen pipeline "
        "architecture, its in-distribution-tuned hyperparameters, precision–recall and calibration, "
        "and the cost-system thresholding."
    )

    try:
        payload = load_payload()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    render_model_deepdive(payload, track="interpolation")


main()
