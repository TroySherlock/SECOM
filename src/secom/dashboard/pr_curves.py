"""Frozen CV-OOF + holdout PR curves for the dashboard (read-only).

The curves are computed once offline by ``secom.benchmark`` and persisted to
``secom_report_cache.json``; this module only reconstructs them for plotting.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from secom.dashboard.data import report_entry
from secom.metrics import PRCurve

DEFAULT_TRACK = "extrapolation"


def _pr_curve(block: dict | None) -> PRCurve | None:
    if not block:
        return None
    return PRCurve(
        recall=np.asarray(block.get("recall", []), dtype=float),
        precision=np.asarray(block.get("precision", []), dtype=float),
        baseline=float(block.get("baseline", 0.0)),
    )


@st.cache_data(show_spinner=False)
def load_pr_curves(
    model_id: str, track: str = DEFAULT_TRACK
) -> tuple[PRCurve | None, PRCurve | None, tuple[float, float] | None]:
    """Frozen CV-OOF + holdout PR curves and a BER operating point on the chosen
    protocol (extrapolation: blocked CV + temporal holdout; interpolation:
    stratified CV + random holdout)."""
    pr = report_entry(track, model_id).get("pr_curve") or {}
    cv_curve = _pr_curve(pr.get("cv"))
    ho_curve = _pr_curve(pr.get("holdout"))
    ber = pr.get("ber_point")
    ber_point = (float(ber[0]), float(ber[1])) if ber else None
    return cv_curve, ho_curve, ber_point
