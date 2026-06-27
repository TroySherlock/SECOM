"""Threshold profile definitions for Stage 2 tuning.

Three of the operating points come from the balanced-error-rate (BER) curve
itself: the ``ber`` profile is the BER-minimising threshold, and
``conservative`` / ``aggressive`` are the high / low ends of the BER tolerance
band (all thresholds within ``BER_BAND_TOLERANCE`` absolute BER points of that
minimum). The fourth, ``economic``, is a genuine fab-economics point: it
minimises expected cost ``C_escape*FN + C_overkill*FP`` at an explicit
escape:overkill cost ratio (``ESCAPE_OVERKILL_COST_RATIO``). Edit the constants
below, then re-run threshold tuning and benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score

ProfileId = Literal["conservative", "ber", "aggressive", "economic"]

# --- BER tolerance band ------------------------------------------------------
# Absolute balanced-error-rate tolerance (in BER points) around the minimum that
# defines the conservative / aggressive operating band. Must match the units of
# the BER values produced by the threshold sweep.
BER_BAND_TOLERANCE = 2.0

# --- Fab economics -----------------------------------------------------------
# Cost of an escape (a failing wafer predicted Pass, i.e. false negative)
# relative to an overkill (a good wafer predicted Fail, i.e. false positive).
# An escape ships a bad die downstream (scrap, RMA, returns); an overkill only
# re-tests/holds a good wafer. Fabs routinely price escapes 10-100x an overkill.
# This is a proxy dollar ratio used to derive the cost-optimal ``economic``
# operating point; it is NOT a calibrated dollar figure.
ESCAPE_OVERKILL_COST_RATIO = 20.0

DEFAULT_PROFILE_ID: ProfileId = "ber"

PROFILE_IDS: tuple[ProfileId, ...] = ("conservative", "ber", "aggressive", "economic")


@dataclass(frozen=True)
class ThresholdProfile:
    profile_id: ProfileId
    beta: float | None
    display_name: str
    description: str
    objective: Literal["fbeta", "ber", "cost"] = "fbeta"


THRESHOLD_PROFILES: dict[ProfileId, ThresholdProfile] = {
    "conservative": ThresholdProfile(
        profile_id="conservative",
        beta=None,
        display_name="Conservative — high threshold",
        description=(
            "Highest threshold still within the BER tolerance band (BER-min + "
            f"{BER_BAND_TOLERANCE:g} pts). Precision-leaning: fewer positives, "
            "fewer false line stops, lower recall."
        ),
        objective="ber",
    ),
    "ber": ThresholdProfile(
        profile_id="ber",
        beta=None,
        display_name="BER-min — balanced",
        description=(
            "Minimise mean CV balanced error rate (BER) on the threshold grid; "
            "symmetric pass/fail misclassification cost. Default deploy threshold."
        ),
        objective="ber",
    ),
    "aggressive": ThresholdProfile(
        profile_id="aggressive",
        beta=None,
        display_name="Aggressive — low threshold",
        description=(
            "Lowest threshold still within the BER tolerance band (BER-min + "
            f"{BER_BAND_TOLERANCE:g} pts). Recall-leaning: catches more fails at "
            "the cost of more false stops."
        ),
        objective="ber",
    ),
    "economic": ThresholdProfile(
        profile_id="economic",
        beta=None,
        display_name=f"Economic — cost-optimal ({ESCAPE_OVERKILL_COST_RATIO:g}:1)",
        description=(
            "Minimise expected cost C_escape*FN + C_overkill*FP at an explicit "
            f"escape:overkill ratio of {ESCAPE_OVERKILL_COST_RATIO:g}:1. Because "
            "an escape costs far more than an overkill, this lands left of "
            "BER-min (recall-leaning): higher catch rate, more overkill."
        ),
        objective="cost",
    ),
}


def threshold_profile_config() -> dict[str, float | str]:
    """Snapshot for JSON artifacts and the dashboard."""
    return {
        "ber_band_tolerance": float(BER_BAND_TOLERANCE),
        "default_profile": DEFAULT_PROFILE_ID,
        "escape_overkill_cost_ratio": float(ESCAPE_OVERKILL_COST_RATIO),
    }


def cost_optimal_threshold(
    thresholds: Sequence[float],
    tprs: Sequence[float],
    tnrs: Sequence[float],
    prevalence: float,
    cost_ratio: float = ESCAPE_OVERKILL_COST_RATIO,
) -> float:
    """Threshold minimising expected cost over a sweep.

    ``tprs`` / ``tnrs`` are the per-threshold true-positive / true-negative
    rates (fractions in [0, 1]) aligned with ``thresholds``. Ties are broken
    toward the lower (more recall-leaning) threshold, matching how a fab
    operates left of balanced when escapes dominate the cost.
    """
    thr = np.asarray(thresholds, dtype=float)
    tpr = np.asarray(tprs, dtype=float)
    tnr = np.asarray(tnrs, dtype=float)
    if thr.size == 0:
        raise ValueError("cost_optimal_threshold needs a non-empty threshold grid")
    p = float(prevalence)
    costs = float(cost_ratio) * p * (1.0 - tpr) + (1.0 - p) * (1.0 - tnr)
    min_cost = float(np.min(costs))
    # tie-break toward the lowest (recall-leaning) threshold within a tiny eps
    eps = 1e-12
    tied = thr[costs <= min_cost + eps]
    return float(np.min(tied))


def catch_overkill_from_confusion(cm) -> dict[str, float | int]:
    """Fab-economics readout from a 2x2 confusion matrix ``[[tn, fp], [fn, tp]]``.

    Returns catch rate (recall / TPR), overkill rate (FPR = 1 - TNR), precision,
    and the raw counts a fab engineer reads directly (fails caught vs total,
    good wafers flagged vs total).
    """
    arr = np.asarray(cm, dtype=float)
    if arr.shape != (2, 2):
        return {}
    tn, fp, fn, tp = arr.ravel()
    fails = tp + fn
    goods = tn + fp
    flagged = tp + fp
    catch_rate = float(tp / fails) if fails else 0.0
    overkill_rate = float(fp / goods) if goods else 0.0
    precision = float(tp / flagged) if flagged else 0.0
    return {
        "catch_rate": catch_rate,
        "overkill_rate": overkill_rate,
        "precision": precision,
        "fails_caught": int(tp),
        "fails_total": int(fails),
        "good_flagged": int(fp),
        "good_total": int(goods),
    }


def fbeta_at_threshold(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    *,
    beta: float,
) -> float:
    """Binary F-beta for positive class (fail) via sklearn."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0))


def resolve_threshold_profiles(tuned_payload: dict) -> dict[str, float]:
    """Map profile_id -> best_threshold from tuned JSON."""
    profiles = tuned_payload.get("threshold_profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("tuned payload missing threshold_profiles")
    out: dict[str, float] = {}
    for pid, data in profiles.items():
        key = str(pid)
        if key in PROFILE_IDS and isinstance(data, dict) and "best_threshold" in data:
            out[key] = float(data["best_threshold"])
    return out


def has_multi_profile_thresholds(tuned_payload: dict) -> bool:
    profiles = tuned_payload.get("threshold_profiles")
    if not isinstance(profiles, dict):
        return False
    keys = {str(k) for k in profiles}
    return all(pid in keys for pid in PROFILE_IDS)
