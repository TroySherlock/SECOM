"""BER-band threshold profile definitions for Stage 2 tuning.

The operating points come from the balanced-error-rate (BER) curve itself: the
``ber`` profile is the BER-minimising threshold, and ``conservative`` /
``aggressive`` are the high / low ends of the BER tolerance band (all thresholds
within ``BER_BAND_TOLERANCE`` absolute BER points of that minimum). Edit
``BER_BAND_TOLERANCE`` below, then re-run threshold tuning and benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score

ProfileId = Literal["conservative", "ber", "aggressive"]

# --- BER tolerance band ------------------------------------------------------
# Absolute balanced-error-rate tolerance (in BER points) around the minimum that
# defines the conservative / aggressive operating band. Must match the units of
# the BER values produced by the threshold sweep.
BER_BAND_TOLERANCE = 2.0

DEFAULT_PROFILE_ID: ProfileId = "ber"

PROFILE_IDS: tuple[ProfileId, ...] = ("conservative", "ber", "aggressive")


@dataclass(frozen=True)
class ThresholdProfile:
    profile_id: ProfileId
    beta: float | None
    display_name: str
    description: str
    objective: Literal["fbeta", "ber"] = "fbeta"


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
}


def threshold_profile_config() -> dict[str, float | str]:
    """Snapshot for JSON artifacts and the dashboard."""
    return {
        "ber_band_tolerance": float(BER_BAND_TOLERANCE),
        "default_profile": DEFAULT_PROFILE_ID,
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
