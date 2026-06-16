"""F-beta threshold profile definitions for Stage 2 tuning.

Edit F0_5_BETA / F2_BETA / F4_BETA below, then re-run threshold tuning and benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score

ProfileId = Literal["f0_5", "f2", "f4", "ber"]

# --- Editable F-beta values (sklearn beta parameter) -------------------------

F0_5_BETA = 0.5  # conservative (precision-weighted)
F2_BETA = 2.0  # neutral (~4:1 recall emphasis); default deployment threshold
F4_BETA = 4.0  # aggressive (~16:1 recall emphasis)

DEFAULT_PROFILE_ID: ProfileId = "f2"

PROFILE_IDS: tuple[ProfileId, ...] = ("f0_5", "f2", "f4", "ber")


@dataclass(frozen=True)
class ThresholdProfile:
    profile_id: ProfileId
    beta: float | None
    display_name: str
    description: str
    objective: Literal["fbeta", "ber"] = "fbeta"


THRESHOLD_PROFILES: dict[ProfileId, ThresholdProfile] = {
    "f0_5": ThresholdProfile(
        profile_id="f0_5",
        beta=F0_5_BETA,
        display_name="F0.5 — conservative",
        description=(
            "Maximise F0.5 on CV validation folds; precision-weighted scoring "
            "(fewer false line stops among the F-beta profiles)."
        ),
    ),
    "f2": ThresholdProfile(
        profile_id="f2",
        beta=F2_BETA,
        display_name="F2 — neutral",
        description=(
            "Maximise F2 on CV validation folds; mid recall emphasis (~4:1). "
            "Used as the default classifier threshold in tuned pipelines."
        ),
    ),
    "f4": ThresholdProfile(
        profile_id="f4",
        beta=F4_BETA,
        display_name="F4 — aggressive",
        description=(
            "Maximise F4 on CV validation folds; stronger recall emphasis (~16:1), "
            "catches more anomalies at the cost of more false stops."
        ),
    ),
    "ber": ThresholdProfile(
        profile_id="ber",
        beta=None,
        display_name="BER — minimum balanced error",
        description=(
            "Minimise mean CV balanced error rate (BER) on the threshold grid; "
            "symmetric pass/fail misclassification cost."
        ),
        objective="ber",
    ),
}


def threshold_profile_config() -> dict[str, float | str]:
    """Snapshot for JSON artifacts and the dashboard."""
    return {
        "f0_5_beta": float(F0_5_BETA),
        "f2_beta": float(F2_BETA),
        "f4_beta": float(F4_BETA),
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
