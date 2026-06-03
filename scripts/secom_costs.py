"""F-beta threshold profile definitions for Stage 2 tuning.

Edit F1_BETA / F2_BETA / F3_BETA below, then re-run threshold tuning and benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score

ProfileId = Literal["f1", "f2", "f3"]

# --- Editable F-beta values (sklearn beta parameter) -------------------------

F1_BETA = 1.0  # conservative (balanced precision/recall)
F2_BETA = 2.0  # neutral (~4:1 recall emphasis); default deployment threshold
F3_BETA = 3.0  # aggressive (~9:1)

DEFAULT_PROFILE_ID: ProfileId = "f2"

PROFILE_IDS: tuple[ProfileId, ...] = ("f1", "f2", "f3")

_LEGACY_NAME_MAP = {
    "ber": "f2",
    "conservative": "f1",
    "aggressive": "f3",
}
# Old tuned JSON used f2/f3/f4 (β=2/3/4); new schema is f1/f2/f3 (β=1/2/3).
_OLD_F234_TO_F123 = {"f2": "f1", "f3": "f2", "f4": "f3"}


@dataclass(frozen=True)
class ThresholdProfile:
    profile_id: ProfileId
    beta: float
    display_name: str
    description: str
    objective: Literal["fbeta"] = "fbeta"


THRESHOLD_PROFILES: dict[ProfileId, ThresholdProfile] = {
    "f1": ThresholdProfile(
        profile_id="f1",
        beta=F1_BETA,
        display_name="F1 — conservative",
        description=(
            "Maximise F1 on CV validation folds; balanced precision and recall "
            "(fewest false line stops among the three profiles)."
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
    "f3": ThresholdProfile(
        profile_id="f3",
        beta=F3_BETA,
        display_name="F3 — aggressive",
        description=(
            "Maximise F3 on CV validation folds; stronger recall emphasis (~9:1), "
            "catches more anomalies at the cost of more false stops."
        ),
    ),
}


def threshold_profile_config() -> dict[str, float | str]:
    """Snapshot for JSON artifacts and the dashboard."""
    return {
        "f1_beta": float(F1_BETA),
        "f2_beta": float(F2_BETA),
        "f3_beta": float(F3_BETA),
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


def normalize_profile_id(profile_id: str, profile_keys: set[str] | None = None) -> str:
    pid = str(profile_id)
    keys = profile_keys or set()
    if "f4" in keys and "f1" not in keys and pid in _OLD_F234_TO_F123:
        return _OLD_F234_TO_F123[pid]
    if pid in PROFILE_IDS:
        return pid
    return _LEGACY_NAME_MAP.get(pid, pid)


def resolve_threshold_profiles(tuned_payload: dict) -> dict[str, float]:
    """Map profile_id -> best_threshold from tuned JSON (legacy-safe)."""
    profiles = tuned_payload.get("threshold_profiles")
    if isinstance(profiles, dict) and profiles:
        keys = {str(k) for k in profiles}
        out: dict[str, float] = {}
        for pid, data in profiles.items():
            norm = normalize_profile_id(pid, keys)
            if isinstance(data, dict) and "best_threshold" in data:
                out[norm] = float(data["best_threshold"])
        if out:
            return out
    legacy_thr = float(tuned_payload.get("classifier_threshold", 0.5))
    return {DEFAULT_PROFILE_ID: legacy_thr}


def has_multi_profile_thresholds(tuned_payload: dict) -> bool:
    profiles = tuned_payload.get("threshold_profiles")
    if not isinstance(profiles, dict):
        return False
    keys = {str(k) for k in profiles}
    resolved = {normalize_profile_id(k, keys) for k in profiles}
    return all(pid in resolved for pid in PROFILE_IDS)
