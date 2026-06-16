"""Train-only drift-stability feature screening.

Compares an early vs late half of the *training* slice (split at the median
measurement time) with a per-sensor two-sample KS test, then applies
Benjamini-Hochberg FDR control. Sensors whose distribution already shifts
within the training window are flagged as drifting and dropped before modeling.

No holdout rows are touched here; the resulting drop list is injected into the
pipeline's sensor-branch DriftStabilityDropper for the drift-filtered variant.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from secom.pipelines import TARGET_COL, TIMESTAMP_COL, _SENSOR_VALUE_PATTERN

try:
    from scipy.stats import false_discovery_control

    def _bh_adjust(pvalues: np.ndarray) -> np.ndarray:
        return np.asarray(false_discovery_control(pvalues, method="bh"), dtype=float)

except ImportError:  # pragma: no cover - scipy < 1.11 fallback

    def _bh_adjust(pvalues: np.ndarray) -> np.ndarray:
        p = np.asarray(pvalues, dtype=float)
        n = p.size
        order = np.argsort(p)
        ranked = p[order] * n / (np.arange(n) + 1)
        adjusted_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty(n, dtype=float)
        out[order] = np.clip(adjusted_sorted, 0.0, 1.0)
        return out


_SENSOR_RE = re.compile(_SENSOR_VALUE_PATTERN)


def sensor_value_columns(df: pd.DataFrame) -> list[str]:
    """Sensor value columns only (c_\\d+), excluding calendar/missing-flag/aux."""
    return [c for c in df.columns if _SENSOR_RE.match(c)]


def compute_stable_features(
    train_df: pd.DataFrame,
    *,
    alpha: float = 0.05,
    timestamp_col: str = TIMESTAMP_COL,
    target_col: str = TARGET_COL,
    max_sample: int = 25,
) -> dict[str, Any]:
    """KS + BH-FDR drift screen over the train slice only.

    Returns a metadata dict including the list of drifting columns to drop.
    The computation never sees holdout rows; mild CV feature-screening is
    accepted for this first pass.
    """
    sensor_cols = sensor_value_columns(train_df)
    ts = pd.to_datetime(train_df[timestamp_col], errors="coerce")
    cutpoint = ts.median()

    past_mask = (ts <= cutpoint).to_numpy()
    present_mask = ~past_mask

    # Median-impute (train medians) for KS comparability across missingness.
    # Cast to float first: some sensors are nullable-integer dtype.
    sensors = train_df[sensor_cols].astype("float64")
    medians = sensors.median(numeric_only=True)
    imputed = sensors.fillna(medians)

    past = imputed.loc[past_mask]
    present = imputed.loc[present_mask]

    tested_cols: list[str] = []
    pvalues: list[float] = []
    for col in sensor_cols:
        a = past[col].to_numpy()
        b = present[col].to_numpy()
        # Constant on either side or empty -> skip (treated as stable).
        if a.size == 0 or b.size == 0:
            continue
        if np.nanstd(a) == 0.0 and np.nanstd(b) == 0.0:
            continue
        stat, pval = ks_2samp(a, b)
        tested_cols.append(col)
        pvalues.append(float(pval))

    dropped_columns: list[str] = []
    if pvalues:
        adjusted = _bh_adjust(np.asarray(pvalues, dtype=float))
        dropped_columns = [
            col for col, padj in zip(tested_cols, adjusted) if padj < alpha
        ]

    return {
        "method": "ks_2samp+bh_fdr",
        "alpha": float(alpha),
        "cutpoint": cutpoint.isoformat() if pd.notna(cutpoint) else None,
        "n_sensors": int(len(sensor_cols)),
        "n_tested": int(len(tested_cols)),
        "n_dropped": int(len(dropped_columns)),
        "n_past_rows": int(past_mask.sum()),
        "n_present_rows": int(present_mask.sum()),
        "dropped_columns": dropped_columns,
        "dropped_sample": dropped_columns[:max_sample],
    }
