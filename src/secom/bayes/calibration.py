"""Out-of-fold isotonic probability calibration for the Bayesian extrapolation head.

The NumPyro elastic-net head returns posterior-mean probabilities; under class
weighting (``pos_weight``) those are systematically inflated (the likelihood is
tilted toward the minority class), and the shrinkage prior + drift add further
miscalibration. We fit a monotonic isotonic map on the blocked-CV out-of-fold
predictions during tuning and apply it inside ``predict_proba`` so the
cost-optimal thresholds and confusion matrices stay valid. Because the map is
monotonic, PR-AUC / ROC-AUC are unchanged.

The fitted map is stored as two breakpoint arrays so it serializes cleanly into
the tuned JSON and is reapplied at predict time via ``np.interp`` (no sklearn
object needs to be persisted).
"""
from __future__ import annotations

import numpy as np

# Minimum OOF points (and both classes present) required to fit a stable map;
# below this we fall back to the identity so we never invent a calibration.
_MIN_POINTS = 10


class IsotonicCalibrator:
    """Monotonic isotonic calibration map fit on out-of-fold probabilities."""

    def __init__(self):
        self.identity_ = True
        self.x_thresholds_: np.ndarray = np.asarray([0.0, 1.0], dtype=float)
        self.y_thresholds_: np.ndarray = np.asarray([0.0, 1.0], dtype=float)

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression

        p = np.asarray(p, dtype=float)
        y = np.asarray(y, dtype=float)
        if p.size < _MIN_POINTS or len(np.unique(y)) < 2:
            self.identity_ = True
            return self
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p, y)
        x = np.asarray(iso.X_thresholds_, dtype=float)
        yv = np.asarray(iso.y_thresholds_, dtype=float)
        # np.interp needs >= 2 strictly usable breakpoints; guard degenerate fits.
        if x.size < 2:
            self.identity_ = True
            return self
        self.identity_ = False
        self.x_thresholds_ = x
        self.y_thresholds_ = yv
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.identity_:
            return p
        # np.interp clamps to endpoint values outside the range == out_of_bounds="clip".
        return np.interp(p, self.x_thresholds_, self.y_thresholds_)

    def to_dict(self) -> dict:
        return {
            "method": "isotonic",
            "identity": bool(self.identity_),
            "x_thresholds": [float(v) for v in self.x_thresholds_],
            "y_thresholds": [float(v) for v in self.y_thresholds_],
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "IsotonicCalibrator":
        cal = cls()
        if not payload:
            return cal
        cal.identity_ = bool(payload.get("identity", False))
        x = payload.get("x_thresholds")
        y = payload.get("y_thresholds")
        if x and y and len(x) >= 2:
            cal.x_thresholds_ = np.asarray(x, dtype=float)
            cal.y_thresholds_ = np.asarray(y, dtype=float)
        else:
            cal.identity_ = True
        return cal
