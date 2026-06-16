"""Post-cluster process gate: Hotelling T² OR Isolation Forest (passing-train reference).

Scores wafers in the same feature space as the classifier's cluster stage
(impute → SmartCorrelatedSelection), fit on passing training wafers only, and
abstains when **either** detector flags out-of-control (OR logic).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest

from secom.hub_interactions import sensor_value_columns
from secom.pipelines import (
    GATE_CORR_THRESHOLD,
    IF_GATE_ALPHA,
    IF_GATE_MAX_SAMPLES,
    IF_GATE_N_ESTIMATORS,
    RANDOM_SEED,
    T2_GATE_ALPHA,
    build_gate_feature_pipeline,
)


class ProcessGate:
    """T² OR Isolation Forest gate on post-cluster sensor features.

    Parameters
    ----------
    t2_alpha : empirical UCL quantile on passing T² (``1 - alpha``).
    if_alpha : empirical quantile on passing IF ``decision_function``; scores
        below this threshold are flagged anomalous.
    """

    def __init__(
        self,
        *,
        t2_alpha: float = T2_GATE_ALPHA,
        if_alpha: float = IF_GATE_ALPHA,
        if_n_estimators: int = IF_GATE_N_ESTIMATORS,
        if_max_samples: str | int | float = IF_GATE_MAX_SAMPLES,
        gate_corr_threshold: float = GATE_CORR_THRESHOLD,
    ):
        self.t2_alpha = float(t2_alpha)
        self.if_alpha = float(if_alpha)
        self.if_n_estimators = int(if_n_estimators)
        self.if_max_samples = if_max_samples
        self.gate_corr_threshold = float(gate_corr_threshold)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "ProcessGate":
        sensor_cols = sensor_value_columns(X_train.columns)
        if not sensor_cols:
            raise ValueError("No sensor columns found for ProcessGate")

        self.feature_pipe_ = build_gate_feature_pipeline(self.gate_corr_threshold)
        y_arr = np.asarray(y_train, dtype=int)
        X_sensors = X_train[sensor_cols]
        self.feature_pipe_.fit(X_sensors, y_arr)

        X_gate = self._transform_gate_features(X_sensors)
        passing = y_arr == 0
        ref_df = X_gate.loc[passing] if isinstance(X_gate, pd.DataFrame) else X_gate[passing]
        ref = np.asarray(ref_df, dtype="float64")
        if ref.shape[0] == 0 or ref.shape[1] == 0:
            raise ValueError("ProcessGate requires passing wafers and nonzero features")

        self.mean_ = ref.mean(axis=0)
        lw = LedoitWolf().fit(ref)
        self.precision_ = lw.get_precision()
        self.shrinkage_ = float(lw.shrinkage_)

        ref_t2 = self._t2_from_matrix(ref)
        self.ucl_ = float(np.quantile(ref_t2, 1.0 - self.t2_alpha))
        self.ucl_chi2_ = float(chi2.ppf(1.0 - self.t2_alpha, df=ref.shape[1]))

        self.iforest_ = IsolationForest(
            n_estimators=self.if_n_estimators,
            max_samples=self.if_max_samples,
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        self.iforest_.fit(ref)
        ref_if = self.iforest_.decision_function(ref)
        self.if_threshold_ = float(np.quantile(ref_if, self.if_alpha))

        self.n_reference_ = int(ref.shape[0])
        self.n_features_ = int(ref.shape[1])
        self.feature_names_ = list(X_gate.columns) if isinstance(X_gate, pd.DataFrame) else []
        return self

    def _transform_gate_features(self, X_sensors: pd.DataFrame) -> pd.DataFrame:
        out = self.feature_pipe_.transform(X_sensors)
        if not isinstance(out, pd.DataFrame):
            out = pd.DataFrame(out, index=X_sensors.index)
        return out.astype("float64")

    def _gate_matrix(self, X: pd.DataFrame) -> np.ndarray:
        sensor_cols = sensor_value_columns(X.columns)
        frame = self._transform_gate_features(X[sensor_cols])
        return frame.to_numpy(dtype="float64")

    def _t2_from_matrix(self, mat: np.ndarray) -> np.ndarray:
        centered = mat - self.mean_
        return np.einsum("ij,jk,ik->i", centered, self.precision_, centered)

    def t2_scores(self, X: pd.DataFrame) -> np.ndarray:
        return self._t2_from_matrix(self._gate_matrix(X))

    def if_scores(self, X: pd.DataFrame) -> np.ndarray:
        return self.iforest_.decision_function(self._gate_matrix(X))

    def t2_ooc(self, X: pd.DataFrame) -> np.ndarray:
        return self.t2_scores(X) > self.ucl_

    def if_ooc(self, X: pd.DataFrame) -> np.ndarray:
        return self.if_scores(X) < self.if_threshold_

    def is_in_control(self, X: pd.DataFrame) -> np.ndarray:
        """In control when neither T² nor IF flags OOC (OR abstention)."""
        return ~(self.t2_ooc(X) | self.if_ooc(X))

    def flag_masks(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        t2 = np.asarray(self.t2_ooc(X), dtype=bool)
        iff = np.asarray(self.if_ooc(X), dtype=bool)
        return {
            "t2_ooc": t2,
            "if_ooc": iff,
            "both_ooc": t2 & iff,
            "either_ooc": t2 | iff,
        }

    def flag_breakdown(self, X: pd.DataFrame) -> dict[str, int]:
        masks = self.flag_masks(X)
        n = len(X)
        return {
            "n_total": n,
            "n_flagged_t2": int(masks["t2_ooc"].sum()),
            "n_flagged_if": int(masks["if_ooc"].sum()),
            "n_flagged_both": int(masks["both_ooc"].sum()),
            "n_flagged_either": int(masks["either_ooc"].sum()),
            "n_in_control": int((~masks["either_ooc"]).sum()),
        }

    def config(self) -> dict:
        return {
            "logic": "or",
            "feature_stage": "post_cluster",
            "t2_alpha": self.t2_alpha,
            "if_alpha": self.if_alpha,
            "if_n_estimators": self.if_n_estimators,
            "if_max_samples": self.if_max_samples,
            "gate_corr_threshold": self.gate_corr_threshold,
            "ucl": getattr(self, "ucl_", None),
            "ucl_chi2_reference": getattr(self, "ucl_chi2_", None),
            "if_threshold": getattr(self, "if_threshold_", None),
            "n_features": getattr(self, "n_features_", None),
            "n_reference_wafers": getattr(self, "n_reference_", None),
            "shrinkage": getattr(self, "shrinkage_", None),
        }


# Backward-compatible alias
T2Gate = ProcessGate
