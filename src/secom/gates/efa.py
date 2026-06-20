"""Regularized-EFA gate: Hotelling T2 + Q (SPE) risk-coverage tool.

``EFAGate`` is a standalone abstention/risk-coverage report (no longer bound to a
protocol): fit on passing training wafers so the statistics measure deviation
from in-control behaviour, then flag out-of-control holdout wafers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import FactorAnalysis

from secom.core import RANDOM_SEED, build_gate_feature_pipeline
from secom.hub_interactions import sensor_value_columns
from secom.pipelines import (
    EFA_GATE_LOGIC,
    EFA_GATE_N_FACTORS,
    EFA_GATE_Q_ALPHA,
    EFA_GATE_T2_ALPHA,
    GATE_CORR_THRESHOLD,
)


class RegularizedEFA:
    """Regularized Exploratory Factor Analysis monitor (non-Bayesian).

    Fits sklearn ``FactorAnalysis`` on an in-control reference matrix. Hotelling
    T2 is the squared Mahalanobis distance of the factor scores using a
    Ledoit-Wolf-shrunk score covariance; Q (SPE) is the squared residual the
    factors do not reconstruct.
    """

    def __init__(self, n_factors: int = EFA_GATE_N_FACTORS, random_state: int = RANDOM_SEED):
        self.n_factors = int(n_factors)
        self.random_state = int(random_state)

    def fit(self, ref: np.ndarray) -> "RegularizedEFA":
        ref = np.asarray(ref, dtype="float64")
        n_samples, n_features = ref.shape
        if n_samples == 0 or n_features == 0:
            raise ValueError("RegularizedEFA requires a non-empty reference matrix")
        k = max(1, min(self.n_factors, n_features, n_samples - 1))
        self.n_factors_ = int(k)
        self.fa_ = FactorAnalysis(n_components=self.n_factors_, random_state=self.random_state)
        self.fa_.fit(ref)

        scores = self.fa_.transform(ref)
        self.score_mean_ = scores.mean(axis=0)
        centered = scores - self.score_mean_
        lw = LedoitWolf().fit(centered)
        self.score_precision_ = lw.get_precision()
        self.shrinkage_ = float(lw.shrinkage_)

        self.t2_ref_ = self._t2_from_scores(scores)
        self.q_ref_ = self._q_from_matrix(ref, scores)
        self.n_reference_ = int(n_samples)
        self.n_features_ = int(n_features)
        return self

    def _t2_from_scores(self, scores: np.ndarray) -> np.ndarray:
        centered = scores - self.score_mean_
        return np.einsum("ij,jk,ik->i", centered, self.score_precision_, centered)

    def _q_from_matrix(self, mat: np.ndarray, scores: np.ndarray) -> np.ndarray:
        recon = scores @ self.fa_.components_ + self.fa_.mean_
        resid = mat - recon
        return np.einsum("ij,ij->i", resid, resid)

    def t2_q(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "fa_"):
            raise RuntimeError("RegularizedEFA must be fit before scoring")
        mat = np.asarray(X, dtype="float64")
        scores = self.fa_.transform(mat)
        return self._t2_from_scores(scores), self._q_from_matrix(mat, scores)


class EFAGate:
    """Regularized EFA -> Hotelling T2 + Q (SPE) abstention / risk-coverage gate."""

    def __init__(
        self,
        *,
        n_factors: int = EFA_GATE_N_FACTORS,
        t2_alpha: float = EFA_GATE_T2_ALPHA,
        q_alpha: float = EFA_GATE_Q_ALPHA,
        gate_corr_threshold: float = GATE_CORR_THRESHOLD,
        logic: str = EFA_GATE_LOGIC,
    ):
        self.n_factors = int(n_factors)
        self.t2_alpha = float(t2_alpha)
        self.q_alpha = float(q_alpha)
        self.gate_corr_threshold = float(gate_corr_threshold)
        logic_norm = str(logic).strip().lower()
        if logic_norm not in ("or", "and"):
            raise ValueError(f"gate logic must be 'or' or 'and', got {logic!r}")
        self.logic = logic_norm

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "EFAGate":
        sensor_cols = sensor_value_columns(X_train.columns)
        if not sensor_cols:
            raise ValueError("No sensor columns found for EFAGate")

        self.feature_pipe_ = build_gate_feature_pipeline(self.gate_corr_threshold)
        y_arr = np.asarray(y_train, dtype=int)
        X_sensors = X_train[sensor_cols]
        self.feature_pipe_.fit(X_sensors, y_arr)

        X_gate = self._transform_gate_features(X_sensors)
        passing = y_arr == 0
        ref_df = X_gate.loc[passing] if isinstance(X_gate, pd.DataFrame) else X_gate[passing]
        ref = np.asarray(ref_df, dtype="float64")
        if ref.shape[0] == 0 or ref.shape[1] == 0:
            raise ValueError("EFAGate requires passing wafers and nonzero features")

        self.efa_ = RegularizedEFA(n_factors=self.n_factors).fit(ref)
        self.t2_ucl_ = float(np.quantile(self.efa_.t2_ref_, 1.0 - self.t2_alpha))
        self.q_ucl_ = float(np.quantile(self.efa_.q_ref_, 1.0 - self.q_alpha))

        self.n_reference_ = int(ref.shape[0])
        self.n_features_ = int(ref.shape[1])
        self.n_factors_ = int(self.efa_.n_factors_)
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

    def t2_q_scores(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.efa_.t2_q(self._gate_matrix(X))

    def t2_scores(self, X: pd.DataFrame) -> np.ndarray:
        return self.t2_q_scores(X)[0]

    def q_scores(self, X: pd.DataFrame) -> np.ndarray:
        return self.t2_q_scores(X)[1]

    def flag_masks(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        t2, q = self.t2_q_scores(X)
        t2_ooc = np.asarray(t2 > self.t2_ucl_, dtype=bool)
        q_ooc = np.asarray(q > self.q_ucl_, dtype=bool)
        return {
            "t2_ooc": t2_ooc,
            "q_ooc": q_ooc,
            "both_ooc": t2_ooc & q_ooc,
            "either_ooc": t2_ooc | q_ooc,
        }

    def ooc_mask(self, X: pd.DataFrame) -> np.ndarray:
        return self.flag_masks(X)["both_ooc" if self.logic == "and" else "either_ooc"]

    def is_in_control(self, X: pd.DataFrame) -> np.ndarray:
        return ~self.ooc_mask(X)

    def ooc_severity(self, X: pd.DataFrame) -> np.ndarray:
        """Per-wafer out-of-control severity in [0, 1] for risk-coverage ranking.

        Each statistic becomes an empirical upper-tail exceedance p-value against
        the passing reference (high T2 or high Q -> high severity); the max of the
        two is the OR-logic severity, independent of the alpha operating point.
        """
        t2, q = self.t2_q_scores(X)
        ref_t2 = self.efa_.t2_ref_
        ref_q = self.efa_.q_ref_
        anomaly_t2 = np.array([float(np.mean(ref_t2 <= v)) for v in t2], dtype="float64")
        anomaly_q = np.array([float(np.mean(ref_q <= v)) for v in q], dtype="float64")
        return np.maximum(anomaly_t2, anomaly_q)

    def flag_breakdown(self, X: pd.DataFrame) -> dict[str, int]:
        masks = self.flag_masks(X)
        ooc = masks["both_ooc" if self.logic == "and" else "either_ooc"]
        return {
            "n_total": int(len(X)),
            "n_flagged_t2": int(masks["t2_ooc"].sum()),
            "n_flagged_q": int(masks["q_ooc"].sum()),
            "n_flagged_both": int(masks["both_ooc"].sum()),
            "n_flagged_either": int(masks["either_ooc"].sum()),
            "n_flagged_ooc": int(ooc.sum()),
            "n_in_control": int((~ooc).sum()),
        }

    def config(self) -> dict:
        return {
            "logic": self.logic,
            "feature_stage": "post_cluster_efa",
            "method": "regularized_efa_t2_q",
            "n_factors": getattr(self, "n_factors_", self.n_factors),
            "t2_alpha": self.t2_alpha,
            "q_alpha": self.q_alpha,
            "gate_corr_threshold": self.gate_corr_threshold,
            "t2_ucl": getattr(self, "t2_ucl_", None),
            "q_ucl": getattr(self, "q_ucl_", None),
            "n_features": getattr(self, "n_features_", None),
            "n_reference_wafers": getattr(self, "n_reference_", None),
            "shrinkage": getattr(getattr(self, "efa_", None), "shrinkage_", None),
        }
