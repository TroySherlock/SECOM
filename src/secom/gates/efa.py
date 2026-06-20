"""Interpolation gate: Regularized EFA -> Hotelling T2 + Q (SPE) statistics.

Provides the in-pipeline ``EFAMonitorFeatures`` (appends ``gate_t2``/``gate_q``)
and the standalone ``InterpProcessGate`` abstention report. Fit on passing
training wafers so the statistics measure deviation from in-control behaviour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import FactorAnalysis
from sklearn.utils.validation import check_is_fitted

from secom.core import RANDOM_SEED, build_gate_feature_pipeline
from secom.hub_interactions import sensor_value_columns
from secom.intrap_pipelines import (
    INTERP_EFA_N_FACTORS,
    INTERP_GATE_CORR_THRESHOLD,
    INTERP_GATE_LOGIC,
    INTERP_Q_GATE_ALPHA,
    INTERP_T2_GATE_ALPHA,
)


class RegularizedEFA:
    """Regularized Exploratory Factor Analysis monitor (non-Bayesian).

    Fits sklearn ``FactorAnalysis`` on an in-control reference matrix. Hotelling
    T2 is the squared Mahalanobis distance of the factor scores using a
    Ledoit-Wolf-shrunk score covariance; Q (SPE) is the squared residual the
    factors do not reconstruct.
    """

    def __init__(self, n_factors: int = INTERP_EFA_N_FACTORS, random_state: int = RANDOM_SEED):
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


class EFAMonitorFeatures(BaseEstimator, TransformerMixin):
    """In-pipeline EFA monitor: append ``gate_t2`` and ``gate_q`` as features."""

    T2_COL = "gate_t2"
    Q_COL = "gate_q"

    def __init__(
        self,
        n_factors: int = INTERP_EFA_N_FACTORS,
        t2_alpha: float = INTERP_T2_GATE_ALPHA,
        q_alpha: float = INTERP_Q_GATE_ALPHA,
    ):
        self.n_factors = int(n_factors)
        self.t2_alpha = float(t2_alpha)
        self.q_alpha = float(q_alpha)

    def fit(self, X, y=None):
        X_df = self._as_dataframe(X)
        self.feature_names_in_ = list(X_df.columns)
        mat = X_df.to_numpy(dtype="float64")
        if y is not None:
            passing = np.asarray(y, dtype=int) == 0
            if passing.sum() >= max(self.n_factors + 1, 5):
                mat = mat[passing]
        self.efa_ = RegularizedEFA(n_factors=self.n_factors).fit(mat)
        self.t2_ucl_ = float(np.quantile(self.efa_.t2_ref_, 1.0 - self.t2_alpha))
        self.q_ucl_ = float(np.quantile(self.efa_.q_ref_, 1.0 - self.q_alpha))
        return self

    def transform(self, X):
        check_is_fitted(self, "efa_")
        X_df = self._as_dataframe(X)
        t2, q = self.efa_.t2_q(X_df.to_numpy(dtype="float64"))
        return pd.DataFrame({self.T2_COL: t2, self.Q_COL: q}, index=X_df.index)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "efa_")
        return np.asarray([self.T2_COL, self.Q_COL], dtype=object)

    @staticmethod
    def _as_dataframe(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)
        df.columns = df.columns.astype(str)
        return df


class InterpProcessGate:
    """Interpolation gate: Regularized EFA -> Hotelling T2 + Q (SPE) abstention."""

    def __init__(
        self,
        *,
        n_factors: int = INTERP_EFA_N_FACTORS,
        t2_alpha: float = INTERP_T2_GATE_ALPHA,
        q_alpha: float = INTERP_Q_GATE_ALPHA,
        gate_corr_threshold: float = INTERP_GATE_CORR_THRESHOLD,
        logic: str = INTERP_GATE_LOGIC,
    ):
        self.n_factors = int(n_factors)
        self.t2_alpha = float(t2_alpha)
        self.q_alpha = float(q_alpha)
        self.gate_corr_threshold = float(gate_corr_threshold)
        logic_norm = str(logic).strip().lower()
        if logic_norm not in ("or", "and"):
            raise ValueError(f"gate logic must be 'or' or 'and', got {logic!r}")
        self.logic = logic_norm

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "InterpProcessGate":
        sensor_cols = sensor_value_columns(X_train.columns)
        if not sensor_cols:
            raise ValueError("No sensor columns found for InterpProcessGate")

        self.feature_pipe_ = build_gate_feature_pipeline(self.gate_corr_threshold)
        y_arr = np.asarray(y_train, dtype=int)
        X_sensors = X_train[sensor_cols]
        self.feature_pipe_.fit(X_sensors, y_arr)

        X_gate = self._transform_gate_features(X_sensors)
        passing = y_arr == 0
        ref_df = X_gate.loc[passing] if isinstance(X_gate, pd.DataFrame) else X_gate[passing]
        ref = np.asarray(ref_df, dtype="float64")
        if ref.shape[0] == 0 or ref.shape[1] == 0:
            raise ValueError("InterpProcessGate requires passing wafers and nonzero features")

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
