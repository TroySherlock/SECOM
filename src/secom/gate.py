"""Post-cluster process gate: Hotelling T² + Isolation Forest (passing-train reference).

Scores wafers in the same feature space as the classifier's cluster stage
(impute → SmartCorrelatedSelection), fit on passing training wafers only. The
abstention rule is configurable via ``logic`` (``secom.pipelines.GATE_LOGIC``):
``"or"`` flags when either detector trips, ``"and"`` only when both agree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import FactorAnalysis
from sklearn.ensemble import IsolationForest
from sklearn.utils.validation import check_is_fitted

from secom.hub_interactions import sensor_value_columns
from secom.pipelines import (
    GATE_CORR_THRESHOLD,
    GATE_LOGIC,
    IF_GATE_ALPHA,
    IF_GATE_MAX_SAMPLES,
    IF_GATE_N_ESTIMATORS,
    INTERP_EFA_N_FACTORS,
    INTERP_GATE_CORR_THRESHOLD,
    INTERP_GATE_LOGIC,
    INTERP_Q_GATE_ALPHA,
    INTERP_T2_GATE_ALPHA,
    RANDOM_SEED,
    T2_GATE_ALPHA,
    build_gate_feature_pipeline,
)


class ProcessGate:
    """Hotelling T² + Isolation Forest gate on post-cluster sensor features.

    Parameters
    ----------
    t2_alpha : empirical UCL quantile on passing T² (``1 - alpha``).
    if_alpha : empirical quantile on passing IF ``decision_function``; scores
        below this threshold are flagged anomalous.
    logic : ``"or"`` abstains when either detector trips; ``"and"`` abstains
        only when both T² and IF agree a wafer is out of control.
    """

    def __init__(
        self,
        *,
        t2_alpha: float = T2_GATE_ALPHA,
        if_alpha: float = IF_GATE_ALPHA,
        if_n_estimators: int = IF_GATE_N_ESTIMATORS,
        if_max_samples: str | int | float = IF_GATE_MAX_SAMPLES,
        gate_corr_threshold: float = GATE_CORR_THRESHOLD,
        logic: str = GATE_LOGIC,
    ):
        self.t2_alpha = float(t2_alpha)
        self.if_alpha = float(if_alpha)
        self.if_n_estimators = int(if_n_estimators)
        self.if_max_samples = if_max_samples
        self.gate_corr_threshold = float(gate_corr_threshold)
        logic_norm = str(logic).strip().lower()
        if logic_norm not in ("or", "and"):
            raise ValueError(f"gate logic must be 'or' or 'and', got {logic!r}")
        self.logic = logic_norm

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

    def ooc_mask(self, X: pd.DataFrame) -> np.ndarray:
        """Out-of-control mask under the configured logic ('or' vs 'and')."""
        return self.flag_masks(X)["both_ooc" if self.logic == "and" else "either_ooc"]

    def is_in_control(self, X: pd.DataFrame) -> np.ndarray:
        """In control unless the configured logic flags the wafer OOC.

        OR: abstain when either T² or IF trips. AND: abstain only when both agree.
        """
        return ~self.ooc_mask(X)

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
        ooc = masks["both_ooc" if self.logic == "and" else "either_ooc"]
        n = len(X)
        return {
            "n_total": n,
            "n_flagged_t2": int(masks["t2_ooc"].sum()),
            "n_flagged_if": int(masks["if_ooc"].sum()),
            "n_flagged_both": int(masks["both_ooc"].sum()),
            "n_flagged_either": int(masks["either_ooc"].sum()),
            "n_flagged_ooc": int(ooc.sum()),
            "n_in_control": int((~ooc).sum()),
        }

    def config(self) -> dict:
        return {
            "logic": self.logic,
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


class RegularizedEFA:
    """Regularized Exploratory Factor Analysis monitor (non-Bayesian).

    Fits sklearn ``FactorAnalysis`` (an MLE factor model that already carries a
    per-feature noise term, regularizing it versus PCA) on an in-control
    reference matrix. Hotelling T² is the squared Mahalanobis distance of the
    factor scores using a Ledoit-Wolf-shrunk score covariance; Q (SPE) is the
    squared residual the factors do not reconstruct. Both are classic MSPC
    statistics on the EFA decomposition.
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
    """In-pipeline EFA monitor: append ``gate_t2`` and ``gate_q`` as features.

    Expects an already cluster-processed numeric frame (impute → cluster sits
    upstream in the gate branch). The EFA reference is fit on passing wafers
    (``y == 0``) so the statistics measure deviation from in-control behaviour;
    when ``y`` is unavailable every row is used as the reference.
    """

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
        return pd.DataFrame(
            {self.T2_COL: t2, self.Q_COL: q},
            index=X_df.index,
        )

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
    """Interpolation gate: Regularized EFA → Hotelling T² + Q (SPE) abstention.

    Mirrors :class:`ProcessGate` but swaps the Isolation Forest for the EFA Q
    statistic. Fit on passing training wafers in the post-cluster space; flags a
    wafer out of control under ``logic`` (``"or"`` if either T² or Q exceeds its
    upper control limit, ``"and"`` only when both do).
    """

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
        """Out-of-control mask under the configured logic ('or' vs 'and')."""
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


# Backward-compatible alias
T2Gate = ProcessGate
