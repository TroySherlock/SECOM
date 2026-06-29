"""PCA-MSPC gate: Hotelling T2 + Q (SPE) risk-coverage tool (fab-standard baseline).

``PCAGate`` is the industry-standard multivariate-SPC monitor: PCA on the
in-control (passing) training wafers, then Hotelling T2 (in-model excursion) and
Q / SPE (out-of-model residual) flag out-of-control holdout wafers. It is the
baseline against which the custom ``BayesGate`` (sBFA -> BGM) is compared; both
share identical preprocessing so the only difference is the latent model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from secom.hub_interactions import sensor_value_columns
from secom.pipelines import (
    GATE_CORR_THRESHOLD,
    PCA_GATE_CLIP,
    PCA_GATE_LOGIC,
    PCA_GATE_N_COMPONENTS,
    PCA_GATE_Q_ALPHA,
    PCA_GATE_T2_ALPHA,
    RANDOM_SEED,
    build_gate_feature_pipeline,
)


class PCAMonitor:
    """Standard PCA-MSPC monitor (fab baseline).

    Fits sklearn ``PCA`` on an in-control reference matrix. Hotelling T2 is the
    eigenvalue-scaled sum of squared scores (sum_k t_k^2 / lambda_k); Q (SPE) is
    the squared residual the retained components do not reconstruct.
    """

    def __init__(
        self, n_components: int = PCA_GATE_N_COMPONENTS, random_state: int = RANDOM_SEED
    ):
        self.n_components = int(n_components)
        self.random_state = int(random_state)

    def fit(self, ref: np.ndarray) -> "PCAMonitor":
        ref = np.asarray(ref, dtype="float64")
        n_samples, n_features = ref.shape
        if n_samples == 0 or n_features == 0:
            raise ValueError("PCAMonitor requires a non-empty reference matrix")
        k = max(1, min(self.n_components, n_features, n_samples - 1))
        self.n_components_ = int(k)
        self.pca_ = PCA(n_components=self.n_components_, random_state=self.random_state)
        self.pca_.fit(ref)
        # Guard against zero/near-zero eigenvalues in the T2 scaling.
        self.explained_variance_ = np.clip(
            np.asarray(self.pca_.explained_variance_, dtype="float64"), 1e-12, None
        )

        scores = self.pca_.transform(ref)
        self.t2_ref_ = self._t2_from_scores(scores)
        self.q_ref_ = self._q_from_matrix(ref, scores)
        self.n_reference_ = int(n_samples)
        self.n_features_ = int(n_features)
        return self

    def _t2_from_scores(self, scores: np.ndarray) -> np.ndarray:
        return np.einsum("ij,ij->i", scores, scores / self.explained_variance_)

    def _q_from_matrix(self, mat: np.ndarray, scores: np.ndarray) -> np.ndarray:
        recon = scores @ self.pca_.components_ + self.pca_.mean_
        resid = mat - recon
        return np.einsum("ij,ij->i", resid, resid)

    def t2_q(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "pca_"):
            raise RuntimeError("PCAMonitor must be fit before scoring")
        mat = np.asarray(X, dtype="float64")
        scores = self.pca_.transform(mat)
        return self._t2_from_scores(scores), self._q_from_matrix(mat, scores)


class PCAGate:
    """Standard PCA -> Hotelling T2 + Q (SPE) abstention / risk-coverage gate."""

    def __init__(
        self,
        *,
        n_components: int = PCA_GATE_N_COMPONENTS,
        t2_alpha: float = PCA_GATE_T2_ALPHA,
        q_alpha: float = PCA_GATE_Q_ALPHA,
        gate_corr_threshold: float = GATE_CORR_THRESHOLD,
        logic: str = PCA_GATE_LOGIC,
        clip: float = PCA_GATE_CLIP,
    ):
        self.n_components = int(n_components)
        self.t2_alpha = float(t2_alpha)
        self.q_alpha = float(q_alpha)
        self.gate_corr_threshold = float(gate_corr_threshold)
        self.clip = float(clip)
        logic_norm = str(logic).strip().lower()
        if logic_norm not in ("or", "and"):
            raise ValueError(f"gate logic must be 'or' or 'and', got {logic!r}")
        self.logic = logic_norm

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "PCAGate":
        sensor_cols = sensor_value_columns(X_train.columns)
        if not sensor_cols:
            raise ValueError("No sensor columns found for PCAGate")
        self.sensor_cols_ = sensor_cols

        self.feature_pipe_ = build_gate_feature_pipeline(self.gate_corr_threshold)
        y_arr = np.asarray(y_train, dtype=int)
        X_sensors = X_train[sensor_cols]
        self.feature_pipe_.fit(X_sensors, y_arr)

        post = self._post_cluster(X_sensors)
        # Robust (median/IQR) scaling fit on all wafers; _scale_clip then clips the
        # tails RobustScaler leaves intact so spikes can't dominate the Gaussian
        # T2 / Q statistics (mirrors the sBFA BayesGate).
        self.scaler_ = RobustScaler().fit(post)

        passing = y_arr == 0
        ref = self._scale_clip(post)[passing]
        if ref.shape[0] == 0 or ref.shape[1] == 0:
            raise ValueError("PCAGate requires passing wafers and nonzero features")

        self.pca_ = PCAMonitor(n_components=self.n_components).fit(ref)
        self.t2_ucl_ = float(np.quantile(self.pca_.t2_ref_, 1.0 - self.t2_alpha))
        self.q_ucl_ = float(np.quantile(self.pca_.q_ref_, 1.0 - self.q_alpha))

        # Reference component scores for the PCA component-space / loadings drift
        # visuals (mirrors BayesGate.ref_scores_; single deterministic fit).
        self.ref_scores_ = self.pca_.pca_.transform(ref)

        self.n_reference_ = int(ref.shape[0])
        self.n_features_ = int(ref.shape[1])
        self.n_components_ = int(self.pca_.n_components_)
        return self

    def _post_cluster(self, X_sensors: pd.DataFrame) -> np.ndarray:
        """Raw sensors -> impute -> SmartCorrelatedSelection (post-cluster frame)."""
        out = self.feature_pipe_.transform(X_sensors)
        if not isinstance(out, pd.DataFrame):
            out = pd.DataFrame(out, index=X_sensors.index)
        self.feature_names_ = list(out.columns)
        return out.to_numpy(dtype="float64")

    def _scale_clip(self, post: np.ndarray) -> np.ndarray:
        """RobustScaler transform clipped to +/-self.clip so spikes can't dominate."""
        return np.clip(self.scaler_.transform(post), -self.clip, self.clip)

    def _gate_matrix(self, X: pd.DataFrame) -> np.ndarray:
        post = self._post_cluster(X[self.sensor_cols_])
        return self._scale_clip(post)

    def t2_q_scores(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.pca_.t2_q(self._gate_matrix(X))

    def q_scores(self, X: pd.DataFrame) -> np.ndarray:
        return self.t2_q_scores(X)[1]

    def diagnostics(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Per-wafer control statistics + OOC flags for the gate-monitor charts.

        One pass yields Hotelling T2, Q/SPE, and the T2/Q/combined out-of-control
        masks at the fitted upper control limits, over all wafers (mirrors
        ``BayesGate.diagnostics`` for the 5.2 distribution-shift / control chart).
        """
        t2, q = self.t2_q_scores(X)
        t2_ooc = np.asarray(t2 > self.t2_ucl_, dtype=bool)
        q_ooc = np.asarray(q > self.q_ucl_, dtype=bool)
        ooc = t2_ooc & q_ooc if self.logic == "and" else t2_ooc | q_ooc
        return {
            "t2": t2,
            "q": q,
            "t2_ooc": t2_ooc,
            "q_ooc": q_ooc,
            "ooc": ooc,
        }

    def factor_scores(self, X: pd.DataFrame) -> np.ndarray:
        """PCA component scores (n, n_components) for the geometry views."""
        return self.pca_.pca_.transform(self._gate_matrix(X))

    def residual_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Per-sensor reconstruction residual in the scaled gate space (n, n_features).

        ``mat - (scores @ components_ + mean_)`` - the part the retained PCA
        components cannot rebuild. Squared row-sums give Q/SPE; per-column squares
        give the equal-weight (PCA-style) contribution of each sensor.
        """
        mat = self._gate_matrix(X)
        scores = self.pca_.pca_.transform(mat)
        recon = scores @ self.pca_.pca_.components_ + self.pca_.pca_.mean_
        return mat - recon

    def loadings(self) -> np.ndarray:
        """PCA loadings ``Pᵀ`` (n_features, n_components)."""
        return self.pca_.pca_.components_.T

    def score_gaussian(self) -> tuple[np.ndarray, np.ndarray]:
        """Component-score Gaussian (mean, covariance) for the Hotelling control ellipse.

        PCA scores are mean-centred and uncorrelated, so the mean is zero and the
        covariance is diagonal with the retained eigenvalues.
        """
        mean = np.zeros(self.pca_.n_components_, dtype="float64")
        cov = np.diag(self.pca_.explained_variance_)
        return mean, cov

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
        ref_t2 = self.pca_.t2_ref_
        ref_q = self.pca_.q_ref_
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
            "feature_stage": "post_cluster_pca_scaled",
            "method": "pca_t2_q",
            "n_components": getattr(self, "n_components_", self.n_components),
            "t2_alpha": self.t2_alpha,
            "q_alpha": self.q_alpha,
            "gate_corr_threshold": self.gate_corr_threshold,
            "clip": self.clip,
            "t2_ucl": getattr(self, "t2_ucl_", None),
            "q_ucl": getattr(self, "q_ucl_", None),
            "n_features": getattr(self, "n_features_", None),
            "n_reference_wafers": getattr(self, "n_reference_", None),
        }
