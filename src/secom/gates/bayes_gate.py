"""Bayesian gate: sBFA -> BayesianGaussianMixture density + Q (SPE).

The EFA gate (``secom.gates.efa.EFAGate``) uses a Regularized EFA -> Hotelling T2
+ Q monitor. ``BayesGate`` is its Bayesian analogue: a *sparse Bayesian factor
analysis* (NumPyro, ADVI) replaces the frequentist EFA, a
``BayesianGaussianMixture`` density on the factor scores replaces the Hotelling
T2 density, and the same Q / SPE reconstruction statistic flags structural
breaks the factor model cannot explain. No Hotelling T2 and no Isolation Forest.

The gate scores wafers in the RAW post-cluster sensor space (impute ->
SmartCorrelatedSelection), not the rolling-Z space, so it stays orthogonal to the
yield models and flags process excursions they do not already capture. It is fit
on the passing training wafers; control limits are empirical passing-wafer
quantiles (low BGM log-density or high SPE trips the gate).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import RobustScaler

from secom.core import RANDOM_SEED, build_gate_feature_pipeline
from secom.hub_interactions import sensor_value_columns
from secom.pipelines import (
    BAYES_GATE_BGM_COMPONENTS,
    BAYES_GATE_CLIP,
    BAYES_GATE_DENSITY_ALPHA,
    BAYES_GATE_LOADING_SCALE,
    BAYES_GATE_LOGIC,
    BAYES_GATE_N_FACTORS,
    BAYES_GATE_N_SEEDS,
    BAYES_GATE_Q_ALPHA,
    BAYES_GATE_SVI_STEPS,
    GATE_CORR_THRESHOLD,
)


class SparseBayesianFactorAnalysis:
    """Sparse Bayesian factor analysis on standardized data (NumPyro, ADVI).

    Marginalises the latent scores analytically: ``x ~ N(0, W Wᵀ + diag(psi))``
    via ``LowRankMultivariateNormal`` (Woodbury), so inference is over the
    sparse loadings ``W`` (Laplace prior) and diagonal noise variances ``psi``
    only. Posterior-mean ``W_``/``psi_`` give closed-form factor scores and
    reconstructions for the gate's density and Q statistics.
    """

    def __init__(
        self,
        *,
        n_factors: int = BAYES_GATE_N_FACTORS,
        loading_scale: float = BAYES_GATE_LOADING_SCALE,
        svi_steps: int = BAYES_GATE_SVI_STEPS,
        seed: int = RANDOM_SEED,
    ):
        self.n_factors = int(n_factors)
        self.loading_scale = float(loading_scale)
        self.svi_steps = int(svi_steps)
        self.seed = int(seed)

    def _model(self, X, n_features: int, k: int):
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist

        W = numpyro.sample(
            "W", dist.Laplace(jnp.zeros((n_features, k)), self.loading_scale)
        )
        psi = numpyro.sample("psi", dist.HalfNormal(jnp.ones(n_features)))
        numpyro.sample(
            "x",
            dist.LowRankMultivariateNormal(
                jnp.zeros(n_features), cov_factor=W, cov_diag=psi + 1e-4
            ),
            obs=X,
        )

    def fit(self, X: np.ndarray) -> "SparseBayesianFactorAnalysis":
        import jax
        import jax.numpy as jnp
        from numpyro.infer import SVI, Trace_ELBO
        from numpyro.infer.autoguide import AutoNormal
        import numpyro.optim as optim

        mat = np.asarray(X, dtype="float64")
        n_samples, n_features = mat.shape
        if n_samples == 0 or n_features == 0:
            raise ValueError("SparseBayesianFactorAnalysis requires a non-empty matrix")
        k = max(1, min(self.n_factors, n_features, n_samples - 1))
        self.n_factors_ = int(k)
        self.n_features_ = int(n_features)

        guide = AutoNormal(self._model)
        svi = SVI(self._model, guide, optim.Adam(0.01), Trace_ELBO())
        rng = jax.random.PRNGKey(self.seed)
        result = svi.run(
            rng,
            self.svi_steps,
            X=jnp.asarray(mat),
            n_features=n_features,
            k=k,
            progress_bar=False,
        )
        rng_s = jax.random.PRNGKey(self.seed + 1)
        samples = guide.sample_posterior(rng_s, result.params, sample_shape=(200,))
        self.W_ = np.asarray(samples["W"]).mean(axis=0)
        self.psi_ = np.asarray(samples["psi"]).mean(axis=0) + 1e-4

        # Precompute the FA score projection: z = (I + Wᵀ Ψ⁻¹ W)⁻¹ Wᵀ Ψ⁻¹ x.
        wt_psi = self.W_.T / self.psi_  # (k, p)
        precision = np.eye(k) + wt_psi @ self.W_  # (k, k)
        self._proj_ = np.linalg.solve(precision, wt_psi)  # (k, p)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "W_"):
            raise RuntimeError("SparseBayesianFactorAnalysis must be fit before scoring")
        mat = np.asarray(X, dtype="float64")
        return mat @ self._proj_.T  # (n, k)

    def reconstruct(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores, dtype="float64") @ self.W_.T  # (n, p)

    def spe(self, X: np.ndarray) -> np.ndarray:
        """Squared prediction error (Q statistic): residual the factors miss."""
        mat = np.asarray(X, dtype="float64")
        resid = mat - self.reconstruct(self.transform(mat))
        return np.einsum("ij,ij->i", resid, resid)


class BayesGate:
    """sBFA -> BGM density + Q (SPE) abstention / risk-coverage gate.

    ``density_alpha`` is the lower-tail quantile on passing BGM log-densities
    (scores below trip the gate), ``q_alpha`` the upper-tail quantile on passing
    SPE, and ``logic`` is ``"or"`` (abstain when either trips) or ``"and"`` (only
    when both agree).
    """

    def __init__(
        self,
        *,
        n_factors: int = BAYES_GATE_N_FACTORS,
        n_mixture_components: int = BAYES_GATE_BGM_COMPONENTS,
        loading_scale: float = BAYES_GATE_LOADING_SCALE,
        density_alpha: float = BAYES_GATE_DENSITY_ALPHA,
        q_alpha: float = BAYES_GATE_Q_ALPHA,
        svi_steps: int = BAYES_GATE_SVI_STEPS,
        gate_corr_threshold: float = GATE_CORR_THRESHOLD,
        n_seeds: int = BAYES_GATE_N_SEEDS,
        logic: str = BAYES_GATE_LOGIC,
        clip: float = BAYES_GATE_CLIP,
    ):
        self.n_factors = int(n_factors)
        self.n_mixture_components = int(n_mixture_components)
        self.loading_scale = float(loading_scale)
        self.density_alpha = float(density_alpha)
        self.q_alpha = float(q_alpha)
        self.svi_steps = int(svi_steps)
        self.gate_corr_threshold = float(gate_corr_threshold)
        self.clip = float(clip)
        self.n_seeds = max(1, int(n_seeds))
        logic_norm = str(logic).strip().lower()
        if logic_norm not in ("or", "and"):
            raise ValueError(f"gate logic must be 'or' or 'and', got {logic!r}")
        self.logic = logic_norm

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "BayesGate":
        sensor_cols = sensor_value_columns(X_train.columns)
        if not sensor_cols:
            raise ValueError("No raw sensor columns found for BayesGate")
        self.sensor_cols_ = sensor_cols

        self.feature_pipe_ = build_gate_feature_pipeline(self.gate_corr_threshold)
        y_arr = np.asarray(y_train, dtype=int)
        X_sensors = X_train[sensor_cols]
        self.feature_pipe_.fit(X_sensors, y_arr)

        post = self._post_cluster(X_sensors)
        # Robust (median/IQR) scaling fit on all wafers; _scale_clip then clips the
        # tails RobustScaler leaves intact so spikes can't dominate the Gaussian
        # sBFA/BGM/SPE.
        self.scaler_ = RobustScaler().fit(post)

        passing = y_arr == 0
        ref = self._scale_clip(post)[passing]
        if ref.shape[0] == 0 or ref.shape[1] == 0:
            raise ValueError("BayesGate requires passing wafers and features")

        # Seed-ensemble: density (BGM log-likelihood) and Q (SPE) are invariant to
        # factor rotation/sign, so averaging them across independent sBFA+BGM fits
        # removes the ADVI run-to-run flip.
        n_components = max(1, min(self.n_mixture_components, ref.shape[0]))
        self.members_: list[tuple[SparseBayesianFactorAnalysis, BayesianGaussianMixture]] = []
        for i in range(self.n_seeds):
            sbfa = SparseBayesianFactorAnalysis(
                n_factors=self.n_factors,
                loading_scale=self.loading_scale,
                svi_steps=self.svi_steps,
                seed=RANDOM_SEED + i,
            ).fit(ref)
            bgm = BayesianGaussianMixture(
                n_components=n_components,
                covariance_type="full",
                random_state=RANDOM_SEED + i,
                max_iter=500,
                reg_covar=1e-4,
            ).fit(sbfa.transform(ref))
            self.members_.append((sbfa, bgm))

        self.ref_density_ = self._density_from_matrix(ref)
        self.ref_q_ = self._q_from_matrix(ref)
        self.density_lcl_ = float(np.quantile(self.ref_density_, self.density_alpha))
        self.q_ucl_ = float(np.quantile(self.ref_q_, 1.0 - self.q_alpha))

        self.n_reference_ = int(ref.shape[0])
        self.n_features_ = int(ref.shape[1])
        self.n_factors_ = int(self.members_[0][0].n_factors_)
        self.n_mixture_components_ = int(n_components)
        return self

    def _post_cluster(self, X_sensors: pd.DataFrame) -> np.ndarray:
        """Raw sensors -> impute -> SmartCorrelatedSelection (post-cluster frame)."""
        out = self.feature_pipe_.transform(X_sensors)
        if not isinstance(out, pd.DataFrame):
            out = pd.DataFrame(out, index=X_sensors.index)
        return out.to_numpy(dtype="float64")

    def _scale_clip(self, post: np.ndarray) -> np.ndarray:
        """RobustScaler transform clipped to +/-self.clip so spikes can't dominate."""
        return np.clip(self.scaler_.transform(post), -self.clip, self.clip)

    def _gate_matrix(self, X: pd.DataFrame) -> np.ndarray:
        post = self._post_cluster(X[self.sensor_cols_])
        return self._scale_clip(post)

    def _density_from_matrix(self, std: np.ndarray) -> np.ndarray:
        per_member = [bgm.score_samples(sbfa.transform(std)) for sbfa, bgm in self.members_]
        return np.mean(per_member, axis=0)

    def _q_from_matrix(self, std: np.ndarray) -> np.ndarray:
        per_member = [sbfa.spe(std) for sbfa, _ in self.members_]
        return np.mean(per_member, axis=0)

    def density_scores(self, X: pd.DataFrame) -> np.ndarray:
        return self._density_from_matrix(self._gate_matrix(X))

    def q_scores(self, X: pd.DataFrame) -> np.ndarray:
        return self._q_from_matrix(self._gate_matrix(X))

    def ooc_severity(self, X: pd.DataFrame) -> np.ndarray:
        """Per-wafer out-of-control severity in [0, 1] for risk-coverage ranking.

        Low density -> high ``anomaly_density``; high SPE -> high ``anomaly_q``.
        The max of the two is the OR-logic severity, independent of the alpha
        operating point.
        """
        std = self._gate_matrix(X)
        density = self._density_from_matrix(std)
        q = self._q_from_matrix(std)
        ref_d = self.ref_density_
        ref_q = self.ref_q_
        anomaly_density = np.array(
            [float(np.mean(ref_d >= d)) for d in density], dtype="float64"
        )
        anomaly_q = np.array([float(np.mean(ref_q <= v)) for v in q], dtype="float64")
        return np.maximum(anomaly_density, anomaly_q)

    def flag_masks(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        density_ooc = np.asarray(self.density_scores(X) < self.density_lcl_, dtype=bool)
        q_ooc = np.asarray(self.q_scores(X) > self.q_ucl_, dtype=bool)
        return {
            "density_ooc": density_ooc,
            "q_ooc": q_ooc,
            "both_ooc": density_ooc & q_ooc,
            "either_ooc": density_ooc | q_ooc,
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
            "n_flagged_density": int(masks["density_ooc"].sum()),
            "n_flagged_q": int(masks["q_ooc"].sum()),
            "n_flagged_both": int(masks["both_ooc"].sum()),
            "n_flagged_either": int(masks["either_ooc"].sum()),
            "n_flagged_ooc": int(ooc.sum()),
            "n_in_control": int((~ooc).sum()),
        }

    def config(self) -> dict:
        return {
            "logic": self.logic,
            "feature_stage": "post_cluster_sbfa",
            "method": "sbfa_bgm_q",
            "n_factors": getattr(self, "n_factors_", self.n_factors),
            "n_mixture_components": getattr(
                self, "n_mixture_components_", self.n_mixture_components
            ),
            "loading_scale": self.loading_scale,
            "density_alpha": self.density_alpha,
            "q_alpha": self.q_alpha,
            "gate_corr_threshold": self.gate_corr_threshold,
            "clip": self.clip,
            "n_seeds": self.n_seeds,
            "density_lcl": getattr(self, "density_lcl_", None),
            "q_ucl": getattr(self, "q_ucl_", None),
            "n_features": getattr(self, "n_features_", None),
            "n_reference_wafers": getattr(self, "n_reference_", None),
        }
