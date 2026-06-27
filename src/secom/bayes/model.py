"""Bayesian elastic-net logistic head (NumPyro), as a scikit-learn estimator.

Slope prior is an explicit elastic net parameterised in sklearn terms via ``C``
(inverse total penalty) and ``l1_ratio`` (L1 vs L2 mix). These map onto a Laplace
(L1) base with a ridge (L2) potential::

    laplace_scale = C / l1_ratio          # beta_j ~ Laplace(0, laplace_scale)
    ridge_w       = (1 - l1_ratio) / (2C) # factor(-ridge_w * sum beta^2)

so the negative log-prior matches sklearn's
``(1/C) * [l1_ratio * ||beta||_1 + ((1 - l1_ratio) / 2) * ||beta||_2^2]``.

The estimator implements the standard ``fit(X, y, sample_weight=None)`` /
``predict_proba`` / ``predict`` contract so it drops into a scikit-learn
``Pipeline`` and ``CalibratedClassifierCV`` exactly where ``LogisticRegression``
would. Probability calibration is therefore handled by the shared wrapper, not
here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class BayesianElasticNetLogistic(ClassifierMixin, BaseEstimator):
    """NumPyro elastic-net logistic regression with a static intercept."""

    def __init__(
        self,
        C: float = 0.1,
        l1_ratio: float = 0.5,
        pos_weight: float = 1.0,
        inference: str = "advi",
        draws: int = 500,
        tune: int = 500,
        svi_steps: int = 1500,
        seed: int = 42,
    ):
        # sklearn convention: store constructor args verbatim (no validation /
        # transformation here) so get_params/set_params/clone round-trip cleanly.
        self.C = C
        self.l1_ratio = l1_ratio
        self.pos_weight = pos_weight
        self.inference = inference
        self.draws = draws
        self.tune = tune
        self.svi_steps = svi_steps
        self.seed = seed

    # --- prior knobs (validated/derived at fit/sample time) ------------------
    def _prior_scales(self) -> tuple[float, float, float]:
        C = float(self.C)
        if C <= 0:
            raise ValueError(f"C must be positive, got {self.C!r}")
        # l1_ratio in (0, 1]; floor avoids an infinite Laplace scale at pure ridge.
        l1_ratio = float(min(max(float(self.l1_ratio), 1e-6), 1.0))
        laplace_scale = C / l1_ratio
        ridge_w = (1.0 - l1_ratio) / (2.0 * C)
        return laplace_scale, ridge_w, l1_ratio

    # --- NumPyro model -------------------------------------------------------
    def _model(self, design, y=None, weights=None):
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist

        _, p = design.shape
        laplace_scale, ridge_w, _ = self._prior_scales()
        beta = numpyro.sample("beta", dist.Laplace(jnp.zeros(p), laplace_scale))
        numpyro.factor("ridge", -ridge_w * jnp.sum(beta**2))

        alpha = numpyro.sample("alpha", dist.Normal(0.0, 5.0))
        logits = alpha + design @ beta
        if y is None:
            numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=None)
        else:
            # Weighted likelihood: scale each observation's log-prob by its
            # (renormalized) weight so the minority class / recent wafers pull
            # harder without changing the effective sample size.
            log_lik = dist.Bernoulli(logits=logits).log_prob(y)
            if weights is not None:
                log_lik = weights * log_lik
            numpyro.factor("obs", jnp.sum(log_lik))

    # --- Fit -----------------------------------------------------------------
    def fit(self, X, y, sample_weight=None) -> "BayesianElasticNetLogistic":
        import jax
        import jax.numpy as jnp

        pos_weight = float(self.pos_weight)
        if pos_weight <= 0:
            raise ValueError(f"pos_weight must be positive, got {self.pos_weight!r}")

        design = np.asarray(X, dtype=float)
        y_np = np.asarray(y, dtype=float)
        self.classes_ = np.unique(y_np)
        self.n_features_in_ = int(design.shape[1])
        self.n_features_ = self.n_features_in_
        self.feature_names_ = (
            list(X.columns) if isinstance(X, pd.DataFrame)
            else [f"x{i}" for i in range(self.n_features_in_)]
        )

        # Combine an optional external sample_weight (e.g. time-decay) with the
        # positive-class up-weighting, then renormalize to mean weight == 1.
        base = (
            np.ones_like(y_np)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        w = base * np.where(y_np == 1.0, pos_weight, 1.0)
        if w.sum() > 0:
            w = w * (len(w) / w.sum())

        Xj = jnp.asarray(design)
        yj = jnp.asarray(y_np)
        wj = jnp.asarray(w)

        rng = jax.random.PRNGKey(int(self.seed))
        if str(self.inference).lower() == "advi":
            self.posterior_ = self._fit_advi(rng, Xj, yj, wj)
        else:
            self.posterior_ = self._fit_nuts(rng, Xj, yj, wj)
        return self

    def _fit_nuts(self, rng, X, y, weights) -> dict:
        from numpyro.infer import MCMC, NUTS

        kernel = NUTS(self._model)
        mcmc = MCMC(
            kernel,
            num_warmup=int(self.tune),
            num_samples=int(self.draws),
            num_chains=1,
            progress_bar=False,
        )
        mcmc.run(rng, design=X, y=y, weights=weights)
        return {k: np.asarray(v) for k, v in mcmc.get_samples().items()}

    def _fit_advi(self, rng, X, y, weights) -> dict:
        import jax
        import numpyro.optim as optim
        from numpyro.infer import SVI, Trace_ELBO
        from numpyro.infer.autoguide import AutoNormal

        guide = AutoNormal(self._model)
        svi = SVI(self._model, guide, optim.Adam(0.01), Trace_ELBO())
        result = svi.run(
            rng, int(self.svi_steps), design=X, y=y, weights=weights,
            progress_bar=False,
        )
        rng_s = jax.random.PRNGKey(int(self.seed) + 1)
        samples = guide.sample_posterior(
            rng_s, result.params, sample_shape=(max(200, int(self.draws)),)
        )
        return {k: np.asarray(v) for k, v in samples.items()}

    # --- Predict -------------------------------------------------------------
    def _logits_samples(self, X) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        beta = self.posterior_["beta"]  # (S, p)
        intercept = self.posterior_["alpha"][:, None]
        return intercept + beta @ design.T  # (S, n)

    def predict_proba(self, X) -> np.ndarray:
        logits = np.clip(self._logits_samples(X), -30.0, 30.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        mean = probs.mean(axis=0)
        return np.column_stack([1.0 - mean, mean])

    def predict(self, X) -> np.ndarray:
        pos = self.predict_proba(X)[:, 1]
        idx = (pos >= 0.5).astype(int)
        return self.classes_[idx]

    def coef_summary(self) -> pd.DataFrame:
        beta = self.posterior_["beta"]
        return pd.DataFrame(
            {
                "feature": self.feature_names_,
                "mean": beta.mean(axis=0),
                "sd": beta.std(axis=0),
                "hdi_low": np.quantile(beta, 0.025, axis=0),
                "hdi_high": np.quantile(beta, 0.975, axis=0),
            }
        )
