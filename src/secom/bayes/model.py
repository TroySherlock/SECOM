"""Bayesian elastic-net logistic head (NumPyro) with optional random-walk intercept.

Slope prior is an explicit elastic net parameterised in sklearn terms via ``C``
(inverse total penalty) and ``l1_ratio`` (L1 vs L2 mix). These map onto a Laplace
(L1) base with a ridge (L2) potential::

    laplace_scale = C / l1_ratio          # beta_j ~ Laplace(0, laplace_scale)
    ridge_w       = (1 - l1_ratio) / (2C) # factor(-ridge_w * sum beta^2)

so the negative log-prior matches sklearn's
``(1/C) * [l1_ratio * ||beta||_1 + ((1 - l1_ratio) / 2) * ||beta||_2^2]``.
Drift is modelled structurally by a non-centred random-walk intercept over time
blocks; out-of-sample prediction uses the last block's state (honest forecast).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_block_index(timestamps, n_blocks: int) -> np.ndarray:
    """Assign rows to ``n_blocks`` contiguous time blocks by timestamp quantile."""
    ts = pd.to_datetime(np.asarray(timestamps), errors="coerce").astype("int64").to_numpy(dtype=float)
    n = len(ts)
    if n == 0:
        return np.zeros(0, dtype=int)
    order = np.argsort(ts, kind="mergesort")
    block = np.zeros(n, dtype=int)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    for b in range(n_blocks):
        block[order[edges[b]:edges[b + 1]]] = b
    return block


class BayesianElasticNetLogistic:
    """NumPyro elastic-net logistic; static or random-walk intercept."""

    def __init__(
        self,
        *,
        rw_intercept: bool = False,
        n_blocks: int = 5,
        C: float = 0.1,
        l1_ratio: float = 0.5,
        pos_weight: float = 1.0,
        inference: str = "nuts",
        draws: int = 500,
        tune: int = 500,
        svi_steps: int = 1500,
        seed: int = 42,
    ):
        self.rw_intercept = bool(rw_intercept)
        self.n_blocks = int(n_blocks)
        self.C = float(C)
        if self.C <= 0:
            raise ValueError(f"C must be positive, got {C!r}")
        # l1_ratio in (0, 1]; floor avoids an infinite Laplace scale at pure ridge.
        self.l1_ratio = float(min(max(l1_ratio, 1e-6), 1.0))
        self.pos_weight = float(pos_weight)
        if self.pos_weight <= 0:
            raise ValueError(f"pos_weight must be positive, got {pos_weight!r}")
        self.inference = str(inference).lower()
        self.draws = int(draws)
        self.tune = int(tune)
        self.svi_steps = int(svi_steps)
        self.seed = int(seed)
        self.calibrator_ = None

    def set_calibrator(self, calibrator) -> "BayesianElasticNetLogistic":
        """Attach a fitted probability calibrator applied inside ``predict_proba``."""
        self.calibrator_ = calibrator
        return self

    # --- NumPyro model -------------------------------------------------------
    def _model(self, design, y=None, block_idx=None, n_blocks=1, weights=None):
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist

        n, p = design.shape
        laplace_scale = self.C / self.l1_ratio
        ridge_w = (1.0 - self.l1_ratio) / (2.0 * self.C)
        beta = numpyro.sample("beta", dist.Laplace(jnp.zeros(p), laplace_scale))
        numpyro.factor("ridge", -ridge_w * jnp.sum(beta**2))

        if self.rw_intercept and n_blocks > 1:
            alpha0 = numpyro.sample("alpha0", dist.Normal(0.0, 5.0))
            sigma_rw = numpyro.sample("sigma_rw", dist.HalfNormal(1.0))
            steps = numpyro.sample(
                "alpha_steps", dist.Normal(jnp.zeros(n_blocks - 1), 1.0)
            )
            walk = jnp.concatenate([jnp.zeros(1), jnp.cumsum(steps) * sigma_rw])
            alpha_blocks = alpha0 + walk
            intercept = alpha_blocks[block_idx]
        else:
            alpha = numpyro.sample("alpha", dist.Normal(0.0, 5.0))
            intercept = alpha

        logits = intercept + design @ beta
        if y is None:
            numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=None)
        else:
            # Class-weighted likelihood: scale each observation's log-prob by its
            # (renormalized) weight so the minority class pulls harder without
            # changing the effective sample size.
            log_lik = dist.Bernoulli(logits=logits).log_prob(y)
            if weights is not None:
                log_lik = weights * log_lik
            numpyro.factor("obs", jnp.sum(log_lik))

    # --- Fit -----------------------------------------------------------------
    def fit(self, design, y, block_idx=None) -> "BayesianElasticNetLogistic":
        import jax
        import jax.numpy as jnp

        X = jnp.asarray(np.asarray(design, dtype=float))
        y_arr = jnp.asarray(np.asarray(y, dtype=float))
        self.n_features_ = int(X.shape[1])
        self.feature_names_ = (
            list(design.columns) if isinstance(design, pd.DataFrame)
            else [f"x{i}" for i in range(self.n_features_)]
        )

        rw = self.rw_intercept and block_idx is not None
        nb = int(self.n_blocks) if rw else 1
        if rw:
            bidx = jnp.asarray(np.asarray(block_idx, dtype=int))
        else:
            bidx = None
        self._rw_active_ = rw
        self._n_blocks_used_ = nb

        # Per-observation class weights, renormalized to preserve the effective
        # sample size (mean weight == 1) so the prior-likelihood balance is stable
        # across pos_weight values.
        y_np = np.asarray(y, dtype=float)
        w = np.where(y_np == 1.0, self.pos_weight, 1.0)
        if w.sum() > 0:
            w = w * (len(w) / w.sum())
        weights = jnp.asarray(w)

        rng = jax.random.PRNGKey(self.seed)
        if self.inference == "advi":
            self.posterior_ = self._fit_advi(rng, X, y_arr, bidx, nb, weights)
        else:
            self.posterior_ = self._fit_nuts(rng, X, y_arr, bidx, nb, weights)
        return self

    def _fit_nuts(self, rng, X, y, bidx, nb, weights) -> dict:
        from numpyro.infer import MCMC, NUTS

        kernel = NUTS(self._model)
        mcmc = MCMC(
            kernel,
            num_warmup=self.tune,
            num_samples=self.draws,
            num_chains=1,
            progress_bar=False,
        )
        mcmc.run(rng, design=X, y=y, block_idx=bidx, n_blocks=nb, weights=weights)
        return {k: np.asarray(v) for k, v in mcmc.get_samples().items()}

    def _fit_advi(self, rng, X, y, bidx, nb, weights) -> dict:
        import jax
        from numpyro.infer import SVI, Trace_ELBO
        from numpyro.infer.autoguide import AutoNormal
        import numpyro.optim as optim

        guide = AutoNormal(self._model)
        svi = SVI(self._model, guide, optim.Adam(0.01), Trace_ELBO())
        result = svi.run(
            rng, self.svi_steps, design=X, y=y, block_idx=bidx, n_blocks=nb,
            weights=weights, progress_bar=False,
        )
        rng_s = jax.random.PRNGKey(self.seed + 1)
        samples = guide.sample_posterior(
            rng_s, result.params, sample_shape=(max(200, self.draws),)
        )
        return {k: np.asarray(v) for k, v in samples.items()}

    # --- Predict -------------------------------------------------------------
    def _logits_samples(self, design) -> np.ndarray:
        X = np.asarray(design, dtype=float)
        beta = self.posterior_["beta"]  # (S, p)
        if self._rw_active_:
            alpha0 = self.posterior_["alpha0"][:, None]
            sigma = self.posterior_["sigma_rw"][:, None]
            steps = self.posterior_["alpha_steps"]  # (S, nb-1)
            walk = np.cumsum(steps, axis=1) * sigma
            alpha_last = alpha0[:, 0] + walk[:, -1]  # forecast = last block state
            intercept = alpha_last[:, None]
        else:
            intercept = self.posterior_["alpha"][:, None]
        return intercept + beta @ X.T  # (S, n)

    def predict_proba(self, design) -> np.ndarray:
        logits = np.clip(self._logits_samples(design), -30.0, 30.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        mean = probs.mean(axis=0)
        if self.calibrator_ is not None:
            mean = np.asarray(self.calibrator_.transform(mean), dtype=float)
        return np.column_stack([1.0 - mean, mean])

    def predict_proba_pos(self, design) -> np.ndarray:
        return self.predict_proba(design)[:, 1]

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
