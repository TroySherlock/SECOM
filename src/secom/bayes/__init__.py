"""Bayesian extrapolation toolkit: nonlinear screening + elastic-net logistic head.

- ``representation``: HSIC-Lasso / random-forest screening + physical interaction
  frame, or sPLS aggregation, producing a stable named design matrix.
- ``model``: ``BayesianElasticNetLogistic`` (NumPyro) with static or random-walk
  intercept, an explicit elastic-net (Laplace L1 + ridge L2) slope prior, tunable
  positive-class weighting, and an optional isotonic calibrator.
- ``calibration``: ``IsotonicCalibrator`` for OOF probability calibration.
- ``harness``: blocked-CV grid search, threshold tuning, holdout + gate-conditional
  evaluation that emit the same JSON shapes the benchmark/dashboard consume.
"""
