"""Bayesian elastic-net logistic head.

- ``model``: ``BayesianElasticNetLogistic`` (NumPyro) - a scikit-learn estimator
  with an explicit elastic-net (Laplace L1 + ridge L2) slope prior, a static
  intercept, and positive-class / sample weighting. It drops into the shared
  ``Pipeline`` + ``CalibratedClassifierCV`` exactly where ``LogisticRegression``
  would, so it tunes and benchmarks through the standard sklearn machinery.
"""
