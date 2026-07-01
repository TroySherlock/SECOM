"""Standalone risk-coverage gates (not pipeline steps).

- ``PCAGate``: standard PCA-MSPC -> Hotelling T2 + Q (``secom.gates.pca``).
- ``BayesGate``: sparse Bayesian factor analysis -> BGM density + Q
  (``secom.gates.bayes_gate``).

Both score the raw post-cluster sensor space. Shared preprocessing is fit on the
full training set; the latent monitor and control limits are fit on passing
wafers.
"""
from secom.gates.bayes_gate import BayesGate, SparseBayesianFactorAnalysis
from secom.gates.pca import PCAGate, PCAMonitor

__all__ = [
    "PCAGate",
    "PCAMonitor",
    "BayesGate",
    "SparseBayesianFactorAnalysis",
]
