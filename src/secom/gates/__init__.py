"""Standalone risk-coverage gates (not pipeline steps).

- ``EFAGate``: Regularized EFA -> Hotelling T2 + Q (``secom.gates.efa``).
- ``BayesGate``: sparse Bayesian factor analysis -> BGM density + Q
  (``secom.gates.bayes_gate``).

Both score the raw post-cluster sensor space and are fit on passing wafers.
"""
from secom.gates.bayes_gate import BayesGate, SparseBayesianFactorAnalysis
from secom.gates.efa import EFAGate, RegularizedEFA

__all__ = [
    "EFAGate",
    "RegularizedEFA",
    "BayesGate",
    "SparseBayesianFactorAnalysis",
]
