"""Process gates.

Interpolation: Regularized-EFA -> Hotelling T2 + Q (``secom.gates.efa``).
Extrapolation: sparse Bayesian factor analysis -> BGM density + Q
(``secom.bayes.gate.ExtrapProcessGate``).
"""
from secom.gates.efa import (
    EFAMonitorFeatures,
    InterpProcessGate,
    RegularizedEFA,
)

__all__ = [
    "RegularizedEFA",
    "EFAMonitorFeatures",
    "InterpProcessGate",
]
