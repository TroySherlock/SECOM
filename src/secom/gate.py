"""Backward-compatible shim. Gates now live in :mod:`secom.gates`.

The extrapolation gate moved to :mod:`secom.bayes.gate` (``ExtrapProcessGate``);
the legacy T2 + Isolation Forest ``ProcessGate`` was retired.
"""
from secom.gates import (  # noqa: F401
    EFAMonitorFeatures,
    InterpProcessGate,
    RegularizedEFA,
)

__all__ = [
    "RegularizedEFA",
    "EFAMonitorFeatures",
    "InterpProcessGate",
]
