"""Deprecated module.

The KS drift-drop screen and the later per-sensor baseline normalizer were both
removed. Drift is now handled by time-decay weighting on the LR/RF cells and by
the standalone ``secom.gates.BayesGate`` (sBFA -> BGM + Q) risk-coverage tool.
"""
