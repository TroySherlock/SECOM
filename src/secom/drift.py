"""Deprecated module.

The KS drift-drop screen and the later per-sensor baseline normalizer were both
removed. Extrapolation drift is now handled structurally by the Bayesian
random-walk-intercept yield models and the sBFA -> BGM + Q extrapolation gate
(see ``secom.bayes.gate.ExtrapProcessGate``).
"""
