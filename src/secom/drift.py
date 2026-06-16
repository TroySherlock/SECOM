"""Deprecated module.

The KS drift-drop screen and the later per-sensor baseline normalizer were both
removed. Extrapolation drift is now handled by time-decay sample weighting
(see ``secom.pipelines.time_decay_weights``) and a post-cluster process gate
(see ``secom.gate.ProcessGate``).
"""
