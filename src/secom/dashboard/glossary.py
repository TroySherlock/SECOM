"""Tier-0 method grounding: plain-English definitions for wafer-RCA statistics.

Deterministic keyed retrieval (no embeddings). Given a facts dict built by
``secom.dashboard.narrator.build_wafer_facts`` we return only the definition
snippets for the statistics that wafer actually cites, so the constrained LLM
can phrase them in plain English without inventing meaning. Wording is kept
consistent with ``PROJECT_DOCUMENTATION.md`` Section 8 (glossary).
"""
from __future__ import annotations

from typing import Any

GLOSSARY_VERSION = "v3"

# Concise, plain-English definitions keyed by the fact concept they explain.
STAT_DEFINITIONS: dict[str, str] = {
    "fail_probability": (
        "P(fail): the model's calibrated probability that the wafer fails; it is "
        "compared against the deploy threshold to decide Fail vs Pass."
    ),
    "threshold": (
        "Deploy threshold: the probability cut-off (tuned to minimise the "
        "balanced error rate); P(fail) at or above it predicts Fail."
    ),
    "fail_probability_credible_interval": (
        "Credible interval: the calibrated 95% credible interval for P(fail), on "
        "the same scale as the point estimate; a wide band means the model is "
        "uncertain about this wafer's risk."
    ),
    "uncertainty": (
        "Uncertainty label: a low/moderate/high summary of the credible interval's "
        "width - how confident the model is in this wafer's risk estimate."
    ),
    "borderline": (
        "Borderline call: P(fail) sits within roughly half-to-twice the deploy "
        "threshold, so the Fail/Pass decision is marginal rather than decisive."
    ),
    "spc_z": (
        "SPC z-score: a robust (median / IQR) standard-score of the sensor "
        "versus the in-control passing baseline; |z| < 2 reads as within the "
        "in-control baseline, |z| >= 2 as elevated/unusual, and |z| > 3 is roughly "
        "the top 0.1% tail."
    ),
    "drifting": (
        "Drifting contributor: a sensor whose level has shifted past the "
        "era-drift threshold from the training era to the holdout era (a temporal "
        "change, not a fixed offset)."
    ),
    "drift_shift": (
        "Drift shift: how far a sensor's mean moved between the training and "
        "holdout eras, in SD units (signed); larger magnitudes mean more era drift."
    ),
    "driver_direction": (
        "Driver direction balance: how many of the top contributors push toward "
        "Fail (raise risk) versus Pass (lower risk) for this wafer."
    ),
    "attribution_robust": (
        "Robust attribution: one whose 95% posterior HDI excludes zero, i.e. the "
        "model is confident the effect is genuinely non-zero rather than noise."
    ),
    "outcome_caught_fail": (
        "Caught fail: a failing wafer the model correctly predicted Fail (true "
        "positive)."
    ),
    "outcome_missed_fail": (
        "Missed fail (escape): a failing wafer the model predicted Pass (false "
        "negative) - the expensive error."
    ),
    "outcome_false_alarm": (
        "False alarm (overkill): a passing wafer the model predicted Fail (false "
        "positive) - a cheap re-test."
    ),
    "outcome_correct_pass": (
        "Correct pass: a passing wafer the model correctly predicted Pass (true "
        "negative)."
    ),
    "process_gate": (
        "Process gate: a standalone MSPC monitor whose control limits and latent "
        "model are fit on passing wafers after shared training-set preprocessing; "
        "it flags (abstains on) wafers that look out-of-control, independently of "
        "the yield model."
    ),
    "hotelling_t2": (
        "Hotelling T2: the Mahalanobis distance of a wafer's PCA component scores "
        "from the in-control mean; a high value flags an excursion along the "
        "learned component directions (the fab-standard PCA-MSPC gate)."
    ),
    "q_spe": (
        "Q (SPE): the squared reconstruction residual the factor model cannot "
        "explain; a high value flags a structural break orthogonal to the known "
        "factors."
    ),
    "bgm_density": (
        "BGM log-density: the wafer's log-likelihood under a Bayesian Gaussian "
        "mixture fit on the factor scores; a low value flags an out-of-control "
        "wafer."
    ),
}


def reference_definitions_for_facts(facts: dict[str, Any]) -> list[str]:
    """Definition snippets for exactly the statistics present in a facts dict.

    Inspects the top-level keys plus the per-contributor flags
    (``spc_z`` / ``drifting`` / ``attribution_robust``) and the ``process_gates``
    block, returning matching snippets in a stable order with no duplicates.
    """
    keys: list[str] = []
    if "fail_probability" in facts:
        keys.append("fail_probability")
    if "threshold" in facts:
        keys.append("threshold")
    if "fail_probability_credible_interval" in facts:
        keys.append("fail_probability_credible_interval")
    if "uncertainty" in facts:
        keys.append("uncertainty")
    if "borderline" in facts:
        keys.append("borderline")
    if "n_raising_contributors" in facts or "n_lowering_contributors" in facts:
        keys.append("driver_direction")

    outcome = facts.get("outcome")
    if outcome:
        keys.append(f"outcome_{outcome}")

    contributors = facts.get("top_contributors") or []
    if any("spc_z" in c for c in contributors):
        keys.append("spc_z")
    if facts.get("n_drifting_contributors") or any(
        c.get("drifting") for c in contributors
    ):
        keys.append("drifting")
    if any("drift_shift" in c for c in contributors):
        keys.append("drift_shift")
    if any("attribution_robust" in c for c in contributors):
        keys.append("attribution_robust")

    gates = facts.get("process_gates") or {}
    if gates:
        keys.append("process_gate")
        if "pca" in gates:
            keys += ["hotelling_t2", "q_spe"]
        if "bayes" in gates:
            keys += ["bgm_density", "q_spe"]

    seen: set[str] = set()
    out: list[str] = []
    for key in keys:
        if key in STAT_DEFINITIONS and key not in seen:
            seen.add(key)
            out.append(STAT_DEFINITIONS[key])
    return out
