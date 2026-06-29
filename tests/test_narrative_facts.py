"""Tests for gate-aware grounded narrative facts and the Tier-0 glossary.

Covers two invariants:
  * ``reference_definitions_for_facts`` returns only real glossary snippets and
    can reach every definition the facts can emit (no orphan keys).
  * ``build_wafer_facts`` stays JSON-serializable once the optional process-gate
    and model-context blocks are appended. The non-Bayesian interpolation
    champion (``hsic_rf``) needs no posterior/artifact fitting, so this runs in a
    clean checkout; it skips if local data is genuinely unavailable.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from secom.dashboard.glossary import (
    STAT_DEFINITIONS,
    reference_definitions_for_facts,
)

_OUTCOMES = ("caught_fail", "missed_fail", "false_alarm", "correct_pass")


def test_definitions_are_nonempty_strings() -> None:
    assert STAT_DEFINITIONS
    for key, text in STAT_DEFINITIONS.items():
        assert isinstance(text, str) and text.strip(), key


def test_reference_definitions_only_emit_known_snippets() -> None:
    facts = {
        "fail_probability": 0.8,
        "threshold": 0.3,
        "outcome": "caught_fail",
        "top_contributors": [{"feature": "c_1", "spc_z": 3.2, "drifting": True}],
    }
    defs = reference_definitions_for_facts(facts)
    assert defs
    assert set(defs).issubset(set(STAT_DEFINITIONS.values()))


def test_reference_definitions_reach_every_snippet() -> None:
    """Across all outcomes + the rich blocks, every glossary snippet is reachable."""
    base = {
        "fail_probability": 0.8,
        "threshold": 0.3,
        "fail_probability_credible_interval": [0.6, 0.95],
        "n_drifting_contributors": 1,
        "top_contributors": [
            {"feature": "c_1", "spc_z": 3.2, "drifting": True, "attribution_robust": True}
        ],
        "process_gates": {
            "pca": {"out_of_control": True, "tripped": ["Hotelling T2 above limit"]},
            "bayes": {"out_of_control": True, "tripped": ["Q/SPE above limit"]},
        },
        "model_context": {"front_end": "HSIC", "n_selected": 50},
    }
    seen: set[str] = set()
    for outcome in _OUTCOMES:
        seen.update(reference_definitions_for_facts({**base, "outcome": outcome}))
    assert seen == set(STAT_DEFINITIONS.values())


def test_empty_facts_yield_no_definitions() -> None:
    assert reference_definitions_for_facts({}) == []


def test_build_wafer_facts_is_json_serializable() -> None:
    from secom.dashboard.explainability import WaferExplanation
    from secom.dashboard.narrator import build_wafer_facts

    local_df = pd.DataFrame(
        {
            "feature": ["c_1", "c_2"],
            "contribution": [0.5, -0.2],
            "coefficient": [1.0, -1.0],
            "spc_z": [3.4, 0.1],
            "drift_flag": [True, False],
        }
    )
    result = WaferExplanation(
        observation_id=-999_999,  # not in any holdout -> gate facts stay empty
        actual_label=1,
        predicted_label=1,
        fail_probability=0.82,
        threshold=0.3,
        local_df=local_df,
        method="unit-test",
        fail_probability_interval=(0.6, 0.95),
    )
    try:
        facts = build_wafer_facts("hsic_rf", result, "interpolation")
    except FileNotFoundError:
        pytest.skip("local artifacts unavailable")

    text = json.dumps(facts)  # must not raise
    assert facts["outcome"] == "caught_fail"
    assert facts["model_id"] == "hsic_rf"
    # The optional blocks are empty-safe: present only when data backs them.
    assert "process_gates" not in facts or isinstance(facts["process_gates"], dict)
    assert "model_context" not in facts or isinstance(facts["model_context"], dict)
    assert "top_contributors" in json.loads(text)
