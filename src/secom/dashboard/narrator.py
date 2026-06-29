"""Plain-English wafer summaries for hsic_bayes (facts + frozen Gemma narratives).

Narratives explain the HSIC -> Bayesian elastic-net head on the temporal holdout.

Batch generation (requires local llama-server):
  python -m secom.cli.build_narratives

Environment variables for batch script:
  OPENAI_BASE_URL  — default http://127.0.0.1:8080/v1
  OPENAI_API_KEY   — any non-empty string (llama.cpp ignores it)
  GEMMA_MODEL      — model id for /chat/completions (default: gemma)
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from secom.dashboard.data import (
    cached_benchmark_results,
    model_info,
    wafer_gate_facts,
)
from secom.dashboard.explainability import (
    WaferExplanation,
    cached_bayes_hdis,
    cached_pls_sensor_robust_map,
    is_pls_model,
)
from secom.dashboard.glossary import GLOSSARY_VERSION, reference_definitions_for_facts
from secom.pipelines import INTERP_NARRATIVES_PATH, NARRATIVES_PATH

# The narrative model is track-dependent: the random/in-distribution track uses
# the RF-selection tree head, the temporal track uses the PLS Bayesian champion.
NARRATIVE_MODEL_BY_TRACK: dict[str, str] = {
    "interpolation": "hsic_rf",
    "extrapolation": "pls_bayes",
}
NARRATIVE_PATH_BY_TRACK = {
    "interpolation": INTERP_NARRATIVES_PATH,
    "extrapolation": NARRATIVES_PATH,
}
# Back-compat default (temporal champion).
NARRATIVE_MODEL_ID = NARRATIVE_MODEL_BY_TRACK["extrapolation"]
PROMPT_VERSION = "statistical-interpreter-v8-natural-language"
# Bumped whenever the shape of build_wafer_facts changes (provenance for frozen
# narratives). v2 adds the process_gates + model_context blocks; v3 renames the
# EFA gate key to the PCA (fab-standard) gate; v4 reports only the BGM gate and
# drops model_context; v5 reports a calibrated fail_probability_credible_interval
# (same scale as P(fail)) plus uncertainty / borderline / direction-count facts
# and per-contributor drift_shift magnitudes.
FACTS_SCHEMA_VERSION = "v5-calibrated-interval"

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "gemma"
LLM_TIMEOUT_SEC = 45
LLM_MAX_TOKENS = 360

_SYSTEM_PROMPT = (
    "You are a Statistical Interpreter for semiconductor wafer screening models. "
    "Restate only the numeric facts in the user JSON. Do not invent fab processes, "
    "equipment names, or root causes. Do not mention features not listed, and do "
    "not describe the model's feature front-end or how many sensors/components it "
    "uses. There is no real sensor-to-tool/chamber mapping, so never describe "
    "sensors as physical equipment. Treat every attribution as associational, not "
    "causal. "
    "Always write statistics and outcomes in natural language; never echo the raw "
    "JSON field names - say 'correct pass' not 'correct_pass', 'fail probability' "
    "not 'fail_probability', 'drift' not 'drift_shift', and 'missed fail (escape)' "
    "not 'missed_fail'. "
    "When an SPC z-score is given, judge it by the 2-sigma convention: restate "
    "|z| >= 2 as 'elevated/unusual vs the in-control baseline' (|z| > 3 is roughly "
    "the top 0.1% tail) and only |z| < 2 as 'within the in-control baseline'. When "
    "a contributor has a drift_shift, it drifted that many SD between the training "
    "and holdout eras (e.g. 'drifted 2.4 SD between eras'); a drifting flag is "
    "'drifting over time'. "
    "The fail_probability_credible_interval is a calibrated 95% credible interval "
    "on the same scale as fail_probability, so you may relate the two and compare "
    "the interval to the deploy threshold. Use the supplied uncertainty label "
    "(low/moderate/high) for how certain the model is - do not invent your own "
    "'wide'/'narrow' wording. Use the borderline flag to decide tone: when true, "
    "call it a marginal/near-threshold call; when false, call it decisive. "
    "When a process_gates block is present, you may state whether the BGM "
    "log-density gate (an independent sensor-space drift monitor) flagged this "
    "wafer out-of-control and whether that corroborates the model's verdict; refer "
    "to it only as 'the BGM log-density gate'. "
    "Deployment note: the extrapolation (PLS-Bayes) track's verdict uses the "
    "cost-optimal economic threshold. "
    "Use any supplied reference definitions only to phrase the statistics in plain "
    "English; never infer causes from them. Write a concise 4-6 sentence root-cause "
    "analysis following the arc: symptom (the verdict, using the exact outcome - "
    "never call a fail a pass), evidence (the top drivers with SPC z-scores, drift "
    "magnitudes, and posterior confidence, plus any gate corroboration). When most "
    "of the largest contributors push against the verdict (compare the raising vs "
    "lowering contributor counts), say so explicitly - e.g. the fail comes from the "
    "net of many smaller signals while the largest individual drivers lean toward "
    "pass. Then give a hypothesis "
    "(a single sensible first point of contact - prefer a drifting contributor, "
    "otherwise the largest-magnitude robust driver - phrased associationally, e.g. "
    "'a sensible first place to look is ...'), and a next action matched to the "
    "outcome. Action by outcome: for a missed_fail (escape) recommend containing "
    "the lot and routing it to manual review; for a caught_fail call it a correct "
    "catch and recommend holding/dispositioning the wafer as a genuine reject "
    "(never describe it as a pass and never say no action is needed); for a "
    "false_alarm suggest a cheap re-test to confirm; for a correct_pass recommend "
    "no action UNLESS the BGM gate fired, in which case route it to manual review."
)


def narrative_model_for_track(track: str) -> str:
    """Designated narrative model id for a dashboard track."""
    return NARRATIVE_MODEL_BY_TRACK.get(track, NARRATIVE_MODEL_ID)


def narratives_path_for_track(track: str):
    """Frozen narratives artifact path for a dashboard track."""
    return NARRATIVE_PATH_BY_TRACK.get(track, NARRATIVES_PATH)


class LLMNarrativeError(RuntimeError):
    """Raised when local Gemma / llama-server fails to return narrative text."""


def _label_text(label: int) -> str:
    return "Fail" if label == 1 else "Pass"


def _robust_map(model_id: str, track: str) -> dict[str, bool]:
    """Attribution-robustness lookup (HDI excludes 0) for Bayesian heads."""
    if model_info(model_id).explainability != "bayesian":
        return {}
    if is_pls_model(model_id):
        return cached_pls_sensor_robust_map(model_id, track)
    hdi = cached_bayes_hdis(model_id, track)
    if hdi is None or hdi.empty:
        return {}
    return {str(r.feature): bool(r.robust) for r in hdi.itertuples()}


def _contributor_rows(local_df: pd.DataFrame, robust_map: dict[str, bool]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    has_coef = "coefficient" in local_df.columns
    for _, row in local_df.iterrows():
        feat = str(row["feature"])
        contribution = float(row["contribution"])
        entry: dict[str, Any] = {
            "feature": feat,
            "contribution": round(contribution, 4),
        }
        if has_coef and pd.notna(row.get("coefficient")):
            entry["direction"] = "positive" if float(row["coefficient"]) > 0 else "negative"
        else:
            entry["direction"] = (
                "raises_fail_risk" if contribution > 0 else "lowers_fail_risk"
            )
        spc = row.get("spc_z")
        if spc is not None and pd.notna(spc):
            entry["spc_z"] = round(float(spc), 2)
        entry["drifting"] = bool(row.get("drift_flag", False))
        shift = row.get("drift_shift")
        if shift is not None and pd.notna(shift):
            entry["drift_shift"] = round(float(shift), 2)
        if feat in robust_map:
            entry["attribution_robust"] = bool(robust_map[feat])
        rows.append(entry)
    return rows


def build_wafer_facts(
    model_id: str, result: WaferExplanation, track: str = "extrapolation"
) -> dict[str, Any]:
    """Build JSON-serializable RCA facts for the track's narrative model.

    Facts are sensor-space (PLS back-projected where needed) and carry SPC
    z-scores, drift flags, and attribution-robustness (HDI) so the Statistical
    Interpreter can restate richer evidence without inventing causes.
    """
    expected = narrative_model_for_track(track)
    if model_id != expected:
        raise ValueError(
            f"Narration for {track!r} only supported for {expected!r}, got {model_id!r}"
        )

    info = model_info(model_id)
    robust_map = _robust_map(model_id, track)
    contributors = _contributor_rows(result.local_df, robust_map)
    n_drifting = sum(1 for c in contributors if c.get("drifting"))
    n_raising = sum(
        1 for c in contributors if c.get("direction") in ("positive", "raises_fail_risk")
    )
    n_lowering = sum(
        1 for c in contributors if c.get("direction") in ("negative", "lowers_fail_risk")
    )
    if result.actual_label == 1:
        outcome = "caught_fail" if result.predicted_label == 1 else "missed_fail"
    else:
        outcome = "false_alarm" if result.predicted_label == 1 else "correct_pass"

    fail_probability = round(result.fail_probability, 4)
    threshold = round(result.threshold, 4)
    facts: dict[str, Any] = {
        "model_id": model_id,
        "model_name": info.display_name,
        "track": track,
        "feature_space": "sensor (PLS back-projected)" if is_pls_model(model_id) else "sensor",
        "wafer_id": str(result.observation_id),
        "actual": _label_text(result.actual_label),
        "predicted": _label_text(result.predicted_label),
        "outcome": outcome,
        "fail_probability": fail_probability,
        "threshold": threshold,
        # Decision sits within 0.5x-2x the deploy threshold - a marginal call.
        "borderline": bool(0.5 * threshold <= fail_probability <= 2.0 * threshold),
        "n_drifting_contributors": int(n_drifting),
        "n_raising_contributors": int(n_raising),
        "n_lowering_contributors": int(n_lowering),
        "method": result.method,
        "top_contributors": contributors,
        "caveat": "associational, not causal; no real tool/chamber mapping exists",
    }
    if result.fail_probability_interval is not None:
        lo, hi = result.fail_probability_interval
        # Calibrated 95% credible interval, same scale as fail_probability.
        facts["fail_probability_credible_interval"] = [round(lo, 4), round(hi, 4)]
        width = hi - lo
        facts["uncertainty"] = (
            "low" if width < 0.10 else "moderate" if width <= 0.30 else "high"
        )

    # Independent corroboration: did the standalone BGM log-density gate flag this
    # same wafer? Narratives report only the custom BGM gate (not the PCA
    # baseline). Optional and empty-safe so page 6's live path still works.
    gate_facts = wafer_gate_facts(
        cached_benchmark_results(), track, result.observation_id
    )
    if gate_facts.get("bayes"):
        facts["process_gates"] = {"bayes": gate_facts["bayes"]}
    return facts


def llm_config() -> tuple[str, str, str]:
    base_url = os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    api_key = os.environ.get("OPENAI_API_KEY", "local")
    model = os.environ.get("GEMMA_MODEL", DEFAULT_MODEL)
    return base_url, api_key, model


def resolve_llm_model(*, base_url: str | None = None, timeout: float = 5.0) -> str:
    """Return GEMMA_MODEL env value, or the first model id from llama-server /models."""
    _, _, configured = llm_config()
    if configured != DEFAULT_MODEL:
        return configured
    resolved_base, _, _ = llm_config()
    base_url = (base_url or resolved_base).rstrip("/")
    req = urllib.request.Request(f"{base_url}/models", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    models = data.get("data") or data.get("models") or []
    if not models:
        raise LLMNarrativeError(f"No models listed at {base_url}/models")
    first = models[0]
    if isinstance(first, dict):
        return str(first.get("id") or first.get("model") or first.get("name"))
    return str(first)


def check_llm_available(*, base_url: str | None = None, timeout: float = 5.0) -> None:
    """Raise LLMNarrativeError if llama-server is not reachable."""
    resolved_base, _, _ = llm_config()
    base_url = (base_url or resolved_base).rstrip("/")
    req = urllib.request.Request(f"{base_url}/models", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            pass
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        raise LLMNarrativeError(
            f"llama-server not reachable at {base_url} — start services.llama-cpp first"
        ) from exc


def generate_narrative_via_llm(
    facts: dict[str, Any],
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    timeout: float = LLM_TIMEOUT_SEC,
) -> str:
    """Call local OpenAI-compatible llama-server; raise LLMNarrativeError on failure."""
    resolved_base, resolved_key, _ = llm_config()
    base_url = (base_url or resolved_base).rstrip("/")
    model = model or resolve_llm_model(base_url=base_url)
    api_key = api_key or resolved_key

    messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    definitions = reference_definitions_for_facts(facts)
    if definitions:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Reference definitions (use only to phrase the statistics in "
                    "plain English; do NOT infer causes from them): "
                    + " ".join(definitions)
                ),
            }
        )
    messages.append(
        {"role": "user", "content": json.dumps(facts, separators=(",", ":"))}
    )
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": LLM_MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        raise LLMNarrativeError(f"Gemma request failed: {exc}") from exc

    choices = data.get("choices") or []
    if not choices:
        raise LLMNarrativeError("Gemma returned no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not content or not str(content).strip():
        content = message.get("reasoning_content")
    if not content or not str(content).strip():
        raise LLMNarrativeError("Gemma returned empty content")
    return str(content).strip()


def load_narratives(
    path: Path | str = NARRATIVES_PATH, expected_model_id: str | None = None
) -> dict[str, Any]:
    """Load frozen narratives artifact; optionally validate its ``model_id``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing narratives file: {path}")
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    if expected_model_id is not None and payload.get("model_id") != expected_model_id:
        raise ValueError(
            f"Expected model_id={expected_model_id!r} in {path}, "
            f"got {payload.get('model_id')!r}"
        )
    if "narratives" not in payload:
        raise ValueError(f"Missing 'narratives' key in {path}")
    return payload


def get_wafer_narrative(wafer_id: object, narratives_payload: dict[str, Any]) -> str | None:
    """Lookup pre-generated text for a holdout wafer id."""
    narratives = narratives_payload.get("narratives") or {}
    return narratives.get(str(wafer_id))


def build_narratives_artifact(
    narratives: dict[str, str],
    *,
    model_id: str = NARRATIVE_MODEL_ID,
    track: str = "extrapolation",
    llm_base_url: str | None = None,
    llm_model: str | None = None,
) -> dict[str, Any]:
    base_url, _, model = llm_config()
    return {
        "model_id": model_id,
        "track": track,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "llm_base_url": llm_base_url or base_url,
        "llm_model": llm_model or model,
        "prompt_version": PROMPT_VERSION,
        "glossary_version": GLOSSARY_VERSION,
        "facts_schema_version": FACTS_SCHEMA_VERSION,
        "narratives": narratives,
    }


def write_narratives_artifact(
    payload: dict[str, Any],
    path: Path | str = NARRATIVES_PATH,
) -> Path:
    """Write narratives JSON atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    tmp.replace(path)
    return path
