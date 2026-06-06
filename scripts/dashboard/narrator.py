"""Plain-English wafer summaries for linear_lr (facts + frozen Gemma narratives).

Batch generation (requires local llama-server):
  python -m scripts.build_wafer_narratives

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

import numpy as np
import pandas as pd

from scripts.dashboard.data import model_info
from scripts.dashboard.explainability import (
    WaferExplanation,
    fit_holdout_pipeline,
    load_holdout_split,
    scaled_matrix,
)
from scripts.secom_pipelines import ID_COL, LINEAR_LR_NARRATIVES_PATH

LINEAR_LR_MODEL_ID = "linear_lr"
PROMPT_VERSION = "statistical-interpreter-v1"

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "gemma"
LLM_TIMEOUT_SEC = 45
LLM_MAX_TOKENS = 120

_SYSTEM_PROMPT = (
    "You are a Statistical Interpreter for semiconductor wafer screening models. "
    "Restate only the numeric facts in the user JSON. Do not invent fab processes, "
    "equipment names, or root causes. Do not mention features not listed. "
    "Write 3–5 concise sentences in plain English."
)


class LLMNarrativeError(RuntimeError):
    """Raised when local Gemma / llama-server fails to return narrative text."""


def _label_text(label: int) -> str:
    return "Fail" if label == 1 else "Pass"


def _linear_z_scores(
    pipeline,
    observation_id: object,
    feature_names: list[str],
) -> dict[str, float]:
    split = load_holdout_split()
    mask = split.test_df[ID_COL] == observation_id
    X_row = split.X_test.loc[mask]
    X_train_scaled, names = scaled_matrix(pipeline, split.X_train)
    row_scaled, _ = scaled_matrix(pipeline, X_row)
    mean = X_train_scaled.mean(axis=0)
    std = X_train_scaled.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    name_to_idx = {str(n): i for i, n in enumerate(names)}
    out: dict[str, float] = {}
    for feat in feature_names:
        idx = name_to_idx.get(str(feat))
        if idx is None:
            continue
        out[str(feat)] = float((row_scaled[0, idx] - mean[idx]) / std[idx])
    return out


def _local_rows(local_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in local_df.iterrows():
        entry: dict[str, Any] = {
            "feature": str(row["feature"]),
            "contribution": float(row["contribution"]),
        }
        if "coefficient" in row.index and pd.notna(row["coefficient"]):
            entry["coefficient"] = float(row["coefficient"])
        rows.append(entry)
    return rows


def build_wafer_facts(model_id: str, result: WaferExplanation) -> dict[str, Any]:
    """Build JSON-serializable facts for linear_lr Gemma narration."""
    if model_id != LINEAR_LR_MODEL_ID:
        raise ValueError(f"Narration only supported for {LINEAR_LR_MODEL_ID!r}, got {model_id!r}")

    info = model_info(model_id)
    local_rows = _local_rows(result.local_df)

    facts: dict[str, Any] = {
        "model_id": model_id,
        "model_name": info.display_name,
        "wafer_id": str(result.observation_id),
        "actual": _label_text(result.actual_label),
        "predicted": _label_text(result.predicted_label),
        "fail_probability": round(result.fail_probability, 4),
        "threshold": round(result.threshold, 4),
        "method": result.method,
        "local_contributors": local_rows,
    }

    pipeline, _tuned = fit_holdout_pipeline(model_id)
    feature_names = [r["feature"] for r in local_rows]
    z_scores = _linear_z_scores(pipeline, result.observation_id, feature_names)
    for row in local_rows:
        feat = row["feature"]
        if feat in z_scores:
            row["z_score"] = round(z_scores[feat], 3)
        coef = row.get("coefficient")
        if coef is not None:
            row["coefficient_sign"] = "positive" if coef > 0 else "negative"

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
    resolved_base, resolved_key, resolved_model = llm_config()
    base_url = (base_url or resolved_base).rstrip("/")
    model = model or resolve_llm_model(base_url=base_url)
    api_key = api_key or resolved_key

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(facts, separators=(",", ":"))},
        ],
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


def load_narratives(path: Path | str = LINEAR_LR_NARRATIVES_PATH) -> dict[str, Any]:
    """Load frozen narratives artifact; validate model_id."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing narratives file: {path}")
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("model_id") != LINEAR_LR_MODEL_ID:
        raise ValueError(
            f"Expected model_id={LINEAR_LR_MODEL_ID!r} in {path}, "
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
    llm_base_url: str | None = None,
    llm_model: str | None = None,
) -> dict[str, Any]:
    base_url, _, model = llm_config()
    return {
        "model_id": LINEAR_LR_MODEL_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "llm_base_url": llm_base_url or base_url,
        "llm_model": llm_model or model,
        "prompt_version": PROMPT_VERSION,
        "narratives": narratives,
    }


def write_narratives_artifact(
    payload: dict[str, Any],
    path: Path | str = LINEAR_LR_NARRATIVES_PATH,
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
