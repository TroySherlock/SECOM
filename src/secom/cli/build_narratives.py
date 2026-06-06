"""Batch-generate frozen Gemma narratives for all linear_lr holdout wafers."""
from __future__ import annotations

import argparse
import sys
import time

from tqdm import tqdm

from secom.dashboard.explainability import cached_wafer_explanation, holdout_wafer_ids
from secom.dashboard.narrator import (
    LINEAR_LR_MODEL_ID,
    LLMNarrativeError,
    build_narratives_artifact,
    build_wafer_facts,
    check_llm_available,
    generate_narrative_via_llm,
    llm_config,
    load_narratives,
    resolve_llm_model,
    write_narratives_artifact,
)
from secom.pipelines import LINEAR_LR_NARRATIVES_PATH

MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 2.0


def _generate_with_retries(facts: dict, *, model: str) -> str:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return generate_narrative_via_llm(facts, model=model)
        except LLMNarrativeError as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * attempt)
    assert last_error is not None
    raise last_error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate frozen Gemma wafer narratives for linear_lr holdout wafers."
    )
    parser.add_argument(
        "--wafer-id",
        type=str,
        default=None,
        help="Regenerate a single holdout wafer (merges into existing artifact if present).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(LINEAR_LR_NARRATIVES_PATH),
        help=f"Output JSON path (default: {LINEAR_LR_NARRATIVES_PATH})",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    base_url, _, _ = llm_config()
    try:
        check_llm_available(base_url=base_url)
        model = resolve_llm_model(base_url=base_url)
    except LLMNarrativeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    wafer_ids = holdout_wafer_ids()
    if args.wafer_id is not None:
        target = args.wafer_id
        if target not in {str(w) for w in wafer_ids} and not any(
            str(w) == target for w in wafer_ids
        ):
            print(f"ERROR: wafer id {target!r} not in holdout split", file=sys.stderr)
            return 1
        wafer_ids = [w for w in wafer_ids if str(w) == target]
        if not wafer_ids:
            wafer_ids = [args.wafer_id]

    existing: dict[str, str] = {}
    output_path = args.output
    try:
        prior = load_narratives(output_path)
        existing = dict(prior.get("narratives") or {})
    except FileNotFoundError:
        pass

    narratives: dict[str, str] = dict(existing) if args.wafer_id else {}

    for wafer_id in tqdm(wafer_ids, desc="Gemma narratives", unit="wafer"):
        result = cached_wafer_explanation(LINEAR_LR_MODEL_ID, wafer_id)
        if result is None:
            print(f"ERROR: wafer {wafer_id!r} not found in holdout split", file=sys.stderr)
            return 1
        facts = build_wafer_facts(LINEAR_LR_MODEL_ID, result)
        key = str(wafer_id)
        try:
            narratives[key] = _generate_with_retries(facts, model=model)
        except LLMNarrativeError as exc:
            print(f"ERROR: wafer {key}: {exc}", file=sys.stderr)
            return 1

    payload = build_narratives_artifact(
        narratives,
        llm_base_url=base_url,
        llm_model=model,
    )
    out = write_narratives_artifact(payload, output_path)
    print(f"Wrote {len(narratives)} narratives to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
