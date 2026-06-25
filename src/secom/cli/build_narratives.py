"""Batch-generate frozen Gemma wafer narratives for both tracks.

The narrative model is track-dependent: the random/in-distribution track uses
``hsic_rf`` and the temporal track uses ``pls_bayes``. Each track is written to
its own artifact (``interp_wafer_narratives.json`` / ``extrap_wafer_narratives.json``).

Requires a local llama-server (see secom.dashboard.narrator for env vars).
"""
from __future__ import annotations

import argparse
import sys
import time

from tqdm import tqdm

from secom.dashboard.explainability import cached_wafer_explanation, holdout_wafer_ids
from secom.dashboard.narrator import (
    NARRATIVE_MODEL_BY_TRACK,
    LLMNarrativeError,
    build_narratives_artifact,
    build_wafer_facts,
    check_llm_available,
    generate_narrative_via_llm,
    llm_config,
    load_narratives,
    narrative_model_for_track,
    narratives_path_for_track,
    resolve_llm_model,
    write_narratives_artifact,
)

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
        description="Generate frozen Gemma wafer narratives per track."
    )
    parser.add_argument(
        "--track",
        choices=sorted(NARRATIVE_MODEL_BY_TRACK),
        default=None,
        help="Only generate for one track (default: both).",
    )
    parser.add_argument(
        "--wafer-id",
        type=str,
        default=None,
        help="Regenerate a single holdout wafer (merges into the existing artifact).",
    )
    return parser.parse_args()


def _run_track(track: str, *, model: str, wafer_id: str | None) -> int:
    narrative_model = narrative_model_for_track(track)
    output_path = str(narratives_path_for_track(track))

    wafer_ids = holdout_wafer_ids(track)
    if wafer_id is not None:
        wafer_ids = [w for w in wafer_ids if str(w) == wafer_id] or [wafer_id]

    existing: dict[str, str] = {}
    try:
        prior = load_narratives(output_path)
        existing = dict(prior.get("narratives") or {})
    except FileNotFoundError:
        pass

    narratives: dict[str, str] = dict(existing) if wafer_id else {}

    for wid in tqdm(wafer_ids, desc=f"Gemma narratives [{track}]", unit="wafer"):
        result = cached_wafer_explanation(narrative_model, wid, track)
        if result is None:
            print(f"ERROR: wafer {wid!r} not found in {track} holdout", file=sys.stderr)
            return 1
        facts = build_wafer_facts(narrative_model, result, track)
        try:
            narratives[str(wid)] = _generate_with_retries(facts, model=model)
        except LLMNarrativeError as exc:
            print(f"ERROR: wafer {wid}: {exc}", file=sys.stderr)
            return 1

    base_url, _, _ = llm_config()
    payload = build_narratives_artifact(
        narratives,
        model_id=narrative_model,
        track=track,
        llm_base_url=base_url,
        llm_model=model,
    )
    out = write_narratives_artifact(payload, output_path)
    print(f"Wrote {len(narratives)} narratives ({track}, {narrative_model}) to {out}")
    return 0


def main() -> int:
    args = _parse_args()
    base_url, _, _ = llm_config()
    try:
        check_llm_available(base_url=base_url)
        model = resolve_llm_model(base_url=base_url)
    except LLMNarrativeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    tracks = [args.track] if args.track else sorted(NARRATIVE_MODEL_BY_TRACK)
    for track in tracks:
        rc = _run_track(track, model=model, wafer_id=args.wafer_id)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
