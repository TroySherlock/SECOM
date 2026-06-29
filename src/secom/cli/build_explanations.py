"""Batch-precompute frozen model-explainability artifacts for the dashboard.

The dashboard reads these instead of fitting models live, so the deployed app
needs none of the heavy modelling deps (numpyro / jax / pymc / pyHSICLasso /
shap). The champion model is track-dependent (``hsic_rf`` for the random /
interpolation track, ``pls_bayes`` for the temporal / extrapolation track); the
page-2 PLS score scatter uses ``pls_enet``.

Outputs ``secom_explanations_cache.json`` keyed ``[track][model_id]`` with
``outcomes`` / ``global`` / ``wafers`` for the champion and ``score_scatter`` for
``pls_enet``.

Run order: ``python -m secom.benchmark`` -> ``python -m secom.cli.build_explanations``
-> ``python -m secom.cli.build_narratives``.
"""
from __future__ import annotations

import argparse
import json
import sys

from tqdm import tqdm

from secom.dashboard.explainability import (
    CHAMPION_BY_TRACK,
    _live_bayes_hdis,
    _live_holdout_outcomes,
    _live_pls_global_importance,
    _live_pls_score_scatter,
    _live_pls_sensor_posterior,
    _live_pls_sensor_robust_map,
    _live_wafer_explanation,
    _live_wafer_sensor_posterior,
    explanation_to_blob,
    holdout_wafer_ids,
    is_pls_model,
)
from secom.pipelines import EXPLANATIONS_CACHE_PATH
from secom.utils import json_safe

SCATTER_MODEL = "pls_enet"


def _df_records(df) -> list | None:
    return df.to_dict(orient="records") if df is not None else None


def _build_global(model_id: str, track: str) -> dict:
    glob = {
        "bayes_hdis": _df_records(_live_bayes_hdis(model_id, track)),
        "pls_robust_map": _live_pls_sensor_robust_map(model_id, track) or None,
        "pls_sensor_posterior": _df_records(_live_pls_sensor_posterior(model_id, track)),
        "global_importance": None,
    }
    if is_pls_model(model_id):
        top, signed, caption = _live_pls_global_importance(model_id, track)
        glob["global_importance"] = {
            "top": top.to_dict(orient="records"),
            "signed": signed.to_dict(orient="records") if signed is not None else None,
            "caption": caption,
        }
    return glob


def _build_wafers(model_id: str, track: str) -> dict:
    pls = is_pls_model(model_id)
    wafers: dict[str, dict] = {}
    for wid in tqdm(
        holdout_wafer_ids(track), desc=f"explanations [{track}/{model_id}]", unit="wafer"
    ):
        result = _live_wafer_explanation(model_id, wid, track)
        if result is None:
            continue
        blob: dict = {"explanation": explanation_to_blob(result)}
        if pls:
            blob["wafer_posterior"] = _df_records(
                _live_wafer_sensor_posterior(model_id, wid, track)
            )
        wafers[str(wid)] = blob
    return wafers


def _build_track(track: str) -> dict:
    champion = CHAMPION_BY_TRACK[track]
    t1, t2, y = _live_pls_score_scatter(track, SCATTER_MODEL)
    return {
        champion: {
            "outcomes": _live_holdout_outcomes(champion, track).to_dict(orient="records"),
            "global": _build_global(champion, track),
            "wafers": _build_wafers(champion, track),
        },
        SCATTER_MODEL: {
            "score_scatter": {"t1": t1.tolist(), "t2": t2.tolist(), "y": y.tolist()},
        },
    }


def _write(entries_by_track: dict, *, merge: bool) -> None:
    payload: dict = {}
    if merge and EXPLANATIONS_CACHE_PATH.exists():
        payload = json.loads(EXPLANATIONS_CACHE_PATH.read_text(encoding="utf-8"))
    for track, entries in entries_by_track.items():
        track_block = dict(payload.get(track) or {})
        track_block.update(entries)
        payload[track] = track_block
    EXPLANATIONS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPLANATIONS_CACHE_PATH.write_text(
        json.dumps(json_safe(payload), indent=2), encoding="utf-8"
    )


def _update_single_wafer(tracks: list[str], wafer_id: str) -> int:
    if not EXPLANATIONS_CACHE_PATH.exists():
        print(
            "ERROR: build the full cache first (run without --wafer-id).", file=sys.stderr
        )
        return 1
    payload = json.loads(EXPLANATIONS_CACHE_PATH.read_text(encoding="utf-8"))
    for track in tracks:
        champion = CHAMPION_BY_TRACK[track]
        result = _live_wafer_explanation(champion, wafer_id, track)
        if result is None:
            print(f"ERROR: wafer {wafer_id!r} not in {track} holdout", file=sys.stderr)
            return 1
        blob: dict = {"explanation": explanation_to_blob(result)}
        if is_pls_model(champion):
            blob["wafer_posterior"] = _df_records(
                _live_wafer_sensor_posterior(champion, wafer_id, track)
            )
        wafers = payload.setdefault(track, {}).setdefault(champion, {}).setdefault("wafers", {})
        wafers[str(wafer_id)] = blob
    EXPLANATIONS_CACHE_PATH.write_text(
        json.dumps(json_safe(payload), indent=2), encoding="utf-8"
    )
    print(f"Updated wafer {wafer_id} for {tracks} in {EXPLANATIONS_CACHE_PATH}")
    return 0


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute frozen model-explainability artifacts per track."
    )
    parser.add_argument(
        "--track",
        choices=sorted(CHAMPION_BY_TRACK),
        default=None,
        help="Only build for one track (default: both).",
    )
    parser.add_argument(
        "--wafer-id",
        type=str,
        default=None,
        help="Recompute a single holdout wafer (merges into the existing cache).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    tracks = [args.track] if args.track else list(CHAMPION_BY_TRACK)

    if args.wafer_id is not None:
        return _update_single_wafer(tracks, args.wafer_id)

    entries_by_track = {track: _build_track(track) for track in tracks}
    merge = len(tracks) != len(CHAMPION_BY_TRACK)
    _write(entries_by_track, merge=merge)
    print(f"Wrote explanations cache for {tracks} to {EXPLANATIONS_CACHE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
