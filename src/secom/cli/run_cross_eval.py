#!/usr/bin/env python3
"""Cross-protocol diagnostic: retune one model under the other track's protocol.

Fills the off-diagonal of the {model family} x {protocol} 2x2 to settle whether
weak extrapolation is a drift effect or a model defect. Standalone and additive:
results land under data/processed/cross_eval/, never the real tuned/benchmark
artifacts.

Examples:
  python -m secom.cli.run_cross_eval --direction bayes_on_interp --model extrap_rf_static
  python -m secom.cli.run_cross_eval --direction sklearn_on_extrap --model intrap_topk_rf
  python -m secom.cli.run_cross_eval --summary
"""
from __future__ import annotations

import argparse

from secom.cross_eval import DIRECTIONS, run_direction, summarize_cross_eval
from secom.tuning.registry import ALL_MODEL_IDS


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direction",
        choices=sorted(DIRECTIONS),
        help="Cross direction: bayes_on_interp or sklearn_on_extrap.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(ALL_MODEL_IDS),
        help="Retune a single model id under the cross protocol.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print the filled 2x2 from existing cross_eval + benchmark results.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.direction or args.model:
        if not (args.direction and args.model):
            raise SystemExit("Both --direction and --model are required to run a retune.")
        run_direction(args.direction, args.model)
    if args.summary or not (args.direction or args.model):
        summarize_cross_eval()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
