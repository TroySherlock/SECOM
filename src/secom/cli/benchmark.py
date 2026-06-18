"""Run the tuned-pipeline benchmark.

  python -m secom.cli.benchmark                      # all models, full rewrite
  python -m secom.cli.benchmark --track extrapolation  # one track (merges JSON)
  python -m secom.cli.benchmark --model extrap_enet    # one pipeline (merges JSON)
"""
from __future__ import annotations

from secom.benchmark import main


if __name__ == "__main__":
    main()
