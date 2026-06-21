"""Run the tuned-pipeline benchmark.

  python -m secom.cli.benchmark                     # all models, full rewrite
  python -m secom.cli.benchmark --model hsic_enet   # one pipeline (merges JSON)
  python -m secom.cli.benchmark --clear-pipeline-cache  # drop joblib preprocess cache first
"""
from __future__ import annotations

from secom.benchmark import main


if __name__ == "__main__":
    main()
