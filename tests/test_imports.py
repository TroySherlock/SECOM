"""Smoke test: every public module imports cleanly (catches cycles / dead refs)."""
from __future__ import annotations

import importlib

import pytest

MODULES = [
    "secom",
    "secom.paths",
    "secom.utils",
    "secom.metrics",
    "secom.costs",
    "secom.pipelines",
    "secom.hub_interactions",
    "secom.artifacts",
    "secom.reporting",
    "secom.benchmark",
    "secom.bayes.model",
    "secom.gates",
    "secom.gates.pca",
    "secom.gates.bayes_gate",
    "secom.tuning",
    "secom.tuning.registry",
    "secom.cli.build_seed",
    "secom.cli.build_narratives",
    "secom.cli.benchmark",
    "secom.cli.run_tuning",
    "secom.dashboard",
    "secom.dashboard.app",
    "secom.dashboard.data",
    "secom.dashboard.stg",
    "secom.dashboard.pr_curves",
    "secom.dashboard.model_views",
    "secom.dashboard.narrator",
    "secom.dashboard.glossary",
    "secom.dashboard.explainability",
    "secom.dashboard.charts",
    "secom.dashboard.charts.eda",
    "secom.dashboard.charts.pipeline",
    "secom.dashboard.charts.benchmark",
    "secom.dashboard.charts.gates",
    "secom.dashboard.charts.explain",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module: str) -> None:
    importlib.import_module(module)


def test_charts_reexports_are_present() -> None:
    """Every name in charts.__all__ is actually exported by the package."""
    from secom.dashboard import charts

    missing = [name for name in charts.__all__ if not hasattr(charts, name)]
    assert not missing, f"charts package missing exports: {missing}"
