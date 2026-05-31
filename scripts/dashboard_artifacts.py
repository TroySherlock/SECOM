"""Load pipeline reporting artifacts for the Pipeline dashboard page."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.dashboard_stg import N_SENSORS
from scripts.secom_pipelines import PIPELINE_ARTIFACTS_PATH

REFERENCE_MODELS = {"linear": "linear_lr", "topk": "topk_rf"}


def artifacts_available(path: Path | None = None) -> bool:
    path = path or PIPELINE_ARTIFACTS_PATH
    return path.is_file()


def load_pipeline_artifacts(path: Path | None = None) -> dict[str, Any]:
    path = path or PIPELINE_ARTIFACTS_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Pipeline artifacts not found at {path}. "
            "Run: python -m scripts.benchmark_models"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def get_reference_artifacts(
    artifacts: dict[str, Any],
    family: str,
) -> dict[str, Any] | None:
    """Return model artifact block for linear or topk reference model."""
    ref_ids = artifacts.get("reference_models", REFERENCE_MODELS)
    model_id = ref_ids.get(family) or REFERENCE_MODELS.get(family)
    if not model_id:
        return None
    models = artifacts.get("models", {})
    return models.get(model_id)


def _stage_int(stages: dict[str, Any], key: str, fallback: int = 0) -> int:
    """Read stage count; support legacy raw_sensors key."""
    if key in stages:
        return int(stages[key])
    if key == "mart_sensors" and "raw_sensors" in stages:
        return int(stages["raw_sensors"])
    return fallback


def build_reduction_profile(artifacts: dict[str, Any]) -> dict[str, int]:
    """Build reduction metrics from reference linear model stages."""
    linear_model = get_reference_artifacts(artifacts, "linear") or {}
    stages = linear_model.get("stages", {})

    stg = _stage_int(stages, "stg_sensors", int(N_SENSORS))
    mart = _stage_int(stages, "mart_sensors")
    after_cluster = _stage_int(stages, "after_cluster")
    auxiliary = _stage_int(stages, "auxiliary_features")
    classifier_input = _stage_int(stages, "classifier_input")
    dbt_dropped = _stage_int(stages, "dbt_dropped_sensors", max(0, stg - mart))
    drop_corr = _stage_int(stages, "drop_correlated")

    return {
        "stg_sensors": stg,
        "mart_sensors": mart,
        "dbt_dropped_sensors": dbt_dropped,
        "after_cluster": after_cluster,
        "auxiliary_features": auxiliary,
        "classifier_input": classifier_input,
        "drop_correlated": drop_corr,
    }
