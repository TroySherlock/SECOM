"""Load benchmark results and pipeline artifacts for Streamlit dashboard pages."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from secom.costs import (
    PROFILE_IDS,
    has_multi_profile_thresholds,
    threshold_profile_config,
)
from secom.pipelines import (
    BENCHMARK_MODEL_IDS,
    BENCHMARK_RESULTS_PATH,
    MODEL_CELLS,
    MODEL_IDS,
    N_SENSORS,
    PIPELINE_ARTIFACTS_PATH,
    REFERENCE_MODELS,
    REPORT_CACHE_PATH,
)
from secom.utils import load_tuned_params

TRACKS: tuple[str, str] = ("extrapolation", "interpolation")


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    display_name: str
    family: str
    classifier: str
    feature_path: str
    description: str
    tuning_notebook: str
    explainability: Literal["linear", "tree", "bayesian"]
    track: str = "both"


# Each of the 9 cells runs on BOTH protocols; the dashboard's track selector
# picks which protocol's tuned artifacts/holdout to view.
_FRONT_END_LABEL = {
    "hsic": "HSIC-Lasso select K → T² → hub pairs",
    "rfsel": "RF-select K → T² → hub pairs",
    "pls": "sPLS components",
}
_FRONT_END_NAME = {"hsic": "HSIC", "rfsel": "RF-select", "pls": "sPLS"}
_CLASSIFIER_LABEL = {
    "enet": "Elastic-net logistic (saga)",
    "rf": "Random forest",
    "bayes": "Bayesian elastic-net logistic (NumPyro)",
}
_CLASSIFIER_NAME = {"enet": "Elastic Net", "rf": "Random Forest", "bayes": "Bayesian enet"}
_EXPLAINABILITY: dict[str, Literal["linear", "tree", "bayesian"]] = {
    "enet": "linear",
    "rf": "tree",
    "bayes": "bayesian",
}


def _build_model_info(model_id: str) -> ModelInfo:
    front_end, classifier_kind = MODEL_CELLS[model_id]
    fe_path = _FRONT_END_LABEL[front_end]
    return ModelInfo(
        model_id=model_id,
        display_name=f"{_FRONT_END_NAME[front_end]} → {_CLASSIFIER_NAME[classifier_kind]}",
        family="Unified 3×3 grid",
        classifier=_CLASSIFIER_LABEL[classifier_kind],
        feature_path=f"raw+rz → impute → cluster → {fe_path} → scale → {_CLASSIFIER_LABEL[classifier_kind]}",
        description=(
            f"{_FRONT_END_LABEL[front_end]} front-end into a "
            f"{_CLASSIFIER_LABEL[classifier_kind]} head; calibrated (isotonic) and "
            "run on both the random-stratified and blocked-temporal protocols."
        ),
        tuning_notebook=f"python -m secom.cli.run_tuning --model {model_id}",
        explainability=_EXPLAINABILITY[classifier_kind],
        track="both",
    )


MODEL_CATALOG: dict[str, ModelInfo] = {mid: _build_model_info(mid) for mid in MODEL_IDS}


def load_benchmark_results(path: Path | str = BENCHMARK_RESULTS_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing benchmark results at `{path}`.\n"
            "Run: `python -m secom.benchmark` or `benchmark_models.ipynb`."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def cv_leaderboard_df(payload: dict[str, Any]) -> pd.DataFrame:
    rows = payload.get("leaderboard") or []
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def cv_leaderboard_blocked_df(payload: dict[str, Any]) -> pd.DataFrame:
    rows = payload.get("leaderboard_blocked") or []
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


HOLDOUT_VIEW_KEYS: dict[str, str] = {
    "temporal": "holdout",
    "random": "holdout_random",
}


def holdout_df(payload: dict[str, Any], key: str = "holdout") -> pd.DataFrame:
    """Holdout rows for a given benchmark key (temporal / random)."""
    rows = payload.get(key) or []
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def holdout_comparison_df(payload: dict[str, Any]) -> pd.DataFrame:
    """One row per model under its own track: that track's CV PR-AUC vs its holdout PR-AUC.

    Interpolation models use stratified CV + random holdout; extrapolation models use
    blocked time CV + temporal holdout. The two tracks have disjoint model ids.
    """
    cv_df = cv_leaderboard_df(payload)
    cv_blocked_df = cv_leaderboard_blocked_df(payload)

    def _metric_map(key: str, metric: str) -> dict[str, float]:
        rows = payload.get(key) or []
        return {
            r["pipeline"]: r.get(metric)
            for r in rows
            if isinstance(r, dict) and "pipeline" in r
        }

    random_pr = _metric_map("holdout_random", "pr_auc")
    temporal_pr = _metric_map("holdout", "pr_auc")
    cv_pr = dict(zip(cv_df.get("pipeline", []), cv_df.get("mean_pr_auc", [])))
    cv_blocked_pr = (
        dict(zip(cv_blocked_df.get("pipeline", []), cv_blocked_df.get("mean_pr_auc", [])))
        if not cv_blocked_df.empty
        else {}
    )

    out_rows = []
    for pipeline in cv_pr:
        out_rows.append(
            {
                "pipeline": pipeline,
                "track": "interpolation",
                "cv_pr_auc": cv_pr.get(pipeline),
                "holdout_pr_auc": random_pr.get(pipeline),
            }
        )
    for pipeline in cv_blocked_pr:
        out_rows.append(
            {
                "pipeline": pipeline,
                "track": "extrapolation",
                "cv_pr_auc": cv_blocked_pr.get(pipeline),
                "holdout_pr_auc": temporal_pr.get(pipeline),
            }
        )
    if not out_rows:
        return pd.DataFrame()
    out = pd.DataFrame(out_rows)
    for col in out.select_dtypes(include="float").columns:
        out[col] = out[col].round(3)
    return out


def holdout_conditional_df(
    payload: dict[str, Any], key: str = "holdout_conditional"
) -> pd.DataFrame:
    """T2-gate conditional metrics + coverage per pipeline (temporal / random)."""
    rows = payload.get(key) or []
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def risk_coverage_df(payload: dict[str, Any]) -> pd.DataFrame:
    """Extrapolation gate risk-coverage sweep: one row per (pipeline, coverage)."""
    rows = payload.get("risk_coverage") or []
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def process_gate_meta(payload: dict[str, Any]) -> dict[str, Any]:
    """Process gate config (track-dependent): the interpolation EFA T²+Q or the
    extrapolation sBFA -> BGM density + Q params, control limits, and logic."""
    meta = payload.get("process_gate")
    if isinstance(meta, dict) and meta:
        return dict(meta)
    # Backward compatibility with older benchmark JSON.
    legacy = payload.get("t2_gate")
    return dict(legacy) if isinstance(legacy, dict) else {}


def t2_gate_meta(payload: dict[str, Any]) -> dict[str, Any]:
    """Alias for :func:`process_gate_meta`."""
    return process_gate_meta(payload)


def time_decay_meta(payload: dict[str, Any]) -> dict[str, Any]:
    """Per-model tuned decay lambda + search (extrapolation path)."""
    meta = payload.get("time_decay")
    return dict(meta) if isinstance(meta, dict) else {}


HOLDOUT_AUC_DISPLAY_COLS = [
    "pipeline",
    "pr_auc",
    "pr_auc_ci_low",
    "pr_auc_ci_high",
    "roc_auc",
    "roc_auc_ci_low",
    "roc_auc_ci_high",
]


def holdout_auc_summary_df(ho_df: pd.DataFrame) -> pd.DataFrame:
    """Holdout point estimates + bootstrap CIs for PR-AUC and ROC-AUC only."""
    cols = [c for c in HOLDOUT_AUC_DISPLAY_COLS if c in ho_df.columns]
    if not cols:
        return pd.DataFrame()
    out = ho_df[cols].copy()
    for col in out.select_dtypes(include="float").columns:
        out[col] = out[col].round(3)
    return out


def model_info(model_id: str) -> ModelInfo:
    if model_id not in MODEL_CATALOG:
        raise KeyError(f"Unknown model_id: {model_id}")
    return MODEL_CATALOG[model_id]


def list_model_ids(payload: dict[str, Any] | None = None) -> list[str]:
    if payload and payload.get("model_ids"):
        return list(payload["model_ids"])
    return list(BENCHMARK_MODEL_IDS)


def _parse_confusion_matrix(raw: object) -> list[list[int]] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, list):
        return raw
    return None


def holdout_confusion_by_profile(
    ho_df: pd.DataFrame,
    pipeline_id: str,
) -> dict[str, list[list[int]] | None]:
    """Holdout confusion matrices per F-beta profile for one pipeline."""
    out: dict[str, list[list[int]] | None] = {pid: None for pid in PROFILE_IDS}
    if ho_df.empty or "pipeline" not in ho_df.columns:
        return out
    rows = ho_df.loc[ho_df["pipeline"] == pipeline_id]
    if rows.empty:
        return out
    row = rows.iloc[0]
    for pid in PROFILE_IDS:
        cm_col = f"{pid}_confusion_matrix"
        if cm_col in row.index:
            out[pid] = _parse_confusion_matrix(row[cm_col])
    return out


def benchmark_has_multi_profile_thresholds(payload: dict[str, Any]) -> bool:
    for model_id in list_model_ids(payload):
        try:
            if has_multi_profile_thresholds(load_tuned_params(model_id)):
                return True
        except FileNotFoundError:
            continue
    return False


def resolved_threshold_profile_config(
    payload: dict[str, Any],
) -> dict[str, float | str]:
    cfg = payload.get("threshold_profile_config")
    if isinstance(cfg, dict) and cfg:
        return dict(cfg)
    frozen = payload.get("frozen_config") or {}
    if isinstance(frozen, dict) and (
        frozen.get("f0_5_beta") is not None or frozen.get("f2_beta") is not None
    ):
        return {
            "f0_5_beta": frozen.get("f0_5_beta"),
            "f2_beta": frozen.get("f2_beta"),
            "f4_beta": frozen.get("f4_beta"),
            "default_profile": frozen.get("default_profile", "f2"),
        }
    return threshold_profile_config()


def artifacts_available(path: Path | None = None) -> bool:
    path = path or PIPELINE_ARTIFACTS_PATH
    return path.is_file()


def load_pipeline_artifacts(path: Path | None = None) -> dict[str, Any]:
    path = path or PIPELINE_ARTIFACTS_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Pipeline artifacts not found at {path}. "
            "Run: python -m secom.benchmark"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_report_cache(path: Path | None = None) -> dict[str, Any]:
    """Frozen PR-curve + global-importance cache written by ``secom.benchmark``."""
    path = path or REPORT_CACHE_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Report cache not found at {path}. Run: python -m secom.benchmark"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def report_entry(track: str, model_id: str) -> dict[str, Any]:
    """Frozen report entry (`pr_curve`, `global_importance`) for one track/model."""
    cache = load_report_cache()
    entry = (cache.get(track) or {}).get(model_id)
    if not entry:
        raise FileNotFoundError(
            f"No frozen report for {model_id!r} ({track}). Run: python -m secom.benchmark"
        )
    return entry


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
