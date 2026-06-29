"""Load benchmark results and pipeline artifacts for Streamlit dashboard pages."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from secom.costs import (
    PROFILE_IDS,
    THRESHOLD_PROFILES,
    catch_overkill_from_confusion,
    has_multi_profile_thresholds,
    threshold_profile_config,
)
from secom.pipelines import (
    BENCHMARK_MODEL_IDS,
    BENCHMARK_RESULTS_PATH,
    EXPLANATIONS_CACHE_PATH,
    MODEL_CELLS,
    MODEL_IDS,
    PIPELINE_ARTIFACTS_PATH,
    REFERENCE_MODELS,
    REPORT_CACHE_PATH,
)
from secom.utils import load_tuned_params


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
    "pls": "PLS components",
}
_FRONT_END_NAME = {"hsic": "HSIC", "rfsel": "RF-select", "pls": "PLS"}
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
            f"{_CLASSIFIER_LABEL[classifier_kind]} head; calibrated (sigmoid / Platt) and "
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


@lru_cache(maxsize=1)
def cached_benchmark_results() -> dict[str, Any]:
    """Process-cached benchmark JSON for the offline narrative batch build.

    The narrator enriches every wafer with gate facts; reading the (large)
    benchmark JSON once per wafer would be wasteful. Cached read-only; callers
    must not mutate the returned dict. Returns ``{}`` when the file is absent so
    the narrative build degrades to drivers-only facts.
    """
    try:
        return load_benchmark_results()
    except FileNotFoundError:
        return {}


def cv_leaderboard_df(payload: dict[str, Any]) -> pd.DataFrame:
    rows = payload.get("leaderboard") or []
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def holdout_df(payload: dict[str, Any], key: str = "holdout") -> pd.DataFrame:
    """Holdout rows for a given benchmark key (temporal / random)."""
    rows = payload.get(key) or []
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def holdout_comparison_df(payload: dict[str, Any]) -> pd.DataFrame:
    """Two rows per model (one per track): in-distribution CV PR-AUC vs holdout PR-AUC.

    Tuning happens once on the stratified CV, so both tracks share that
    in-distribution CV PR-AUC as the reference. Interpolation rows compare it to
    the random holdout; extrapolation rows compare it to the temporal holdout
    (the drift gap).
    """
    cv_df = cv_leaderboard_df(payload)

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
    for pipeline in cv_pr:
        out_rows.append(
            {
                "pipeline": pipeline,
                "track": "extrapolation",
                "cv_pr_auc": cv_pr.get(pipeline),
                "holdout_pr_auc": temporal_pr.get(pipeline),
            }
        )
    if not out_rows:
        return pd.DataFrame()
    out = pd.DataFrame(out_rows)
    for col in out.select_dtypes(include="float").columns:
        out[col] = out[col].round(3)
    return out


# --- Gate reports (both gates x both tracks, persisted in `gate_reports`) ----
# Dashboard track -> benchmark protocol key used inside `gate_reports`.
_TRACK_TO_PROTOCOL = {"interpolation": "random", "extrapolation": "temporal"}
# Metric radio label -> column stem in the gate/holdout rows.
DELTA_METRIC_COLS = {"PR-AUC": "pr_auc", "ROC-AUC": "roc_auc"}


def _gate_block(payload: dict[str, Any], track: str, gate: str) -> dict[str, Any]:
    proto = _TRACK_TO_PROTOCOL.get(track, track)
    reports = payload.get("gate_reports") or {}
    return ((reports.get(proto) or {}).get(gate)) or {}


def gate_conditional_df(payload: dict[str, Any], track: str, gate: str) -> pd.DataFrame:
    """Conditional + coverage rows for one gate on one track (pca | bayes)."""
    rows = _gate_block(payload, track, gate).get("conditional") or []
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def gate_risk_coverage_df(payload: dict[str, Any], track: str, gate: str) -> pd.DataFrame:
    """Risk-coverage sweep rows for one gate on one track."""
    rows = _gate_block(payload, track, gate).get("risk_coverage") or []
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def gate_config(payload: dict[str, Any], track: str, gate: str) -> dict[str, Any]:
    """Fitted-gate config (control limits, factors, logic) for one gate/track."""
    return dict(_gate_block(payload, track, gate).get("config") or {})


def gate_diagnostics(
    payload: dict[str, Any], track: str, gate: str = "bayes"
) -> dict[str, Any]:
    """Frozen per-wafer control statistics for the gate-monitor charts (Tier 1).

    Returns the ``{reference, holdout, limits}`` block persisted by the benchmark
    (BGM density/Q over reference + holdout wafers, control limits, and temporal
    timestamps). Empty dict when absent (older benchmark JSON / PCA gate).
    """
    return dict(_gate_block(payload, track, gate).get("diagnostics") or {})


# PCA / BGM gate -> (holdout flag key, plain-English label) for the wafer-level
# OOC corroboration facts. ``ooc`` is the gate's overall verdict for the wafer.
_GATE_TRIP_SPECS: dict[str, list[tuple[str, str]]] = {
    "pca": [
        ("t2_ooc", "Hotelling T2 above limit"),
        ("q_ooc", "Q/SPE above limit"),
    ],
    "bayes": [
        ("density_ooc", "BGM density below limit"),
        ("q_ooc", "Q/SPE above limit"),
    ],
}


def wafer_gate_facts(
    payload: dict[str, Any], track: str, observation_id: object
) -> dict[str, Any]:
    """Per-wafer PCA + BGM process-gate corroboration for one wafer.

    For each standalone gate, looks up this wafer's frozen control statistics by
    ``observation_id`` (written by the benchmark) and reports whether the gate
    independently flagged it out-of-control and which control statistic tripped.
    Returns ``{}`` when ids are absent (benchmark not yet re-run) or the wafer is
    not in the holdout, so callers degrade gracefully to drivers-only facts.
    """
    if observation_id is None:
        return {}
    try:
        target_id = int(observation_id)
    except (TypeError, ValueError):
        return {}
    out: dict[str, Any] = {}
    for gate, specs in _GATE_TRIP_SPECS.items():
        holdout = gate_diagnostics(payload, track, gate).get("holdout") or {}
        ids = holdout.get("observation_id")
        if not ids:
            continue
        try:
            idx = list(ids).index(target_id)
        except ValueError:
            continue

        def _flag(key: str) -> bool:
            arr = holdout.get(key) or []
            return bool(arr[idx]) if idx < len(arr) else False

        tripped = [label for key, label in specs if _flag(key)]
        out[gate] = {
            "out_of_control": _flag("ooc"),
            "tripped": tripped,
        }
    return out


def bgm_ooc_wafers(
    payload: dict[str, Any], track: str = "extrapolation"
) -> pd.DataFrame:
    """Passing (in-control) holdout wafers the BGM gate flagged out-of-control.

    From the frozen BGM holdout block, keep wafers with ``ooc`` true and
    ``y_true == 0`` - wafers that passed inspection (good yield) yet trip the
    density/Q gate, the cleanest "process drift, not yield" examples. Columns:
    ``observation_id``, ``ts``, ``density``, ``density_ooc``, ``q_ooc``; sorted by
    lowest density first (most out-of-control). Empty when ids/diagnostics absent.
    """
    holdout = gate_diagnostics(payload, track, "bayes").get("holdout") or {}
    ids = holdout.get("observation_id")
    if not ids:
        return pd.DataFrame()
    n = len(ids)

    def _col(key: str, fill: Any = None) -> list[Any]:
        arr = list(holdout.get(key) or [])
        return arr + [fill] * (n - len(arr)) if len(arr) < n else arr[:n]

    df = pd.DataFrame(
        {
            "observation_id": [int(i) for i in ids],
            "ts": _col("ts", ""),
            "density": _col("density", float("nan")),
            "density_ooc": _col("density_ooc", False),
            "q_ooc": _col("q_ooc", False),
            "y_true": _col("y_true", 0),
            "ooc": _col("ooc", False),
        }
    )
    keep = df[(df["ooc"].astype(bool)) & (df["y_true"].astype(int) == 0)].copy()
    if keep.empty:
        return pd.DataFrame()
    return (
        keep.drop(columns=["y_true", "ooc"])
        .sort_values("density", ascending=True, na_position="last")
        .reset_index(drop=True)
    )


def bgm_ooc_map(payload: dict[str, Any], track: str = "extrapolation") -> dict[int, bool]:
    """Per-wafer BGM-gate out-of-control verdict for every holdout wafer.

    Maps ``observation_id -> ooc`` from the frozen BGM holdout block, covering
    both passing and failing wafers (unlike :func:`bgm_ooc_wafers`, which keeps
    only passing OOC wafers). Returns ``{}`` when ids/diagnostics are absent
    (older benchmark JSON, or a track with no frozen gate diagnostics) so callers
    can degrade gracefully to a gate-free picker.
    """
    holdout = gate_diagnostics(payload, track, "bayes").get("holdout") or {}
    ids = holdout.get("observation_id")
    ooc = holdout.get("ooc")
    if not ids or ooc is None:
        return {}
    out: dict[int, bool] = {}
    for i, flag in zip(ids, ooc):
        try:
            out[int(i)] = bool(flag)
        except (TypeError, ValueError):
            continue
    return out


def gate_drift_stats(
    payload: dict[str, Any],
    gate: str = "bayes",
    track: str = "extrapolation",
    stats: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Honest in-control population drift scalars per control statistic.

    For each statistic (``stats`` is a list of ``(holdout_key, label)``; defaults
    to the BGM log-density + Q/SPE) computes the 2-sample Kolmogorov-Smirnov
    distance (+ p-value) and a separability AUC between the passing-train
    ``reference`` and the **passing** wafers of the holdout (fails removed via the
    frozen ``y_true``). Restricting to passing wafers isolates sensor/process drift
    from the yield mix; for the extrapolation track this is passing-early-train vs
    passing-late-holdout. Computed over all passing wafers, so it is the
    statistically solid drift evidence. Empty when diagnostics are absent.
    """
    import numpy as np
    from scipy.stats import ks_2samp
    from sklearn.metrics import roc_auc_score

    if stats is None:
        stats = [("density", "BGM log-density"), ("q", "Q / SPE")]
    diag = gate_diagnostics(payload, track, gate)
    ref = diag.get("reference") or {}
    hold = diag.get("holdout") or {}
    y_hold = np.asarray(hold.get("y_true", []), dtype=int)
    rows = []
    for key, label in stats:
        a = np.asarray(ref.get(key, []), dtype=float)
        b = np.asarray(hold.get(key, []), dtype=float)
        # Restrict the holdout to passing wafers so the scalar measures in-control
        # process drift, not the pass/fail mix.
        if y_hold.size == b.size and y_hold.size:
            b = b[y_hold == 0]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size == 0 or b.size == 0:
            continue
        ks = ks_2samp(a, b)
        labels = np.concatenate([np.zeros(a.size), np.ones(b.size)])
        values = np.concatenate([a, b])
        try:
            auc = float(roc_auc_score(labels, values))
        except ValueError:
            auc = float("nan")
        rows.append(
            {
                "statistic": label,
                "ks_distance": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                # Directionless separability: 0.5 = no drift, 1.0 = fully separable.
                "separability_auc": max(auc, 1.0 - auc),
                "n_reference": int(a.size),
                "n_holdout": int(b.size),
            }
        )
    return pd.DataFrame(rows)


def sbfa_diagnostics(
    payload: dict[str, Any], track: str = "extrapolation", gate: str = "bayes"
) -> dict[str, Any]:
    """Frozen member-0 sBFA artifacts (loadings, BGM envelope, factor scores).

    Returns the ``sbfa`` sub-block of the gate diagnostics (Tier-2 latent-space /
    root-cause visuals). Empty dict when absent (random track / older JSON).
    """
    return dict(gate_diagnostics(payload, track, gate).get("sbfa") or {})


def pca_component_diagnostics(
    payload: dict[str, Any], track: str = "extrapolation"
) -> dict[str, Any]:
    """Frozen PCA component artifacts (loadings, score Gaussian, component scores).

    Returns the ``components`` sub-block of the PCA gate diagnostics (5.2
    component-space / root-cause visuals). Empty dict when absent (random track /
    older JSON).
    """
    return dict(gate_diagnostics(payload, track, "pca").get("components") or {})


def gate_contrast(payload: dict[str, Any], track: str = "extrapolation") -> dict[str, Any]:
    """Frozen PCA-vs-sBFA justification artifacts (page 5.4).

    Returns the ``contrast`` sub-block of the temporal gate reports (per-wafer T2
    vs BGM density, BGM mode weights, per-sensor Ψ, and one example wafer's
    residuals). Empty dict when absent (older JSON / benchmark not re-run).
    """
    proto = _TRACK_TO_PROTOCOL.get(track, track)
    reports = payload.get("gate_reports") or {}
    return dict(((reports.get(proto) or {}).get("contrast")) or {})


def gate_disagreement_summary(contrast: dict[str, Any]) -> dict[str, int]:
    """Quadrant counts of passing wafers by which gate(s) would abstain.

    Compares per-wafer PCA Hotelling T2 against its UCL and BGM log-density
    against its LCL. The ``pca_only`` count is the headline: healthy wafers PCA
    flags out-of-control that the multimodal BGM keeps in-control.
    """
    import numpy as np

    ref = contrast.get("reference") or {}
    t2 = np.asarray(ref.get("pca_t2", []), dtype=float)
    dens = np.asarray(ref.get("bgm_density", []), dtype=float)
    if t2.size == 0 or dens.size != t2.size:
        return {}
    pca_ooc = t2 > float(ref.get("pca_t2_ucl", np.inf))
    bgm_ooc = dens < float(ref.get("bgm_density_lcl", -np.inf))
    return {
        "n_total": int(t2.size),
        "pca_only": int((pca_ooc & ~bgm_ooc).sum()),
        "bgm_only": int((~pca_ooc & bgm_ooc).sum()),
        "both": int((pca_ooc & bgm_ooc).sum()),
        "neither": int((~pca_ooc & ~bgm_ooc).sum()),
    }


def sensor_noise_df(contrast: dict[str, Any]) -> pd.DataFrame:
    """Per-sensor sBFA noise variance Ψ, descending (heteroscedasticity spectrum)."""
    import numpy as np

    psi = np.asarray(contrast.get("psi", []), dtype=float)
    names = list(contrast.get("feature_names", []))
    if psi.size == 0 or len(names) != psi.size:
        return pd.DataFrame()
    out = pd.DataFrame({"sensor": names, "psi": psi})
    return out.sort_values("psi", ascending=False).reset_index(drop=True)


def contribution_compare_df(contrast: dict[str, Any]) -> pd.DataFrame:
    """Per-sensor equal-weight (PCA) vs noise-weighted (sBFA) contribution.

    For the frozen example wafer: equal-weight = ``resid²``; noise-weighted =
    ``resid² / Ψ``. Each column is normalised to its own max so the comparison is
    about *rank*, not absolute scale. Sorted by the noise-weighted contribution.
    """
    import numpy as np

    ex = contrast.get("example_wafer") or {}
    rp = np.asarray(ex.get("pca_resid", []), dtype=float)
    rs = np.asarray(ex.get("sbfa_resid", []), dtype=float)
    psi = np.asarray(contrast.get("psi", []), dtype=float)
    names = list(contrast.get("feature_names", []))
    if rp.size == 0 or rs.size != rp.size or psi.size != rp.size or len(names) != rp.size:
        return pd.DataFrame()
    pca_contrib = rp**2
    sbfa_contrib = rs**2 / np.clip(psi, 1e-12, None)
    out = pd.DataFrame(
        {
            "sensor": names,
            "pca_contribution": pca_contrib / (pca_contrib.max() or 1.0),
            "sbfa_contribution": sbfa_contrib / (sbfa_contrib.max() or 1.0),
        }
    )
    return out.sort_values("sbfa_contribution", ascending=False).reset_index(drop=True)


def factor_drift_ranking(sbfa: dict[str, Any]) -> pd.DataFrame:
    """Rank sBFA factors by in-dist->holdout score drift, with heavy-loading sensors.

    For each latent factor, the KS distance between the reference and holdout
    score distributions; the heavy-loading sensors are the largest-|loading|
    features for that factor. Sorted by drift descending. Empty when no sbfa.
    """
    import numpy as np
    from scipy.stats import ks_2samp

    ref = np.asarray(sbfa.get("reference_scores", []), dtype=float)
    hold = np.asarray(sbfa.get("holdout_scores", []), dtype=float)
    loadings = np.asarray(sbfa.get("loadings", []), dtype=float)
    names = list(sbfa.get("feature_names", []))
    if ref.ndim != 2 or hold.ndim != 2 or ref.shape[1] == 0:
        return pd.DataFrame()

    n_factors = ref.shape[1]
    rows = []
    for f in range(n_factors):
        a = ref[:, f]
        b = hold[:, f]
        ks = ks_2samp(a[np.isfinite(a)], b[np.isfinite(b)])
        top_sensors = ""
        if loadings.ndim == 2 and f < loadings.shape[1] and names:
            order = np.argsort(np.abs(loadings[:, f]))[::-1][:3]
            top_sensors = ", ".join(
                str(names[i]) for i in order if i < len(names)
            )
        rows.append(
            {
                "factor": f + 1,
                "factor_idx": f,
                "ks_distance": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "mean_shift": float(np.nanmean(b) - np.nanmean(a)),
                "top_sensors": top_sensors,
            }
        )
    return pd.DataFrame(rows).sort_values("ks_distance", ascending=False).reset_index(drop=True)


def holdout_delta_df(payload: dict[str, Any], metric: str) -> pd.DataFrame:
    """Per model: interpolation vs extrapolation holdout metric and their delta.

    ``metric`` is a column stem (``pr_auc`` / ``roc_auc``). ``delta`` is
    interpolation - extrapolation (positive = interpolation higher = drift cost).
    """
    interp = holdout_df(payload, "holdout_random")
    extrap = holdout_df(payload, "holdout")
    if interp.empty or extrap.empty or metric not in interp or metric not in extrap:
        return pd.DataFrame()
    i = interp[["pipeline", metric]].rename(columns={metric: "interpolation"})
    e = extrap[["pipeline", metric]].rename(columns={metric: "extrapolation"})
    out = i.merge(e, on="pipeline", how="inner")
    out["delta"] = out["interpolation"] - out["extrapolation"]
    return out


def gate_vs_gate_df(payload: dict[str, Any], track: str, metric: str) -> pd.DataFrame:
    """Per model PCA-minus-Bayes conditional metric on one track."""
    pca = gate_conditional_df(payload, track, "pca")
    bayes = gate_conditional_df(payload, track, "bayes")
    col = f"conditional_{metric}"
    if pca.empty or bayes.empty or col not in pca or col not in bayes:
        return pd.DataFrame()
    e = pca[["pipeline", col]].rename(columns={col: "pca"})
    b = bayes[["pipeline", col]].rename(columns={col: "bayes"})
    out = e.merge(b, on="pipeline", how="inner")
    out["delta"] = out["pca"] - out["bayes"]
    return out.dropna(subset=["delta"])


HOLDOUT_AUC_DISPLAY_COLS = [
    "pipeline",
    "pr_auc",
    "pr_auc_ci_low",
    "pr_auc_ci_high",
    "roc_auc",
    "roc_auc_ci_low",
    "roc_auc_ci_high",
    "ber_percent",
    "ber_percent_ci_low",
    "ber_percent_ci_high",
]


def holdout_auc_summary_df(ho_df: pd.DataFrame) -> pd.DataFrame:
    """Holdout point estimates + bootstrap CIs for PR-AUC, ROC-AUC, and BER."""
    cols = [c for c in HOLDOUT_AUC_DISPLAY_COLS if c in ho_df.columns]
    if not cols:
        return pd.DataFrame()
    out = ho_df[cols].copy()
    for col in out.select_dtypes(include="float").columns:
        out[col] = out[col].round(3)
    return out


def operating_table_df(ho_df: pd.DataFrame, pipeline_id: str) -> pd.DataFrame:
    """Per-threshold-profile operating point for one pipeline, in fab vocabulary.

    One row per profile (``conservative/ber/aggressive/economic``). Catch rate
    (recall), overkill rate (false-alarm rate = 1 - TNR), precision, and the raw
    fails-caught / good-flagged counts come from each profile's holdout confusion
    matrix already stored in ``holdout_df``; BER% is kept as the secondary
    (prevalence-free, symmetric) summary on the right. Empty when the
    pipeline/columns are absent.
    """
    if ho_df.empty or "pipeline" not in ho_df.columns:
        return pd.DataFrame()
    rows = ho_df.loc[ho_df["pipeline"] == pipeline_id]
    if rows.empty:
        return pd.DataFrame()
    row = rows.iloc[0]
    out = []
    for pid in PROFILE_IDS:
        thr = row.get(f"{pid}_threshold")
        ber = row.get(f"{pid}_ber_percent")
        if thr is None and ber is None:
            continue
        label = THRESHOLD_PROFILES[pid].display_name if pid in THRESHOLD_PROFILES else pid
        cm = _parse_confusion_matrix(row.get(f"{pid}_confusion_matrix"))
        fab = catch_overkill_from_confusion(cm) if cm is not None else {}
        out.append(
            {
                "Profile": label,
                "Threshold": round(float(thr), 4) if thr is not None else None,
                "Catch %": round(100 * fab["catch_rate"], 1) if fab else None,
                "Overkill %": round(100 * fab["overkill_rate"], 1) if fab else None,
                "Precision %": round(100 * fab["precision"], 1) if fab else None,
                "Fails caught": (
                    f"{fab['fails_caught']}/{fab['fails_total']}" if fab else None
                ),
                "Good flagged": (
                    f"{fab['good_flagged']}/{fab['good_total']}" if fab else None
                ),
                "BER %": round(float(ber), 1) if ber is not None else None,
            }
        )
    return pd.DataFrame(out)


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
    if isinstance(frozen, dict) and frozen.get("ber_band_tolerance") is not None:
        return {
            "ber_band_tolerance": frozen.get("ber_band_tolerance"),
            "default_profile": frozen.get("default_profile", "ber"),
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


@lru_cache(maxsize=None)
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


@lru_cache(maxsize=None)
def load_explanations_cache(path: Path | None = None) -> dict[str, Any]:
    """Frozen per-wafer explainability cache written by ``secom.cli.build_explanations``."""
    path = path or EXPLANATIONS_CACHE_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Explanations cache not found at {path}. "
            "Run: python -m secom.cli.build_explanations"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def explanation_entry(track: str, model_id: str) -> dict[str, Any]:
    """Frozen explainability entry (`outcomes`, `wafers`, `global`, ...) for one track/model."""
    cache = load_explanations_cache()
    entry = (cache.get(track) or {}).get(model_id)
    if not entry:
        raise FileNotFoundError(
            f"No frozen explanations for {model_id!r} ({track}). "
            "Run: python -m secom.cli.build_explanations"
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


@dataclass
class PipelineContext:
    """Loaded pipeline artifacts and the slices the Pipeline pages render."""

    available: bool
    artifact_caption: str
    linear_ref: dict[str, Any] | None
    topk_ref: dict[str, Any] | None
    cluster_example: dict[str, Any] | None
    models: dict[str, Any]


def load_pipeline_context() -> PipelineContext:
    """Load pipeline artifacts and slice out what the Pipeline pages need.

    Streamlit-free: when ``available`` is False the caller decides how to
    surface it (the renderers fall back to illustrative values).
    """
    artifacts: dict[str, Any] | None = None
    artifact_caption = ""
    available = artifacts_available()
    if available:
        artifacts = load_pipeline_artifacts()
        generated = artifacts.get("generated_at", "")[:19].replace("T", " ")
        artifact_caption = (
            f"From holdout training fit (`secom_pipeline_artifacts.json`, {generated} UTC)."
        )
    shared = (artifacts or {}).get("shared", {})
    return PipelineContext(
        available=available,
        artifact_caption=artifact_caption,
        linear_ref=get_reference_artifacts(artifacts, "linear") if artifacts else None,
        topk_ref=get_reference_artifacts(artifacts, "topk") if artifacts else None,
        cluster_example=shared.get("spearman_cluster_example"),
        models=(artifacts or {}).get("models", {}),
    )
