"""Model explainability: global importance and holdout wafer-level breakdowns."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from secom.bayes.model import BayesianElasticNetLogistic
from secom.costs import DEFAULT_PROFILE_ID, resolve_threshold_profiles
from secom.dashboard.data import model_info, report_entry
from secom.metrics import predict_with_threshold
from secom.pipelines import (
    BENCHMARK_MODEL_IDS,
    ID_COL,
    MODEL_CELLS,
    TARGET_COL,
    feature_columns,
    load_mart,
    split_train_test,
    split_train_test_random,
)
from secom.reporting import _bayes_mean_coef, scaled_matrix
from secom.tuning.registry import build_tuned_pipeline, fit_pipeline_weighted
from secom.utils import (
    fitted_base_classifier,
    load_tuned_params,
)

LOCAL_TOP_N = 5
GLOBAL_SENSOR_TOP_N = 15

DEFAULT_TRACK = "extrapolation"

# The page exposes exactly one champion per track (matches the narrative model):
# the random/in-distribution track uses the RF-selection tree head; the temporal
# track uses the PLS Bayesian champion.
CHAMPION_BY_TRACK: dict[str, str] = {
    "interpolation": "hsic_rf",
    "extrapolation": "pls_bayes",
}


def champion_for_track(track: str) -> str:
    """The single model surfaced on the RCA page for ``track``."""
    return CHAMPION_BY_TRACK.get(track, CHAMPION_BY_TRACK[DEFAULT_TRACK])


def is_pls_model(model_id: str) -> bool:
    """True for the sPLS front-end cells (pls_enet / pls_rf / pls_bayes)."""
    return MODEL_CELLS.get(model_id, ("", ""))[0] == "pls"


@dataclass(frozen=True)
class HoldoutSplit:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_cols: list[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass(frozen=True)
class WaferExplanation:
    observation_id: object
    actual_label: int
    predicted_label: int
    fail_probability: float
    threshold: float
    local_df: pd.DataFrame
    method: str
    # Best-effort posterior P(fail) credible interval for Bayesian heads, taken
    # from the base estimator's posterior logits *before* sigmoid calibration
    # (the calibrated point estimate may fall outside it). None for other heads.
    fail_probability_interval: tuple[float, float] | None = None
    fail_probability_uncalibrated: float | None = None


@st.cache_data(show_spinner=False)
def load_holdout_split(track: str = DEFAULT_TRACK) -> HoldoutSplit:
    df = load_mart()
    cols = feature_columns(df)
    if track == "interpolation":
        train_df, test_df = split_train_test_random(df)
    else:
        train_df, test_df = split_train_test(df)
    X_train = train_df[cols]
    X_test = test_df[cols]
    y_train = train_df[TARGET_COL].astype(int)
    y_test = test_df[TARGET_COL].astype(int)
    return HoldoutSplit(
        train_df=train_df,
        test_df=test_df,
        feature_cols=cols,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


def _tuned_for(model_id: str, track: str) -> dict:
    """Single in-distribution tuned params, reused for both tracks."""
    return load_tuned_params(model_id)


@st.cache_resource(show_spinner="Fitting pipeline on train split…")
def fit_holdout_pipeline(model_id: str, track: str = DEFAULT_TRACK):
    if model_id not in BENCHMARK_MODEL_IDS:
        raise ValueError(f"Unknown model_id: {model_id}")
    split = load_holdout_split(track)
    tuned = _tuned_for(model_id, track)
    pipeline = build_tuned_pipeline(model_id, tuned)
    pipeline, _ = fit_pipeline_weighted(
        pipeline, split.X_train, split.y_train, None
    )
    return pipeline, tuned


def _deploy_threshold(tuned: dict) -> float:
    profiles = resolve_threshold_profiles(tuned)
    return float(profiles[DEFAULT_PROFILE_ID])


def _shap_positive_class_values(shap_output: object, n_features: int) -> np.ndarray:
    """Extract positive-class SHAP vector length n_features."""
    if isinstance(shap_output, list):
        return np.asarray(shap_output[1]).reshape(-1)[:n_features]
    arr = np.asarray(shap_output)
    if arr.ndim == 3:
        return arr[0, :, 1]
    if arr.ndim == 2:
        if arr.shape[0] == 1 and arr.shape[1] == n_features:
            return arr[0, :]
        if arr.shape[1] >= 2 and arr.shape[0] == n_features:
            return arr[:, 1]
    return arr.reshape(-1)[:n_features]


def _shap_contribution_vector(pipeline, row_scaled: np.ndarray, n_features: int) -> np.ndarray:
    """Full signed SHAP vector (positive class) for one scaled row."""
    import shap

    estimator = fitted_base_classifier(pipeline)
    explainer = shap.TreeExplainer(estimator)
    sv = explainer.shap_values(row_scaled.reshape(1, -1))
    return _shap_positive_class_values(sv, n_features)


# --- PLS sensor back-projection ----------------------------------------------
# PLS heads emit latent scores pls_0..pls_{k-1}, so attributions land on
# components, not sensors. PLS is linear, so we map them back to the post-cluster
# sensor space through the fitted PLSRegression loadings, unifying all 9 models
# into one sensor vocabulary. This is a linear approximation (the RobustScaler
# sits between the components and the head), so it is labelled as such in the UI.
def _fitted_pls(pipeline):
    """Return the fitted ``PLSFeatures`` transformer inside a pls_* pipeline."""
    preprocess = pipeline.named_steps["preprocess"]
    branch = preprocess.named_transformers_["sensor_branch"]
    return branch.named_steps["front_end"]


def _pls_loadings(pipeline) -> tuple[np.ndarray, list[str]]:
    """(loadings, sensor_names) where loadings is (n_sensors, n_components)."""
    front = _fitted_pls(pipeline)
    loadings = np.asarray(front.pls_.x_loadings_, dtype=float)
    sensors = [str(s) for s in front.feature_names_in_]
    return loadings, sensors


@st.cache_data(show_spinner="Projecting PLS components…")
def cached_pls_score_scatter(
    track: str = "interpolation",
    model_id: str = "pls_enet",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training-split latent scores (component 1 vs 2) for a PLS front-end.

    Uses the cached live fit so page 2 can show the supervised projection that PLS
    builds. Reuses ``PLSRegression.x_scores_`` (the training scores fit inside the
    pipeline), aligned to the train labels. If the model has a single component,
    the second axis is returned as zeros.
    """
    pipeline, _ = fit_holdout_pipeline(model_id, track)
    front = _fitted_pls(pipeline)
    scores = np.asarray(front.pls_.x_scores_, dtype=float)
    t1 = scores[:, 0]
    t2 = scores[:, 1] if scores.shape[1] > 1 else np.zeros_like(t1)
    y = load_holdout_split(track).y_train.to_numpy().astype(int)
    return t1, t2, y


def _component_index(name: str) -> int | None:
    s = str(name)
    if not s.startswith("pls_"):
        return None
    try:
        return int(s.split("_", 1)[1])
    except ValueError:
        return None


def _split_components(
    values: np.ndarray, names: np.ndarray, n_components: int
) -> tuple[np.ndarray, list[tuple[str, float]]]:
    """Split a per-feature vector into a component-indexed array + aux passthrough."""
    comp = np.zeros(n_components, dtype=float)
    aux: list[tuple[str, float]] = []
    for value, name in zip(values, names):
        c = _component_index(name)
        if c is not None and 0 <= c < n_components:
            comp[c] += float(value)
        else:
            aux.append((str(name), float(value)))
    return comp, aux


def pls_local_sensor_contributions(
    pipeline, contrib: np.ndarray, names: np.ndarray, top_n: int = LOCAL_TOP_N
) -> pd.DataFrame:
    """Back-project component contributions (coef_c x score_c) onto sensors.

    Each component's contribution is spread across sensors by its signed loading,
    normalised by the component's total |loading| so magnitude is conserved.
    Non-sensor aux features (calendar, missing flags) are excluded so the view
    is a pure sensor RCA.
    """
    loadings, sensors = _pls_loadings(pipeline)
    n_components = loadings.shape[1]
    comp_contrib, _aux = _split_components(np.asarray(contrib, dtype=float), names, n_components)
    col_abs = np.abs(loadings).sum(axis=0)
    col_abs[col_abs == 0] = 1.0
    weights = loadings / col_abs  # (n_sensors, n_components), sign-aware
    sensor_contrib = weights @ comp_contrib
    rows = list(zip(sensors, sensor_contrib))
    df = pd.DataFrame(rows, columns=["feature", "contribution"])
    df["abs_contribution"] = df["contribution"].abs()
    return df.nlargest(top_n, "abs_contribution")


def pls_global_sensor_importance(
    pipeline, importance: np.ndarray, names: np.ndarray, signed: np.ndarray | None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Back-project per-component importance onto sensors for the global view.

    ``importance`` is a non-negative per-feature vector (|coef| or mean|SHAP|);
    ``signed`` (optional, linear/Bayesian heads) carries direction. Returns the
    top sensor-space importance table plus an optional signed subset. Non-sensor
    aux features (calendar, missing flags) are excluded (pure sensor RCA).
    """
    loadings, sensors = _pls_loadings(pipeline)
    n_components = loadings.shape[1]
    comp_imp, _aux_imp = _split_components(np.asarray(importance, dtype=float), names, n_components)
    abs_load = np.abs(loadings)
    sensor_imp = abs_load @ comp_imp
    records = [(s, float(v)) for s, v in zip(sensors, sensor_imp)]
    full = pd.DataFrame(records, columns=["feature", "importance"])
    full["abs_coefficient"] = full["importance"]
    top = full.nlargest(GLOBAL_SENSOR_TOP_N, "importance").reset_index(drop=True)

    signed_df: pd.DataFrame | None = None
    if signed is not None:
        comp_signed, _aux_signed = _split_components(
            np.asarray(signed, dtype=float), names, n_components
        )
        sensor_signed = loadings @ comp_signed
        srecords = [(s, float(v)) for s, v in zip(sensors, sensor_signed)]
        sfull = pd.DataFrame(srecords, columns=["feature", "coefficient"])
        pos = sfull.loc[sfull["coefficient"] > 0].nlargest(8, "coefficient")
        neg = sfull.loc[sfull["coefficient"] < 0].nsmallest(8, "coefficient")
        signed_df = pd.concat([pos, neg], ignore_index=True)
    return top, signed_df


def _bayes_fail_interval(
    pipeline, row_scaled: np.ndarray, q: tuple[float, float] = (0.025, 0.975)
) -> tuple[tuple[float, float] | None, float | None]:
    """Posterior P(fail) credible interval from the base Bayesian estimator.

    Uses the uncalibrated posterior logits (pre-sigmoid), so this is a spread
    around the *uncalibrated* mean, not the calibrated point estimate.
    """
    est = fitted_base_classifier(pipeline)
    if not isinstance(est, BayesianElasticNetLogistic):
        return None, None
    try:
        logits = est._logits_samples(row_scaled.reshape(1, -1))
    except Exception:
        return None, None
    logits = np.clip(np.asarray(logits, dtype=float), -30.0, 30.0).ravel()
    if logits.size == 0:
        return None, None
    probs = 1.0 / (1.0 + np.exp(-logits))
    lo, hi = (float(v) for v in np.quantile(probs, q))
    return (lo, hi), float(probs.mean())


def _robust_z(value: float, ref_values) -> float:
    """Robust SPC z-score (median / IQR) of one value vs a reference sample."""
    ref = np.asarray(ref_values, dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size < 5 or not np.isfinite(value):
        return float("nan")
    med = float(np.median(ref))
    q1, q3 = np.quantile(ref, [0.25, 0.75])
    iqr = float(q3 - q1)
    scale = iqr / 1.349 if iqr > 0 else float(ref.std())
    if not scale or not np.isfinite(scale):
        return float("nan")
    return float((value - med) / scale)


def _enrich_local_df(
    local_df: pd.DataFrame, split: "HoldoutSplit", row: pd.Series, track: str
) -> pd.DataFrame:
    """Annotate contributors with an SPC z-score, an era-drift flag, and a group.

    - ``spc_z``: robust z of this wafer's raw value vs the in-control (passing)
      training distribution. NaN for derived features (interactions, T²) that
      have no raw column.
    - ``drift_shift`` / ``drift_flag``: training-era -> holdout-era mean shift in
      SD units (reused era-drift logic), flagged past the z-threshold.
    - ``group``: data-driven correlation-cluster id (not real equipment).
    """
    drift = cached_era_drift()
    shift_map = drift["shift"]
    z_threshold = float(drift["z_threshold"])
    groups = cached_sensor_groups(track)
    train_df = split.train_df
    pass_mask = train_df[TARGET_COL].astype(int) == 0

    spc, drift_shift, drift_flag, group = [], [], [], []
    for feat in local_df["feature"].astype(str):
        if feat in train_df.columns:
            spc.append(_robust_z(row.get(feat, np.nan), train_df.loc[pass_mask, feat]))
        else:
            spc.append(float("nan"))
        s = shift_map.get(feat)
        drift_shift.append(float(s) if s is not None else float("nan"))
        drift_flag.append(bool(s is not None and abs(float(s)) > z_threshold))
        group.append(groups.get(feat, "—"))

    out = local_df.copy()
    out["spc_z"] = spc
    out["drift_shift"] = drift_shift
    out["drift_flag"] = drift_flag
    out["group"] = group
    return out


def wafer_explanation(
    model_id: str,
    pipeline,
    tuned: dict,
    observation_id: object,
    track: str = DEFAULT_TRACK,
) -> WaferExplanation | None:
    split = load_holdout_split(track)
    test_df = split.test_df
    if observation_id not in test_df[ID_COL].values:
        return None

    mask = test_df[ID_COL] == observation_id
    row = test_df.loc[mask].iloc[0]
    X_row = split.X_test.loc[mask]
    y_true = int(row[TARGET_COL])

    fail_proba = float(pipeline.predict_proba(X_row)[0, 1])
    threshold = _deploy_threshold(tuned)
    y_pred = int(predict_with_threshold(np.array([fail_proba]), threshold)[0])

    kind = model_info(model_id).explainability
    X_scaled, names = scaled_matrix(pipeline, X_row)
    row_scaled = X_scaled[0]
    pls = is_pls_model(model_id)

    coefs: np.ndarray | None = None
    if kind == "linear":
        coefs = np.asarray(fitted_base_classifier(pipeline).coef_, dtype=float).ravel()
        contrib_full = coefs * row_scaled
        base_method = "coefficient × scaled feature value"
    elif kind == "bayesian":
        coefs = np.asarray(_bayes_mean_coef(pipeline), dtype=float).ravel()
        contrib_full = coefs * row_scaled
        base_method = "posterior-mean coefficient × scaled feature value"
    elif kind == "tree":
        contrib_full = _shap_contribution_vector(pipeline, row_scaled, len(names))
        base_method = "SHAP values for this wafer (TreeExplainer)"
    else:
        raise ValueError(model_id)

    if pls:
        local_df = pls_local_sensor_contributions(pipeline, contrib_full, names)
        method = f"Local: {base_method}, back-projected to sensors via PLS loadings."
    else:
        local_df = pd.DataFrame({"feature": np.asarray(names), "contribution": contrib_full})
        if coefs is not None:
            local_df["coefficient"] = coefs
        local_df["abs_contribution"] = local_df["contribution"].abs()
        local_df = local_df.nlargest(LOCAL_TOP_N, "abs_contribution")
        method = f"Local: {base_method}."

    local_df = _enrich_local_df(local_df, split, row, track)

    interval, uncal_mean = (None, None)
    if kind == "bayesian":
        interval, uncal_mean = _bayes_fail_interval(pipeline, row_scaled)

    return WaferExplanation(
        observation_id=observation_id,
        actual_label=y_true,
        predicted_label=y_pred,
        fail_probability=fail_proba,
        threshold=threshold,
        local_df=local_df,
        method=method,
        fail_probability_interval=interval,
        fail_probability_uncalibrated=uncal_mean,
    )


@st.cache_data(show_spinner=False)
def holdout_wafer_ids(track: str = DEFAULT_TRACK) -> list:
    split = load_holdout_split(track)
    ids = split.test_df[ID_COL].tolist()
    return sorted(ids, key=str)


# Outcome taxonomy for the color-coded wafer picker.
OUTCOME_ORDER = {"missed_fail": 0, "false_alarm": 1, "caught_fail": 2, "correct_pass": 3}
OUTCOME_LABELS = {
    "missed_fail": "missed fail",
    "false_alarm": "false alarm",
    "caught_fail": "caught fail",
    "correct_pass": "correct pass",
}
OUTCOME_EMOJI = {
    "missed_fail": "🔴",
    "false_alarm": "🟠",
    "caught_fail": "🟢",
    "correct_pass": "⚪",
}


def _classify_outcome(actual: int, predicted: int) -> str:
    if actual == 1:
        return "caught_fail" if predicted == 1 else "missed_fail"
    return "false_alarm" if predicted == 1 else "correct_pass"


@st.cache_data(show_spinner="Scoring holdout wafers…")
def cached_holdout_outcomes(model_id: str, track: str = DEFAULT_TRACK) -> pd.DataFrame:
    """Per-wafer verdict for the champion: actual/predicted/proba/outcome.

    Used to color-code the wafer picker so misses (false negatives) and false
    alarms surface first. Errors are sorted ahead of correct calls.
    """
    pipeline, tuned = fit_holdout_pipeline(model_id, track)
    split = load_holdout_split(track)
    proba = np.asarray(pipeline.predict_proba(split.X_test)[:, 1], dtype=float)
    threshold = _deploy_threshold(tuned)
    pred = np.asarray(predict_with_threshold(proba, threshold), dtype=int)
    actual = split.test_df[TARGET_COL].astype(int).to_numpy()
    ids = split.test_df[ID_COL].to_numpy()
    outcome = [_classify_outcome(int(a), int(p)) for a, p in zip(actual, pred)]
    df = pd.DataFrame(
        {
            "observation_id": ids,
            "actual": actual,
            "predicted": pred,
            "proba": proba,
            "outcome": outcome,
        }
    )
    df["_order"] = df["outcome"].map(OUTCOME_ORDER)
    df = (
        df.sort_values(["_order", "proba"], ascending=[True, False])
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    return df


def key_findings(
    result: "WaferExplanation", local_df: pd.DataFrame, track: str
) -> dict:
    """Deterministic RCA headline facts from a wafer explanation + enriched local_df."""
    actual = "Fail" if result.actual_label == 1 else "Pass"
    predicted = "Fail" if result.predicted_label == 1 else "Pass"
    outcome = _classify_outcome(result.actual_label, result.predicted_label)
    findings: dict = {
        "p_fail": result.fail_probability,
        "threshold": result.threshold,
        "actual": actual,
        "predicted": predicted,
        "outcome": outcome,
        "outcome_label": OUTCOME_LABELS[outcome],
        "correct": result.actual_label == result.predicted_label,
    }
    if local_df is not None and not local_df.empty:
        top = local_df.loc[local_df["contribution"].abs().idxmax()]
        findings["top_driver"] = str(top["feature"])
        findings["top_direction"] = (
            "toward Fail" if float(top["contribution"]) > 0 else "toward Pass"
        )
        findings["top_spc_z"] = float(top.get("spc_z", float("nan")))
        findings["top_drifting"] = bool(top.get("drift_flag", False))
        n = len(local_df)
        findings["n_contributors"] = n
        flags = local_df.get("drift_flag")
        findings["n_drifting"] = (
            int(flags.fillna(False).astype(bool).sum()) if flags is not None else 0
        )
        if "robust" in local_df.columns:
            findings["n_robust"] = int(
                local_df["robust"].fillna(False).astype(bool).sum()
            )
    return findings


@st.cache_data(show_spinner=False)
def cached_global_importance(
    model_id: str, track: str = DEFAULT_TRACK
) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    """Read the frozen global-importance payload written by ``secom.benchmark``."""
    entry = report_entry(track, model_id)
    importance = (entry or {}).get("global_importance")
    if not importance:
        raise FileNotFoundError(
            f"No frozen global importance for {model_id!r} ({track}). "
            "Run: python -m secom.benchmark"
        )
    top = pd.DataFrame(importance.get("top") or [])
    signed_records = importance.get("signed")
    signed = pd.DataFrame(signed_records) if signed_records else None
    return top, signed, str(importance.get("caption", ""))


@st.cache_data(show_spinner="Building wafer explanation…")
def cached_wafer_explanation(
    model_id: str, observation_id: object, track: str = DEFAULT_TRACK
) -> WaferExplanation | None:
    pipeline, tuned = fit_holdout_pipeline(model_id, track)
    return wafer_explanation(model_id, pipeline, tuned, observation_id, track)


def _shap_mean_abs_vector(pipeline, X_train: pd.DataFrame, n_features: int) -> np.ndarray:
    """Full mean |SHAP| importance vector over a train subsample (tree heads)."""
    import shap

    from secom.pipelines import RANDOM_SEED

    X_bg, _ = scaled_matrix(pipeline, X_train)
    if len(X_bg) > 200:
        rng = np.random.default_rng(RANDOM_SEED)
        X_bg = X_bg[rng.choice(len(X_bg), size=200, replace=False)]
    estimator = fitted_base_classifier(pipeline)
    explainer = shap.TreeExplainer(estimator)
    sv = explainer.shap_values(X_bg)
    if isinstance(sv, list):
        values = np.asarray(sv[1])
    elif np.asarray(sv).ndim == 3:
        values = np.asarray(sv)[:, :, 1]
    else:
        values = np.asarray(sv)
    return np.mean(np.abs(values), axis=0)[:n_features]


def _global_vectors(
    model_id: str, pipeline, X_train: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """(importance, names, signed_or_None) in scaled feature space for one model."""
    kind = model_info(model_id).explainability
    _, names = scaled_matrix(pipeline, X_train.iloc[:1])
    if kind == "linear":
        coefs = np.asarray(fitted_base_classifier(pipeline).coef_, dtype=float).ravel()
        return np.abs(coefs), names, coefs
    if kind == "bayesian":
        coefs = np.asarray(_bayes_mean_coef(pipeline), dtype=float).ravel()
        return np.abs(coefs), names, coefs
    if kind == "tree":
        return _shap_mean_abs_vector(pipeline, X_train, len(names)), names, None
    raise ValueError(model_id)


_PLS_COMPONENT_NOTE = {
    "linear": "Component importance = |elastic-net coefficient|.",
    "bayesian": "Component importance = |posterior-mean coefficient|.",
    "tree": "Component importance = mean |SHAP| over a train subsample.",
}


def _bayes_posterior_summary(pipeline) -> tuple[np.ndarray, ...]:
    """Posterior (mean, sd, hdi_low, hdi_high) arrays averaged over calibration copies."""
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "estimator_"):  # FixedThresholdClassifier
        classifier = classifier.estimator_
    calibrated = getattr(classifier, "calibrated_classifiers_", None)
    if calibrated:
        summaries = [c.estimator.coef_summary() for c in calibrated]
    else:
        summaries = [fitted_base_classifier(pipeline).coef_summary()]
    cols = ("mean", "sd", "hdi_low", "hdi_high")
    return tuple(
        np.mean([s[col].to_numpy(dtype=float) for s in summaries], axis=0) for col in cols
    )


@st.cache_data(show_spinner="Reading posterior credible intervals…")
def cached_bayes_hdis(model_id: str, track: str = DEFAULT_TRACK) -> pd.DataFrame | None:
    """Posterior coefficient HDIs (feature/component space) for Bayesian heads.

    Returns None for non-Bayesian models. ``robust`` flags attributions whose 95%
    HDI excludes zero (robustly nonzero vs posterior noise).
    """
    if model_info(model_id).explainability != "bayesian":
        return None
    pipeline, _ = fit_holdout_pipeline(model_id, track)
    split = load_holdout_split(track)
    _, names = scaled_matrix(pipeline, split.X_train.iloc[:1])
    mean, sd, hdi_low, hdi_high = _bayes_posterior_summary(pipeline)
    df = pd.DataFrame(
        {
            "feature": [str(n) for n in names],
            "mean": mean,
            "sd": sd,
            "hdi_low": hdi_low,
            "hdi_high": hdi_high,
        }
    )
    df["robust"] = (df["hdi_low"] > 0) | (df["hdi_high"] < 0)
    df["abs_mean"] = df["mean"].abs()
    return df


@st.cache_data(show_spinner=False)
def cached_pls_sensor_robust_map(model_id: str, track: str = DEFAULT_TRACK) -> dict:
    """Map each sensor -> robust(bool) for pls_bayes via its dominant component.

    A sensor is called robust when the component it loads onto most strongly
    (weighted by posterior |mean|) has an HDI that excludes zero. Empty for
    non-pls / non-Bayesian models.
    """
    if not (is_pls_model(model_id) and model_info(model_id).explainability == "bayesian"):
        return {}
    hdi_df = cached_bayes_hdis(model_id, track)
    if hdi_df is None or hdi_df.empty:
        return {}
    pipeline, _ = fit_holdout_pipeline(model_id, track)
    loadings, sensors = _pls_loadings(pipeline)
    n_comp = loadings.shape[1]
    robust = np.zeros(n_comp, dtype=bool)
    mean = np.zeros(n_comp, dtype=float)
    for _, r in hdi_df.iterrows():
        c = _component_index(r["feature"])
        if c is not None and 0 <= c < n_comp:
            robust[c] = bool(r["robust"])
            mean[c] = float(r["mean"])
    score = np.abs(loadings) * np.abs(mean)[None, :]
    out: dict[str, bool] = {}
    for i, s in enumerate(sensors):
        row = score[i]
        cstar = int(np.argmax(row)) if row.any() else int(np.argmax(np.abs(loadings[i])))
        out[str(s)] = bool(robust[cstar])
    return out


def _bayes_beta_draws(pipeline) -> np.ndarray:
    """Stacked posterior beta draws ``(S_total, p)`` across calibration copies."""
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "estimator_"):  # FixedThresholdClassifier
        classifier = classifier.estimator_
    calibrated = getattr(classifier, "calibrated_classifiers_", None)
    estimators = (
        [c.estimator for c in calibrated]
        if calibrated
        else [fitted_base_classifier(pipeline)]
    )
    draws = [
        np.asarray(est.posterior_["beta"], dtype=float)
        for est in estimators
        if getattr(est, "posterior_", None) is not None and "beta" in est.posterior_
    ]
    return np.vstack(draws) if draws else np.empty((0, 0))


def _split_component_draws(
    draws: np.ndarray, names: np.ndarray, n_components: int
) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    """Split per-feature draws ``(S, p)`` into component draws + aux passthrough."""
    comp = np.zeros((draws.shape[0], n_components), dtype=float)
    aux: list[tuple[str, np.ndarray]] = []
    for j, name in enumerate(names):
        c = _component_index(name)
        if c is not None and 0 <= c < n_components:
            comp[:, c] += draws[:, j]
        else:
            aux.append((str(name), draws[:, j]))
    return comp, aux


def _posterior_forest_frame(
    sensor_draws: np.ndarray,
    sensors: list[str],
    aux: list[tuple[str, np.ndarray]],
) -> pd.DataFrame:
    """Per-feature posterior mean + 95% HDI from sensor-space draws."""
    blocks = [sensor_draws]
    feats = [str(s) for s in sensors]
    if aux:
        blocks.append(np.column_stack([v for _, v in aux]))
        feats += [name for name, _ in aux]
    allm = np.column_stack(blocks)
    df = pd.DataFrame(
        {
            "feature": feats,
            "mean": allm.mean(axis=0),
            "hdi_low": np.quantile(allm, 0.025, axis=0),
            "hdi_high": np.quantile(allm, 0.975, axis=0),
        }
    )
    df["robust"] = (df["hdi_low"] > 0) | (df["hdi_high"] < 0)
    df["abs_mean"] = df["mean"].abs()
    return df


@st.cache_data(show_spinner="Sampling sensor-space posterior…")
def cached_pls_sensor_posterior(
    model_id: str, track: str = DEFAULT_TRACK, top_n: int = GLOBAL_SENSOR_TOP_N
) -> pd.DataFrame | None:
    """Global signed sensor-coefficient posterior for pls_bayes (forest plot).

    Back-projects each posterior draw's component coefficients through the PLS
    loadings (``sensor = comp @ loadings.T``) to get a *true* sensor-space
    posterior, then reports mean + 95% HDI per sensor. ``robust`` = HDI clears 0.
    None for non-pls / non-Bayesian models.
    """
    if not (is_pls_model(model_id) and model_info(model_id).explainability == "bayesian"):
        return None
    pipeline, _ = fit_holdout_pipeline(model_id, track)
    split = load_holdout_split(track)
    _, names = scaled_matrix(pipeline, split.X_train.iloc[:1])
    draws = _bayes_beta_draws(pipeline)
    if draws.size == 0:
        return None
    loadings, sensors = _pls_loadings(pipeline)
    comp_draws, _aux = _split_component_draws(draws, names, loadings.shape[1])
    sensor_draws = comp_draws @ loadings.T  # (S, n_sensors), signed
    # Sensors only: non-sensor aux features (calendar, missing flags) are excluded.
    df = _posterior_forest_frame(sensor_draws, sensors, [])
    return df.nlargest(top_n, "abs_mean").reset_index(drop=True)


@st.cache_data(show_spinner="Sampling wafer posterior…")
def wafer_sensor_posterior(
    model_id: str, observation_id: object, track: str = DEFAULT_TRACK, top_n: int = LOCAL_TOP_N
) -> pd.DataFrame | None:
    """Per-sensor contribution posterior for ONE wafer (pls_bayes forest plot).

    Contribution draw per component = ``beta_draw * scaled_score`` (the wafer's
    PLS scores are fixed), back-projected to sensors with the same sign-aware
    loading weights as ``pls_local_sensor_contributions``. None otherwise.
    """
    if not (is_pls_model(model_id) and model_info(model_id).explainability == "bayesian"):
        return None
    pipeline, _ = fit_holdout_pipeline(model_id, track)
    split = load_holdout_split(track)
    test_df = split.test_df
    if observation_id not in test_df[ID_COL].values:
        return None
    mask = test_df[ID_COL] == observation_id
    X_scaled, names = scaled_matrix(pipeline, split.X_test.loc[mask])
    row_scaled = X_scaled[0]
    draws = _bayes_beta_draws(pipeline)
    if draws.size == 0:
        return None
    loadings, sensors = _pls_loadings(pipeline)
    contrib_draws = draws * row_scaled[None, :]
    comp_contrib, _aux = _split_component_draws(contrib_draws, names, loadings.shape[1])
    col_abs = np.abs(loadings).sum(axis=0)
    col_abs[col_abs == 0] = 1.0
    weights = loadings / col_abs  # (n_sensors, n_components), sign-aware
    sensor_draws = comp_contrib @ weights.T
    # Sensors only: non-sensor aux features (calendar, missing flags) are excluded.
    df = _posterior_forest_frame(sensor_draws, sensors, [])
    return df.nlargest(top_n, "abs_mean").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def cached_era_drift(z_threshold: float = 2.0) -> dict:
    """Per-sensor training-era -> holdout-era mean shift (SD units) over the mart."""
    from secom.dashboard.stg import per_sensor_era_drift

    df = load_mart()
    cols = [
        str(c)
        for c in df.columns
        if str(c).startswith("c_") and not str(c).endswith("__missing")
    ]
    shift = per_sensor_era_drift(df, sensor_cols=cols)
    return {
        "shift": {str(k): float(v) for k, v in shift.items()},
        "z_threshold": float(z_threshold),
    }


@st.cache_data(show_spinner="Clustering sensors into data-driven groups…")
def cached_sensor_groups(track: str = DEFAULT_TRACK, n_groups: int = 8) -> dict:
    """Data-driven correlation clusters of the raw sensors (NOT real equipment).

    Hierarchical (average-linkage) clustering on ``1 - |corr|`` of the passing
    training sensors. Each ``c_N`` and its ``c_N_rz`` variant share a group id.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    split = load_holdout_split(track)
    train = split.train_df
    raw = [
        str(c)
        for c in split.feature_cols
        if str(c).startswith("c_")
        and not str(c).endswith("__missing")
        and not str(c).endswith("_rz")
    ]
    if len(raw) < 2:
        return {}
    X = train[raw].astype(float)
    X = X.fillna(X.median(numeric_only=True))
    corr = X.corr().fillna(0.0).to_numpy()
    dist = 1.0 - np.abs(corr)
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    dist[dist < 0] = 0.0
    linked = linkage(squareform(dist, checks=False), method="average")
    n_clusters = int(min(max(2, n_groups), len(raw)))
    labels = fcluster(linked, t=n_clusters, criterion="maxclust")
    mapping: dict[str, str] = {}
    for col, lab in zip(raw, labels):
        gid = f"G{int(lab)}"
        mapping[str(col)] = gid
        mapping[f"{col}_rz"] = gid
    return mapping


@st.cache_data(show_spinner="Computing sensor-space global importance…")
def cached_global_sensor_importance(
    model_id: str, track: str = DEFAULT_TRACK
) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    """Global importance in one sensor vocabulary across all 9 models.

    Selection/tree front-ends already speak sensor names, so their frozen
    importance is reused as-is. PLS front-ends emit latent components, so they
    are fit live and back-projected to sensors via the PLS loadings.
    """
    if not is_pls_model(model_id):
        return cached_global_importance(model_id, track)
    pipeline, _ = fit_holdout_pipeline(model_id, track)
    split = load_holdout_split(track)
    importance, names, signed_vec = _global_vectors(model_id, pipeline, split.X_train)
    top, signed_df = pls_global_sensor_importance(pipeline, importance, names, signed_vec)
    kind = model_info(model_id).explainability
    caption = (
        "Sensor-space importance back-projected from PLS latent components via the "
        "fitted PLSRegression loadings (linear approximation). "
        + _PLS_COMPONENT_NOTE.get(kind, "")
    )
    return top, signed_df, caption
