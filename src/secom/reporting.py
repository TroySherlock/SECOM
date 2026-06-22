"""Pure (no-Streamlit) report computations frozen into the benchmark cache.

These power the dashboard's PR-curve and global-importance views. They run once
offline inside ``secom.benchmark`` and the results are persisted to
``secom_report_cache.json``; the dashboard only reads that cache.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from secom.dashboard.data import model_info
from secom.metrics import pr_curve_points, threshold_pr_point
from secom.pipelines import RANDOM_SEED
from secom.tuning.registry import fit_pipeline_weighted
from secom.utils import fitted_base_classifier

GLOBAL_TOP_N = 15
SHAP_BACKGROUND_ROWS = 200


# --- Cross-validated out-of-fold probabilities -------------------------------
def collect_cv_oof_proba(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv,
) -> tuple[pd.Series, np.ndarray]:
    """Out-of-fold positive-class probabilities over ``cv`` (in-distribution)."""
    proba = np.full(len(y_train), np.nan)
    for train_idx, val_idx in cv.split(X_train, y_train):
        fold_pipe, _ = fit_pipeline_weighted(
            clone(pipeline),
            X_train.iloc[train_idx],
            y_train.iloc[train_idx],
            None,
        )
        proba[val_idx] = fold_pipe.predict_proba(X_train.iloc[val_idx])[:, 1]
    mask = ~np.isnan(proba)
    y_oof = y_train.iloc[mask].reset_index(drop=True)
    return y_oof, np.asarray(proba[mask], dtype=float)


# --- PR-curve payload --------------------------------------------------------
def pr_curve_payload(
    y_cv: pd.Series | None,
    score_cv: np.ndarray | None,
    y_ho: pd.Series,
    score_ho: np.ndarray,
    ber_threshold: float | None = None,
) -> dict:
    """JSON-serializable CV + holdout PR curves and a BER operating point.

    ``y_cv``/``score_cv`` may be ``None`` (e.g. the temporal track has no CV), in
    which case the ``cv`` branch is empty and only the holdout curve is stored.
    """
    if y_cv is None or score_cv is None or len(y_cv) == 0:
        cv_block = {"recall": [], "precision": [], "baseline": 0.0}
    else:
        cv = pr_curve_points(y_cv, score_cv)
        cv_block = {
            "recall": cv.recall.tolist(),
            "precision": cv.precision.tolist(),
            "baseline": float(cv.baseline),
        }
    ho = pr_curve_points(y_ho, score_ho)
    ber_point = None
    if ber_threshold is not None:
        recall, precision = threshold_pr_point(y_ho, score_ho, float(ber_threshold))
        ber_point = [float(recall), float(precision)]
    return {
        "cv": cv_block,
        "holdout": {
            "recall": ho.recall.tolist(),
            "precision": ho.precision.tolist(),
            "baseline": float(ho.baseline),
        },
        "ber_point": ber_point,
    }


# --- Global feature importance ----------------------------------------------
def scaled_matrix(pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Transform to scaled feature space; return (matrix, feature_names)."""
    preprocess = pipeline.named_steps["preprocess"]
    scale = pipeline.named_steps["scale"]
    X_pre = preprocess.transform(X)
    X_scaled = scale.transform(X_pre)
    names = scale.get_feature_names_out()
    return np.asarray(X_scaled, dtype=float), np.asarray(names, dtype=object)


def _bayes_mean_coef(pipeline) -> np.ndarray:
    """Posterior-mean elastic-net coefficients averaged over the calibration copies."""
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "estimator_"):  # FixedThresholdClassifier
        classifier = classifier.estimator_
    calibrated = getattr(classifier, "calibrated_classifiers_", None)
    if calibrated:
        means = [c.estimator.coef_summary()["mean"].to_numpy() for c in calibrated]
        return np.mean(means, axis=0)
    return fitted_base_classifier(pipeline).coef_summary()["mean"].to_numpy()


def _coef_tables(names, coefs) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = pd.DataFrame({"feature": names, "coefficient": np.asarray(coefs).ravel()})
    full["abs_coefficient"] = np.abs(full["coefficient"])
    top = full.nlargest(GLOBAL_TOP_N, "abs_coefficient").copy()
    top["importance"] = top["abs_coefficient"]
    pos = full.loc[full["coefficient"] > 0].nlargest(8, "coefficient")
    neg = full.loc[full["coefficient"] < 0].nsmallest(8, "coefficient")
    signed = pd.concat([pos, neg], ignore_index=True)
    return top, signed


def global_importance_linear(pipeline, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top |coef| table and signed coef subset for chart (elastic-net LR)."""
    _, names = scaled_matrix(pipeline, X.iloc[:1])
    coefs = fitted_base_classifier(pipeline).coef_.ravel()
    return _coef_tables(names, coefs)


def global_importance_bayesian(pipeline, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Posterior-mean coefficient table + signed subset for the Bayesian head."""
    _, names = scaled_matrix(pipeline, X.iloc[:1])
    coefs = _bayes_mean_coef(pipeline)
    return _coef_tables(names, coefs)


def global_importance_shap(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    import shap

    X_bg, names = scaled_matrix(pipeline, X)
    if len(X_bg) > SHAP_BACKGROUND_ROWS:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(len(X_bg), size=SHAP_BACKGROUND_ROWS, replace=False)
        X_bg = X_bg[idx]

    estimator = fitted_base_classifier(pipeline)
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_bg)
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[1])
    elif np.asarray(shap_values).ndim == 3:
        values = np.asarray(shap_values)[:, :, 1]
    else:
        values = np.asarray(shap_values)

    mean_abs = np.mean(np.abs(values), axis=0)
    full = pd.DataFrame({"feature": names, "importance": mean_abs})
    return full.nlargest(GLOBAL_TOP_N, "importance")


def global_importance(
    model_id: str, pipeline, X: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    """Returns (top_df, optional_signed_coef_df, method_caption).

    ``X`` is the (track-appropriate) train design used for feature names and the
    SHAP background sample.
    """
    kind = model_info(model_id).explainability
    if kind == "bayesian":
        top, signed = global_importance_bayesian(pipeline, X)
        return (
            top,
            signed,
            "Global view uses posterior-mean elastic-net coefficients (averaged over "
            "the calibration copies) on the scaled design (Laplace L1 + ridge L2 priors).",
        )
    if kind == "linear":
        top, signed = global_importance_linear(pipeline, X)
        return (
            top,
            signed,
            "Global view uses elastic-net coefficients on scaled features "
            "(underlying logistic inside shared pipeline calibration).",
        )
    if kind == "tree":
        top = global_importance_shap(pipeline, X)
        return top, None, "Global view uses mean |SHAP| from TreeExplainer on a train subsample."
    raise ValueError(model_id)


def compute_global_importance(
    model_id: str, fitted_pipeline, X_train: pd.DataFrame
) -> dict:
    """JSON-serializable global importance for the report cache."""
    top, signed, caption = global_importance(model_id, fitted_pipeline, X_train)
    return {
        "top": top.to_dict(orient="records"),
        "signed": signed.to_dict(orient="records") if signed is not None else None,
        "caption": caption,
    }
