"""Stage 1+2 representation for the Bayesian extrapolation models.

Three front-ends produce a stable, named design matrix from the rolling-Z sensor
columns:

- ``hsic`` : HSIC-Lasso nonlinear screening of K sensors.
- ``rf``   : random-forest importance screening of K sensors.
- ``spls`` : supervised PLS aggregation into K components (no interactions).

For ``hsic``/``rf`` the top ``n_hubs`` selected sensors are expanded into
pairwise products + squares (the physical interaction frame), so features keep
their sensor identity across Leave-Future-Out windows (no basis rotation).
Sensors are robustly scaled (median/IQR) before screening to match the interp
track and resist SECOM's outliers.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from secom.core import _SENSOR_VALUE_PATTERN_EXTRAP, RANDOM_SEED, RF_N_ESTIMATORS
from secom.hub_interactions import (
    PLSFeatures,
    _interaction_column_name,
    _interaction_frame,
)

VALID_METHODS = ("hsic", "rf", "spls")


def extrap_sensor_columns(columns) -> list[str]:
    """rolling-Z sensor columns only (``c_<id>_rz``); falls back to raw if absent."""
    import re

    pat = re.compile(_SENSOR_VALUE_PATTERN_EXTRAP)
    rz = [c for c in map(str, columns) if c.endswith("_rz") and pat.match(c)]
    if rz:
        return rz
    return [c for c in map(str, columns) if pat.match(c)]


def _ensure_numpy_linalg_compat() -> None:
    """pyHSICLasso's nlars references the NumPy<2 private alias ``np.linalg.linalg``.

    NumPy 2.x removed it, which turns nlars's singular-matrix recovery
    (``except np.linalg.linalg.LinAlgError``) into an ``AttributeError``. Restore
    the alias so the library's own noise-injection fallback works.
    """
    if not hasattr(np.linalg, "linalg"):
        np.linalg.linalg = np.linalg  # type: ignore[attr-defined]


def _select_hsic(X: np.ndarray, y: np.ndarray, k: int) -> list[int]:
    import contextlib
    import io
    import warnings

    from pyHSICLasso import HSICLasso

    _ensure_numpy_linalg_compat()
    hl = HSICLasso()
    hl.input(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
    # Block HSIC Lasso (library default B/M): exact HSIC is O(n^2) per feature and
    # far too slow across the tuning grid. redirect_stdout swallows the library's
    # progress prints; catch_warnings drops the "B must be an exact divisor" notice.
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hl.classification(int(k))
    idx = [int(i) for i in hl.get_index()]
    return idx[:k]


def _select_rf(X: np.ndarray, y: np.ndarray, k: int) -> list[int]:
    """Top-k sensors by random-forest impurity importance (mirrors intrap_topk_rf).

    Fixed ``random_state``/``n_estimators`` keep the selected set stable across
    Leave-Future-Out windows so the downstream features don't rotate.
    """
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
    order = np.argsort(rf.feature_importances_)[::-1]
    return [int(i) for i in order[:k]]


class ExtrapRepresentation:
    """Fit-on-train screening/aggregation + interaction frame -> design matrix."""

    def __init__(self, method: str = "hsic", *, k: int = 20, n_hubs: int = 5):
        method = str(method).lower()
        if method not in VALID_METHODS:
            raise ValueError(f"method must be one of {VALID_METHODS}, got {method!r}")
        self.method = method
        self.k = int(k)
        self.n_hubs = int(n_hubs)

    def fit(self, X: pd.DataFrame, y) -> "ExtrapRepresentation":
        cols = extrap_sensor_columns(X.columns)
        if not cols:
            raise ValueError("No sensor columns found for ExtrapRepresentation")
        self.sensor_cols_ = cols
        mat = X[cols].to_numpy(dtype=float)

        self.medians_ = np.nanmedian(mat, axis=0)
        self.medians_ = np.where(np.isnan(self.medians_), 0.0, self.medians_)
        mat = self._impute(mat)
        # Robust (median/IQR) scaling: matches the interp track and resists the
        # outliers/spikes the plain mean/std z-score over-weighted.
        self.scaler_ = RobustScaler().fit(mat)
        mat = self.scaler_.transform(mat)

        y_arr = np.asarray(y, dtype=int)
        if self.method == "spls":
            k = max(1, min(self.k, mat.shape[1], mat.shape[0] - 1))
            self.pls_ = PLSFeatures(n_components=k).fit(
                pd.DataFrame(mat, columns=cols), y_arr
            )
            self.output_names_ = list(self.pls_.output_names_)
            self.selected_cols_ = []
            self.hub_cols_ = []
        else:
            k = max(1, min(self.k, mat.shape[1]))
            if self.method == "hsic":
                sel = _select_hsic(mat, y_arr, k)
            else:
                sel = _select_rf(mat, y_arr, k)
            if not sel:
                sel = list(range(min(k, mat.shape[1])))
            self.selected_idx_ = sel
            self.selected_cols_ = [cols[i] for i in sel]
            self.hub_cols_ = self.selected_cols_[: max(0, self.n_hubs)]
            self.pair_list_ = list(combinations(self.hub_cols_, 2))
            self.square_cols_ = [f"sq_{c}" for c in self.hub_cols_]
            self.pair_cols_ = [
                _interaction_column_name(a, b) for a, b in self.pair_list_
            ]
            self.output_names_ = (
                list(self.selected_cols_)
                + list(self.square_cols_)
                + list(self.pair_cols_)
            )

        # Standardize the *full* design (main effects + squares + pairs, or PLS
        # scores) so every column is unit-variance and the head's single shared
        # elastic-net prior is scale-coherent. Squares/products are otherwise on
        # a different (outlier-fattened) scale than the linear terms.
        design = self._build_design(X)
        self.design_scaler_ = StandardScaler().fit(design.to_numpy(dtype=float))
        return self

    def _build_design(self, X: pd.DataFrame) -> pd.DataFrame:
        """Unscaled design (sensor scaling + interaction frame), before StandardScaler."""
        mat = X[self.sensor_cols_].to_numpy(dtype=float)
        mat = self._impute(mat)
        mat = self.scaler_.transform(mat)
        std_df = pd.DataFrame(mat, columns=self.sensor_cols_, index=X.index)

        if self.method == "spls":
            scores = self.pls_.transform(std_df)
            return pd.DataFrame(
                np.asarray(scores, dtype=float),
                columns=self.output_names_,
                index=X.index,
            )

        sel_df = std_df[self.selected_cols_]
        parts = [sel_df]
        if self.hub_cols_:
            squares = pd.DataFrame(
                {f"sq_{c}": std_df[c].to_numpy(dtype=float) ** 2 for c in self.hub_cols_},
                index=X.index,
            )
            parts.append(squares)
        if self.pair_list_:
            parts.append(_interaction_frame(std_df, self.pair_list_))
        design = pd.concat(parts, axis=1)
        return design[self.output_names_]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        design = self._build_design(X)
        scaled = self.design_scaler_.transform(design.to_numpy(dtype=float))
        return pd.DataFrame(scaled, columns=self.output_names_, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def _impute(self, mat: np.ndarray) -> np.ndarray:
        out = mat.copy()
        inds = np.where(np.isnan(out))
        if inds[0].size:
            out[inds] = np.take(self.medians_, inds[1])
        return out

    def config(self) -> dict:
        return {
            "method": self.method,
            "k": self.k,
            "n_hubs": self.n_hubs,
            "n_selected": len(getattr(self, "selected_cols_", [])),
            "n_design_features": len(getattr(self, "output_names_", [])),
            "selected_sensors": list(getattr(self, "selected_cols_", [])),
            "hub_sensors": list(getattr(self, "hub_cols_", [])),
        }
