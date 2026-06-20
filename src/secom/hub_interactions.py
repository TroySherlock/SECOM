"""RF hub interaction features and Hotelling T² for the shared sensor preprocess path."""
from __future__ import annotations

import re
from itertools import combinations
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.validation import check_is_fitted

SENSOR_COLUMN_PATTERN = re.compile(r"^c_\d+$")
DEFAULT_T2_COL = "hotelling_t2"


def sensor_value_columns(columns: Iterable[str]) -> list[str]:
    """Return raw sensor value columns (c_<id>), excluding __missing flags."""
    return [col for col in columns if SENSOR_COLUMN_PATTERN.fullmatch(str(col))]


def _columns_matching(columns: Iterable[str], pattern: re.Pattern[str]) -> list[str]:
    return [col for col in columns if pattern.fullmatch(str(col))]


def _ensure_numpy_linalg_compat() -> None:
    """pyHSICLasso's nlars references the NumPy<2 private alias ``np.linalg.linalg``.

    NumPy 2.x removed it, which turns nlars's singular-matrix recovery
    (``except np.linalg.linalg.LinAlgError``) into an ``AttributeError``. Restore
    the alias so the library's own noise-injection fallback works.
    """
    if not hasattr(np.linalg, "linalg"):
        np.linalg.linalg = np.linalg  # type: ignore[attr-defined]


def _select_hsic(X: np.ndarray, y: np.ndarray, k: int) -> list[int]:
    """Top-k feature indices by Block HSIC-Lasso nonlinear screening (ranked)."""
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


class MahalanobisT2Features(BaseEstimator, TransformerMixin):
    """Hotelling T² via LedoitWolf (or EmpiricalCovariance) squared Mahalanobis distance."""

    def __init__(
        self,
        score_columns: str | None = None,
        t2_col: str = DEFAULT_T2_COL,
        shrinkage: bool = True,
    ):
        self.score_columns = score_columns
        self.t2_col = t2_col
        self.shrinkage = shrinkage

    def fit(self, X, y=None):
        X_df = self._as_dataframe(X)
        self.feature_names_in_ = list(X_df.columns)
        cols = self._t2_columns(X_df)
        X_arr = X_df[cols].to_numpy(dtype=float) if cols else np.empty((len(X_df), 0))
        n_features = X_arr.shape[1]
        n_samples = X_arr.shape[0]
        self.t2_columns_ = cols

        if n_features == 0 or n_samples == 0:
            self.cov_ = None
            return self

        self.cov_ = self._fit_covariance(X_arr)
        return self

    def transform(self, X):
        check_is_fitted(self, "t2_columns_")
        X_df = self._as_dataframe(X)
        index = X_df.index
        n_rows = len(X_df)

        if self.cov_ is None or not self.t2_columns_:
            t2 = np.zeros(n_rows, dtype=float)
        else:
            X_arr = X_df[self.t2_columns_].to_numpy(dtype=float)
            t2 = self.cov_.mahalanobis(X_arr)

        t2_df = pd.DataFrame({self.t2_col: t2}, index=index)
        return pd.concat([X_df, t2_df], axis=1)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "t2_columns_")
        base = (
            list(input_features)
            if input_features is not None
            else list(self.feature_names_in_)
        )
        return np.asarray(base + [self.t2_col], dtype=object)

    def _t2_columns(self, X_df: pd.DataFrame) -> list[str]:
        if self.score_columns is not None:
            pattern = re.compile(self.score_columns)
            return _columns_matching(X_df.columns, pattern)
        cols = sensor_value_columns(X_df.columns)
        return cols if cols else list(X_df.columns)

    def _fit_covariance(self, X_arr: np.ndarray):
        try:
            if self.shrinkage:
                return LedoitWolf().fit(X_arr)
            return EmpiricalCovariance().fit(X_arr)
        except np.linalg.LinAlgError:
            return LedoitWolf().fit(X_arr)

    @staticmethod
    def _as_dataframe(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)
        df.columns = df.columns.astype(str)
        return df


class PLSFeatures(BaseEstimator, TransformerMixin):
    """PLS (PLS-DA) reduction: emit X-scores ``pls_0..pls_{k-1}`` as a frame.

    Wraps ``PLSRegression`` so the latent components flow through the pandas
    ColumnTransformer with stable names. ``fit`` consumes the binary target
    (supervised reduction); ``transform`` returns only the X-side scores.
    """

    def __init__(self, n_components: int = 10):
        self.n_components = n_components

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("PLSFeatures requires y (supervised PLS reduction)")
        X_df = self._as_dataframe(X)
        self.feature_names_in_ = list(X_df.columns)
        n_samples, n_features = X_df.shape
        k = max(1, min(int(self.n_components), n_features, n_samples - 1))
        self.n_components_ = int(k)
        self.pls_ = PLSRegression(n_components=self.n_components_, scale=True)
        self.pls_.fit(
            X_df.to_numpy(dtype=float),
            np.asarray(y, dtype=float),
        )
        self.output_names_ = [f"pls_{i}" for i in range(self.n_components_)]
        return self

    def transform(self, X):
        check_is_fitted(self, "pls_")
        X_df = self._as_dataframe(X)
        scores = self.pls_.transform(X_df.to_numpy(dtype=float))
        scores = np.asarray(scores, dtype=float).reshape(len(X_df), -1)
        return pd.DataFrame(scores, index=X_df.index, columns=self.output_names_)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "pls_")
        return np.asarray(self.output_names_, dtype=object)

    @staticmethod
    def _as_dataframe(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)
        df.columns = df.columns.astype(str)
        return df


def _interaction_column_name(a: str, b: str) -> str:
    return f"interact_{a}__{b}"


def _hub_sensors_from_select(select: SelectFromModel, n_hubs: int) -> list[str]:
    support = select.get_support()
    names_in = list(select.feature_names_in_)
    importances = select.estimator_.feature_importances_
    ranked = [
        (str(name), float(imp))
        for name, keep, imp in zip(names_in, support, importances)
        if keep
    ]
    ranked.sort(key=lambda x: x[1], reverse=True)
    n = min(int(n_hubs), len(ranked))
    if n <= 0:
        return []
    return [name for name, _ in ranked[:n]]


def _interaction_frame(X_sel: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    if not pairs:
        return pd.DataFrame(index=X_sel.index)
    names = [_interaction_column_name(a, b) for a, b in pairs]
    values = np.column_stack(
        [
            X_sel[a].to_numpy(dtype=float) * X_sel[b].to_numpy(dtype=float)
            for a, b in pairs
        ]
    )
    return pd.DataFrame(values, index=X_sel.index, columns=names)


class LinearSelectT2HubBlock(BaseEstimator, TransformerMixin):
    """RF top-k selection, Hotelling T², and hub pair interactions."""

    def __init__(
        self,
        top_k: int = 15,
        n_hubs: int = 8,
    ):
        self.top_k = top_k
        self.n_hubs = n_hubs

    def fit(self, X, y=None):
        from secom.pipelines import random_forest_classifier

        X_df = self._as_dataframe(X)
        self.feature_names_in_ = list(X_df.columns)

        self.select_ = SelectFromModel(
            random_forest_classifier(),
            max_features=int(self.top_k),
            threshold=-np.inf,
        )
        self.select_.fit(X_df, y)
        X_sel = self._selected_dataframe(X_df)

        self.t2_ = MahalanobisT2Features()
        self.t2_.fit(X_sel, y)

        self.hubs_ = _hub_sensors_from_select(self.select_, self.n_hubs)
        self.interaction_pairs_ = list(combinations(self.hubs_, 2))
        self.interaction_names_ = [
            _interaction_column_name(a, b) for a, b in self.interaction_pairs_
        ]
        self.n_interaction_features_ = len(self.interaction_names_)
        return self

    def transform(self, X):
        check_is_fitted(self, "select_")
        X_df = self._as_dataframe(X)
        X_sel = self._selected_dataframe(X_df)

        t2_df = self.t2_.transform(X_sel)[[DEFAULT_T2_COL]]
        parts: list[pd.DataFrame] = [X_sel, t2_df]

        if self.interaction_pairs_:
            interact_df = _interaction_frame(X_sel, self.interaction_pairs_)
            parts.append(interact_df)

        return pd.concat(parts, axis=1)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "t2_")
        names: list[str] = list(self._selected_sensor_columns())
        names.append(DEFAULT_T2_COL)
        names.extend(list(self.interaction_names_))
        return np.asarray(names, dtype=object)

    def _selected_sensor_columns(self) -> list[str]:
        check_is_fitted(self, "select_")
        support = self.select_.get_support()
        names_in = list(self.select_.feature_names_in_)
        return [str(name) for name, keep in zip(names_in, support) if keep]

    def _selected_dataframe(self, X_df: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "select_")
        cols = self._selected_sensor_columns()
        X_sel = self.select_.transform(X_df)
        if isinstance(X_sel, pd.DataFrame):
            out = X_sel.copy()
            if list(out.columns) != cols:
                out.columns = cols
            return out
        return pd.DataFrame(X_sel, index=X_df.index, columns=cols)

    @staticmethod
    def _as_dataframe(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)
        df.columns = df.columns.astype(str)
        return df


class HSICSelectHubBlock(BaseEstimator, TransformerMixin):
    """HSIC-Lasso top-k selection, Hotelling T2, and hub pair interactions.

    The selection-front-end analogue of ``LinearSelectT2HubBlock`` but with
    nonlinear Block HSIC-Lasso screening instead of RF impurity. Fits on the
    post-cluster matrix (already reduced by ``SmartCorrelatedSelection``).
    """

    def __init__(
        self,
        top_k: int = 15,
        n_hubs: int = 8,
    ):
        self.top_k = top_k
        self.n_hubs = n_hubs

    def fit(self, X, y=None):
        X_df = self._as_dataframe(X)
        self.feature_names_in_ = list(X_df.columns)
        cols = list(X_df.columns)
        k = max(1, min(int(self.top_k), len(cols)))
        idx = _select_hsic(X_df.to_numpy(dtype=float), np.asarray(y, dtype=int), k)
        if not idx:
            idx = list(range(k))
        self.selected_columns_ = [str(cols[i]) for i in idx]
        X_sel = X_df[self.selected_columns_]

        self.t2_ = MahalanobisT2Features()
        self.t2_.fit(X_sel, y)

        n = max(0, int(self.n_hubs))
        self.hubs_ = list(self.selected_columns_[:n])
        self.interaction_pairs_ = list(combinations(self.hubs_, 2))
        self.interaction_names_ = [
            _interaction_column_name(a, b) for a, b in self.interaction_pairs_
        ]
        self.n_interaction_features_ = len(self.interaction_names_)
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_columns_")
        X_df = self._as_dataframe(X)
        X_sel = X_df[self.selected_columns_]

        t2_df = self.t2_.transform(X_sel)[[DEFAULT_T2_COL]]
        parts: list[pd.DataFrame] = [X_sel, t2_df]
        if self.interaction_pairs_:
            parts.append(_interaction_frame(X_sel, self.interaction_pairs_))
        return pd.concat(parts, axis=1)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "t2_")
        names: list[str] = list(self.selected_columns_)
        names.append(DEFAULT_T2_COL)
        names.extend(list(self.interaction_names_))
        return np.asarray(names, dtype=object)

    @staticmethod
    def _as_dataframe(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)
        df.columns = df.columns.astype(str)
        return df


def extract_hub_interaction_info(block: LinearSelectT2HubBlock) -> dict[str, Any]:
    """Artifact summary from a fitted LinearSelectT2HubBlock."""
    check_is_fitted(block, "select_")
    hubs = [str(h) for h in getattr(block, "hubs_", [])]
    pairs = getattr(block, "interaction_pairs_", list(combinations(hubs, 2)))
    interaction_pairs = [
        {"a": a, "b": b, "column": _interaction_column_name(a, b)}
        for a, b in pairs
    ]
    return {
        "top_k_requested": int(block.top_k),
        "n_hubs_requested": int(block.n_hubs),
        "n_hubs_selected": len(hubs),
        "n_interaction_features": int(getattr(block, "n_interaction_features_", 0)),
        "hub_sensors": hubs,
        "interaction_pairs": interaction_pairs,
    }
