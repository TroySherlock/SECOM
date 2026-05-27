"""PLS and Mahalanobis MSPC feature transformers."""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_is_fitted

SENSOR_COLUMN_PATTERN = re.compile(r"^c_\d+$")
PLS_SCORE_COLUMN_PATTERN = re.compile(r"^pls_\d+$")

DEFAULT_T2_COL = "hotelling_t2"
DEFAULT_Q_COL = "q_statistic"


def sensor_value_columns(columns: Iterable[str]) -> list[str]:
    """Return raw sensor value columns (c_<id>), excluding __missing flags."""
    return [col for col in columns if SENSOR_COLUMN_PATTERN.fullmatch(str(col))]


def _columns_matching(columns: Iterable[str], pattern: re.Pattern[str]) -> list[str]:
    return [col for col in columns if pattern.fullmatch(str(col))]


class PLSWithQFeatures(BaseEstimator, TransformerMixin):
    """Fit PLS on c_* sensors; emit pls_* scores and q_statistic (PLS SPE)."""

    def __init__(
        self,
        n_components: int = 20,
        scale: bool = True,
        q_col: str = DEFAULT_Q_COL,
    ):
        self.n_components = n_components
        self.scale = scale
        self.q_col = q_col

    def fit(self, X, y=None):
        X_df = self._as_dataframe(X)
        self.feature_names_in_ = list(X_df.columns)
        X_arr = self._sensor_array(X)
        n_features = X_arr.shape[1]
        n_samples = X_arr.shape[0]
        if n_features == 0 or n_samples == 0:
            self.n_components_ = 0
            self.pls_ = None
            return self

        if y is None:
            raise ValueError("PLSWithQFeatures.fit requires target y.")

        y_arr = np.asarray(y, dtype=float).reshape(-1, 1)
        max_components = min(n_features, n_samples - 1)
        if max_components < 1:
            self.n_components_ = 0
            self.pls_ = None
            return self

        n_comp = min(int(self.n_components), max_components)
        self.pls_ = PLSRegression(n_components=n_comp, scale=self.scale)
        self.pls_.fit(X_arr, y_arr)
        self.n_components_ = int(self.pls_.n_components)
        return self

    def transform(self, X):
        check_is_fitted(self, "n_components_")
        X_df = self._as_dataframe(X)
        index = X_df.index
        n_rows = len(X_df)
        if self.n_components_ == 0 or self.pls_ is None:
            return pd.DataFrame(
                {self.q_col: np.zeros(n_rows, dtype=float)},
                index=index,
            )

        X_arr = self._sensor_array(X)
        scores = self.pls_.transform(X_arr)
        reconstructed = self.pls_.inverse_transform(scores)
        residuals = X_arr - reconstructed

        out = {f"pls_{i}": scores[:, i] for i in range(self.n_components_)}
        out[self.q_col] = np.sum(residuals**2, axis=1)
        return pd.DataFrame(out, index=index)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "n_components_")
        names = [f"pls_{i}" for i in range(self.n_components_)]
        names.append(self.q_col)
        return np.asarray(names, dtype=object)

    def explained_variance_ratio(self) -> list[float]:
        """Per-component fraction of X variance explained (for reporting)."""
        check_is_fitted(self, "n_components_")
        if self.n_components_ == 0 or self.pls_ is None:
            return []
        scores = self.pls_.x_scores_
        total = float(np.sum(scores**2))
        if total <= 0:
            return [0.0] * self.n_components_
        return [float(np.sum(scores[:, i] ** 2) / total) for i in range(self.n_components_)]

    def _sensor_array(self, X) -> np.ndarray:
        X_df = self._as_dataframe(X)
        cols = sensor_value_columns(X_df.columns)
        if not cols:
            return np.empty((len(X_df), 0), dtype=float)
        return X_df[cols].to_numpy(dtype=float)

    @staticmethod
    def _as_dataframe(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)
        df.columns = df.columns.astype(str)
        return df


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


# Backward-compatible alias
PLSProcessControlFeatures = PLSWithQFeatures
