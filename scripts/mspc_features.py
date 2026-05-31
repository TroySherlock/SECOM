"""Mahalanobis Hotelling T² feature transformer for SECOM sensor paths."""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.utils.validation import check_is_fitted

SENSOR_COLUMN_PATTERN = re.compile(r"^c_\d+$")

DEFAULT_T2_COL = "hotelling_t2"


def sensor_value_columns(columns: Iterable[str]) -> list[str]:
    """Return raw sensor value columns (c_<id>), excluding __missing flags."""
    return [col for col in columns if SENSOR_COLUMN_PATTERN.fullmatch(str(col))]


def _columns_matching(columns: Iterable[str], pattern: re.Pattern[str]) -> list[str]:
    return [col for col in columns if pattern.fullmatch(str(col))]


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
