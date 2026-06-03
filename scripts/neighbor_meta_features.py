"""KNN neighbor fail-rate meta feature on RF-selected sensor space."""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted

DEFAULT_NEIGHBOR_FAIL_COL = "neighbor_fail_rate"


class NeighborFailRateFeatures(BaseEstimator, TransformerMixin):
    """Mean training-label rate among k nearest neighbors in sensor space.

    Requires y at fit. Neighbors use Mahalanobis distance (T² covariance precision)
    or unweighted Euclidean when distance_metric='uniform'.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        score_col: str = DEFAULT_NEIGHBOR_FAIL_COL,
        distance_metric: Literal["mahalanobis", "uniform"] = "mahalanobis",
        n_jobs: int = 1,
    ):
        self.n_neighbors = n_neighbors
        self.score_col = score_col
        self.distance_metric = distance_metric
        self.n_jobs = n_jobs

    def set_precision_matrix(self, precision: np.ndarray) -> None:
        self._precision_matrix = np.asarray(precision, dtype=float)

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("NeighborFailRateFeatures.fit requires target y.")
        X_df = self._as_dataframe(X)
        self.feature_names_in_ = list(X_df.columns)
        self.y_train_ = np.asarray(y, dtype=float).reshape(-1)
        self.train_index_ = list(X_df.index)

        X_arr = X_df.to_numpy(dtype=float)
        n_train = X_arr.shape[0]
        n_features = X_arr.shape[1]

        if n_train == 0 or n_features == 0:
            self.nn_ = None
            self.n_neighbors_fit_ = 0
            return self

        metric, metric_params = self._neighbor_metric_config()
        k = min(int(self.n_neighbors), n_train)
        self.n_neighbors_fit_ = int(k)
        self.nn_ = NearestNeighbors(
            n_neighbors=k,
            metric=metric,
            metric_params=metric_params,
            n_jobs=int(self.n_jobs),
        )
        self.nn_.fit(X_arr)
        return self

    def transform(self, X):
        check_is_fitted(self, "y_train_")
        X_df = self._as_dataframe(X)
        n_rows = len(X_df)
        if self.nn_ is None or self.n_neighbors_fit_ == 0:
            return pd.DataFrame(
                {self.score_col: np.zeros(n_rows, dtype=float)},
                index=X_df.index,
            )

        X_arr = X_df.to_numpy(dtype=float)
        k_query = min(self.n_neighbors_fit_ + 1, len(self.y_train_))
        distances, indices = self.nn_.kneighbors(X_arr, n_neighbors=k_query)

        rates = np.zeros(n_rows, dtype=float)
        for i in range(n_rows):
            neigh_idx = indices[i]
            neigh_dist = distances[i]
            mask = neigh_dist > 1e-12
            if not np.any(mask) and len(neigh_idx) > 1:
                mask = np.ones(len(neigh_idx), dtype=bool)
                mask[0] = False
            valid_idx = neigh_idx[mask][: self.n_neighbors_fit_]
            if len(valid_idx) == 0:
                rates[i] = float(np.mean(self.y_train_))
            else:
                rates[i] = float(np.mean(self.y_train_[valid_idx]))

        return pd.DataFrame({self.score_col: rates}, index=X_df.index)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return np.asarray([self.score_col], dtype=object)

    def _neighbor_metric_config(self) -> tuple[str, dict]:
        if self.distance_metric == "uniform":
            return "euclidean", {}
        precision = getattr(self, "_precision_matrix", None)
        if precision is None:
            raise ValueError(
                "Mahalanobis neighbor distance requires set_precision_matrix() "
                "from the fitted T² covariance."
            )
        return "mahalanobis", {"VI": precision}

    @staticmethod
    def _as_dataframe(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)
        df.columns = df.columns.astype(str)
        return df
