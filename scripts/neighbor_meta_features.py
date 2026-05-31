"""KNN neighbor fail-rate meta feature for SECOM sensor branches."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from scripts.hub_interactions import LinearSelectT2HubBlock

DEFAULT_NEIGHBOR_FAIL_COL = "neighbor_fail_rate"
HUB_STEP_NAME = "select_t2_hubs"
NEIGHBOR_STEP_NAME = "neighbor_fail_rate"


class NeighborFailRateFeatures(BaseEstimator, TransformerMixin):
    """Append mean training-label rate among k RF-weighted nearest neighbors.

    Requires y at fit. Neighbors are drawn only from the training rows seen at
    fit (safe inside CV train folds). Call set_column_weights before fit when
    using RF-weighted distances from LinearSelectT2HubBlock.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        score_col: str = DEFAULT_NEIGHBOR_FAIL_COL,
        n_jobs: int = 1,
    ):
        self.n_neighbors = n_neighbors
        self.score_col = score_col
        self.n_jobs = n_jobs

    def set_column_weights(self, weights: np.ndarray) -> None:
        self._column_weights = np.asarray(weights, dtype=float)

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("NeighborFailRateFeatures.fit requires target y.")
        X_df = self._as_dataframe(X)
        self.feature_names_in_ = list(X_df.columns)
        self.y_train_ = np.asarray(y, dtype=float).reshape(-1)
        self.train_index_ = list(X_df.index)

        X_weighted = self._weighted_array(X_df)
        n_train = X_weighted.shape[0]
        n_features = X_weighted.shape[1]

        if n_train == 0 or n_features == 0:
            self.nn_ = None
            self.n_neighbors_fit_ = 0
            return self

        k = min(int(self.n_neighbors), n_train)
        self.n_neighbors_fit_ = int(k)
        self.nn_ = NearestNeighbors(
            n_neighbors=k,
            metric="minkowski",
            n_jobs=int(self.n_jobs),
        )
        self.nn_.fit(X_weighted)
        return self

    def transform(self, X):
        check_is_fitted(self, "y_train_")
        X_df = self._as_dataframe(X)
        n_rows = len(X_df)
        if self.nn_ is None or self.n_neighbors_fit_ == 0:
            return pd.concat(
                [
                    X_df,
                    pd.DataFrame(
                        {self.score_col: np.zeros(n_rows, dtype=float)},
                        index=X_df.index,
                    ),
                ],
                axis=1,
            )

        X_weighted = self._weighted_array(X_df)
        k_query = min(self.n_neighbors_fit_ + 1, len(self.y_train_))
        distances, indices = self.nn_.kneighbors(X_weighted, n_neighbors=k_query)

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

        score_df = pd.DataFrame({self.score_col: rates}, index=X_df.index)
        return pd.concat([X_df, score_df], axis=1)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        base = (
            list(input_features)
            if input_features is not None
            else list(self.feature_names_in_)
        )
        return np.asarray(base + [self.score_col], dtype=object)

    def _weighted_array(self, X_df: pd.DataFrame) -> np.ndarray:
        X_arr = X_df.to_numpy(dtype=float)
        weights = getattr(self, "_column_weights", None)
        if weights is None or len(weights) != X_arr.shape[1]:
            return X_arr
        return X_arr * weights.reshape(1, -1)

    @staticmethod
    def _as_dataframe(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)
        df.columns = df.columns.astype(str)
        return df


class SensorBranchPipeline(Pipeline):
    """Sensor-branch Pipeline that passes hub RF weights into NeighborFailRateFeatures."""

    def _handoff_hub_weights(
        self,
        Xt: pd.DataFrame,
        fitted_steps: list[tuple[str, BaseEstimator]],
        neighbor_step: NeighborFailRateFeatures,
    ) -> None:
        hub = dict(fitted_steps).get(HUB_STEP_NAME)
        if hub is None:
            return
        check_is_fitted(hub, "select_")
        neighbor_step.set_column_weights(hub.column_importance_weights(Xt.columns))

    def fit(self, X, y=None, **params):
        self._validate_steps()
        Xt = X
        fitted_steps: list[tuple[str, BaseEstimator]] = []
        for name, step in self.steps:
            step = clone(step)
            step_params = self._get_step_fit_params(name, params)
            if name == NEIGHBOR_STEP_NAME and isinstance(step, NeighborFailRateFeatures):
                self._handoff_hub_weights(Xt, fitted_steps, step)
            Xt = step.fit_transform(Xt, y, **step_params)
            fitted_steps.append((name, step))
        self.steps = fitted_steps
        return self

    def fit_transform(self, X, y=None, **params):
        self._validate_steps()
        Xt = X
        fitted_steps: list[tuple[str, BaseEstimator]] = []
        for name, step in self.steps:
            step = clone(step)
            step_params = self._get_step_fit_params(name, params)
            if name == NEIGHBOR_STEP_NAME and isinstance(step, NeighborFailRateFeatures):
                self._handoff_hub_weights(Xt, fitted_steps, step)
            Xt = step.fit_transform(Xt, y, **step_params)
            fitted_steps.append((name, step))
        self.steps = fitted_steps
        return Xt

    def _get_step_fit_params(self, name: str, params: dict) -> dict:
        prefix = f"{name}__"
        return {
            key[len(prefix) :]: value
            for key, value in params.items()
            if key.startswith(prefix)
        }
