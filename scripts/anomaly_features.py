"""Unsupervised anomaly scores for SECOM sensor feature branches."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.utils.validation import check_is_fitted

DEFAULT_SCORE_COL = "isolation_forest_score"


class IsolationForestScoreFeatures(BaseEstimator, TransformerMixin):
    """Append Isolation Forest decision_function as a single anomaly score column.

    Fit uses X only (y ignored). Higher scores are more inlier-like; more negative
    values are more anomalous (sklearn convention). Refit inside each CV train fold
    via Pipeline — no label leakage, but do not fit on validation rows.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: str | float = "auto",
        score_col: str = DEFAULT_SCORE_COL,
        random_state: int = 42,
        n_jobs: int = 1,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.score_col = score_col
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        X_df = self._as_dataframe(X)
        self.feature_names_in_ = list(X_df.columns)
        X_arr = X_df.to_numpy(dtype=float)

        if X_arr.size == 0 or X_arr.shape[1] == 0:
            self.iforest_ = None
            return self

        self.iforest_ = IsolationForest(
            n_estimators=int(self.n_estimators),
            contamination=self.contamination,
            random_state=int(self.random_state),
            n_jobs=int(self.n_jobs),
        )
        self.iforest_.fit(X_arr)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        X_df = self._as_dataframe(X)
        n_rows = len(X_df)

        if self.iforest_ is None:
            scores = np.zeros(n_rows, dtype=float)
        else:
            X_arr = X_df.to_numpy(dtype=float)
            scores = self.iforest_.decision_function(X_arr)

        score_df = pd.DataFrame({self.score_col: scores}, index=X_df.index)
        return pd.concat([X_df, score_df], axis=1)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        base = (
            list(input_features)
            if input_features is not None
            else list(self.feature_names_in_)
        )
        return np.asarray(base + [self.score_col], dtype=object)

    @staticmethod
    def _as_dataframe(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)
        df.columns = df.columns.astype(str)
        return df
