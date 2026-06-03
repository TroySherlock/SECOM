"""RF hub interaction features for the shared sensor preprocess path."""
from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.validation import check_is_fitted

from scripts.mspc_features import DEFAULT_T2_COL, MahalanobisT2Features


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
        from scripts.secom_pipelines import random_forest_classifier

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
