"""Blocked expanding-window time-series CV with local stratification guard."""
from __future__ import annotations

import numpy as np
import pandas as pd

from secom.pipelines import (
    BLOCKED_MIN_VAL_FAILS,
    N_BLOCKED_SPLITS,
    TARGET_COL,
    TIMESTAMP_COL,
)


class BlockedTimeSeriesCV:
    """Expanding-window blocked CV on measurement time with fail-count guard.

    Rows must align positionally with the timestamps and targets passed at
    construction (``X`` from sklearn does not include ``measurement_ts``).
    """

    def __init__(
        self,
        timestamps: np.ndarray | pd.Series,
        y: np.ndarray | pd.Series,
        *,
        n_splits: int = N_BLOCKED_SPLITS,
        min_val_fails: int = BLOCKED_MIN_VAL_FAILS,
    ):
        self.timestamps = pd.to_datetime(np.asarray(timestamps), errors="coerce")
        self.y = np.asarray(y, dtype=int)
        self.n_splits = int(n_splits)
        self.min_val_fails = int(min_val_fails)
        self._folds: list[tuple[np.ndarray, np.ndarray]] | None = None

    def _build_folds(self) -> list[tuple[np.ndarray, np.ndarray]]:
        n = len(self.y)
        if n == 0:
            return []

        order = np.lexsort(
            (
                np.arange(n, dtype=np.int64),
                self.timestamps.view("int64"),
            )
        )
        ts_sorted = self.timestamps[order]
        y_sorted = self.y[order]

        t_min = ts_sorted.min()
        t_max = ts_sorted.max()
        if pd.isna(t_min) or pd.isna(t_max) or t_min == t_max:
            return []

        edges = pd.date_range(t_min, t_max, periods=self.n_splits + 1)
        block_ids = np.searchsorted(edges[1:].values, ts_sorted.values, side="right")
        block_ids = np.clip(block_ids, 0, self.n_splits - 1)

        folds: list[tuple[np.ndarray, np.ndarray]] = []
        for val_block in range(1, self.n_splits):
            val_blocks = {val_block}
            while True:
                val_mask = np.isin(block_ids, list(val_blocks))
                n_fails = int(y_sorted[val_mask].sum())
                if n_fails >= self.min_val_fails or val_block == 0:
                    break
                val_block -= 1
                val_blocks.add(val_block)

            val_mask = np.isin(block_ids, list(val_blocks))
            train_mask = block_ids < min(val_blocks)
            if not train_mask.any() or not val_mask.any():
                continue
            if int(y_sorted[val_mask].sum()) < self.min_val_fails:
                continue

            train_idx = order[train_mask]
            val_idx = order[val_mask]
            folds.append((train_idx, val_idx))

        return folds

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if self._folds is None:
            self._folds = self._build_folds()
        return len(self._folds)

    def split(self, X, y=None, groups=None):
        if self._folds is None:
            self._folds = self._build_folds()
        for train_idx, val_idx in self._folds:
            yield train_idx, val_idx


def make_blocked_time_cv(train_df: pd.DataFrame) -> BlockedTimeSeriesCV:
    """Factory: blocked CV aligned to a temporal train dataframe.

    Timestamps and targets are passed in the dataframe's original row order so
    the indices yielded by ``split`` align positionally with ``X`` (which shares
    that order). ``BlockedTimeSeriesCV`` sorts internally by time and maps fold
    masks back to original positions, so no pre-sorting is needed here.
    """
    ts = pd.to_datetime(train_df[TIMESTAMP_COL], errors="coerce").to_numpy()
    return BlockedTimeSeriesCV(
        ts,
        train_df[TARGET_COL].astype(int).to_numpy(),
    )
