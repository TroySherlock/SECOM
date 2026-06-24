"""Load and summarize dbt stg_secom for the Introduction dashboard."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import duckdb
import pandas as pd

from secom.pipelines import DB_PATH, TARGET_COL, TIMESTAMP_COL

STG_RELATION = "public.stg_secom"
ROW_INDEX_COL = "row_index"

_SENSOR_PATTERN = re.compile(r"^c_\d+$")


@dataclass(frozen=True)
class StgSnapshot:
    df: pd.DataFrame
    sensor_cols: list[str]
    stats: dict[str, Any]
    default_sensor: str


def build_stg_snapshot(db_path=DB_PATH) -> StgSnapshot:
    """Load stg_secom and precompute columns, summary stats, and default sensor once."""
    df = load_stg_secom(db_path)
    sensor_cols = sensor_columns(df)
    stats = stg_summary_stats(df)
    default_sensor = ""
    if sensor_cols:
        variances = df[sensor_cols].var(numeric_only=True).sort_values(ascending=False)
        default_sensor = str(variances.index[0])
    return StgSnapshot(
        df=df,
        sensor_cols=sensor_cols,
        stats=stats,
        default_sensor=default_sensor,
    )


def stg_available(db_path=DB_PATH) -> bool:
    if not db_path.exists():
        return False
    try:
        with duckdb.connect(str(db_path), read_only=True) as con:
            con.execute(f"select 1 from {STG_RELATION} limit 1").fetchone()
        return True
    except Exception:
        return False


def load_stg_secom(db_path=DB_PATH) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Missing `{db_path.name}`. Run: dbt run -s stg_secom")
    with duckdb.connect(str(db_path), read_only=True) as con:
        df = con.execute(f"select * from {STG_RELATION}").df()
    df = df.reset_index(drop=True)
    df[ROW_INDEX_COL] = df.index
    return df


def sensor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if _SENSOR_PATTERN.fullmatch(str(c))]


def stg_summary_stats(df: pd.DataFrame) -> dict[str, Any]:
    sensor_cols = sensor_columns(df)
    y = df[TARGET_COL].astype(int)
    ts = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    miss_rate = df[sensor_cols].isna().mean() if sensor_cols else pd.Series(dtype=float)

    date_min = ts.min()
    date_max = ts.max()
    if pd.notna(date_min) and pd.notna(date_max):
        date_min_str = f"{date_min:%Y-%m-%d}"
        date_max_str = f"{date_max:%Y-%m-%d}"
        date_range = f"{date_min_str} – {date_max_str}"
    else:
        date_min_str = "—"
        date_max_str = "—"
        date_range = "—"

    return {
        "n_obs": len(df),
        "n_pass": int((y == 0).sum()),
        "n_fail": int((y == 1).sum()),
        "fail_rate": float(y.mean()),
        "n_sensors": len(sensor_cols),
        "date_min": date_min_str,
        "date_max": date_max_str,
        "date_range": date_range,
        "missing_cell_pct": 100 * float(df[sensor_cols].isna().mean().mean()) if sensor_cols else 0.0,
        "high_miss_sensor_pct": 100 * float((miss_rate > 0.5).mean()) if len(miss_rate) else 0.0,
    }


def era_drift_summary(
    df: pd.DataFrame,
    *,
    timestamp_col: str = TIMESTAMP_COL,
    sensor_cols: list[str] | None = None,
    test_size: float = 0.20,
    z_threshold: float = 2.0,
) -> dict[str, Any]:
    """Quantify era drift: share of sensors whose holdout-era mean shifts past
    ``z_threshold`` SD from the training-era baseline (first ``1 - test_size`` by time).

    Mirrors the standardisation in ``charts.fig_sensor_drift_heatmap`` so the headline
    scalar and the heatmap agree. Returns counts plus the median absolute shift.
    """
    sensor_cols = sensor_cols or sensor_columns(df)
    empty = {
        "n_evaluated": 0,
        "n_drifted": 0,
        "pct_drifted": 0.0,
        "median_abs_shift": 0.0,
        "z_threshold": float(z_threshold),
    }
    if not sensor_cols:
        return empty

    ts = pd.to_datetime(df[timestamp_col], errors="coerce")
    order = ts.sort_values().index
    values = df.loc[order, sensor_cols].astype(float)
    values = values.fillna(values.median(numeric_only=True))

    n_rows = len(values)
    train_n = int(round(n_rows * (1.0 - test_size)))
    if n_rows < 4 or train_n < 2 or train_n >= n_rows:
        return empty

    base = values.iloc[:train_n]
    base_std = base.std(ddof=0)
    keep = base_std[base_std > 0].index.tolist()
    if not keep:
        return empty

    z = (values[keep] - base.mean()[keep]) / base_std[keep]
    holdout_shift = z.iloc[train_n:].mean().abs()
    n_drifted = int((holdout_shift > z_threshold).sum())
    n_eval = len(keep)
    return {
        "n_evaluated": n_eval,
        "n_drifted": n_drifted,
        "pct_drifted": 100.0 * n_drifted / n_eval if n_eval else 0.0,
        "median_abs_shift": float(holdout_shift.median()),
        "z_threshold": float(z_threshold),
    }


def slice_stg_for_display(
    df: pd.DataFrame,
    *,
    n_sensor_cols: int,
    target_filter: str,
    max_rows: int | None = None,
) -> pd.DataFrame:
    label_cols = [TIMESTAMP_COL, TARGET_COL]
    sensors = sensor_columns(df)
    n_sensor_cols = min(n_sensor_cols, len(sensors))
    cols = label_cols + sensors[:n_sensor_cols]
    out = df[cols].copy()
    if target_filter == "Pass (0)":
        out = out.loc[out[TARGET_COL] == 0]
    elif target_filter == "Fail (1)":
        out = out.loc[out[TARGET_COL] == 1]
    out = out.reset_index(drop=True)
    if max_rows is not None:
        out = out.head(max_rows)
    return out
