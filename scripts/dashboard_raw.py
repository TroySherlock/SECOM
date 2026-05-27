"""Load raw SECOM files for the dashboard browser."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS_PATH = REPO_ROOT / "data" / "secom_labels.data"
SENSORS_PATH = REPO_ROOT / "data" / "secom.data"
ZIP_PATH = REPO_ROOT / "data" / "secom.zip"

RAW_TARGET_COL = "raw_target"
TIMESTAMP_COL = "timestamp"
N_SENSORS = 591


def raw_files_available() -> bool:
    return LABELS_PATH.is_file() and SENSORS_PATH.is_file()


def load_raw_secom() -> pd.DataFrame:
    """Load unprocessed SECOM labels + sensor matrix (-1/+1 labels, NaN sensors)."""
    if not raw_files_available():
        raise FileNotFoundError(
            f"Missing raw files. Expected `{LABELS_PATH.name}` and `{SENSORS_PATH.name}` under data/."
        )

    parsed_labels: list[dict[str, object]] = []
    with LABELS_PATH.open(encoding="utf-8") as f:
        for line in f:
            cleaned = line.strip()
            if not cleaned:
                continue
            parts = cleaned.split(maxsplit=1)
            if len(parts) == 2:
                parsed_labels.append({"raw_target": int(parts[0]), "timestamp": parts[1]})

    df_labels = pd.DataFrame(parsed_labels)
    df_sensors = pd.read_csv(
        SENSORS_PATH,
        sep=r"\s+",
        header=None,
        engine="python",
        na_values=["NaN", "nan"],
    )
    df_sensors.columns = [f"c_{i}" for i in range(df_sensors.shape[1])]

    return pd.concat([df_labels.reset_index(drop=True), df_sensors.reset_index(drop=True)], axis=1)


def sensor_column_names(n: int = N_SENSORS) -> list[str]:
    return [f"c_{i}" for i in range(n)]


def slice_raw_for_display(
    df: pd.DataFrame,
    *,
    n_sensor_cols: int,
    raw_target_filter: str,
) -> pd.DataFrame:
    label_cols = [RAW_TARGET_COL, TIMESTAMP_COL]
    sensors = sensor_column_names()
    n_sensor_cols = min(n_sensor_cols, len(sensors))
    cols = label_cols + sensors[:n_sensor_cols]

    out = df[cols].copy()
    if raw_target_filter == "Pass (-1)":
        out = out.loc[out[RAW_TARGET_COL] == -1]
    elif raw_target_filter == "Fail (+1)":
        out = out.loc[out[RAW_TARGET_COL] == 1]
    return out.reset_index(drop=True)
