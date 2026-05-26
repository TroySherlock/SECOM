#!/usr/bin/env python3
"""Build seeds/raw_secom.csv from raw SECOM label and sensor files."""
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    print("Step 1: Processing secom_labels.data...")
    parsed_labels = []
    labels_path = REPO_ROOT / "data" / "secom_labels.data"
    with labels_path.open(encoding="utf-8") as f:
        for line in f:
            cleaned_line = line.strip()
            if not cleaned_line:
                continue
            parts = cleaned_line.split(maxsplit=1)
            if len(parts) == 2:
                raw_target = int(parts[0])
                timestamp = parts[1]
                binary_target = 0 if raw_target == -1 else 1
                parsed_labels.append({"target": binary_target, "timestamp": timestamp})

    df_labels = pd.DataFrame(parsed_labels).reset_index(drop=True)

    print("Step 2: Processing secom.data...")
    df_sensors = pd.read_csv(
        REPO_ROOT / "data" / "secom.data", sep=r"\s+", header=None, engine="python"
    )
    df_sensors = df_sensors.reset_index(drop=True)
    df_sensors.columns = [f"c_{i}" for i in range(df_sensors.shape[1])]

    print("Step 3: Merging data matrices...")
    df_complete = pd.concat([df_labels, df_sensors], axis=1)

    out_path = REPO_ROOT / "seeds" / "raw_secom.csv"
    print(f"Step 4: Writing {out_path}...")
    df_complete.to_csv(out_path, index=False)
    print(f"Done. Shape {df_complete.shape}")


if __name__ == "__main__":
    main()
