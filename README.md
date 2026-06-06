# SECOM defect detection

Semiconductor manufacturing defect classification on the [UCI SECOM dataset](https://archive.ics.uci.edu/ml/datasets/SECOM): dbt + DuckDB for feature prep, scikit-learn pipelines for modeling, and a Streamlit dashboard for exploration and explainability.

## Data setup

1. Download the sensor matrix from UCI and place it at `data/secom.data` (not in git; ~5 MB whitespace-separated file).
2. Label and metadata files are already in-repo:
   - `data/secom_labels.data`
   - `data/secom.names`

Local-only paths (gitignored): `data/secom.data`, `data/secom.duckdb`, `seeds/raw_secom.csv`.

## Environment

```bash
direnv allow   # or: nix-shell
```

This creates `.venv`, installs the `secom` package in editable mode, and sets `DBT_PROFILES_DIR` to the repo root.

## Build pipeline

From repo root, in order:

```bash
python -m secom.cli.build_seed   # data/*.data → seeds/raw_secom.csv
dbt seed && dbt run              # DuckDB: stg → int → mart
```

Optional ML artifacts for the dashboard:

```bash
python -m secom.cli.benchmark         # CV leaderboard + holdout metrics
python -m secom.cli.build_narratives  # frozen Gemma wafer summaries (linear_lr)
```

Launch the dashboard:

```bash
streamlit run streamlit_app.py
```

## Tuning

- Per-model notebooks: `tuning/linear_lr.ipynb`, `topk_rf.ipynb`, `topk_knn.ipynb`, `topk_xgb.ipynb`
- All models: `tuning/tune_all.ipynb`
- Benchmark notebook: `tuning/benchmark_models.ipynb`
- CLI: `python -m secom.cli.run_tuning`

Frozen hyperparameters land in `data/processed/tuned/<model_id>.json`.

## dbt models

```
raw_secom (seed) → stg_secom → int_secom_features → int_secom_column_metadata → mart_secom_features
```

Training and the dashboard read `public.mart_secom_features` via `secom.pipelines.load_mart()`.

## Project layout

| Path | Purpose |
|------|---------|
| `src/secom/` | Installable library, CLIs, dashboard helpers |
| `pages/` | Streamlit multipage app |
| `models/`, `macros/`, `seeds/` | dbt project |
| `data/processed/` | Benchmark, tuning, and narrative JSON artifacts |
| `tuning/` | Experiment notebooks |
