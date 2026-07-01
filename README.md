# SECOM Wafer Defect Detection

> End-to-end semiconductor defect-detection analytics on the [UCI SECOM dataset](https://archive.ics.uci.edu/ml/datasets/SECOM) — from SQL/dbt feature engineering through drift-aware ML modeling to a deployed, explainable dashboard.

<p align="center">
  <a href="https://79kpwjksc9d23km8arddpn.streamlit.app/"><strong>🚀 Live Dashboard →</strong></a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="dbt" src="https://img.shields.io/badge/dbt-DuckDB-FF694B?logo=dbt&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white">
</p>

---

## Overview

SECOM is an imbalanced, rare-event classification problem: **1,567 wafers × 590 sensors** at a **~6.7% fail rate**. This project treats it like a production fab-analytics workflow, framed around the **escape vs. overkill** cost trade-off:

- **Data engineering** — a tested `dbt` + `DuckDB` ELT pipeline (`seed → stg → int → mart`).
- **Feature engineering** — robust-z sensor twins, Spearman correlation pruning, hub-pair interactions.
- **Modeling** — a unified 3×3 grid of feature front-ends × classifier heads, probability-calibrated.
- **Drift-aware validation** — repeated stratified CV plus random *and* temporal forward holdouts with bootstrap CIs.
- **Explainability** — per-wafer root-cause analysis (SHAP, SPC z-scores, Bayesian credible intervals) and MSPC abstention gates.
- **Generative AI** — fact-grounded (RAG-style) wafer narratives from a local LLM, frozen to JSON.

The [live dashboard](https://79kpwjksc9d23km8arddpn.streamlit.app/) runs **read-only** on precomputed artifacts — no model refitting at runtime.

## Key results

| Track | Champion | ROC-AUC | PR-AUC | BER |
|-------|----------|:-------:|:------:|:---:|
| In-distribution (random holdout) | `hsic_rf` | 0.80 | 0.34 | 22.8% |
| Temporal drift (forward holdout) | `pls_bayes` | 0.78 | 0.20 | 26.7%¹ |

¹ At a 20:1 cost-optimal threshold, `pls_bayes` catches **88% (15/17)** of failing wafers on the temporal holdout.

## Quickstart

```bash
# 1. Environment (creates .venv, installs the secom package, sets DBT_PROFILES_DIR)
direnv allow            # or: nix-shell

# 2. Build the data pipeline
python -m secom.cli.build_seed   # data/*.data → seeds/raw_secom.csv
dbt seed && dbt run              # DuckDB: stg → int → mart

# 3. Launch the dashboard
streamlit run streamlit_app.py
```

### Data setup

1. Download the sensor matrix from UCI and place it at `data/secom.data` (not in git; ~5 MB whitespace-separated).
2. Label and metadata files are already in-repo: `data/secom_labels.data`, `data/secom.names`.

Local-only paths (gitignored): `data/secom.data`, `data/secom.duckdb`, `seeds/raw_secom.csv`.

## Rebuilding artifacts

The dashboard reads frozen JSON artifacts. To regenerate them:

```bash
python -m secom.cli.run_tuning          # hyperparameters → data/processed/tuned/<model_id>.json
python -m secom.cli.benchmark           # CV leaderboard + holdout metrics
python -m secom.cli.build_explanations  # frozen per-wafer explanations, posteriors, importances
python -m secom.cli.build_narratives    # frozen Gemma wafer summaries (local llama.cpp)
```

## Architecture

```
raw_secom (seed) → stg_secom → int_secom_features → int_secom_column_metadata → mart_secom_features
```

Training and the dashboard read `public.mart_secom_features` via `secom.pipelines.load_mart()`.

## Project layout

| Path | Purpose |
|------|---------|
| `src/secom/` | Installable library, CLIs, dashboard helpers |
| `app_pages/` | Streamlit multipage app |
| `streamlit_app.py` | Dashboard entry point |
| `models/`, `macros/`, `seeds/` | dbt project |
| `data/processed/` | Benchmark, tuning, explanation, and narrative JSON artifacts |
| `tests/` | pytest smoke + unit tests |

## Tech stack

**Python** · **pandas / NumPy / scikit-learn / SciPy** · **NumPyro** (Bayesian) · **SHAP** · **dbt** + **DuckDB** · **Streamlit** · **llama.cpp** (local LLM) · **pytest** / **ruff**
