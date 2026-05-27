"""Pipeline flowchart for the SECOM Streamlit dashboard."""
from __future__ import annotations

import json

import streamlit as st

from theme.gruvbox_material import GRUVBOX

PIPELINE_MERMAID = """
flowchart LR
  raw["Raw SECOM files"]
  duck["DuckDB"]
  dbt["dbt models"]
  mart["mart_secom_features"]
  tune["Tuning (CV)"]
  bench["Benchmark (CV)"]
  dash["Streamlit dashboard"]

  raw --> duck --> dbt --> mart
  mart --> tune --> bench --> dash
  mart --> dash
"""

PREPROCESSING_MERMAID = """
flowchart LR
  stg["stg_secom"]
  int_f["int_secom_features"]
  meta["int_secom_column_metadata"]
  mart["mart_secom_features"]
  imp["Median impute"]
  cluster["Spearman cluster"]
  pls["PLS + Q"]
  rfk["RF top-k (CV)"]
  t2["Hotelling T²"]
  scale["RobustScaler"]
  clf["Classifier"]

  stg --> int_f --> meta --> mart --> imp --> cluster
  cluster --> pls --> t2
  cluster --> rfk --> t2
  t2 --> scale --> clf
"""


def _mermaid_html(diagram: str, height: int = 300) -> str:
    g = GRUVBOX
    theme_vars = {
        "primaryColor": g["bg_soft"],
        "primaryTextColor": g["fg"],
        "primaryBorderColor": g["border"],
        "lineColor": g["fg_muted"],
        "secondaryColor": g["bg"],
        "tertiaryColor": g["bg_soft"],
        "background": g["bg"],
        "mainBkg": g["bg_soft"],
        "nodeBorder": g["border"],
        "clusterBkg": g["bg_soft"],
        "titleColor": g["fg"],
        "edgeLabelBackground": g["bg"],
        "nodeTextColor": g["fg"],
    }
    theme_json = json.dumps(theme_vars)
    diagram_text = diagram.strip()
    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <style>
    body {{
      margin: 0;
      padding: 0.5rem;
      background: {g["bg"]};
      font-family: sans-serif;
    }}
    .mermaid {{
      display: flex;
      justify-content: center;
    }}
  </style>
</head>
<body>
  <pre class="mermaid">{diagram_text}</pre>
  <script>
    (async () => {{
      mermaid.initialize({{
        startOnLoad: false,
        theme: "base",
        themeVariables: {theme_json},
        flowchart: {{ useMaxWidth: true, htmlLabels: true }},
      }});
      await mermaid.run({{ querySelector: ".mermaid" }});
    }})();
  </script>
</body>
</html>
"""


def _html_flex_flowchart() -> str:
    """No-JS fallback: Gruvbox boxes and arrows."""
    g = GRUVBOX
    nodes = ["Raw SECOM", "DuckDB", "dbt", "mart", "Dashboard"]
    parts = []
    for i, n in enumerate(nodes):
        parts.append(
            f'<div style="background:{g["bg_soft"]};color:{g["fg"]};border:1px solid {g["border"]};'
            f'padding:0.5rem 0.75rem;border-radius:6px;font-size:12px;text-align:center;white-space:nowrap;">{n}</div>'
        )
        if i < len(nodes) - 1:
            parts.append(f'<span style="color:{g["fg_muted"]};margin:0 0.2rem;">→</span>')
    row = "".join(parts)
    branch = (
        f'<div style="margin-top:0.75rem;font-size:11px;color:{g["fg_muted"]};text-align:center;">'
        f'mart → Tuning (CV) → Benchmark (CV) → dashboard</div>'
    )
    return f"""
<div style="background:{g["bg"]};padding:0.75rem;font-family:sans-serif;">
  <div style="display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:0.15rem;">{row}</div>
  {branch}
</div>
"""


def _render_mermaid(diagram: str, *, height: int = 300) -> None:
    try:
        st.iframe(_mermaid_html(diagram, height=height), height=height)
    except Exception:
        if diagram.strip() == PIPELINE_MERMAID.strip():
            st.markdown(_html_flex_flowchart(), unsafe_allow_html=True)
        else:
            st.code(diagram.strip(), language="text")


def render_pipeline_flowchart() -> None:
    _render_mermaid(PIPELINE_MERMAID, height=300)


def render_preprocessing_flowchart() -> None:
    _render_mermaid(PREPROCESSING_MERMAID, height=340)
