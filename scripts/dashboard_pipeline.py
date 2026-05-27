"""Pipeline flowchart for the SECOM Streamlit dashboard."""
from __future__ import annotations

import json

import streamlit as st

from theme.gruvbox_material import GRUVBOX

PIPELINE_MERMAID = """
flowchart LR
  raw["Raw SECOM files"]
  dbt["dbt models"]
  duck["DuckDB"]
  mart["mart_secom_features"]
  tune["Tuning"]
  bench["benchmark_models"]
  dash["Streamlit dashboard"]

  raw --> dbt --> duck --> mart --> dash
  mart --> tune --> bench --> dash
"""


def _mermaid_html(diagram: str) -> str:
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
    nodes = ["Raw SECOM", "dbt", "DuckDB", "mart", "Dashboard"]
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
        f'mart → tuning → benchmark → dashboard</div>'
    )
    return f"""
<div style="background:{g["bg"]};padding:0.75rem;font-family:sans-serif;">
  <div style="display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:0.15rem;">{row}</div>
  {branch}
</div>
"""


def render_pipeline_flowchart() -> None:
    try:
        st.iframe(_mermaid_html(PIPELINE_MERMAID), height=300)
    except Exception:
        st.markdown(_html_flex_flowchart(), unsafe_allow_html=True)
