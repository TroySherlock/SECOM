"""Pipeline flowchart for the SECOM Streamlit dashboard."""
from __future__ import annotations

import html

import streamlit as st

try:
    import streamlit.components.v1 as components
except ImportError:  # pragma: no cover
    components = None  # type: ignore[assignment]

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
  hubs["RF + T² + hub pairs"]
  scale["RobustScaler"]
  clf["Classifier"]

  stg --> int_f --> meta --> mart --> imp --> cluster --> hubs --> scale --> clf
"""

_MERMAID_INIT = """
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
mermaid.initialize({ startOnLoad: false, theme: 'dark' });
const el = document.querySelector('.mermaid');
await mermaid.run({ nodes: [el] });
const svg = document.querySelector('.mermaid svg');
if (svg) {
    const h = Math.ceil(svg.getBoundingClientRect().height) + 16;
    document.body.style.margin = '0';
    document.body.style.overflow = 'hidden';
    window.parent.postMessage({ type: 'streamlit:setFrameHeight', height: h }, '*');
}
"""


def _render_mermaid(diagram: str, *, height: int = 140) -> None:
    code = html.escape(diagram.strip())
    html_doc = f"""
        <body style="margin:0;padding:0;">
        <pre class="mermaid">{code}</pre>
        <script type="module">
        {_MERMAID_INIT}
        </script>
        </body>
        """
    if hasattr(st, "iframe"):
        try:
            st.iframe(html_doc, height="content")
            return
        except (TypeError, ValueError):
            pass
    if components is not None:
        components.html(html_doc, height=height, scrolling=False)
    else:  # pragma: no cover
        st.markdown(f"```mermaid\n{diagram.strip()}\n```")


def render_pipeline_flowchart() -> None:
    _render_mermaid(PIPELINE_MERMAID)


def render_preprocessing_flowchart() -> None:
    _render_mermaid(PREPROCESSING_MERMAID)
