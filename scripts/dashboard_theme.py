"""Gruvbox Material Light theme for the SECOM Streamlit dashboard."""
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

STG_TABLE_MAX_ROWS = 50

from theme.gruvbox_material import (  # noqa: E402
    CALLOUT_STYLES,
    GRUVBOX,
    GRUVBOX_CODE,
    GRUVBOX_EDU_TABLE,
    GRUVBOX_TABLE,
    CalloutKind,
    PLOTLY_COLORWAY,
)

LEGEND_TOP_RIGHT: dict[str, Any] = dict(
    orientation="v",
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99,
    bgcolor="rgba(242,229,188,0.85)",
    bordercolor=GRUVBOX["border"],
    font=dict(color=GRUVBOX["fg"]),
)


def style_stg_dataframe(df: pd.DataFrame, *, target_col: str = "target"):
    """Pandas Styler for st.dataframe: bulk body colors + per-cell target only."""
    t = GRUVBOX_TABLE
    g = GRUVBOX
    body_props = {"background-color": t["bg"], "color": t["fg_light"]}
    missing_props = f"background-color: {g['fail']}; color: {t['fg_light']}; font-weight: 600;"

    def missing_cell(val: object) -> str:
        if pd.isna(val):
            return missing_props
        return ""

    def target_cell(val: object) -> str:
        if pd.isna(val):
            return missing_props
        if int(val) == 1:
            return f"background-color: {g['fail']}; color: {t['fg_light']}; font-weight: 600;"
        return f"background-color: {g['pass']}; color: {t['fg_light']}; font-weight: 600;"

    other_cols = [c for c in df.columns if c != target_col]
    styler = df.style
    if other_cols:
        styler = styler.set_properties(**body_props, subset=other_cols)
        styler = styler.map(missing_cell, subset=other_cols)
    if target_col in df.columns:
        styler = styler.map(target_cell, subset=[target_col])
    return styler


def render_callout(
    kind: CalloutKind,
    content: str,
    *,
    key_suffix: str = "",
) -> None:
    """Notes: st.info. Edu/pipeline: keyed container (.st-key-secom_callout_*)."""
    text = content.strip()
    if kind == "note":
        st.info(text)
        return
    slug = key_suffix or str(abs(hash(text)) % 10**8)
    with st.container(key=f"secom_callout_{kind}_{slug}"):
        st.markdown(text)


def display_stg_dataframe(
    data: pd.DataFrame | pd.io.formats.style.Styler,
    *,
    height: int = 400,
) -> None:
    if isinstance(data, pd.DataFrame):
        data = style_stg_dataframe(data)
    st.dataframe(data, width="stretch", height=height, hide_index=True)


def streamlit_css() -> str:
    g = GRUVBOX
    t = GRUVBOX_TABLE
    c = GRUVBOX_CODE
    et = GRUVBOX_EDU_TABLE
    n = CALLOUT_STYLES["note"]
    e = CALLOUT_STYLES["edu"]
    p = CALLOUT_STYLES["pipeline"]
    return f"""
    <style>
    .stApp {{
        background-color: {g["bg"]};
        color: {g["fg"]};
    }}
    header[data-testid="stHeader"] {{
        background-color: {g["bg_soft"]};
        border-bottom: 1px solid {g["border"]};
    }}
    [data-testid="stMetric"] {{
        background-color: {g["bg_soft"]};
        border: 1px solid {g["border"]};
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }}
    [data-testid="stMetricLabel"] {{
        color: {g["fg_muted"]};
    }}
    [data-testid="stMetricValue"] {{
        color: {g["fg"]};
        font-size: 1.1rem;
        overflow-wrap: anywhere;
    }}
    div[data-testid="stExpander"] details {{
        background-color: {g["bg_soft"]};
        border: 1px solid {g["border"]};
        border-radius: 8px;
    }}
    div[data-testid="stExpander"] summary {{
        color: {g["fg"]};
    }}
    [data-testid="stSubheader"],
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {{
        color: {g["fg"]};
    }}
    [data-testid="stSelectbox"] label,
    [data-testid="stCheckbox"] label {{
        color: {g["fg"]};
    }}
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p,
    [data-testid="stCaptionContainer"] span {{
        color: {g["fg"]} !important;
        background-color: transparent !important;
    }}
    h1, h2, h3, p, li, span {{
        color: {g["fg"]};
    }}
    code, [data-testid="stMarkdownContainer"] code {{
        background-color: {c["bg"]};
        color: {c["fg"]};
        border: 1px solid {c["border"]};
        padding: 0.1em 0.35em;
        border-radius: 4px;
    }}
    [data-testid="stCaptionContainer"] code {{
        background-color: {c["bg"]} !important;
        color: {c["fg"]} !important;
        border: 1px solid {c["border"]} !important;
        padding: 0.1em 0.35em;
        border-radius: 4px;
    }}
    div[data-testid="stAlert"] {{
        background-color: {n["bg"]};
        border: none;
        color: {n["fg"]};
        border-radius: 8px;
    }}
    div[data-testid="stAlert"] p, div[data-testid="stAlert"] li {{
        color: {n["fg"]};
    }}
    div[data-testid="stElementContainer"]:has([class*="st-key-secom_callout_edu_"]),
    div[data-testid="stVerticalBlock"][class*="st-key-secom_callout_edu_"],
    [class*="st-key-secom_callout_edu_"] {{
        background-color: {e["bg"]} !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        border-radius: 8px;
        padding: 0.85rem 1.1rem 0.65rem 1.1rem;
        margin: 0.35rem 0 0.85rem 0;
    }}
    div[data-testid="stElementContainer"]:has([class*="st-key-secom_callout_pipeline_"]),
    div[data-testid="stVerticalBlock"][class*="st-key-secom_callout_pipeline_"],
    [class*="st-key-secom_callout_pipeline_"] {{
        background-color: {p["bg"]} !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        border-radius: 8px;
        padding: 0.85rem 1.1rem 0.65rem 1.1rem;
        margin: 0.35rem 0 0.85rem 0;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"]:has([class*="st-key-secom_callout_"]) {{
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }}
    div[data-testid="stElementContainer"]:has([class*="st-key-secom_callout_edu_"]) table,
    div[data-testid="stElementContainer"]:has([class*="st-key-secom_callout_edu_"]) [data-testid="stMarkdownContainer"] table,
    [class*="st-key-secom_callout_edu_"] table {{
        border-collapse: collapse;
        width: 100%;
        margin: 0.5rem 0;
        background-color: {et["cell_bg"]} !important;
    }}
    div[data-testid="stElementContainer"]:has([class*="st-key-secom_callout_edu_"]) table th,
    div[data-testid="stElementContainer"]:has([class*="st-key-secom_callout_edu_"]) table td,
    div[data-testid="stElementContainer"]:has([class*="st-key-secom_callout_edu_"]) [data-testid="stMarkdownContainer"] table th,
    div[data-testid="stElementContainer"]:has([class*="st-key-secom_callout_edu_"]) [data-testid="stMarkdownContainer"] table td,
    [class*="st-key-secom_callout_edu_"] table th,
    [class*="st-key-secom_callout_edu_"] table td {{
        background-color: {et["cell_bg"]} !important;
        color: {e["fg"]} !important;
        padding: 0.35rem 0.55rem;
        text-align: left;
        border: none !important;
    }}
    div[data-testid="stElementContainer"]:has([class*="st-key-secom_callout_edu_"]) table thead th,
    div[data-testid="stElementContainer"]:has([class*="st-key-secom_callout_edu_"]) [data-testid="stMarkdownContainer"] table thead th,
    [class*="st-key-secom_callout_edu_"] table thead th {{
        background-color: {et["header_bg"]} !important;
        color: {e["fg"]} !important;
        font-weight: 600;
    }}
    div[data-testid="stElementContainer"]:has([class*="st-key-secom_callout_edu_"]) table tbody tr,
    div[data-testid="stElementContainer"]:has([class*="st-key-secom_callout_edu_"]) [data-testid="stMarkdownContainer"] table tbody tr {{
        background-color: {et["cell_bg"]} !important;
    }}
    [class*="st-key-secom_callout_edu_"] code,
    [class*="st-key-secom_callout_pipeline_"] code {{
        background-color: {g["bg"]};
        color: {c["fg"]};
        border: 1px solid {g["bg"]};
    }}
    [data-testid="stDataFrame"] {{
        border: none;
        border-radius: 8px;
        --gdg-bg-header: {t["header_bg"]};
        --gdg-bg-header-hovered: {t["bg"]};
        --gdg-bg-header-has-focus: {t["header_bg"]};
        --gdg-text-header: {t["fg_light"]};
        --gdg-text-header-selected: {t["fg_light"]};
        --gdg-bg-icon-header: {t["header_bg"]};
        --gdg-fg-icon-header: {t["fg_light"]};
        --gdg-bg-cell: {t["bg"]};
        --gdg-bg-cell-medium: {t["bg_alt"]};
        --gdg-text-dark: {t["fg_light"]};
        --gdg-text-medium: {t["fg_muted"]};
        --gdg-border-color: {t["border"]};
        --gdg-horizontal-border-color: {t["border"]};
    }}
    </style>
    """


def plotly_chart(fig: go.Figure, *, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", key=key)


def base_plotly_layout(**overrides: Any) -> dict[str, Any]:
    g = GRUVBOX
    layout: dict[str, Any] = dict(
        paper_bgcolor=g["bg"],
        plot_bgcolor=g["bg_soft"],
        font=dict(color=g["fg"], family="sans-serif"),
        margin=dict(l=48, r=24, t=64, b=48),
        colorway=PLOTLY_COLORWAY,
        title=dict(font=dict(color=g["fg"], size=16)),
        legend=LEGEND_TOP_RIGHT,
        xaxis=dict(
            gridcolor=g["border"],
            linecolor=g["border"],
            tickfont=dict(color=g["fg_muted"]),
            title=dict(font=dict(color=g["fg"])),
        ),
        yaxis=dict(
            gridcolor=g["border"],
            linecolor=g["border"],
            tickfont=dict(color=g["fg_muted"]),
            title=dict(font=dict(color=g["fg"])),
        ),
    )
    layout.update(overrides)
    return layout


def apply_plotly_theme(fig: go.Figure, **layout_overrides: Any) -> go.Figure:
    fig.update_layout(**base_plotly_layout(**layout_overrides))
    return fig
