"""Canonical Gruvbox Material Light palette (sainnhe/gruvbox-material)."""
from __future__ import annotations

from typing import Literal

# Core surface + semantic colors
GRUVBOX: dict[str, str] = {
    "bg": "#fbf1c7",
    "bg_soft": "#f2e5bc",
    "fg": "#654735",
    "fg_muted": "#928374",
    "pass": "#50a14f",
    "fail": "#ea6962",
    "accent": "#4e9acc",
    "border": "#e6d8bb",
    "orange": "#e78a4e",
    "yellow": "#d8a657",
    "purple": "#b16286",
}

GRUVBOX_TABLE: dict[str, str] = {
    "bg": "#3c3836",
    "bg_alt": "#32302f",
    "header_bg": "#504945",
    "fg_light": "#fbf1c7",
    "fg_muted": "#bdae93",
    "border": "#928374",
}

GRUVBOX_CODE: dict[str, str] = {
    "bg": GRUVBOX["bg"],
    "border": GRUVBOX["bg"],
    "fg": GRUVBOX["fg"],
}

GRUVBOX_EDU_TABLE: dict[str, str] = {
    "cell_bg": GRUVBOX["bg_soft"],
    "header_bg": GRUVBOX["yellow"],
}

CalloutKind = Literal["note", "edu", "pipeline"]

CALLOUT_STYLES: dict[CalloutKind, dict[str, str]] = {
    "note": {
        "bg": "rgba(78, 154, 204, 0.14)",
        "border": GRUVBOX["accent"],
        "fg": GRUVBOX["fg"],
    },
    "edu": {
        "bg": "#f0dfbc",
        "border": GRUVBOX["yellow"],
        "fg": GRUVBOX["fg"],
    },
    "pipeline": {
        "bg": "#ebe0e8",
        "border": GRUVBOX["purple"],
        "fg": GRUVBOX["fg"],
    },
}

PLOTLY_COLORWAY: list[str] = [
    GRUVBOX["pass"],
    GRUVBOX["fail"],
    GRUVBOX["accent"],
    GRUVBOX["orange"],
    GRUVBOX["yellow"],
]

# Keys map to [theme] in .streamlit/config.toml
STREAMLIT_THEME: dict[str, str] = {
    "primaryColor": GRUVBOX["accent"],
    "backgroundColor": GRUVBOX["bg"],
    "secondaryBackgroundColor": GRUVBOX["bg_soft"],
    "textColor": GRUVBOX["fg"],
    "borderColor": GRUVBOX["border"],
}
