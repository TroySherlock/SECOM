"""SECOM Streamlit dashboard package."""
from scripts.dashboard.app import (
    configure_page,
    ensure_repo_on_path,
    inject_dashboard_styles,
    render_blue_note,
)

__all__ = [
    "configure_page",
    "ensure_repo_on_path",
    "inject_dashboard_styles",
    "render_blue_note",
]
