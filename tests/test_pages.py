"""Smoke test: each Streamlit page runs without raising an uncaught exception.

Uses ``streamlit.testing`` to execute every page against the existing frozen
artifacts. Some pages fit models live, so the timeout is generous. Pages that
cannot find their local data (e.g. a missing ``data/secom.duckdb``) are skipped
rather than failed, so the suite still passes in a stripped-down checkout.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGES_DIR = REPO_ROOT / "app_pages"
PAGE_FILES = sorted(p.name for p in PAGES_DIR.glob("*.py"))

# Substrings that indicate a missing-data environment rather than a code bug.
_DATA_ERROR_HINTS = ("duckdb", "no such file", "not found", "catalog error", "io error")


@pytest.mark.parametrize("page", PAGE_FILES)
def test_page_runs(page: str) -> None:
    at = AppTest.from_file(str(PAGES_DIR / page), default_timeout=180)
    at.run()
    if at.exception:
        message = " ".join(str(e.message or e.value or "") for e in at.exception).lower()
        if any(hint in message for hint in _DATA_ERROR_HINTS):
            pytest.skip(f"{page}: local data artifacts unavailable")
        raise AssertionError(f"{page} raised: {message}")
