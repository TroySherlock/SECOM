"""Shared tqdm / joblib progress helpers for notebooks and CLIs."""
from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def tqdm_joblib_context(total: int, desc: str, *, leave: bool = True):
    """Wrap parallel joblib work with a tqdm bar when tqdm-joblib is installed."""
    try:
        from tqdm.auto import tqdm
        from tqdm_joblib import tqdm_joblib

        with tqdm_joblib(tqdm(total=total, desc=desc, leave=leave)):
            yield
    except ImportError:
        yield
