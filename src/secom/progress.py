"""Shared tqdm / joblib progress helpers for notebooks and CLIs."""
from __future__ import annotations

from contextlib import contextmanager


def disable_tqdm_monitor() -> None:
    """Stop tqdm's background TMonitor thread (avoids the WeakSet race that
    raises 'RuntimeError: Set changed size during iteration' when many nested
    bars are created/closed quickly). The monitor only does stall auto-refresh,
    so disabling it does not affect correctness or normal bar display."""
    try:
        from tqdm.auto import tqdm

        tqdm.monitor_interval = 0
    except ImportError:
        pass


disable_tqdm_monitor()


@contextmanager
def tqdm_joblib_context(total: int, desc: str, *, leave: bool = True):
    """Wrap parallel joblib work with a tqdm bar when tqdm-joblib is installed."""
    try:
        from tqdm.auto import tqdm
        from tqdm_joblib import tqdm_joblib

        disable_tqdm_monitor()
        with tqdm_joblib(tqdm(total=total, desc=desc, leave=leave)):
            yield
    except ImportError:
        yield
