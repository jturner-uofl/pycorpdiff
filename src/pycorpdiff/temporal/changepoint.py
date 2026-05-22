"""Changepoint detection on temporal frequency / similarity series.

Wraps the ``ruptures`` library; importable only when the ``temporal``
extra is installed.
"""

from __future__ import annotations

import pandas as pd


def detect_changepoints(
    series: pd.Series,
    method: str = "pelt",
    penalty: float | str = "bic",
) -> pd.DataFrame:
    """Detect changepoints in a temporal frequency series.

    Returns a DataFrame with columns ``period`` and ``method``.
    """
    raise NotImplementedError("detect_changepoints() lands in Phase 7")
