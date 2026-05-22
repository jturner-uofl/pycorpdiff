"""Interrupted time-series analysis around a known event date.

Wraps ``statsmodels``; importable only when the ``temporal`` extra is
installed.
"""

from __future__ import annotations

import pandas as pd


def interrupted_time_series(
    series: pd.Series,
    event_date: str,
) -> pd.DataFrame:
    """Fit an interrupted-time-series model around ``event_date``.

    Returns a one-row DataFrame with level-change and slope-change
    coefficients, standard errors, and confidence intervals.
    """
    raise NotImplementedError("interrupted_time_series() lands in Phase 7")
