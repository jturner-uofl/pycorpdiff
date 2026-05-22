"""Interrupted time-series analysis around a known event date.

Implements the standard segmented-regression specification:

    y_t = β₀ + β₁·t + β₂·post_t + β₃·(t − t_event)·post_t + ε

where ``post_t`` is 1 when ``t ≥ t_event`` and 0 otherwise. The
coefficients of interest:

- ``β₂`` (level change): the immediate step at the intervention.
- ``β₃`` (slope change): how the post-period trend differs from
  the pre-period trend.

Reference
---------
Wagner, A. K., Soumerai, S. B., Zhang, F., & Ross-Degnan, D. (2002).
Segmented regression analysis of interrupted time series studies in
medication use research. *Journal of Clinical Pharmacy and
Therapeutics*, 27(4), 299-309.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def interrupted_time_series(
    series: pd.Series,
    event_date: str | pd.Period | pd.Timestamp,
) -> pd.DataFrame:
    """Fit a segmented-regression ITS model around ``event_date``.

    Parameters
    ----------
    series
        A series indexed by period (or datetime). Index values must
        compare against ``event_date`` to produce the pre/post split.
    event_date
        Where to place the intervention. Anything pandas can compare to
        the series index — a string, a :class:`pandas.Period`, a
        :class:`pandas.Timestamp`.

    Returns
    -------
    pandas.DataFrame
        Four rows (one per coefficient), columns ``term``, ``coef``,
        ``std_err``, ``t``, ``p_value``, ``ci_lower``, ``ci_upper``.
        ``term`` values: ``"intercept"``, ``"time"``, ``"level_change"``,
        ``"slope_change"``.
    """
    try:
        import statsmodels.api as sm
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "interrupted_time_series requires statsmodels. "
            "Install with: pip install 'pycorpdiff[temporal]'"
        ) from exc

    if len(series) < 4:
        raise ValueError(f"need at least 4 observations; got {len(series)}")
    if series.isna().any():
        raise ValueError("series contains NaN values; impute or drop them first")

    y = series.to_numpy(dtype=float)
    # Use a 0-based time index — coefficients are then interpretable as
    # "per period" effects without anchoring to a Unix epoch.
    t = np.arange(len(series), dtype=float)

    event_norm = _normalise_event(event_date, series)
    # Find the first index where index value is >= event_date.
    idx = series.index
    post_mask = np.array([_ge(period, event_norm) for period in idx], dtype=int)
    if post_mask.sum() == 0:
        raise ValueError(f"event_date={event_date!r} is after the last period")
    if post_mask.sum() == len(series):
        raise ValueError(f"event_date={event_date!r} is before the first period")

    event_t = float(t[post_mask.astype(bool)][0])
    time_after = (t - event_t) * post_mask

    x = np.column_stack([np.ones_like(t), t, post_mask.astype(float), time_after])
    model = sm.OLS(y, x).fit()

    conf_int = model.conf_int()
    terms = ["intercept", "time", "level_change", "slope_change"]
    return pd.DataFrame(
        {
            "term": terms,
            "coef": model.params,
            "std_err": model.bse,
            "t": model.tvalues,
            "p_value": model.pvalues,
            "ci_lower": conf_int[:, 0],
            "ci_upper": conf_int[:, 1],
        }
    )


def _normalise_event(
    event: str | pd.Period | pd.Timestamp, series: pd.Series
) -> pd.Period | pd.Timestamp:
    """Coerce ``event`` to the same type as the series index for comparison.

    The pandas type hierarchy here is a bit fiddly: ``Period`` and
    ``Timestamp`` aren't directly comparable, and ``Period(event,
    freq=BaseOffset)`` isn't accepted by mypy even though it works at
    runtime. We route through string forms where types disagree.
    """
    sample = series.index[0]
    if isinstance(sample, pd.Period):
        if isinstance(event, pd.Period):
            return event
        freqstr: str = str(sample.freqstr)
        return pd.Period(str(event), freq=freqstr)
    if isinstance(event, pd.Period):
        return pd.Timestamp(str(event))
    return pd.Timestamp(event)


def _ge(period_index_value: object, event: object) -> bool:
    """``period_index_value >= event`` with sensible coercions."""
    return bool(period_index_value >= event)  # type: ignore[operator]
