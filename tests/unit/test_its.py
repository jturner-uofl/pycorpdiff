"""Tests for ``interrupted_time_series``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pycorpdiff.temporal.its import interrupted_time_series


def test_its_detects_level_change_on_synthetic_step() -> None:
    # 20 periods at level 0, then 20 periods at level 5. ITS should
    # report a significant positive level_change ≈ 5.
    rng = np.random.default_rng(seed=42)
    index = pd.period_range(start="2000", periods=40, freq="Y")
    values = np.concatenate([rng.normal(0, 0.5, 20), rng.normal(5, 0.5, 20)])
    series = pd.Series(values, index=index)
    df = interrupted_time_series(series, event_date="2020")
    level = df[df["term"] == "level_change"].iloc[0]
    assert level["coef"] > 3.0  # we engineered a +5 jump
    assert level["p_value"] < 0.01


def test_its_detects_slope_change_on_synthetic_trend() -> None:
    # No level jump but slope changes from 0 to +0.5 per period.
    index = pd.period_range(start="2000", periods=40, freq="Y")
    t = np.arange(40, dtype=float)
    pre_slope = 0.0
    post_slope = 0.5
    event_t = 20.0
    trend = pre_slope * t + np.where(t >= event_t, post_slope * (t - event_t), 0.0)
    rng = np.random.default_rng(seed=7)
    series = pd.Series(trend + rng.normal(0, 0.2, 40), index=index)
    df = interrupted_time_series(series, event_date="2020")
    slope = df[df["term"] == "slope_change"].iloc[0]
    assert slope["coef"] > 0.3
    assert slope["p_value"] < 0.01


def test_its_returns_expected_schema() -> None:
    index = pd.period_range(start="2000", periods=20, freq="Y")
    series = pd.Series(np.linspace(0, 10, 20) + 0.01, index=index)
    df = interrupted_time_series(series, event_date="2010")
    assert list(df["term"]) == ["intercept", "time", "level_change", "slope_change"]
    assert list(df.columns) == [
        "term",
        "coef",
        "std_err",
        "t",
        "p_value",
        "ci_lower",
        "ci_upper",
    ]


def test_its_event_after_series_raises() -> None:
    index = pd.period_range(start="2000", periods=10, freq="Y")
    series = pd.Series(np.arange(10, dtype=float) + 0.01, index=index)
    with pytest.raises(ValueError, match="after the last period"):
        interrupted_time_series(series, event_date="2050")


def test_its_event_before_series_raises() -> None:
    index = pd.period_range(start="2000", periods=10, freq="Y")
    series = pd.Series(np.arange(10, dtype=float) + 0.01, index=index)
    with pytest.raises(ValueError, match="before the first period"):
        interrupted_time_series(series, event_date="1990")


def test_its_rejects_short_series() -> None:
    with pytest.raises(ValueError, match="at least 4 observations"):
        interrupted_time_series(pd.Series([1.0, 2.0]), event_date="2020")


def test_its_rejects_nan() -> None:
    index = pd.period_range(start="2000", periods=5, freq="Y")
    s = pd.Series([1.0, 2.0, np.nan, 3.0, 4.0], index=index)
    with pytest.raises(ValueError, match="NaN"):
        interrupted_time_series(s, event_date="2002")
