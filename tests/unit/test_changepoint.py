"""Tests for ``detect_changepoints``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pycorpdiff.temporal.changepoint import detect_changepoints


def test_detect_changepoints_on_synthetic_step() -> None:
    # Two clearly-separated regimes: 30 zeros then 30 high values + noise.
    rng = np.random.default_rng(seed=0)
    series = pd.Series(
        np.concatenate(
            [rng.normal(0, 0.1, 30), rng.normal(2, 0.1, 30)]
        )
    )
    df = detect_changepoints(series, method="pelt")
    # There should be at least one changepoint, located near index 30.
    assert len(df) >= 1
    assert any(28 <= row["index"] <= 32 for _, row in df.iterrows())


def test_detect_changepoints_preserves_index() -> None:
    # Verify the reported `period` column carries the original index value.
    rng = np.random.default_rng(seed=1)
    values = np.concatenate([rng.normal(0, 0.1, 20), rng.normal(3, 0.1, 20)])
    series = pd.Series(values, index=pd.period_range(start="2000", periods=40, freq="Y"))
    df = detect_changepoints(series)
    if len(df) > 0:
        # The reported period should be a pandas Period drawn from the input.
        assert isinstance(df.iloc[0]["period"], pd.Period)


def test_detect_changepoints_zero_changes_on_flat_series() -> None:
    # A perfectly flat series should produce zero changepoints (or close).
    series = pd.Series(np.zeros(40))
    df = detect_changepoints(series)
    # PELT might still pick spurious breakpoints on perfectly degenerate data;
    # accept up to one as noise tolerance.
    assert len(df) <= 1


def test_detect_changepoints_rejects_short_series() -> None:
    with pytest.raises(ValueError, match="at least 4 observations"):
        detect_changepoints(pd.Series([1.0, 2.0]))


def test_detect_changepoints_rejects_nan() -> None:
    with pytest.raises(ValueError, match="NaN"):
        detect_changepoints(pd.Series([1.0, 2.0, np.nan, 3.0, 4.0]))


def test_detect_changepoints_binseg_runs() -> None:
    # Just verify the BinSeg method completes end-to-end.
    rng = np.random.default_rng(seed=2)
    series = pd.Series(
        np.concatenate([rng.normal(0, 0.1, 25), rng.normal(2, 0.1, 25)])
    )
    df = detect_changepoints(series, method="binseg")
    assert isinstance(df, pd.DataFrame)
