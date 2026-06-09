"""Tests for Kleinberg-style burst detection on temporal trajectories."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pycorpdiff as pcd
from pycorpdiff.temporal.burstiness import (
    BurstinessResult,
    burstiness_from_table,
    kleinberg_bursts,
)

# ----------------------------------------------------------------------
# Pure algorithm (`kleinberg_bursts`)
# ----------------------------------------------------------------------


def test_all_zero_counts_returns_all_zero_states() -> None:
    counts = np.zeros(10, dtype=np.int64)
    totals = np.full(10, 1000, dtype=np.int64)
    states = kleinberg_bursts(counts, totals)
    assert states.tolist() == [0] * 10


def test_uniform_rate_returns_all_zero_states() -> None:
    """If rate is constant across periods, no burst should fire."""
    counts = np.full(15, 10, dtype=np.int64)
    totals = np.full(15, 1000, dtype=np.int64)
    states = kleinberg_bursts(counts, totals, s=2.0, gamma=1.0)
    # Every period at the base rate — nothing to burst on.
    assert (states == 0).all()


def test_clear_burst_period_gets_elevated_state() -> None:
    """A sustained spike must be tagged with state >= 1."""
    counts = np.concatenate(
        [
            np.full(5, 1, dtype=np.int64),    # quiet
            np.full(5, 40, dtype=np.int64),   # burst (40x base)
            np.full(5, 1, dtype=np.int64),    # quiet again
        ]
    )
    totals = np.full(15, 100, dtype=np.int64)
    states = kleinberg_bursts(counts, totals, s=2.0, gamma=1.0)
    # The middle window should hit state >= 1.
    assert (states[5:10] >= 1).all()
    # Pre- and post-spike should fall back to state 0.
    assert states[0] == 0
    assert states[-1] == 0


def test_higher_gamma_yields_fewer_or_equal_bursts() -> None:
    """Increasing gamma makes the algorithm more conservative."""
    rng = np.random.default_rng(7)
    base = np.full(40, 5, dtype=np.int64)
    base[15:25] = 25
    counts = base + rng.integers(-1, 2, size=40, endpoint=False).clip(min=0)
    totals = np.full(40, 100, dtype=np.int64)
    loose = kleinberg_bursts(counts, totals, gamma=0.5)
    strict = kleinberg_bursts(counts, totals, gamma=10.0)
    assert (strict > 0).sum() <= (loose > 0).sum()


def test_invalid_s_raises() -> None:
    with pytest.raises(ValueError, match="s must be > 1"):
        kleinberg_bursts([1, 2, 3], [10, 10, 10], s=1.0)


def test_invalid_n_states_raises() -> None:
    with pytest.raises(ValueError, match="n_states must be >= 2"):
        kleinberg_bursts([1, 2, 3], [10, 10, 10], n_states=1)


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        kleinberg_bursts([1, 2, 3], [10, 10])


def test_negative_counts_raise() -> None:
    with pytest.raises(ValueError, match="counts must be non-negative"):
        kleinberg_bursts([1, -2, 3], [10, 10, 10])


def test_zero_totals_raise() -> None:
    with pytest.raises(ValueError, match="positive denominator"):
        kleinberg_bursts([1, 2, 3], [10, 0, 10])


def test_empty_series_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        kleinberg_bursts([], [])


def test_states_bounded_by_n_states() -> None:
    counts = np.concatenate([np.full(5, 1), np.full(5, 100), np.full(5, 1)])
    totals = np.full(15, 100, dtype=np.int64)
    states = kleinberg_bursts(counts, totals, n_states=3)
    assert states.max() < 3
    assert states.min() >= 0


# ----------------------------------------------------------------------
# Trajectory-table convenience (`burstiness_from_table`)
# ----------------------------------------------------------------------


def test_burstiness_from_table_returns_result_dataclass() -> None:
    table = pd.DataFrame(
        {
            "period": pd.PeriodIndex(
                ["2010", "2011", "2012", "2013", "2014"], freq="Y"
            ),
            "term": ["x"] * 5,
            "count": [1, 1, 50, 1, 1],
            "total": [100, 100, 100, 100, 100],
            "relfreq": [0.01, 0.01, 0.5, 0.01, 0.01],
        }
    )
    result = burstiness_from_table(table, target="x")
    assert isinstance(result, BurstinessResult)
    assert result.target == "x"
    assert len(result.states) == 5
    assert result.bursts["start"].iloc[0] == result.bursts["end"].iloc[0]  # spike of length 1


def test_burstiness_from_table_unknown_target_raises() -> None:
    table = pd.DataFrame(
        {
            "period": pd.PeriodIndex(["2010"], freq="Y"),
            "term": ["x"],
            "count": [1],
            "total": [100],
            "relfreq": [0.01],
        }
    )
    with pytest.raises(ValueError, match="not present in trajectory"):
        burstiness_from_table(table, target="missing")


def test_burstiness_from_table_missing_total_column_raises() -> None:
    table = pd.DataFrame(
        {"period": [2010], "term": ["x"], "count": [1], "relfreq": [0.01]}
    )
    with pytest.raises(ValueError, match="total"):
        burstiness_from_table(table, target="x")


# ----------------------------------------------------------------------
# TemporalTrajectory.burstiness wiring
# ----------------------------------------------------------------------


def test_trajectory_burstiness_pipeline_runs() -> None:
    corpus = pcd.load_hansard_sample()
    tr = pcd.track(corpus.slice(topic="immigration"), ["criminal"]).over_time(freq="Y")
    result = tr.burstiness(s=2.0, gamma=1.0)
    assert isinstance(result, BurstinessResult)
    # Bursts table has expected columns.
    assert set(result.bursts.columns) >= {
        "start", "end", "n_periods", "max_state", "total_count"
    }


def test_trajectory_burstiness_requires_target_for_multi_target() -> None:
    corpus = pcd.load_hansard_sample()
    tr = pcd.track(corpus, ["climate", "nhs"]).over_time(freq="Y")
    with pytest.raises(ValueError, match="trajectory carries"):
        tr.burstiness()


def test_trajectory_burstiness_unknown_target_raises() -> None:
    corpus = pcd.load_hansard_sample()
    tr = pcd.track(corpus, ["climate"]).over_time(freq="Y")
    with pytest.raises(ValueError, match="not in trajectory targets"):
        tr.burstiness(target="bogus")


def test_burstiness_summary_string_for_zero_bursts() -> None:
    table = pd.DataFrame(
        {
            "period": pd.PeriodIndex(["2010", "2011", "2012"], freq="Y"),
            "term": ["x"] * 3,
            "count": [5, 5, 5],
            "total": [100, 100, 100],
            "relfreq": [0.05, 0.05, 0.05],
        }
    )
    result = burstiness_from_table(table, target="x")
    s = result.summary()
    assert "no bursts" in s or "0 burst" in s
