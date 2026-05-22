"""Known-answer tests for the Wilson score interval."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pycorpdiff.stats import wilson_ci


def test_wilson_ci_known_value() -> None:
    # For x=10, n=100 at 95% the Wilson CI is approximately (0.0552, 0.1747).
    # (Reference: Newcombe 1998, Table 1.)
    lo, hi = wilson_ci(np.array([10], dtype=np.int64), np.array([100], dtype=np.int64))
    assert math.isclose(lo[0], 0.05533, abs_tol=1e-3)
    assert math.isclose(hi[0], 0.17467, abs_tol=1e-3)


def test_wilson_ci_zero_count_lower_is_zero() -> None:
    # With x=0 the Wilson lower bound is exactly 0; upper bound is finite.
    lo, hi = wilson_ci(np.array([0], dtype=np.int64), np.array([100], dtype=np.int64))
    assert math.isclose(lo[0], 0.0, abs_tol=1e-9)
    assert hi[0] > 0.0


def test_wilson_ci_full_count_upper_is_one() -> None:
    lo, hi = wilson_ci(np.array([100], dtype=np.int64), np.array([100], dtype=np.int64))
    assert math.isclose(hi[0], 1.0, abs_tol=1e-9)
    assert lo[0] < 1.0


def test_wilson_ci_zero_n_returns_nan() -> None:
    lo, hi = wilson_ci(np.array([0], dtype=np.int64), np.array([0], dtype=np.int64))
    assert math.isnan(lo[0]) and math.isnan(hi[0])


def test_wilson_ci_bounded_to_unit_interval() -> None:
    # For tiny n the unclamped formula can spill outside [0, 1] at the edges.
    lo, hi = wilson_ci(np.array([5], dtype=np.int64), np.array([5], dtype=np.int64))
    assert 0.0 <= lo[0] <= 1.0
    assert 0.0 <= hi[0] <= 1.0


def test_wilson_ci_vectorised() -> None:
    x = np.array([10, 50, 0], dtype=np.int64)
    n = np.array([100, 100, 100], dtype=np.int64)
    lo, hi = wilson_ci(x, n)
    assert lo.shape == (3,) and hi.shape == (3,)
    # Sanity: x=10 gives narrower lower than x=50; x=0 lower is 0.
    assert lo[0] < lo[1]
    assert math.isclose(lo[2], 0.0, abs_tol=1e-9)


def test_wilson_ci_invalid_confidence_raises() -> None:
    with pytest.raises(ValueError, match="confidence"):
        wilson_ci(np.array([1], dtype=np.int64), np.array([10], dtype=np.int64), confidence=0.0)
    with pytest.raises(ValueError, match="confidence"):
        wilson_ci(np.array([1], dtype=np.int64), np.array([10], dtype=np.int64), confidence=1.5)


def test_wilson_ci_99_percent_wider_than_95() -> None:
    lo95, hi95 = wilson_ci(
        np.array([10], dtype=np.int64), np.array([100], dtype=np.int64), confidence=0.95
    )
    lo99, hi99 = wilson_ci(
        np.array([10], dtype=np.int64), np.array([100], dtype=np.int64), confidence=0.99
    )
    assert lo99[0] < lo95[0]
    assert hi99[0] > hi95[0]
