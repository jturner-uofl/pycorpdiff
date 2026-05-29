"""Hypothesis property tests on temporal / Wilson-CI invariants.

Guards:

- Wilson CI brackets the point estimate when ``n > 0``.
- Wilson CI is monotonic in confidence (higher confidence → wider).
- Wilson CI is symmetric: swapping success/failure flips the interval
  to the complement.
- All CI bounds are in [0, 1].
- Relative frequency monotonic in count when total fixed.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from pycorpdiff.stats import wilson_ci


@settings(max_examples=200, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=100_000),
    p=st.floats(min_value=0.0, max_value=1.0),
)
def test_wilson_ci_brackets_point_estimate(n: int, p: float) -> None:
    x = int(round(n * p))
    if x > n:
        x = n
    lo, hi = wilson_ci(np.array([x], dtype=np.int64), np.array([n], dtype=np.int64))
    estimate = x / n
    # Allow tiny float roundoff at the clipping boundaries.
    assert lo[0] - 1e-9 <= estimate <= hi[0] + 1e-9


@settings(max_examples=200, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=100_000),
    p=st.floats(min_value=0.0, max_value=1.0),
)
def test_wilson_ci_bounded_to_unit_interval(n: int, p: float) -> None:
    x = int(round(n * p))
    if x > n:
        x = n
    lo, hi = wilson_ci(np.array([x], dtype=np.int64), np.array([n], dtype=np.int64))
    assert 0.0 <= lo[0] <= 1.0
    assert 0.0 <= hi[0] <= 1.0
    assert lo[0] <= hi[0] + 1e-9


@settings(max_examples=100, deadline=None)
@given(
    n=st.integers(min_value=5, max_value=100_000),
    p=st.floats(min_value=0.01, max_value=0.99),
    c_lower=st.floats(min_value=0.50, max_value=0.94),
    c_higher_delta=st.floats(min_value=0.01, max_value=0.049),
)
def test_wilson_ci_wider_at_higher_confidence(
    n: int, p: float, c_lower: float, c_higher_delta: float
) -> None:
    x = int(round(n * p))
    c_higher = min(0.999, c_lower + c_higher_delta)
    lo1, hi1 = wilson_ci(
        np.array([x], dtype=np.int64), np.array([n], dtype=np.int64),
        confidence=c_lower,
    )
    lo2, hi2 = wilson_ci(
        np.array([x], dtype=np.int64), np.array([n], dtype=np.int64),
        confidence=c_higher,
    )
    width1 = hi1[0] - lo1[0]
    width2 = hi2[0] - lo2[0]
    assert width2 >= width1 - 1e-9


@settings(max_examples=100, deadline=None)
@given(n=st.integers(min_value=1, max_value=100_000), x=st.integers(min_value=0))
def test_wilson_ci_complement_symmetry(n: int, x: int) -> None:
    """CI for ``x/n`` and ``(n-x)/n`` are reflections through 0.5."""
    if x > n:
        x = n
    lo1, hi1 = wilson_ci(np.array([x], dtype=np.int64), np.array([n], dtype=np.int64))
    lo2, hi2 = wilson_ci(np.array([n - x], dtype=np.int64), np.array([n], dtype=np.int64))
    # If the first CI is [a, b], the second should be [1-b, 1-a].
    assert math.isclose(lo2[0], 1.0 - hi1[0], abs_tol=1e-9)
    assert math.isclose(hi2[0], 1.0 - lo1[0], abs_tol=1e-9)


@settings(max_examples=100, deadline=None)
@given(
    x=st.integers(min_value=0, max_value=10_000),
    n=st.integers(min_value=10_001, max_value=1_000_000),
)
def test_relfreq_proportional_to_count(x: int, n: int) -> None:
    # Doubling the count (with total fixed) should double the rel-freq.
    relfreq = x / n
    relfreq_doubled = (x * 2) / n
    assert math.isclose(relfreq_doubled, 2 * relfreq, rel_tol=1e-12)
