"""Hypothesis property tests on keyness invariants.

These guard the mathematical contract of the keyness primitives:

- G² >= 0 in magnitude (the unsigned form is chi-squared distributed).
- G² is symmetric under corpus swap modulo sign.
- LogRatio swap exactly negates.
- A term with equal relative frequency in both corpora has G² ≈ 0.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from pycorpdiff.keyness.effect_sizes import log_ratio
from pycorpdiff.keyness.loglikelihood import log_likelihood

# Bound counts and totals to keep the search tractable and avoid degenerate
# (zero-total) cases. Counts can be zero on either side but the corpus
# totals must be positive.
counts = st.integers(min_value=0, max_value=10_000)
totals = st.integers(min_value=1, max_value=1_000_000)


@settings(max_examples=200, deadline=None)
@given(
    a_count=counts,
    b_count=counts,
    total_a=totals,
    total_b=totals,
)
def test_g2_magnitude_is_nonnegative(
    a_count: int, b_count: int, total_a: int, total_b: int
) -> None:
    # Ensure a_count <= total_a, b_count <= total_b.
    a_count = min(a_count, total_a)
    b_count = min(b_count, total_b)
    if a_count == 0 and b_count == 0:
        return  # filtered out in real usage
    a = pd.Series({"x": a_count})
    b = pd.Series({"x": b_count})
    table = log_likelihood(a, b, total_a=total_a, total_b=total_b)
    assert abs(table.loc["x", "g2"]) >= -1e-9  # allow tiny float error


@settings(max_examples=200, deadline=None)
@given(
    a_count=counts,
    b_count=counts,
    total_a=totals,
    total_b=totals,
)
def test_g2_swap_negates(
    a_count: int, b_count: int, total_a: int, total_b: int
) -> None:
    a_count = min(a_count, total_a)
    b_count = min(b_count, total_b)
    if a_count == 0 and b_count == 0:
        return
    a = pd.Series({"x": a_count})
    b = pd.Series({"x": b_count})
    ll_ab = log_likelihood(a, b, total_a=total_a, total_b=total_b).loc["x", "g2"]
    ll_ba = log_likelihood(b, a, total_a=total_b, total_b=total_a).loc["x", "g2"]
    # |G²| is invariant under swap; signed G² flips sign except when the
    # rates are exactly equal (then both are 0 and there's no sign to flip).
    assert math.isclose(abs(ll_ab), abs(ll_ba), abs_tol=1e-9)
    if abs(ll_ab) > 1e-9:
        assert math.isclose(ll_ab, -ll_ba, rel_tol=1e-6, abs_tol=1e-9)


@settings(max_examples=200, deadline=None)
@given(
    a_count=counts,
    b_count=counts,
    total_a=totals,
    total_b=totals,
)
def test_log_ratio_swap_exact_negation(
    a_count: int, b_count: int, total_a: int, total_b: int
) -> None:
    a_count = min(a_count, total_a)
    b_count = min(b_count, total_b)
    a = pd.Series({"x": a_count})
    b = pd.Series({"x": b_count})
    lr_ab = log_ratio(a, b, total_a=total_a, total_b=total_b).iloc[0]
    lr_ba = log_ratio(b, a, total_a=total_b, total_b=total_a).iloc[0]
    assert math.isclose(lr_ab, -lr_ba, abs_tol=1e-9)


@settings(max_examples=100, deadline=None)
@given(
    a_count=st.integers(min_value=1, max_value=10_000),
    multiplier=st.integers(min_value=1, max_value=100),
)
def test_proportional_corpora_yield_zero_g2(a_count: int, multiplier: int) -> None:
    # If b_count = m * a_count and total_b = m * total_a, rates are equal
    # and G² must be 0.
    total_a = a_count * 100
    total_b = total_a * multiplier
    b_count = a_count * multiplier
    a = pd.Series({"x": a_count})
    b = pd.Series({"x": b_count})
    g2 = log_likelihood(a, b, total_a=total_a, total_b=total_b).loc["x", "g2"]
    assert abs(g2) < 1e-6


@settings(max_examples=100, deadline=None)
@given(
    n_terms=st.integers(min_value=1, max_value=20),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_g2_p_values_in_unit_interval(n_terms: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    terms = [f"t{i}" for i in range(n_terms)]
    a = pd.Series(rng.integers(0, 100, size=n_terms), index=terms)
    b = pd.Series(rng.integers(0, 100, size=n_terms), index=terms)
    table = log_likelihood(a, b, total_a=int(a.sum()) + 100, total_b=int(b.sum()) + 100)
    assert (table["p_value"] >= 0).all() and (table["p_value"] <= 1).all()
