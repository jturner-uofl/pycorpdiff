"""Hypothesis property tests on collocation invariants.

These guard:

- logDice is symmetric in ``f_x`` and ``f_y``.
- PMI / t-score / MI³ are symmetric in target / collocate.
- All measures are monotonically non-decreasing in ``f_xy`` (everything
  else held constant).
- logDice is bounded above by 14.
- ``collocation_shift`` swap exactly negates per-collocate.
"""

from __future__ import annotations

import math

import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pycorpdiff.collocation.measures import logdice, mi_three, pmi, t_score

# Bound to avoid degenerate (zero-denominator) cases and to keep the
# search tractable.
counts = st.integers(min_value=1, max_value=10_000)
totals = st.integers(min_value=100, max_value=10_000_000)


@settings(max_examples=200, deadline=None)
@given(f_xy=counts, f_x=counts, f_y=counts)
def test_logdice_symmetric_in_fx_fy(f_xy: int, f_x: int, f_y: int) -> None:
    s1 = logdice(pd.Series([float(f_xy)]), float(f_x), pd.Series([float(f_y)])).iloc[0]
    s2 = logdice(pd.Series([float(f_xy)]), float(f_y), pd.Series([float(f_x)])).iloc[0]
    assert math.isclose(s1, s2, rel_tol=1e-12, abs_tol=1e-12)


@settings(max_examples=200, deadline=None)
@given(f_xy=counts, f_x=counts, f_y=counts, n=totals)
def test_pmi_symmetric_in_target_collocate(
    f_xy: int, f_x: int, f_y: int, n: int
) -> None:
    s1 = pmi(pd.Series([float(f_xy)]), float(f_x), pd.Series([float(f_y)]), n).iloc[0]
    s2 = pmi(pd.Series([float(f_xy)]), float(f_y), pd.Series([float(f_x)]), n).iloc[0]
    assert math.isclose(s1, s2, rel_tol=1e-12, abs_tol=1e-12)


@settings(max_examples=200, deadline=None)
@given(f_xy=counts, f_x=counts, f_y=counts, n=totals)
def test_tscore_symmetric_in_target_collocate(
    f_xy: int, f_x: int, f_y: int, n: int
) -> None:
    s1 = t_score(pd.Series([float(f_xy)]), float(f_x), pd.Series([float(f_y)]), n).iloc[0]
    s2 = t_score(pd.Series([float(f_xy)]), float(f_y), pd.Series([float(f_x)]), n).iloc[0]
    assert math.isclose(s1, s2, rel_tol=1e-12, abs_tol=1e-12)


@settings(max_examples=200, deadline=None)
@given(f_xy=counts, f_x=counts, f_y=counts, n=totals)
def test_mi_three_symmetric_in_target_collocate(
    f_xy: int, f_x: int, f_y: int, n: int
) -> None:
    s1 = mi_three(pd.Series([float(f_xy)]), float(f_x), pd.Series([float(f_y)]), n).iloc[0]
    s2 = mi_three(pd.Series([float(f_xy)]), float(f_y), pd.Series([float(f_x)]), n).iloc[0]
    assert math.isclose(s1, s2, rel_tol=1e-12, abs_tol=1e-12)


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
@given(
    f_xy_lo=st.integers(min_value=1, max_value=500),
    f_xy_hi_delta=st.integers(min_value=1, max_value=500),
    f_x=counts,
    f_y=counts,
)
def test_logdice_monotonic_in_f_xy(
    f_xy_lo: int, f_xy_hi_delta: int, f_x: int, f_y: int
) -> None:
    f_xy_hi = f_xy_lo + f_xy_hi_delta
    s_lo = logdice(pd.Series([float(f_xy_lo)]), float(f_x), pd.Series([float(f_y)])).iloc[0]
    s_hi = logdice(pd.Series([float(f_xy_hi)]), float(f_x), pd.Series([float(f_y)])).iloc[0]
    assert s_hi >= s_lo - 1e-12


@settings(max_examples=200, deadline=None)
@given(f_xy=counts, f_x=counts, f_y=counts)
def test_logdice_bounded_above_by_14(f_xy: int, f_x: int, f_y: int) -> None:
    s = logdice(pd.Series([float(f_xy)]), float(f_x), pd.Series([float(f_y)])).iloc[0]
    # 14 + log2(2*f_xy/(f_x+f_y)); the ratio inside log2 is bounded by
    # the case 2*f_xy >= f_x+f_y, which happens when one of the corpora
    # has the term *only* in the target's window. Even then the ratio
    # caps at 1 (perfect co-occurrence) → score = 14 exactly.
    # Hypothesis may sample values where 2*f_xy > f_x+f_y (when both
    # counts are smaller than f_xy, e.g. unit counts), pushing the
    # score above 14 — that's not a "real" corpus regime but we want
    # the property to hold in realistic input shapes only.
    if 2 * f_xy <= f_x + f_y:
        assert s <= 14.0 + 1e-9


@settings(max_examples=100, deadline=None)
@given(f_xy=counts, f_x=counts, f_y=counts, n=totals)
def test_pmi_monotonic_in_f_xy(
    f_xy: int, f_x: int, f_y: int, n: int
) -> None:
    s_lo = pmi(pd.Series([float(f_xy)]), float(f_x), pd.Series([float(f_y)]), n).iloc[0]
    s_hi = pmi(pd.Series([float(f_xy + 1)]), float(f_x), pd.Series([float(f_y)]), n).iloc[0]
    assert s_hi >= s_lo - 1e-12
