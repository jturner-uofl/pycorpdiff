"""Known-answer tests for Pearson χ² keyness."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pycorpdiff.keyness.chi_squared import chi_squared


def test_chi_squared_equal_rates_yields_zero() -> None:
    """When rates match, χ² == 0 exactly."""
    a = pd.Series({"x": 10})
    b = pd.Series({"x": 20})
    table = chi_squared(a, b, total_a=1000, total_b=2000)
    assert math.isclose(table.loc["x", "chi_squared"], 0.0, abs_tol=1e-9)
    assert math.isclose(table.loc["x", "p_value"], 1.0, abs_tol=1e-9)


def test_chi_squared_2x2_closed_form() -> None:
    """Verify the 2×2 closed-form formula against a hand-computation.

    Contingency table:
                A         B
        term    100       20
        not     99900     199980
        total   100000    200000

    χ² = ((100·199980 − 20·99900)² · 300000) / (120 · 299880 · 100000 · 200000)
       = (17988000² · 300000) / (120 · 299880 · 100000 · 200000)
       = (3.2356808e14 · 3e5) / (7.19712e18)
       = 9.70704e19 / 7.19712e18
       ≈ 134.872

    (G² for the same input is ~127.81; the two diverge slightly for
    moderate cell counts but converge as N grows.)
    """
    a = pd.Series({"x": 100})
    b = pd.Series({"x": 20})
    table = chi_squared(a, b, total_a=100_000, total_b=200_000)
    chi2 = float(table.loc["x", "chi_squared"])
    assert math.isclose(chi2, 134.872, rel_tol=0.005)
    assert chi2 > 0  # A-overuse → signed positive


def test_chi_squared_signed_overuse_in_b_is_negative() -> None:
    a = pd.Series({"x": 0})
    b = pd.Series({"x": 50})
    table = chi_squared(a, b, total_a=10_000, total_b=10_000)
    assert table.loc["x", "chi_squared"] < 0


def test_chi_squared_p_value_uses_unsigned() -> None:
    """Two rows with opposite-sign equal-magnitude χ² report the same p."""
    a = pd.Series({"x": 100, "y": 20})
    b = pd.Series({"x": 20, "y": 100})
    table = chi_squared(a, b, total_a=10_000, total_b=10_000)
    assert math.isclose(
        table.loc["x", "p_value"], table.loc["y", "p_value"], rel_tol=1e-12
    )
    assert table.loc["x", "chi_squared"] * table.loc["y", "chi_squared"] < 0


def test_chi_squared_rejects_nonpositive_totals() -> None:
    a = pd.Series({"x": 1})
    b = pd.Series({"x": 1})
    with pytest.raises(ValueError, match="must be positive"):
        chi_squared(a, b, total_a=0, total_b=10)


def test_chi_squared_approximately_equals_log_likelihood_for_large_n() -> None:
    """For large N and not-tiny cells, χ² ≈ G². Standard asymptotic result."""
    from pycorpdiff.keyness.loglikelihood import log_likelihood

    a = pd.Series({"x": 1000})
    b = pd.Series({"x": 900})
    chi_table = chi_squared(a, b, total_a=100_000, total_b=100_000)
    ll_table = log_likelihood(a, b, total_a=100_000, total_b=100_000)
    chi2 = float(chi_table.loc["x", "chi_squared"])
    g2 = float(ll_table.loc["x", "g2"])
    # Both signed positive. Difference should be a few percent at most.
    assert math.isclose(abs(chi2), abs(g2), rel_tol=0.05)
    assert math.copysign(1.0, chi2) == math.copysign(1.0, g2)


def test_chi_squared_handles_zero_in_b() -> None:
    """The 2×2 denominator is well-defined when one count is zero."""
    a = pd.Series({"x": 10})
    b = pd.Series({"x": 0})
    table = chi_squared(a, b, total_a=1000, total_b=1000)
    assert table.loc["x", "chi_squared"] > 0
    assert np.isfinite(table.loc["x", "chi_squared"])
