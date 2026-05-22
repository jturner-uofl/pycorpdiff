"""Known-answer tests for Dunning G².

Reference values are computed by hand from the standard contingency-table
formula (and cross-checked against Rayson's online LL Wizard at
http://ucrel.lancs.ac.uk/llwizard.html). Each test states the expected
value to four decimal places; the assertion tolerance is set to 1e-3 to
absorb floating-point variation across BLAS implementations.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pycorpdiff.keyness.loglikelihood import log_likelihood


def test_equal_rates_yield_zero_g2() -> None:
    # 10/1000 vs 20/2000 — identical relative frequency, so G² == 0.
    a = pd.Series({"x": 10})
    b = pd.Series({"x": 20})
    table = log_likelihood(a, b, total_a=1000, total_b=2000)
    assert math.isclose(table.loc["x", "g2"], 0.0, abs_tol=1e-9)
    assert math.isclose(table.loc["x", "p_value"], 1.0, abs_tol=1e-9)


def test_overuse_in_a_is_signed_positive() -> None:
    # 10/1000 in A vs 0/1000 in B. Expected values are both 5; the second
    # term collapses to 0 by the 0*log(0)=0 convention, so:
    # G² = 2 * 10 * ln(10/5) = 20 * ln(2) = 13.8629...
    a = pd.Series({"x": 10})
    b = pd.Series({"x": 0})
    table = log_likelihood(a, b, total_a=1000, total_b=1000)
    expected_g2 = 20.0 * math.log(2.0)
    assert math.isclose(table.loc["x", "g2"], expected_g2, rel_tol=1e-9)
    assert table.loc["x", "g2"] > 0


def test_overuse_in_b_is_signed_negative() -> None:
    a = pd.Series({"x": 0})
    b = pd.Series({"x": 10})
    table = log_likelihood(a, b, total_a=1000, total_b=1000)
    expected_g2 = -20.0 * math.log(2.0)
    assert math.isclose(table.loc["x", "g2"], expected_g2, rel_tol=1e-9)


def test_rayson_canonical_example() -> None:
    # The classic Rayson worked example: 12000/1M vs 10000/1M.
    # Expected G² ≈ 182.0694 (computed by hand from the same formula
    # that drives the LL Wizard).
    a = pd.Series({"the": 12000})
    b = pd.Series({"the": 10000})
    table = log_likelihood(a, b, total_a=1_000_000, total_b=1_000_000)
    assert math.isclose(table.loc["the", "g2"], 182.0694, abs_tol=1e-3)


def test_large_disparity_yields_large_g2() -> None:
    # 100/100k vs 20/200k = rates 1e-3 vs 1e-4 (10× over-representation).
    # E1 = 100k * 120 / 300k = 40; E2 = 200k * 120 / 300k = 80.
    # G² = 2*(100*ln(100/40) + 20*ln(20/80))
    #    = 2*(100*ln(2.5) + 20*ln(0.25))
    #    = 2*(91.6291 - 27.7259) = 127.8065
    a = pd.Series({"x": 100})
    b = pd.Series({"x": 20})
    table = log_likelihood(a, b, total_a=100_000, total_b=200_000)
    assert math.isclose(table.loc["x", "g2"], 127.8065, abs_tol=1e-3)


def test_p_value_uses_unsigned_g2() -> None:
    # Two rows with equal |G²| but opposite signs must report the same p-value.
    a = pd.Series({"x": 100, "y": 20})
    b = pd.Series({"x": 20, "y": 100})
    table = log_likelihood(a, b, total_a=10_000, total_b=10_000)
    assert math.isclose(table.loc["x", "p_value"], table.loc["y", "p_value"], rel_tol=1e-12)
    assert table.loc["x", "g2"] > 0
    assert table.loc["y", "g2"] < 0


def test_union_aligns_indices_with_zeros() -> None:
    # Terms unique to one side should appear with the other side's count = 0.
    a = pd.Series({"only_a": 10, "shared": 5})
    b = pd.Series({"shared": 5, "only_b": 7})
    table = log_likelihood(a, b, total_a=100, total_b=100)
    assert set(table.index) == {"only_a", "shared", "only_b"}
    assert int(table.loc["only_a", "count_b"]) == 0
    assert int(table.loc["only_b", "count_a"]) == 0
    assert math.isclose(table.loc["shared", "g2"], 0.0, abs_tol=1e-9)


def test_rejects_nonpositive_totals() -> None:
    a = pd.Series({"x": 1})
    b = pd.Series({"x": 1})
    with pytest.raises(ValueError, match="must be positive"):
        log_likelihood(a, b, total_a=0, total_b=10)
    with pytest.raises(ValueError, match="must be positive"):
        log_likelihood(a, b, total_a=10, total_b=-1)


def test_expected_columns_match_contingency_table() -> None:
    a = pd.Series({"x": 30})
    b = pd.Series({"x": 10})
    table = log_likelihood(a, b, total_a=1000, total_b=1000)
    # expected_a = N_a * (O1+O2) / (N_a+N_b) = 1000 * 40 / 2000 = 20.
    assert math.isclose(table.loc["x", "expected_a"], 20.0, rel_tol=1e-9)
    assert math.isclose(table.loc["x", "expected_b"], 20.0, rel_tol=1e-9)


def test_vectorised_run_matches_per_row() -> None:
    # Running once on a Series of length N should equal running N times
    # individually on length-1 Series.
    a = pd.Series({"x": 30, "y": 5, "z": 100})
    b = pd.Series({"x": 10, "y": 50, "z": 95})
    bulk = log_likelihood(a, b, total_a=1_000, total_b=1_000)
    for term in ("x", "y", "z"):
        single = log_likelihood(
            pd.Series({term: a[term]}),
            pd.Series({term: b[term]}),
            total_a=1_000,
            total_b=1_000,
        )
        assert math.isclose(
            bulk.loc[term, "g2"], single.loc[term, "g2"], rel_tol=1e-12
        )


def test_results_are_finite() -> None:
    rng = np.random.default_rng(seed=42)
    terms = [f"t{i}" for i in range(50)]
    a = pd.Series(rng.integers(0, 100, size=50), index=terms)
    b = pd.Series(rng.integers(0, 100, size=50), index=terms)
    table = log_likelihood(a, b, total_a=int(a.sum()) + 100, total_b=int(b.sum()) + 100)
    assert np.isfinite(table["g2"]).all()
    assert (table["p_value"] >= 0).all() and (table["p_value"] <= 1).all()
