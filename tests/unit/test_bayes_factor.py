"""Known-answer tests for the BIC-approximated Bayes factor (Wilson 2013)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pycorpdiff.keyness.bayes import bayes_factor


def test_equal_rates_yields_bf_below_one() -> None:
    # G² ≈ 0, so BIC = -ln(N) and BF = exp(-ln(N)/2) = 1/sqrt(N).
    a = pd.Series({"x": 10})
    b = pd.Series({"x": 20})
    bf = bayes_factor(a, b, total_a=1_000, total_b=2_000)
    expected = 1.0 / math.sqrt(3_000)
    assert math.isclose(bf["x"], expected, rel_tol=1e-6)


def test_bf_matches_manual_formula() -> None:
    # 10/1000 vs 0/1000 → G² = 20*ln(2) = 13.8629, N = 2000.
    # BIC = 13.8629 - ln(2000) = 13.8629 - 7.6009 = 6.2620
    # BF = exp(6.2620 / 2) = exp(3.1310) ≈ 22.89.
    a = pd.Series({"x": 10})
    b = pd.Series({"x": 0})
    bf = bayes_factor(a, b, total_a=1_000, total_b=1_000)
    g2_abs = 20.0 * math.log(2.0)
    bic = g2_abs - math.log(2_000)
    expected = math.exp(bic / 2.0)
    assert math.isclose(bf["x"], expected, rel_tol=1e-6)


def test_very_strong_evidence_overflows_to_inf_without_warning() -> None:
    # G² well over 1000 → BIC/2 > 700 → exp overflows float64. The function
    # must suppress the runtime warning (we run with filterwarnings=error)
    # and return inf rather than crashing.
    a = pd.Series({"x": 1_000_000})
    b = pd.Series({"x": 0})
    bf = bayes_factor(a, b, total_a=1_000_000, total_b=1_000_000)
    assert math.isinf(bf["x"]) and bf["x"] > 0


def test_bf_swap_symmetry() -> None:
    # BF depends only on |G²|, so it must be invariant under corpus swap.
    a = pd.Series({"x": 50, "y": 10})
    b = pd.Series({"x": 10, "y": 50})
    bf_ab = bayes_factor(a, b, total_a=1_000, total_b=1_000)
    bf_ba = bayes_factor(b, a, total_a=1_000, total_b=1_000)
    np.testing.assert_allclose(bf_ab.to_numpy(), bf_ba.to_numpy(), rtol=1e-9)
