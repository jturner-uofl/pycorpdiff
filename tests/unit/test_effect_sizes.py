"""Known-answer tests for LogRatio (Hardie) and %DIFF (Gabrielatos)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pycorpdiff.keyness.effect_sizes import log_ratio, percent_diff


def test_log_ratio_equal_rates_is_zero() -> None:
    a = pd.Series({"x": 10})
    b = pd.Series({"x": 20})
    lr = log_ratio(a, b, total_a=1_000, total_b=2_000)
    # With smoothing α=0.5, rates become 10.5/1000 vs 20.5/2000 — very
    # nearly equal but not exactly. Should be within 0.05 of zero.
    assert abs(lr["x"]) < 0.05


def test_log_ratio_unsmoothed_doubling() -> None:
    # 20/1000 in A vs 10/1000 in B is a 2× over-rep — LogRatio ≈ 1.
    a = pd.Series({"x": 20})
    b = pd.Series({"x": 10})
    lr = log_ratio(a, b, total_a=1_000, total_b=1_000, smoothing=0.5)
    # With smoothing: log2((20.5/1000) / (10.5/1000)) = log2(20.5/10.5)
    expected = math.log2(20.5 / 10.5)
    assert math.isclose(lr["x"], expected, rel_tol=1e-9)


def test_log_ratio_handles_zero_in_b() -> None:
    # Without smoothing this would be log2(N/0) = +inf. With α=0.5 it's
    # finite: log2((10.5/1000) / (0.5/1000)) = log2(21) ≈ 4.39.
    a = pd.Series({"x": 10})
    b = pd.Series({"x": 0})
    lr = log_ratio(a, b, total_a=1_000, total_b=1_000)
    expected = math.log2(10.5 / 0.5)
    assert math.isclose(lr["x"], expected, rel_tol=1e-9)
    assert np.isfinite(lr["x"])


def test_log_ratio_swap_negates() -> None:
    # log_ratio(a, b) == -log_ratio(b, a) — exact when smoothing is symmetric.
    a = pd.Series({"x": 30, "y": 5})
    b = pd.Series({"x": 10, "y": 50})
    lr_ab = log_ratio(a, b, total_a=1_000, total_b=1_000)
    lr_ba = log_ratio(b, a, total_a=1_000, total_b=1_000)
    np.testing.assert_allclose(lr_ab.to_numpy(), -lr_ba.to_numpy(), rtol=1e-12)


def test_log_ratio_rejects_nonpositive_smoothing() -> None:
    a = pd.Series({"x": 1})
    b = pd.Series({"x": 1})
    with pytest.raises(ValueError, match="smoothing"):
        log_ratio(a, b, total_a=10, total_b=10, smoothing=0)


def test_percent_diff_equal_rates_is_zero() -> None:
    a = pd.Series({"x": 10})
    b = pd.Series({"x": 20})
    pd_result = percent_diff(a, b, total_a=1_000, total_b=2_000)
    assert math.isclose(pd_result["x"], 0.0, abs_tol=1e-9)


def test_percent_diff_doubling_is_100_percent() -> None:
    # 20/1000 vs 10/1000 → 100% more frequent in A.
    a = pd.Series({"x": 20})
    b = pd.Series({"x": 10})
    pd_result = percent_diff(a, b, total_a=1_000, total_b=1_000)
    assert math.isclose(pd_result["x"], 100.0, rel_tol=1e-9)


def test_percent_diff_novel_in_a_is_infinite() -> None:
    # b=0 means the term is novel in A — division by zero, +inf is correct.
    a = pd.Series({"x": 10})
    b = pd.Series({"x": 0})
    pd_result = percent_diff(a, b, total_a=1_000, total_b=1_000)
    assert math.isinf(pd_result["x"]) and pd_result["x"] > 0
