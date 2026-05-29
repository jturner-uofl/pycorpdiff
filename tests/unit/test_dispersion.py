"""Known-answer tests for Juilland's D and Gries's DP."""

from __future__ import annotations

import math

import pandas as pd

from pycorpdiff.keyness.dispersion import dispersion_dp, juilland_d


def _dtm(rows: list[dict[str, int]]) -> pd.DataFrame:
    """Build a DataFrame of integer counts; missing terms become 0."""
    return pd.DataFrame.from_records(rows).fillna(0).astype("int64")


def test_juilland_d_perfectly_even() -> None:
    # Term appears once in every document of equal size — D should be 1.
    dtm = _dtm([{"x": 1, "filler": 9}] * 5)
    d = juilland_d(dtm)
    assert math.isclose(d["x"], 1.0, rel_tol=1e-12)


def test_juilland_d_concentrated_in_one_doc() -> None:
    # Term appears only in doc 0 — D should be low.
    dtm = _dtm([{"x": 10, "filler": 0}, {"x": 0, "filler": 10}, {"x": 0, "filler": 10}])
    d = juilland_d(dtm)
    # Three equally-sized docs, rates [1.0, 0.0, 0.0]; mean=1/3, sd=sqrt(2/9)=0.4714
    # CV = 0.4714 / 0.3333 = 1.4142; D = 1 - 1.4142/sqrt(2) = 1 - 1.0 = 0.0
    assert math.isclose(d["x"], 0.0, abs_tol=1e-9)


def test_juilland_d_single_doc_is_nan() -> None:
    dtm = _dtm([{"x": 5}])
    d = juilland_d(dtm)
    assert math.isnan(d["x"])


def test_juilland_d_absent_term_is_zero() -> None:
    # A term present in dtm.columns with all zero counts → D = 0 (no spread).
    dtm = pd.DataFrame({"x": [0, 0, 0], "y": [1, 1, 1]}, dtype="int64")
    d = juilland_d(dtm)
    assert math.isclose(d["x"], 0.0, abs_tol=1e-12)


def test_dispersion_dp_perfectly_even() -> None:
    # Same proportion of term in each doc → DP = 0.
    dtm = _dtm([{"x": 1, "filler": 9}] * 5)
    dp = dispersion_dp(dtm)
    assert math.isclose(dp["x"], 0.0, abs_tol=1e-12)


def test_dispersion_dp_total_concentration() -> None:
    # Five equally-sized docs, term only in doc 0.
    # observed = [1, 0, 0, 0, 0]; expected = [0.2]*5.
    # DP = 0.5 * (0.8 + 0.2 + 0.2 + 0.2 + 0.2) = 0.5 * 1.6 = 0.8.
    dtm = _dtm([{"x": 10, "filler": 0}] + [{"x": 0, "filler": 10}] * 4)
    dp = dispersion_dp(dtm)
    assert math.isclose(dp["x"], 0.8, rel_tol=1e-9)


def test_dispersion_dp_empty_matrix_returns_empty_series() -> None:
    dtm = pd.DataFrame(dtype="int64")
    dp = dispersion_dp(dtm)
    assert len(dp) == 0


def test_dispersion_dp_handles_term_absent_everywhere() -> None:
    # Should not blow up on a 0/0 column; we replace with expected so |Δ|=0.
    dtm = pd.DataFrame({"x": [0, 0, 0], "y": [3, 7, 5]}, dtype="int64")
    dp = dispersion_dp(dtm)
    assert math.isclose(dp["x"], 0.0, abs_tol=1e-12)
    # 'y' has doc sizes [3, 7, 5], term counts [3, 7, 5] — perfectly aligned.
    assert math.isclose(dp["y"], 0.0, abs_tol=1e-12)
