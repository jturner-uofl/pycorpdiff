"""Known-answer tests for the four collocation measures."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pycorpdiff.collocation.measures import logdice, mi_three, pmi, t_score


def test_logdice_perfect_cooccurrence_is_fourteen() -> None:
    # 2 * f_xy == f_x + f_y → logDice = 14 + log2(1) = 14 (the upper bound).
    score = logdice(pd.Series([100.0]), f_x=100.0, f_y=pd.Series([100.0])).iloc[0]
    assert math.isclose(score, 14.0, rel_tol=1e-12)


def test_logdice_sketchengine_example() -> None:
    # f_xy=100, f_x=200, f_y=300 → 14 + log2(200/500) ≈ 12.6781
    score = logdice(pd.Series([100.0]), f_x=200.0, f_y=pd.Series([300.0])).iloc[0]
    expected = 14.0 + math.log2(200.0 / 500.0)
    assert math.isclose(score, expected, rel_tol=1e-9)


def test_logdice_zero_joint_yields_neg_inf() -> None:
    score = logdice(pd.Series([0.0]), f_x=10.0, f_y=pd.Series([10.0])).iloc[0]
    assert math.isinf(score) and score < 0


def test_logdice_symmetric_in_target_and_collocate() -> None:
    # logDice depends only on the sum f_x + f_y, so swapping has no effect.
    s1 = logdice(pd.Series([50.0]), f_x=200.0, f_y=pd.Series([100.0])).iloc[0]
    s2 = logdice(pd.Series([50.0]), f_x=100.0, f_y=pd.Series([200.0])).iloc[0]
    assert math.isclose(s1, s2, rel_tol=1e-12)


def test_pmi_known_value() -> None:
    # f_xy=100, f_x=200, f_y=300, N=1M → log2(100*1M / (200*300)) = log2(1666.67) ≈ 10.703
    score = pmi(pd.Series([100.0]), f_x=200.0, f_y=pd.Series([300.0]), n=1_000_000).iloc[0]
    expected = math.log2(100.0 * 1_000_000.0 / (200.0 * 300.0))
    assert math.isclose(score, expected, rel_tol=1e-9)


def test_pmi_independence_is_zero() -> None:
    # If f_xy = f_x * f_y / N (the expected count under independence),
    # PMI = log2(1) = 0.
    n = 1_000_000
    f_x, f_y = 200.0, 300.0
    f_xy = f_x * f_y / n  # = 0.06; integer would be silly, but math holds.
    score = pmi(pd.Series([f_xy]), f_x=f_x, f_y=pd.Series([f_y]), n=n).iloc[0]
    assert math.isclose(score, 0.0, abs_tol=1e-12)


def test_pmi_symmetric_in_target_collocate() -> None:
    s1 = pmi(pd.Series([50.0]), f_x=200.0, f_y=pd.Series([100.0]), n=1_000_000).iloc[0]
    s2 = pmi(pd.Series([50.0]), f_x=100.0, f_y=pd.Series([200.0]), n=1_000_000).iloc[0]
    assert math.isclose(s1, s2, rel_tol=1e-12)


def test_t_score_known_value() -> None:
    # f_xy=100, expected = 200*300/1M = 0.06, t = (100-0.06)/sqrt(100) ≈ 9.994
    score = t_score(pd.Series([100.0]), f_x=200.0, f_y=pd.Series([300.0]), n=1_000_000).iloc[0]
    expected = (100.0 - 200.0 * 300.0 / 1_000_000.0) / math.sqrt(100.0)
    assert math.isclose(score, expected, rel_tol=1e-9)


def test_t_score_at_independence_is_near_zero() -> None:
    # When f_xy is the expected count under independence, t-score ≈ 0.
    score = t_score(
        pd.Series([100.0]), f_x=10_000.0, f_y=pd.Series([10_000.0]), n=1_000_000
    ).iloc[0]
    # expected = 10000*10000/1e6 = 100; observed = 100 → numerator 0.
    assert math.isclose(score, 0.0, abs_tol=1e-12)


def test_mi_three_known_value() -> None:
    # f_xy=100, f_x=200, f_y=300, N=1M
    # MI³ = log2(100^3 * 1e6 / (200*300)) = log2(1e6 * 1e6 / 60000)
    score = mi_three(pd.Series([100.0]), f_x=200.0, f_y=pd.Series([300.0]), n=1_000_000).iloc[0]
    expected = math.log2(100.0**3 * 1_000_000.0 / (200.0 * 300.0))
    assert math.isclose(score, expected, rel_tol=1e-9)


def test_mi_three_dominates_pmi_for_frequent_pairs() -> None:
    # MI³ ranks frequent pairs higher than PMI does (relative to a rare pair).
    f_xy_freq = pd.Series([100.0])
    f_xy_rare = pd.Series([5.0])
    pmi_freq = pmi(f_xy_freq, 200.0, pd.Series([300.0]), 1_000_000).iloc[0]
    pmi_rare = pmi(f_xy_rare, 10.0, pd.Series([15.0]), 1_000_000).iloc[0]
    mi3_freq = mi_three(f_xy_freq, 200.0, pd.Series([300.0]), 1_000_000).iloc[0]
    mi3_rare = mi_three(f_xy_rare, 10.0, pd.Series([15.0]), 1_000_000).iloc[0]
    # PMI's bias means the rare pair scores comparable-or-higher; MI³
    # corrects that: freq > rare under MI³.
    assert mi3_freq > mi3_rare
    # And the relative ordering flips compared to PMI's bias:
    assert (mi3_freq - mi3_rare) > (pmi_freq - pmi_rare)


def test_vectorised_run_matches_per_row() -> None:
    f_xy = pd.Series([10.0, 50.0, 200.0], index=["a", "b", "c"])
    f_y = pd.Series([100.0, 500.0, 1500.0], index=["a", "b", "c"])
    bulk = logdice(f_xy, f_x=300.0, f_y=f_y)
    for col in ("a", "b", "c"):
        single = logdice(
            pd.Series({col: float(f_xy[col])}),
            f_x=300.0,
            f_y=pd.Series({col: float(f_y[col])}),
        )
        assert math.isclose(bulk[col], single[col], rel_tol=1e-12)


def test_all_measures_produce_finite_on_realistic_inputs() -> None:
    rng = np.random.default_rng(seed=0)
    n_collocs = 50
    f_xy = pd.Series(rng.integers(1, 100, size=n_collocs).astype(float))
    f_y = pd.Series(rng.integers(50, 5000, size=n_collocs).astype(float))
    assert np.isfinite(logdice(f_xy, 100.0, f_y)).all()
    assert np.isfinite(pmi(f_xy, 100.0, f_y, 1_000_000)).all()
    assert np.isfinite(t_score(f_xy, 100.0, f_y, 1_000_000)).all()
    assert np.isfinite(mi_three(f_xy, 100.0, f_y, 1_000_000)).all()
