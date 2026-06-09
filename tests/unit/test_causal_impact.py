"""Tests for Bayesian causal-impact analysis."""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pandas as pd
import pytest

import pycorpdiff as pcd
from pycorpdiff.temporal.causal_impact import (
    CausalImpactResult,
    causal_impact,
)

statsmodels = pytest.importorskip("statsmodels")


def _quiet(fn):
    """Suppress statsmodels convergence warnings on tiny fixtures."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*args, **kwargs)
    return wrapper


def _engineered_series() -> pd.Series:
    """A clean step-change series: flat-zero for 20 periods, then +0.04
    each period for 10 periods. Ground-truth effect is +0.04 per period.

    Pre-window is 20 to satisfy the default ``min_pre_periods=15``
    safety rail introduced in 0.1.0a21; tests that need the legacy
    short pre-window pass ``min_pre_periods=4`` explicitly."""
    idx = pd.period_range("2000", periods=30, freq="Y")
    pre = np.zeros(20)
    post = np.full(10, 0.04)
    return pd.Series(np.concatenate([pre, post]), index=idx)


def _hansard_trajectory() -> pcd.TemporalTrajectory:
    corpus = pcd.load_hansard_sample()
    immigration = corpus.slice(topic="immigration")
    return pcd.track(immigration, "criminal").over_time(freq="Y")


@_quiet
def test_returns_causal_impact_result() -> None:
    ci = causal_impact(_engineered_series(), event_date="2020", n_samples=300)
    assert isinstance(ci, CausalImpactResult)


@_quiet
def test_pre_post_split_is_correct() -> None:
    ci = causal_impact(_engineered_series(), event_date="2020", n_samples=300)
    assert ci.n_pre == 20
    assert ci.n_post == 10
    assert ci.level == 0.95


@_quiet
def test_table_has_required_columns() -> None:
    ci = causal_impact(_engineered_series(), event_date="2020", n_samples=300)
    required = {
        "period", "observed", "counterfactual",
        "counterfactual_lower", "counterfactual_upper",
        "pointwise_effect", "pointwise_lower", "pointwise_upper",
        "cumulative_effect", "cumulative_lower", "cumulative_upper",
    }
    assert required.issubset(set(ci.table.columns))


@_quiet
def test_table_length_matches_post_window() -> None:
    ci = causal_impact(_engineered_series(), event_date="2020", n_samples=300)
    assert len(ci.table) == ci.n_post


@_quiet
def test_engineered_step_recovers_known_effect() -> None:
    """Ground-truth step of +0.04 per period must be in the 95% CrI."""
    ci = causal_impact(_engineered_series(), event_date="2020", n_samples=500)
    assert ci.metrics["avg_effect_lower"] <= 0.04 <= ci.metrics["avg_effect_upper"]
    assert ci.metrics["avg_effect"] == pytest.approx(0.04, abs=0.005)


@_quiet
def test_engineered_step_significant_p() -> None:
    """An 8-period +0.04 step on flat zero is unmissably non-null."""
    ci = causal_impact(_engineered_series(), event_date="2020", n_samples=500)
    assert ci.metrics["p_no_effect_mc"] < 0.05


@_quiet
def test_null_series_yields_high_p() -> None:
    """A series with NO effect should produce an MC distribution centred
    near zero — the avg_effect interval should contain zero."""
    rng = np.random.default_rng(0)
    idx = pd.period_range("2000", periods=30, freq="Y")
    s = pd.Series(0.05 + rng.normal(0, 0.01, 30), index=idx)
    ci = causal_impact(s, event_date="2015", n_samples=500)
    # Effect should be roughly centred on zero.
    assert abs(ci.metrics["avg_effect"]) < 0.02
    # And the interval should contain zero.
    assert ci.metrics["avg_effect_lower"] <= 0.0 <= ci.metrics["avg_effect_upper"]


@_quiet
def test_intervals_contain_point_estimates() -> None:
    ci = causal_impact(_engineered_series(), event_date="2020", n_samples=300)
    # Counterfactual interval brackets counterfactual point.
    assert (ci.table["counterfactual_lower"] <= ci.table["counterfactual"]).all()
    assert (ci.table["counterfactual"] <= ci.table["counterfactual_upper"]).all()
    # Pointwise / cumulative intervals bracket their means.
    assert (ci.table["pointwise_lower"] <= ci.table["pointwise_effect"] + 1e-9).all()
    assert (ci.table["pointwise_effect"] - 1e-9 <= ci.table["pointwise_upper"]).all()
    assert (ci.table["cumulative_lower"] <= ci.table["cumulative_effect"] + 1e-9).all()
    assert (ci.table["cumulative_effect"] - 1e-9 <= ci.table["cumulative_upper"]).all()


@_quiet
def test_invalid_level_raises() -> None:
    with pytest.raises(ValueError, match=r"level must be in \(0, 1\)"):
        causal_impact(_engineered_series(), event_date="2020", level=1.5)


@_quiet
def test_too_few_samples_raises() -> None:
    with pytest.raises(ValueError, match="n_samples must be >= 100"):
        causal_impact(_engineered_series(), event_date="2020", n_samples=10)


@_quiet
def test_too_short_pre_window_raises() -> None:
    """Under-power pre window blocked by the min_pre_periods safety rail."""
    idx = pd.period_range("2005", periods=10, freq="Y")
    s = pd.Series(np.arange(10) * 0.01, index=idx)
    with pytest.raises(ValueError, match="pre-event window too short"):
        causal_impact(s, event_date="2007", min_post_periods=1, max_pre_post_ratio=20)


@_quiet
def test_event_after_last_period_raises() -> None:
    """If the event is at or after the last period, the post-window safety
    rail fires before the pre-window one."""
    idx = pd.period_range("2005", periods=10, freq="Y")
    s = pd.Series(np.arange(10) * 0.01, index=idx)
    with pytest.raises(ValueError, match="post-event window too short"):
        causal_impact(s, event_date="2030", min_pre_periods=4, max_pre_post_ratio=20)


@_quiet
def test_min_pre_periods_default_blocks_underpowered_run() -> None:
    """Default min_pre_periods=15 blocks runs that would silently
    under-power BSTS (the asylum case study §5.8e leverage finding)."""
    idx = pd.period_range("2005", periods=20, freq="Y")
    s = pd.Series(np.arange(20) * 0.01, index=idx)
    # Event at 2015 → pre=10, post=10. 10 < default min_pre_periods=15.
    with pytest.raises(ValueError, match="min_pre_periods=15"):
        causal_impact(s, event_date="2015")
    # Explicit override succeeds.
    result = causal_impact(s, event_date="2015", min_pre_periods=4)
    assert result.n_pre == 10
    assert result.n_post == 10


@_quiet
def test_min_post_periods_blocks_short_tail() -> None:
    """Default min_post_periods=8 blocks runs that would fit BSTS
    to a tiny tail (the asylum case study §5.8c placebo finding for end-of-series
    events)."""
    idx = pd.period_range("2000", periods=24, freq="Y")
    s = pd.Series(np.arange(24) * 0.01, index=idx)
    # Event at 2020 → pre=20, post=4. 4 < default min_post_periods=8.
    with pytest.raises(ValueError, match="post-event window too short"):
        causal_impact(s, event_date="2020", max_pre_post_ratio=20)
    # Explicit override succeeds.
    result = causal_impact(s, event_date="2020", min_post_periods=1, max_pre_post_ratio=20)
    assert result.n_post == 4


@_quiet
def test_max_pre_post_ratio_blocks_asymmetric_split() -> None:
    """Default max_pre_post_ratio=5 blocks runs that BSTS would handle
    as 'find a step change in this short tail'."""
    idx = pd.period_range("2000", periods=30, freq="Y")
    s = pd.Series(np.arange(30) * 0.01, index=idx)
    # Event at 2026 → pre=26, post=4. ratio=6.5 > default 5.
    with pytest.raises(ValueError, match="pre/post asymmetry too large"):
        causal_impact(s, event_date="2026", min_post_periods=4)
    # Explicit override succeeds.
    result = causal_impact(
        s, event_date="2026", min_post_periods=4, max_pre_post_ratio=20,
    )
    assert result.n_pre == 26
    assert result.n_post == 4


@_quiet
def test_seed_makes_results_reproducible() -> None:
    s = _engineered_series()
    ci_a = causal_impact(s, event_date="2020", seed=42, n_samples=300)
    ci_b = causal_impact(s, event_date="2020", seed=42, n_samples=300)
    pd.testing.assert_frame_equal(ci_a.table, ci_b.table)


@_quiet
def test_temporal_trajectory_method_wires_target() -> None:
    """TemporalTrajectory.causal_impact populates the target name on the
    returned result (the standalone function leaves it blank since it
    doesn't know what the series represents)."""
    tr = _hansard_trajectory()
    ci = tr.causal_impact(event_date="2016", n_samples=300, min_pre_periods=4, min_post_periods=1, max_pre_post_ratio=20)
    assert ci.target == "criminal"


@_quiet
def test_multi_target_trajectory_needs_explicit_target() -> None:
    corpus = pcd.load_hansard_sample()
    tr = pcd.track(corpus.slice(topic="immigration"), ["criminal", "family"]).over_time(
        freq="Y"
    )
    with pytest.raises(ValueError, match="pass target= to pick one"):
        tr.causal_impact(event_date="2016", n_samples=300, min_pre_periods=4, min_post_periods=1, max_pre_post_ratio=20)


@_quiet
def test_unknown_target_raises() -> None:
    tr = _hansard_trajectory()
    with pytest.raises(ValueError, match="not in trajectory targets"):
        tr.causal_impact(event_date="2016", target="unicorn", n_samples=300, min_pre_periods=4, min_post_periods=1, max_pre_post_ratio=20)


@_quiet
def test_summary_string() -> None:
    tr = _hansard_trajectory()
    ci = tr.causal_impact(event_date="2016", n_samples=300, min_pre_periods=4, min_post_periods=1, max_pre_post_ratio=20)
    s = ci.summary()
    assert "CausalImpactResult" in s
    assert "criminal" in s
    assert "avg effect" in s


@_quiet
def test_summary_handles_zero_counterfactual_gracefully() -> None:
    """When the counterfactual mean is ~0, relative_effect is NaN; the
    summary should print 'n/a' rather than '+nan%'."""
    s = _engineered_series()
    ci = causal_impact(s, event_date="2020", n_samples=300)
    summary = ci.summary()
    assert "nan" not in summary.lower() or "n/a" in summary


@_quiet
def test_to_html_and_to_json() -> None:
    tr = _hansard_trajectory()
    ci = tr.causal_impact(event_date="2016", n_samples=300, min_pre_periods=4, min_post_periods=1, max_pre_post_ratio=20)
    html = ci.to_html()
    assert "<table" in html
    json_str = ci.to_json()
    assert "observed" in json_str


@_quiet
def test_plot_is_three_panel_vconcat() -> None:
    pytest.importorskip("altair")
    tr = _hansard_trajectory()
    ci = tr.causal_impact(event_date="2016", n_samples=300, min_pre_periods=4, min_post_periods=1, max_pre_post_ratio=20)
    chart = ci.plot()
    spec = chart.to_dict()
    assert "vconcat" in spec
    assert len(spec["vconcat"]) == 3


@_quiet
def test_plot_has_dashed_counterfactual() -> None:
    pytest.importorskip("altair")
    tr = _hansard_trajectory()
    ci = tr.causal_impact(event_date="2016", n_samples=300, min_pre_periods=4, min_post_periods=1, max_pre_post_ratio=20)
    spec = ci.plot().to_dict()
    # First panel layers — find a dashed line.
    panel1 = spec["vconcat"][0]
    has_dashed = False
    for layer in panel1.get("layer", []):
        mark = layer.get("mark", {})
        if isinstance(mark, dict) and isinstance(
            mark.get("strokeDash"), list
        ):
            has_dashed = True
            break
    assert has_dashed


@_quiet
def test_params_recorded() -> None:
    s = _engineered_series()
    ci = causal_impact(
        s, event_date="2020", n_samples=300, seed=7, model="local level"
    )
    assert ci.params["model"] == "local level"
    assert ci.params["n_samples"] == 300
    assert ci.params["seed"] == 7


@_quiet
def test_exported_at_package_root() -> None:
    assert pcd.CausalImpactResult is CausalImpactResult
    assert pcd.causal_impact is causal_impact


@_quiet
def test_higher_level_widens_cri() -> None:
    s = _engineered_series()
    ci_50 = causal_impact(s, event_date="2020", level=0.50, n_samples=500, seed=0)
    ci_95 = causal_impact(s, event_date="2020", level=0.95, n_samples=500, seed=0)
    width_50 = (ci_50.table["pointwise_upper"] - ci_50.table["pointwise_lower"]).mean()
    width_95 = (ci_95.table["pointwise_upper"] - ci_95.table["pointwise_lower"]).mean()
    assert width_95 > width_50


@_quiet
def test_cumulative_effect_monotone_when_effect_constant_sign() -> None:
    """A purely-positive effect series should yield monotone cumulative."""
    ci = causal_impact(_engineered_series(), event_date="2020", n_samples=300)
    cumdiff = np.diff(ci.table["cumulative_effect"].to_numpy())
    # All increments should be > 0 (or very near).
    assert (cumdiff > -1e-6).all()
    # The cumulative metric is the Monte Carlo mean of the cumulative
    # path; it agrees with avg_effect × n_post up to MC noise (~1/√N).
    assert ci.metrics["cumulative_effect"] == pytest.approx(
        ci.metrics["avg_effect"] * ci.n_post, abs=5e-3
    )
