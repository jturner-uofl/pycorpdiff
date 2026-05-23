"""Tests for Bayesian online changepoint detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pycorpdiff as pcd
from pycorpdiff.temporal.bocpd import BocpdResult, bocpd


def _engineered_step(t1: int = 50, t2: int = 50, seed: int = 0) -> pd.Series:
    """A clean step series: t1 obs around 0.01, then t2 obs around 0.04.

    The engineered changepoint is at index t1.
    """
    rng = np.random.default_rng(seed)
    pre = rng.normal(0.01, 0.005, t1)
    post = rng.normal(0.04, 0.005, t2)
    idx = pd.period_range("2000", periods=t1 + t2, freq="M")
    return pd.Series(np.concatenate([pre, post]), index=idx)


def _stationary_noise(n: int = 80, seed: int = 0) -> pd.Series:
    """A flat-Gaussian series with no real changepoint."""
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2000", periods=n, freq="M")
    return pd.Series(rng.normal(0.02, 0.005, n), index=idx)


def test_returns_bocpd_result() -> None:
    r = bocpd(_engineered_step(), hazard=0.02)
    assert isinstance(r, BocpdResult)
    assert r.hazard == 0.02


def test_output_lengths_match_input() -> None:
    s = _engineered_step()
    r = bocpd(s, hazard=0.02)
    assert len(r.map_run_length) == len(s)
    assert len(r.cp_probability) == len(s)
    assert r.run_length_posterior.shape[0] == len(s)


def test_posterior_sums_to_one_per_step() -> None:
    """At every step, the run-length posterior is a proper distribution."""
    r = bocpd(_engineered_step(), hazard=0.02)
    row_sums = r.run_length_posterior.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)


def test_map_run_length_collapses_at_engineered_changepoint() -> None:
    """The canonical BOCPD diagnostic: MAP run length drops sharply at
    the true changepoint."""
    s = _engineered_step(t1=50, t2=50)
    r = bocpd(s, hazard=0.02, mu_0=0.01, beta_0=0.0001)
    # Right before the change: large MAP (close to t).
    pre_max = int(r.map_run_length.iloc[40:50].max())
    # Right after: small MAP.
    post_min = int(r.map_run_length.iloc[50:55].min())
    assert pre_max >= 30
    assert post_min <= 5
    # And the collapse should happen at the engineered index.
    assert int(r.map_run_length.iloc[52]) < pre_max


def test_map_run_length_grows_monotone_in_stable_regime() -> None:
    """Within a stationary regime the MAP run length should keep growing."""
    r = bocpd(_stationary_noise(n=80), hazard=0.01, mu_0=0.02, beta_0=0.0001)
    # After a burn-in of 10 steps, the MAP should mostly be monotone.
    map_arr = r.map_run_length.iloc[10:].to_numpy()
    monotone_fraction = float((np.diff(map_arr) >= 0).mean())
    assert monotone_fraction > 0.9


def test_stationary_noise_yields_few_detected_changepoints() -> None:
    """A noise series should produce few or no detected changepoints."""
    r = bocpd(_stationary_noise(n=80), hazard=0.01, mu_0=0.02, beta_0=0.0001)
    detected = r.detected_changepoints(threshold=3)
    # Allow a couple from the initial burn-in transient.
    assert len(detected) <= 5


def test_to_df_round_trip() -> None:
    r = bocpd(_engineered_step(), hazard=0.02)
    df = r.to_df()
    assert set(df.columns) == {"period", "value", "map_run_length", "cp_probability"}
    assert len(df) == len(r.series)


def test_to_html_and_to_json() -> None:
    r = bocpd(_engineered_step(t1=10, t2=10), hazard=0.02)
    html = r.to_html()
    assert "<table" in html
    js = r.to_json()
    assert "map_run_length" in js


def test_invalid_hazard_raises() -> None:
    with pytest.raises(ValueError, match=r"hazard must be in \(0, 1\)"):
        bocpd(_engineered_step(t1=10, t2=10), hazard=0.0)
    with pytest.raises(ValueError, match=r"hazard must be in \(0, 1\)"):
        bocpd(_engineered_step(t1=10, t2=10), hazard=1.5)


def test_negative_hyperparams_raise() -> None:
    s = _engineered_step(t1=10, t2=10)
    with pytest.raises(ValueError, match="must be positive"):
        bocpd(s, kappa_0=-1.0)
    with pytest.raises(ValueError, match="must be positive"):
        bocpd(s, alpha_0=-1.0)


def test_too_short_series_raises() -> None:
    s = pd.Series([0.01], index=pd.period_range("2020", periods=1, freq="M"))
    with pytest.raises(ValueError, match="at least 2 observations"):
        bocpd(s)


def test_max_run_length_truncates_posterior_width() -> None:
    s = _engineered_step(t1=30, t2=30)
    r = bocpd(s, hazard=0.02, max_run_length=10)
    assert r.run_length_posterior.shape[1] == 11  # 0..10
    assert int(r.map_run_length.max()) <= 10


def test_higher_hazard_yields_more_detected() -> None:
    """A larger hazard prior → more changepoints declared, all else equal."""
    s = _stationary_noise(n=80)
    r_low = bocpd(s, hazard=0.005, mu_0=0.02, beta_0=0.0001)
    r_high = bocpd(s, hazard=0.2, mu_0=0.02, beta_0=0.0001)
    assert len(r_high.detected_changepoints(threshold=3)) >= len(
        r_low.detected_changepoints(threshold=3)
    )


def test_detected_changepoints_threshold_filter() -> None:
    r = bocpd(_engineered_step(), hazard=0.02, mu_0=0.01, beta_0=0.0001)
    strict = r.detected_changepoints(threshold=1)
    relaxed = r.detected_changepoints(threshold=5)
    # Tighter threshold ⊆ looser threshold.
    assert set(strict["period"]).issubset(set(relaxed["period"]))


def test_summary_string() -> None:
    r = bocpd(_engineered_step(), hazard=0.02)
    s = r.summary()
    assert "BocpdResult" in s
    assert "hazard" in s


def test_params_recorded() -> None:
    r = bocpd(
        _engineered_step(),
        hazard=0.02,
        mu_0=0.01,
        kappa_0=2.0,
        alpha_0=1.5,
        beta_0=0.0001,
        max_run_length=20,
    )
    assert r.params["mu_0"] == 0.01
    assert r.params["kappa_0"] == 2.0
    assert r.params["alpha_0"] == 1.5
    assert r.params["beta_0"] == 0.0001
    assert r.params["max_run_length"] == 20


def test_temporal_trajectory_wires_through() -> None:
    """TemporalTrajectory.changepoints_online() delegates to bocpd()."""
    corpus = pcd.load_hansard_sample()
    tr = pcd.track(corpus.slice(topic="immigration"), "criminal").over_time(freq="Y")
    r = tr.changepoints_online(hazard=0.02, mu_0=0.0, beta_0=0.0001)
    assert isinstance(r, BocpdResult)
    assert len(r.series) == tr.table["period"].nunique()


def test_trajectory_method_requires_target_when_multi() -> None:
    corpus = pcd.load_hansard_sample()
    tr = pcd.track(
        corpus.slice(topic="immigration"), ["criminal", "family"]
    ).over_time(freq="Y")
    with pytest.raises(ValueError, match="pass target= to pick one"):
        tr.changepoints_online()


def test_trajectory_method_unknown_target_raises() -> None:
    corpus = pcd.load_hansard_sample()
    tr = pcd.track(corpus.slice(topic="immigration"), "criminal").over_time(freq="Y")
    with pytest.raises(ValueError, match="not in trajectory targets"):
        tr.changepoints_online(target="unicorn")


def test_exported_at_package_root() -> None:
    assert pcd.bocpd is bocpd
    assert pcd.BocpdResult is BocpdResult


def test_plot_is_three_panel_vconcat() -> None:
    pytest.importorskip("altair")
    r = bocpd(_engineered_step(), hazard=0.02, mu_0=0.01, beta_0=0.0001)
    chart = r.plot()
    spec = chart.to_dict()
    assert "vconcat" in spec
    assert len(spec["vconcat"]) == 3


def test_plot_marks_detected_changepoints_in_first_panel() -> None:
    pytest.importorskip("altair")
    r = bocpd(_engineered_step(), hazard=0.02, mu_0=0.01, beta_0=0.0001)
    spec = r.plot().to_dict()
    panel1 = spec["vconcat"][0]
    # Find a dashed rule (the changepoint marker).
    has_rule = False
    for layer in panel1.get("layer", []):
        mark = layer.get("mark", {})
        if (
            isinstance(mark, dict)
            and mark.get("type") == "rule"
            and isinstance(mark.get("strokeDash"), list)
        ):
            has_rule = True
            break
    assert has_rule


def test_engineered_changepoint_recovered_in_signal() -> None:
    """End-to-end: the detected changepoint set must include indices
    within 5 steps of the engineered changepoint at t=50."""
    s = _engineered_step(t1=50, t2=50)
    r = bocpd(s, hazard=0.02, mu_0=0.01, beta_0=0.0001)
    detected = r.detected_changepoints(threshold=3)
    # Find the integer positions of detected periods in the original series.
    positions = [s.index.get_loc(p) for p in detected["period"]]
    assert any(48 <= pos <= 55 for pos in positions), (
        f"engineered changepoint at t=50 not detected; got positions {positions}"
    )
