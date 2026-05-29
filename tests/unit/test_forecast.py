"""Tests for forward prediction on :class:`TemporalTrajectory`."""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pandas as pd
import pytest

import pycorpdiff as pcd
from pycorpdiff.temporal.forecast import (
    ForecastResult,
    forecast_trajectory,
)

statsmodels = pytest.importorskip("statsmodels")


@pytest.fixture
def trajectory() -> pcd.TemporalTrajectory:
    """A trajectory long enough for ETS auto-selection (≥ 8 periods)."""
    corpus = pcd.load_hansard_sample()
    immigration = corpus.slice(topic="immigration")
    return pcd.track(immigration, ["criminal", "family"]).over_time(freq="Y")


@pytest.fixture
def short_trajectory() -> pcd.TemporalTrajectory:
    """A trajectory short enough that auto-selection falls to Holt."""
    corpus = pcd.load_hansard_sample()
    df = corpus.docs[corpus.docs["year"] >= 2018].copy()
    sub = pcd.from_dataframe(
        df,
        text_col="text",
        meta_cols=tuple(c for c in df.columns if c != "text"),
    )
    return pcd.track(sub.slice(topic="immigration"), "criminal").over_time(freq="Y")


def _suppress_statsmodels_warnings(fn):
    """statsmodels emits a fair number of convergence warnings on tiny data;
    they are not failures and we don't want them as test noise. We use
    ``functools.wraps`` so pytest's fixture introspection still works."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*args, **kwargs)
    return wrapper


@_suppress_statsmodels_warnings
def test_returns_forecast_result(trajectory: pcd.TemporalTrajectory) -> None:
    fc = trajectory.forecast(horizon=3)
    assert isinstance(fc, ForecastResult)
    assert fc.horizon == 3
    assert fc.level == 0.95
    assert fc.targets == ["criminal", "family"]
    assert fc.freq == "Y"


@_suppress_statsmodels_warnings
def test_forecast_table_shape(trajectory: pcd.TemporalTrajectory) -> None:
    fc = trajectory.forecast(horizon=4)
    # 2 targets × 4 horizon = 8 rows
    assert len(fc.forecast) == 8
    assert set(fc.forecast.columns) == {
        "period", "term", "point", "ci_lower", "ci_upper"
    }


@_suppress_statsmodels_warnings
def test_forecast_periods_extend_history(trajectory: pcd.TemporalTrajectory) -> None:
    """Forecast periods must start one step after the last historical period."""
    last_hist = trajectory.table["period"].max()
    fc = trajectory.forecast(horizon=4)
    first_fc = fc.forecast["period"].min()
    assert first_fc == last_hist + 1


@_suppress_statsmodels_warnings
def test_prediction_intervals_contain_point_estimate(
    trajectory: pcd.TemporalTrajectory,
) -> None:
    fc = trajectory.forecast(horizon=4)
    assert (fc.forecast["ci_lower"] <= fc.forecast["point"]).all()
    assert (fc.forecast["point"] <= fc.forecast["ci_upper"]).all()


@_suppress_statsmodels_warnings
def test_logit_transform_keeps_pi_in_unit_interval(
    trajectory: pcd.TemporalTrajectory,
) -> None:
    """With logit_transform=True (default), PIs must be in [0, 1]."""
    fc = trajectory.forecast(horizon=4, level=0.99)
    assert (fc.forecast["ci_lower"] >= 0.0).all()
    assert (fc.forecast["ci_upper"] <= 1.0).all()
    assert (fc.forecast["point"] >= 0.0).all()
    assert (fc.forecast["point"] <= 1.0).all()


@_suppress_statsmodels_warnings
def test_logit_transform_off_can_exceed_unit_interval(
    trajectory: pcd.TemporalTrajectory,
) -> None:
    """Disabling the logit transform allows PIs outside [0, 1].

    Doesn't have to *actually* exceed — just verifies the forecast
    runs and produces a valid table when the constraint is dropped."""
    fc = trajectory.forecast(horizon=4, logit_transform=False)
    assert len(fc.forecast) > 0
    assert fc.params["logit_transform"] is False


@_suppress_statsmodels_warnings
def test_wider_level_gives_wider_pi(trajectory: pcd.TemporalTrajectory) -> None:
    fc_50 = trajectory.forecast(horizon=4, level=0.50)
    fc_95 = trajectory.forecast(horizon=4, level=0.95)
    width_50 = (fc_50.forecast["ci_upper"] - fc_50.forecast["ci_lower"]).mean()
    width_95 = (fc_95.forecast["ci_upper"] - fc_95.forecast["ci_lower"]).mean()
    assert width_95 > width_50


@_suppress_statsmodels_warnings
def test_target_filter(trajectory: pcd.TemporalTrajectory) -> None:
    """target=... restricts the forecast to one term."""
    fc = trajectory.forecast(horizon=3, target="criminal")
    assert fc.targets == ["criminal"]
    assert set(fc.forecast["term"]) == {"criminal"}
    assert len(fc.forecast) == 3


@_suppress_statsmodels_warnings
def test_unknown_target_raises(trajectory: pcd.TemporalTrajectory) -> None:
    with pytest.raises(ValueError, match="not in trajectory targets"):
        trajectory.forecast(horizon=2, target="unicorn")


@_suppress_statsmodels_warnings
def test_horizon_zero_raises(trajectory: pcd.TemporalTrajectory) -> None:
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        forecast_trajectory(trajectory.table, horizon=0)


@_suppress_statsmodels_warnings
def test_invalid_level_raises(trajectory: pcd.TemporalTrajectory) -> None:
    with pytest.raises(ValueError, match=r"level must be in \(0, 1\)"):
        forecast_trajectory(trajectory.table, horizon=2, level=1.5)


@_suppress_statsmodels_warnings
def test_method_explicit_ets(trajectory: pcd.TemporalTrajectory) -> None:
    fc = trajectory.forecast(horizon=2, method="ets")
    assert fc.method == "ets"
    assert len(fc.forecast) == 4  # 2 targets × 2 horizon


@_suppress_statsmodels_warnings
def test_method_explicit_holt(trajectory: pcd.TemporalTrajectory) -> None:
    fc = trajectory.forecast(horizon=2, method="holt")
    assert fc.method == "holt"
    assert len(fc.forecast) == 4


@_suppress_statsmodels_warnings
def test_unknown_method_raises(trajectory: pcd.TemporalTrajectory) -> None:
    with pytest.raises(ValueError, match="unknown method"):
        trajectory.forecast(horizon=2, method="bogus")  # type: ignore[arg-type]


@_suppress_statsmodels_warnings
def test_auto_falls_back_to_holt_on_short_series(
    short_trajectory: pcd.TemporalTrajectory,
) -> None:
    """With < 8 obs, auto should pick holt; ets shouldn't be forced."""
    # The fixture has 6 periods (2018-2023). Auto → holt. Verify by
    # ensuring it doesn't raise and produces a sensible point.
    fc = short_trajectory.forecast(horizon=2)
    assert len(fc.forecast) == 2
    # Point is in [0, 1] thanks to logit transform.
    assert (fc.forecast["point"] >= 0.0).all()
    assert (fc.forecast["point"] <= 1.0).all()


@_suppress_statsmodels_warnings
def test_too_few_observations_raises() -> None:
    """A series of 2-3 observations is too short for either method."""
    df = pd.DataFrame(
        {
            "period": pd.period_range("2020", periods=3, freq="Y"),
            "term": ["x", "x", "x"],
            "count": [1, 2, 3],
            "total": [100, 100, 100],
            "relfreq": [0.01, 0.02, 0.03],
            "ci_lower": [0.0, 0.0, 0.0],
            "ci_upper": [0.05, 0.05, 0.05],
        }
    )
    with pytest.raises(ValueError, match="at least 4 observations"):
        forecast_trajectory(df, targets=["x"], horizon=2)


@_suppress_statsmodels_warnings
def test_to_df_returns_forecast_table(trajectory: pcd.TemporalTrajectory) -> None:
    fc = trajectory.forecast(horizon=3)
    pd.testing.assert_frame_equal(fc.to_df(), fc.forecast)


@_suppress_statsmodels_warnings
def test_to_combined_stitches_history_and_forecast(
    trajectory: pcd.TemporalTrajectory,
) -> None:
    fc = trajectory.forecast(horizon=3)
    combined = fc.to_combined()
    assert "kind" in combined.columns
    assert set(combined["kind"]) == {"observed", "forecast"}
    # Expect history_rows + forecast_rows
    expected = len(fc.history) + len(fc.forecast)
    assert len(combined) == expected


@_suppress_statsmodels_warnings
def test_to_html_and_to_json(trajectory: pcd.TemporalTrajectory, tmp_path) -> None:
    fc = trajectory.forecast(horizon=3)
    html = fc.to_html()
    assert "<table" in html
    json_str = fc.to_json()
    assert "point" in json_str
    # path round-trip
    html_path = tmp_path / "fc.html"
    fc.to_html(html_path)
    assert html_path.exists()


@_suppress_statsmodels_warnings
def test_summary_string(trajectory: pcd.TemporalTrajectory) -> None:
    fc = trajectory.forecast(horizon=3)
    s = fc.summary()
    assert "ForecastResult" in s
    assert "horizon=3" in s


@_suppress_statsmodels_warnings
def test_plot_returns_layered_chart(trajectory: pcd.TemporalTrajectory) -> None:
    pytest.importorskip("altair")
    fc = trajectory.forecast(horizon=4)
    chart = fc.plot()
    spec = chart.to_dict()
    # Should be a layered chart (history layers + forecast layers + seam).
    assert "layer" in spec
    assert len(spec["layer"]) >= 6


@_suppress_statsmodels_warnings
def test_plot_marks_forecast_dashed(trajectory: pcd.TemporalTrajectory) -> None:
    pytest.importorskip("altair")
    fc = trajectory.forecast(horizon=4)
    spec = fc.plot().to_dict()
    has_dashed_line = False
    for layer in spec["layer"]:
        mark = layer.get("mark", {})
        if isinstance(mark, dict) and mark.get("strokeDash") == [6, 4]:
            has_dashed_line = True
            break
    assert has_dashed_line


@_suppress_statsmodels_warnings
def test_exported_at_package_root() -> None:
    assert pcd.ForecastResult is ForecastResult
    assert pcd.forecast_trajectory is forecast_trajectory


@_suppress_statsmodels_warnings
def test_params_recorded(trajectory: pcd.TemporalTrajectory) -> None:
    fc = trajectory.forecast(horizon=2, logit_transform=False)
    assert fc.params["logit_transform"] is False


@_suppress_statsmodels_warnings
def test_pi_widens_with_horizon(trajectory: pcd.TemporalTrajectory) -> None:
    """The prediction interval should grow as we forecast further out —
    a fundamental property of any state-space forecasting method."""
    fc = trajectory.forecast(horizon=6, target="criminal", method="ets")
    widths = (fc.forecast["ci_upper"] - fc.forecast["ci_lower"]).to_numpy()
    # Monotonically non-decreasing (allow tiny float noise).
    assert (np.diff(widths) >= -1e-9).all()


@_suppress_statsmodels_warnings
def test_forecasts_are_non_negative_rates(trajectory: pcd.TemporalTrajectory) -> None:
    """With the default logit transform, point estimates are valid rates."""
    fc = trajectory.forecast(horizon=4)
    assert (fc.forecast["point"] >= 0.0).all()
