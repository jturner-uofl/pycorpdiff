"""Tests for :func:`pycorpdiff.forecast_semantic_drift`."""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pandas as pd
import pytest

import pycorpdiff as pcd
from pycorpdiff.temporal.forecast import forecast_semantic_drift

statsmodels = pytest.importorskip("statsmodels")


def _quiet(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*args, **kwargs)
    return wrapper


def _semantic_trajectory_df() -> pd.DataFrame:
    """Real semantic_trajectory output from the Hansard fixture."""
    corpus = pcd.load_hansard_sample()
    immigration = corpus.slice(topic="immigration")
    return pcd.semantic_trajectory(
        immigration,
        target="immigrant",
        time_col="date",
        freq="Y",
        embedder=pcd.HashEmbedder(dim=32),
        window=4,
    )


def _multi_target_df() -> pd.DataFrame:
    corpus = pcd.load_hansard_sample()
    immigration = corpus.slice(topic="immigration")
    return pcd.semantic_trajectory(
        immigration,
        target=["immigrant", "family"],
        time_col="date",
        freq="Y",
        embedder=pcd.HashEmbedder(dim=32),
        window=4,
    )


@_quiet
def test_returns_expected_columns() -> None:
    fc = forecast_semantic_drift(_semantic_trajectory_df(), horizon=3)
    assert set(fc.columns) == {"period", "target", "point", "ci_lower", "ci_upper"}


@_quiet
def test_horizon_controls_row_count_per_target() -> None:
    fc = forecast_semantic_drift(_semantic_trajectory_df(), horizon=5)
    # Single target × horizon=5 → 5 rows.
    assert len(fc) == 5


@_quiet
def test_multi_target_returns_one_row_per_target_per_horizon() -> None:
    fc = forecast_semantic_drift(_multi_target_df(), horizon=4)
    assert len(fc) == 8  # 2 targets × 4 horizon
    assert set(fc["target"]) == {"immigrant", "family"}


@_quiet
def test_targets_filter() -> None:
    fc = forecast_semantic_drift(
        _multi_target_df(), targets=["immigrant"], horizon=2
    )
    assert set(fc["target"]) == {"immigrant"}
    assert len(fc) == 2


@_quiet
def test_unknown_target_raises() -> None:
    with pytest.raises(ValueError, match="unknown targets"):
        forecast_semantic_drift(
            _semantic_trajectory_df(), targets=["unicorn"], horizon=2
        )


@_quiet
def test_missing_columns_raises() -> None:
    df = pd.DataFrame({"period": [], "target": []})
    with pytest.raises(ValueError, match="missing required columns"):
        forecast_semantic_drift(df, horizon=2)


@_quiet
def test_lower_bound_clipped_at_zero() -> None:
    """Cosine distance is non-negative — the lower PI must never go
    negative, even when the raw forecast would."""
    fc = forecast_semantic_drift(_semantic_trajectory_df(), horizon=4, level=0.99)
    assert (fc["ci_lower"] >= 0.0).all()
    assert (fc["point"] >= 0.0).all()


@_quiet
def test_pi_brackets_point_estimate() -> None:
    fc = forecast_semantic_drift(_semantic_trajectory_df(), horizon=4)
    assert (fc["ci_lower"] <= fc["point"] + 1e-9).all()
    assert (fc["point"] - 1e-9 <= fc["ci_upper"]).all()


@_quiet
def test_periods_extend_history() -> None:
    history = _semantic_trajectory_df()
    last_period = history["period"].max()
    fc = forecast_semantic_drift(history, horizon=3)
    assert fc["period"].min() == last_period + 1


@_quiet
def test_wider_level_gives_wider_pi() -> None:
    history = _semantic_trajectory_df()
    fc_50 = forecast_semantic_drift(history, horizon=4, level=0.50)
    fc_95 = forecast_semantic_drift(history, horizon=4, level=0.95)
    width_50 = (fc_50["ci_upper"] - fc_50["ci_lower"]).mean()
    width_95 = (fc_95["ci_upper"] - fc_95["ci_lower"]).mean()
    assert width_95 > width_50


@_quiet
def test_horizon_zero_raises() -> None:
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        forecast_semantic_drift(_semantic_trajectory_df(), horizon=0)


@_quiet
def test_invalid_level_raises() -> None:
    with pytest.raises(ValueError, match=r"level must be in \(0, 1\)"):
        forecast_semantic_drift(_semantic_trajectory_df(), horizon=2, level=2.0)


@_quiet
def test_too_short_history_raises() -> None:
    """Need at least 4 observations to forecast."""
    df = pd.DataFrame(
        {
            "period": pd.period_range("2020", periods=3, freq="Y"),
            "target": ["x", "x", "x"],
            "distance_from_baseline": [0.1, 0.2, 0.3],
        }
    )
    with pytest.raises(ValueError, match="at least 4 observations"):
        forecast_semantic_drift(df, horizon=2)


@_quiet
def test_method_explicit_ets_and_holt() -> None:
    fc_ets = forecast_semantic_drift(_semantic_trajectory_df(), horizon=2, method="ets")
    fc_holt = forecast_semantic_drift(_semantic_trajectory_df(), horizon=2, method="holt")
    # Both produce 2-row outputs.
    assert len(fc_ets) == 2 and len(fc_holt) == 2


@_quiet
def test_pi_widens_with_horizon() -> None:
    fc = forecast_semantic_drift(_semantic_trajectory_df(), horizon=6, method="ets")
    widths = (fc["ci_upper"] - fc["ci_lower"]).to_numpy()
    assert (np.diff(widths) >= -1e-9).all()


@_quiet
def test_exported_at_package_root() -> None:
    assert pcd.forecast_semantic_drift is forecast_semantic_drift


@_quiet
def test_plot_returns_layered_chart() -> None:
    pytest.importorskip("altair")
    from pycorpdiff.viz import semantic_forecast_plot

    history = _semantic_trajectory_df()
    fc = forecast_semantic_drift(history, horizon=4)
    chart = semantic_forecast_plot(history, fc)
    spec = chart.to_dict()
    assert "layer" in spec
    # history line + points + seam + forecast band + line + points = 6
    assert len(spec["layer"]) == 6


@_quiet
def test_plot_marks_forecast_dashed() -> None:
    pytest.importorskip("altair")
    from pycorpdiff.viz import semantic_forecast_plot

    history = _semantic_trajectory_df()
    fc = forecast_semantic_drift(history, horizon=4)
    spec = semantic_forecast_plot(history, fc).to_dict()
    has_dashed = False
    for layer in spec["layer"]:
        mark = layer.get("mark", {})
        if isinstance(mark, dict) and mark.get("strokeDash") == [6, 4]:
            has_dashed = True
            break
    assert has_dashed
