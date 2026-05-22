"""Smoke tests for altair viz: shape, not pixel-perfect rendering.

We can't render to pixels in CI without a browser, so each test asserts
on the chart's JSON spec (returned by ``chart.to_dict()``). This is
enough to catch typos in column names, broken transforms, and dropped
encodings — the actual visual fidelity check is the manual rendering in
the tutorial notebook.
"""

from __future__ import annotations

import pandas as pd
import pytest

import pycorpdiff as pcd

pytest.importorskip("altair")


@pytest.fixture
def keyness_result() -> pcd.KeynessResult:
    a = pcd.from_dataframe(
        pd.DataFrame(
            {
                "text": [
                    "the migrant worker arrived and the migrant family settled",
                    "the migrant community grew the migrant worker thrived",
                    "the migrant worker and migrant rights advanced",
                ]
            }
        ),
        text_col="text",
    )
    b = pcd.from_dataframe(
        pd.DataFrame(
            {
                "text": [
                    "the migrant criminal threat and the migrant invasion grew",
                    "the migrant threat and the migrant crime increased",
                    "the migrant invasion of migrant criminal gangs spread",
                ]
            }
        ),
        text_col="text",
    )
    return pcd.compare(a, b).keyness(min_count=1)


@pytest.fixture
def collocation_result() -> pcd.CollocationShiftResult:
    a = pcd.from_dataframe(
        pd.DataFrame(
            {
                "text": [
                    "the migrant worker arrived and the migrant family settled",
                    "the migrant community grew the migrant worker thrived",
                ]
            }
        ),
        text_col="text",
    )
    b = pcd.from_dataframe(
        pd.DataFrame(
            {
                "text": [
                    "the migrant criminal threat and the migrant invasion grew",
                    "the migrant invasion of migrant criminal gangs spread",
                ]
            }
        ),
        text_col="text",
    )
    return pcd.compare(a, b).collocation_shift("migrant", window=3, min_count=1)


@pytest.fixture
def trajectory_result() -> pcd.TemporalTrajectory:
    rows = [
        {"text": "the migrant worker arrived", "date": "2020-01-15"},
        {"text": "the migrant family settled", "date": "2020-06-10"},
        {"text": "the migrant criminal threat grew", "date": "2022-02-01"},
        {"text": "the migrant invasion grew", "date": "2022-08-15"},
    ]
    corpus = pcd.from_dataframe(pd.DataFrame(rows), text_col="text")
    return pcd.track(corpus, ["worker", "criminal"]).over_time(freq="Y", time_col="date")


def test_keyness_volcano_returns_layered_chart(keyness_result: pcd.KeynessResult) -> None:
    chart = keyness_result.plot()
    spec = chart.to_dict()
    # A volcano is a layered chart (points + labels).
    assert "layer" in spec


def test_keyness_bar_plot_uses_g2_encoding(keyness_result: pcd.KeynessResult) -> None:
    chart = keyness_result.plot(kind="bar", n=5)
    spec = chart.to_dict()
    enc = spec["encoding"]
    assert enc["x"]["field"] == "g2"
    assert enc["y"]["field"] == "term"


def test_keyness_unknown_kind_raises(keyness_result: pcd.KeynessResult) -> None:
    with pytest.raises(ValueError, match="unknown kind"):
        keyness_result.plot(kind="bogus")


def test_collocation_plot_shape(collocation_result: pcd.CollocationShiftResult) -> None:
    chart = collocation_result.plot(n=5)
    spec = chart.to_dict()
    enc = spec["encoding"]
    assert enc["x"]["field"] == "shift"
    assert enc["y"]["field"] == "collocate"


def test_trajectory_plot_layers(trajectory_result: pcd.TemporalTrajectory) -> None:
    chart = trajectory_result.plot()
    spec = chart.to_dict()
    # Layered: CI band, line, points.
    assert "layer" in spec
    assert len(spec["layer"]) == 3


def test_trajectory_plot_converts_period_to_timestamp(
    trajectory_result: pcd.TemporalTrajectory,
) -> None:
    # The period column should be coerced to datetime so altair gives it a
    # temporal axis. We assert on the encoded type indirectly via the spec.
    chart = trajectory_result.plot()
    spec = chart.to_dict()
    # The first layer's x encoding is what carries the temporal type.
    layer_enc = spec["layer"][0]["encoding"]
    assert layer_enc["x"]["type"] == "temporal"


def test_pcd_viz_functions_accept_bare_dataframe(
    keyness_result: pcd.KeynessResult,
) -> None:
    # Users without a Result should be able to plot a tidy DataFrame.
    from pycorpdiff.viz import keyness_volcano

    chart = keyness_volcano(keyness_result.table)
    assert "layer" in chart.to_dict()
