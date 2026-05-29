"""End-to-end tests for TemporalTrajectory.changepoints / interrupted_time_series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pycorpdiff as pcd


@pytest.fixture
def event_corpus() -> pcd.Corpus:
    """Generate a 40-year corpus with a clear discourse shift at 2010."""
    rng = np.random.default_rng(seed=0)
    rows = []
    for year in range(1990, 2030):
        # Pre-event: 'worker' dominant. Post-event: 'criminal' dominant.
        if year < 2010:
            template = ["the migrant worker arrived and the migrant family settled"]
            n = 4
        else:
            template = [
                "the migrant criminal threat and the migrant invasion grew worse"
            ]
            n = 4
        for _ in range(n):
            month = rng.integers(1, 13)
            rows.append(
                {"text": template[0], "date": f"{year}-{int(month):02d}-15"}
            )
    return pcd.from_dataframe(
        pd.DataFrame(rows), text_col="text", meta_cols=("date",)
    )


def test_trajectory_changepoints_detects_known_event(event_corpus: pcd.Corpus) -> None:
    trajectory = pcd.track(event_corpus, "criminal").over_time(
        freq="Y", time_col="date"
    )
    df = trajectory.changepoints()
    assert len(df) >= 1
    # The engineered shift is at 2010. The detected changepoint should
    # land within 2 years.
    detected_years = {int(str(p)) for p in df["period"]}
    assert any(2008 <= y <= 2012 for y in detected_years), (
        f"expected changepoint near 2010; got {detected_years}"
    )


def test_trajectory_changepoints_requires_target_for_multi(
    event_corpus: pcd.Corpus,
) -> None:
    trajectory = pcd.track(event_corpus, ["worker", "criminal"]).over_time(
        freq="Y", time_col="date"
    )
    with pytest.raises(ValueError, match="pass target= to pick one"):
        trajectory.changepoints()


def test_trajectory_changepoints_picks_specified_target(
    event_corpus: pcd.Corpus,
) -> None:
    trajectory = pcd.track(event_corpus, ["worker", "criminal"]).over_time(
        freq="Y", time_col="date"
    )
    df = trajectory.changepoints(target="criminal")
    assert isinstance(df, pd.DataFrame)


def test_trajectory_its_detects_engineered_step(event_corpus: pcd.Corpus) -> None:
    trajectory = pcd.track(event_corpus, "criminal").over_time(
        freq="Y", time_col="date"
    )
    df = trajectory.interrupted_time_series(event_date="2010")
    level = df[df["term"] == "level_change"].iloc[0]
    # 'criminal' jumps from ~0 frequency to ~0.13 (1/8 tokens per doc),
    # so the level-change coefficient should be positive and significant.
    assert level["coef"] > 0.05
    assert level["p_value"] < 0.05
