"""Tests for ``semantic_trajectory`` and ``Tracker.semantic_over_time``."""

from __future__ import annotations

import math

import pandas as pd
import pytest

import pycorpdiff as pcd


def _shifting_corpus() -> pcd.Corpus:
    """3-year corpus where 'migrant' has dramatically different contexts each year."""
    rows = []
    # Year 2020: humanising contexts
    rows += [
        {"text": "the migrant worker arrived and the family settled", "date": "2020-03-15"},
        {"text": "the migrant community welcomed new arrivals warmly", "date": "2020-06-15"},
        {"text": "the migrant family settled and thrived together", "date": "2020-09-15"},
    ]
    # Year 2021: neutral / mixed
    rows += [
        {"text": "the migrant arrived at the border this year", "date": "2021-03-15"},
        {"text": "the migrant population shifted across regions", "date": "2021-06-15"},
        {"text": "the migrant statistics confirm steady arrivals", "date": "2021-09-15"},
    ]
    # Year 2022: criminalising contexts
    rows += [
        {"text": "the migrant criminal threat grew worse this year", "date": "2022-03-15"},
        {"text": "the migrant invasion of criminal gangs spread further", "date": "2022-06-15"},
        {"text": "the migrant criminal element alarmed residents nationwide", "date": "2022-09-15"},
    ]
    return pcd.from_dataframe(pd.DataFrame(rows), text_col="text")


def test_semantic_trajectory_returns_tidy_frame() -> None:
    corpus = _shifting_corpus()
    df = pcd.semantic_trajectory(
        corpus, "migrant", time_col="date", freq="Y",
        embedder=pcd.HashEmbedder(dim=64),
    )
    assert list(df.columns) == [
        "period", "target", "n_contexts",
        "similarity_to_baseline", "distance_from_baseline",
    ]
    # 3 periods × 1 target = 3 rows.
    assert len(df) == 3
    assert set(df["target"]) == {"migrant"}


def test_semantic_trajectory_baseline_has_zero_distance() -> None:
    """The baseline period's distance to itself should be exactly 0."""
    corpus = _shifting_corpus()
    df = pcd.semantic_trajectory(
        corpus, "migrant", time_col="date", freq="Y",
        embedder=pcd.HashEmbedder(dim=64),
    )
    # First populated period (2020) is the default baseline.
    baseline_row = df[df["period"] == pd.Period("2020", freq="Y")].iloc[0]
    assert math.isclose(baseline_row["distance_from_baseline"], 0.0, abs_tol=1e-12)
    assert math.isclose(baseline_row["similarity_to_baseline"], 1.0, abs_tol=1e-12)


def test_semantic_trajectory_explicit_baseline_period() -> None:
    """Passing baseline_period= anchors the trajectory at that period."""
    corpus = _shifting_corpus()
    df = pcd.semantic_trajectory(
        corpus, "migrant", time_col="date", freq="Y",
        embedder=pcd.HashEmbedder(dim=64),
        baseline_period="2022",
    )
    # 2022 is the baseline → distance ≈ 0 there.
    row_2022 = df[df["period"] == pd.Period("2022", freq="Y")].iloc[0]
    assert math.isclose(row_2022["distance_from_baseline"], 0.0, abs_tol=1e-12)


def test_semantic_trajectory_changing_contexts_yield_positive_drift() -> None:
    """When the contexts of 'migrant' change across periods, distance > 0."""
    corpus = _shifting_corpus()
    df = pcd.semantic_trajectory(
        corpus, "migrant", time_col="date", freq="Y",
        embedder=pcd.HashEmbedder(dim=64),
    )
    # 2022 is dramatically different from 2020 — expect distance > 0.1.
    row_2022 = df[df["period"] == pd.Period("2022", freq="Y")].iloc[0]
    assert row_2022["distance_from_baseline"] > 0.1


def test_semantic_trajectory_records_context_counts() -> None:
    corpus = _shifting_corpus()
    df = pcd.semantic_trajectory(
        corpus, "migrant", time_col="date", freq="Y",
        embedder=pcd.HashEmbedder(dim=64),
    )
    # Each year has 3 documents, each with one 'migrant' occurrence.
    assert (df["n_contexts"] == 3).all()


def test_semantic_trajectory_multi_target() -> None:
    corpus = _shifting_corpus()
    df = pcd.semantic_trajectory(
        corpus, ["migrant", "the"], time_col="date", freq="Y",
        embedder=pcd.HashEmbedder(dim=64),
    )
    # 3 periods × 2 targets = 6 rows; sorted by target then period.
    assert len(df) == 6
    assert set(df["target"]) == {"migrant", "the"}


def test_semantic_trajectory_target_absent_emits_nan() -> None:
    corpus = _shifting_corpus()
    df = pcd.semantic_trajectory(
        corpus, "unicorn", time_col="date", freq="Y",
        embedder=pcd.HashEmbedder(dim=64),
    )
    assert len(df) == 3
    assert (df["n_contexts"] == 0).all()
    assert df["distance_from_baseline"].isna().all()


def test_semantic_trajectory_baseline_without_contexts_raises() -> None:
    corpus = _shifting_corpus()
    # 'unicorn' doesn't appear in any period, so baseline_period='2020' has no contexts.
    with pytest.raises(ValueError, match="baseline_period"):
        pcd.semantic_trajectory(
            corpus, "unicorn", time_col="date", freq="Y",
            embedder=pcd.HashEmbedder(dim=64),
            baseline_period="2020",
        )


def test_tracker_semantic_over_time_wrapper() -> None:
    """The Tracker convenience method delegates correctly."""
    corpus = _shifting_corpus()
    df = pcd.track(corpus, "migrant").semantic_over_time(
        time_col="date", freq="Y", embedder=pcd.HashEmbedder(dim=64),
    )
    assert len(df) == 3
    assert "distance_from_baseline" in df.columns


def test_semantic_trajectory_deterministic_under_hash_embedder() -> None:
    """Same corpus + same embedder → identical trajectory across runs."""
    corpus = _shifting_corpus()
    df1 = pcd.semantic_trajectory(
        corpus, "migrant", time_col="date", freq="Y",
        embedder=pcd.HashEmbedder(dim=32),
    )
    df2 = pcd.semantic_trajectory(
        corpus, "migrant", time_col="date", freq="Y",
        embedder=pcd.HashEmbedder(dim=32),
    )
    pd.testing.assert_frame_equal(df1, df2)


def test_semantic_trajectory_works_on_corpus_slice() -> None:
    """A slice of the corpus should be a valid input."""
    rows = [
        {"text": "the migrant worker arrived and settled", "date": "2020-03-15", "outlet": "A"},
        {"text": "the migrant criminal threat grew", "date": "2022-03-15", "outlet": "A"},
        {"text": "the dog barked loudly", "date": "2020-03-15", "outlet": "B"},
    ]
    corpus = pcd.from_dataframe(
        pd.DataFrame(rows), text_col="text", meta_cols=("outlet", "date")
    )
    slice_a = corpus.slice(outlet="A")
    df = pcd.semantic_trajectory(
        slice_a, "migrant", time_col="date", freq="Y",
        embedder=pcd.HashEmbedder(dim=32),
    )
    assert len(df) >= 1
    assert (df["target"] == "migrant").all()
