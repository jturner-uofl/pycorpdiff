"""Tests for ``TemporalCorpus``, ``Tracker.over_time``, and trajectories."""

from __future__ import annotations

import math

import pandas as pd
import pytest

import pycorpdiff as pcd


@pytest.fixture
def dated_corpus() -> pcd.Corpus:
    """Three years of mini news, with shifting term frequencies."""
    rows = [
        {"text": "the immigrant worker arrived", "date": "2020-01-15"},
        {"text": "the immigrant family settled here", "date": "2020-06-10"},
        {"text": "the immigrant worker thrived", "date": "2021-03-20"},
        {"text": "the immigrant criminal threat increased", "date": "2022-02-01"},
        {"text": "the immigrant invasion grew dangerous", "date": "2022-08-15"},
        {"text": "the immigrant criminal threat persisted", "date": "2022-11-30"},
    ]
    return pcd.from_dataframe(pd.DataFrame(rows), text_col="text", meta_cols=("date",))


def test_temporal_corpus_periods_yearly(dated_corpus: pcd.Corpus) -> None:
    tc = dated_corpus.by_time("date", freq="Y")
    periods = tc.periods()
    assert [str(p) for p in periods] == ["2020", "2021", "2022"]


def test_temporal_corpus_periods_quarterly(dated_corpus: pcd.Corpus) -> None:
    tc = dated_corpus.by_time("date", freq="Q")
    quarters = [str(p) for p in tc.periods()]
    # Only the populated quarters should appear.
    assert "2020Q1" in quarters
    assert "2022Q3" in quarters
    assert "2020Q3" not in quarters  # no doc in this quarter


def test_temporal_corpus_slice_returns_correct_subset(dated_corpus: pcd.Corpus) -> None:
    tc = dated_corpus.by_time("date", freq="Y")
    slice_2022 = tc.slice("2022")
    assert len(slice_2022) == 3
    assert all("2022" in str(d) for d in slice_2022.docs["date"])


def test_temporal_corpus_iter_slices_in_order(dated_corpus: pcd.Corpus) -> None:
    tc = dated_corpus.by_time("date", freq="Y")
    pairs = list(tc.iter_slices())
    period_labels = [str(p) for p, _ in pairs]
    assert period_labels == ["2020", "2021", "2022"]
    # 2022 had three documents; first 'criminal' appears in that bucket.
    period_2022, slice_2022 = pairs[2]
    assert "criminal" in slice_2022.vocab().index


def test_tracker_over_time_returns_trajectory(dated_corpus: pcd.Corpus) -> None:
    tr = pcd.track(dated_corpus, "criminal").over_time(freq="Y", time_col="date")
    assert isinstance(tr, pcd.TemporalTrajectory)
    df = tr.table
    assert list(df.columns) == [
        "period",
        "term",
        "count",
        "total",
        "relfreq",
        "ci_lower",
        "ci_upper",
    ]
    # Three periods (2020, 2021, 2022), one term → three rows.
    assert len(df) == 3


def test_tracker_over_time_counts_match_corpus(dated_corpus: pcd.Corpus) -> None:
    tr = pcd.track(dated_corpus, "criminal").over_time(freq="Y", time_col="date")
    df = tr.table.set_index("period")
    # 'criminal' appears twice in 2022, zero elsewhere.
    assert df.loc[pd.Period("2020", "Y"), "count"] == 0
    assert df.loc[pd.Period("2021", "Y"), "count"] == 0
    assert df.loc[pd.Period("2022", "Y"), "count"] == 2


def test_tracker_over_time_multiple_targets(dated_corpus: pcd.Corpus) -> None:
    tr = pcd.track(dated_corpus, ["worker", "criminal"]).over_time(
        freq="Y", time_col="date"
    )
    df = tr.table
    # Two terms × three periods = six rows.
    assert len(df) == 6
    # Sorted by term then period.
    assert df["term"].is_monotonic_increasing or df["term"].iloc[0] == "criminal"
    by_term = df.groupby("term")
    assert by_term.get_group("criminal")["count"].sum() == 2
    assert by_term.get_group("worker")["count"].sum() == 2


def test_tracker_over_time_relfreq_matches_count_div_total(
    dated_corpus: pcd.Corpus,
) -> None:
    tr = pcd.track(dated_corpus, "the").over_time(freq="Y", time_col="date")
    df = tr.table
    for _, row in df.iterrows():
        if row["total"] > 0:
            expected = row["count"] / row["total"]
            assert math.isclose(row["relfreq"], expected, rel_tol=1e-12)


def test_tracker_over_time_wilson_ci_brackets_pointestimate(
    dated_corpus: pcd.Corpus,
) -> None:
    tr = pcd.track(dated_corpus, "the").over_time(freq="Y", time_col="date")
    df = tr.table
    for _, row in df.iterrows():
        if row["count"] > 0:
            assert row["ci_lower"] <= row["relfreq"] <= row["ci_upper"]
            assert 0.0 <= row["ci_lower"] <= 1.0
            assert 0.0 <= row["ci_upper"] <= 1.0


def test_tracker_trajectory_alias(dated_corpus: pcd.Corpus) -> None:
    via_over_time = pcd.track(dated_corpus, "criminal").over_time(
        freq="Y", time_col="date"
    )
    via_trajectory = pcd.track(dated_corpus, "criminal").trajectory(
        freq="Y", time_col="date"
    )
    pd.testing.assert_frame_equal(via_over_time.table, via_trajectory.table)


def test_tracker_over_time_zero_count_yields_zero_ci_lower(
    dated_corpus: pcd.Corpus,
) -> None:
    tr = pcd.track(dated_corpus, "criminal").over_time(freq="Y", time_col="date")
    # In 2020 'criminal' doesn't appear; Wilson lower must be 0.
    row = tr.table.set_index("period").loc[pd.Period("2020", "Y")]
    assert math.isclose(row["ci_lower"], 0.0, abs_tol=1e-12)


def test_temporal_trajectory_summary_includes_period_count(
    dated_corpus: pcd.Corpus,
) -> None:
    tr = pcd.track(dated_corpus, "criminal").over_time(freq="Y", time_col="date")
    summary = tr.summary()
    assert "criminal" in summary
    assert "periods=3" in summary
