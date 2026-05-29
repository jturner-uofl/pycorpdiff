"""Tests for the bundled Hansard sample."""

from __future__ import annotations

import pycorpdiff as pcd


def test_load_hansard_sample_returns_corpus() -> None:
    corpus = pcd.load_hansard_sample()
    assert isinstance(corpus, pcd.Corpus)
    # The generator ships ~190-200 speeches; this is a stability check.
    assert 150 < len(corpus) < 250


def test_hansard_sample_has_expected_columns() -> None:
    corpus = pcd.load_hansard_sample()
    expected = {"speech_id", "text", "topic", "frame", "party", "date", "year"}
    assert expected <= set(corpus.docs.columns)


def test_hansard_sample_spans_expected_year_range() -> None:
    corpus = pcd.load_hansard_sample()
    assert int(corpus.docs["year"].min()) == 2005
    assert int(corpus.docs["year"].max()) == 2023


def test_hansard_sample_has_four_topics() -> None:
    corpus = pcd.load_hansard_sample()
    topics = set(corpus.docs["topic"])
    assert topics == {"immigration", "brexit", "nhs", "climate"}


def test_hansard_sample_has_four_parties() -> None:
    corpus = pcd.load_hansard_sample()
    parties = set(corpus.docs["party"])
    assert parties == {"Labour", "Conservative", "Liberal Democrat", "SNP"}


def test_hansard_sample_supports_slicing() -> None:
    corpus = pcd.load_hansard_sample()
    immigration = corpus.slice(topic="immigration")
    assert len(immigration) > 30
    assert (immigration.docs["topic"] == "immigration").all()


def test_hansard_sample_drives_keyness_analysis() -> None:
    """End-to-end: load → slice → compare → keyness on the bundled corpus."""
    corpus = pcd.load_hansard_sample()
    humanising = corpus.slice(frame="humanising")
    criminalising = corpus.slice(frame="criminalising")
    result = pcd.compare(humanising, criminalising).keyness(min_count=3)
    assert isinstance(result, pcd.KeynessResult)
    # 'criminal' should be B-leaning (negative signed G²) since the
    # criminalising-frame templates use it heavily.
    df = result.table.set_index("term")
    if "criminal" in df.index:
        assert df.loc["criminal", "g2"] < 0


def test_hansard_sample_supports_temporal_track() -> None:
    """The corpus has a date column; track() should work directly."""
    corpus = pcd.load_hansard_sample()
    trajectory = pcd.track(corpus, "criminal").over_time(
        freq="Y", time_col="date"
    )
    assert isinstance(trajectory, pcd.TemporalTrajectory)
    # 'criminal' should appear more after 2016 than before (engineered shift).
    df = trajectory.table.set_index("period")
    pre = df.loc[df.index < df.index[df.index.year >= 2016].min(), "count"].sum()
    post = df.loc[df.index >= df.index[df.index.year >= 2016].min(), "count"].sum()
    assert post > pre


def test_hansard_sample_is_deterministic() -> None:
    """Two loads return identical content."""
    a = pcd.load_hansard_sample()
    b = pcd.load_hansard_sample()
    # Same document texts in same order.
    assert a.docs["text"].tolist() == b.docs["text"].tolist()
    assert a.docs["topic"].tolist() == b.docs["topic"].tolist()
