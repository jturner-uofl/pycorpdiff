"""End-to-end tests for ``Result.explain()`` on keyness + collocation shift."""

from __future__ import annotations

import pandas as pd
import pytest

import pycorpdiff as pcd


@pytest.fixture
def two_outlet_corpus() -> pcd.Corpus:
    a_docs = [
        "the migrant worker arrived and the migrant family settled",
        "the migrant community grew the migrant worker thrived",
        "the migrant worker and the migrant family stayed",
        "the migrant worker and migrant rights advanced",
    ]
    b_docs = [
        "the migrant criminal threat and the migrant invasion grew",
        "the migrant threat and the migrant crime increased",
        "the migrant invasion of migrant criminal gangs spread",
        "the migrant criminal gangs and migrant invasion stayed",
    ]
    rows = [{"text": d, "outlet": "A"} for d in a_docs] + [
        {"text": d, "outlet": "B"} for d in b_docs
    ]
    return pcd.from_dataframe(pd.DataFrame(rows), text_col="text", meta_cols=("outlet",))


def test_keyness_result_explain_returns_concordance(
    two_outlet_corpus: pcd.Corpus,
) -> None:
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    result = pcd.compare(a, b).keyness(min_count=1)
    concordance = result.explain("migrant", n=3, window=2)
    assert isinstance(concordance, pcd.ConcordanceResult)
    # Both corpora should contribute up to n lines.
    by_corpus = concordance.table.groupby("corpus").size()
    assert by_corpus.loc["outlet='A'"] <= 3
    assert by_corpus.loc["outlet='B'"] <= 3
    # Every line's keyword column is the target.
    assert (concordance.table["keyword"] == "migrant").all()


def test_keyness_explain_without_source_corpora_raises() -> None:
    # Construct a KeynessResult by hand (no corpus refs) — explain must
    # refuse rather than silently returning empty evidence. The path
    # where ``corpus_a`` is present and ``corpus_b`` is ``None`` (used
    # by ``against_baseline``) is covered separately and now *succeeds*
    # with single-side KWIC; the bare case here still raises.
    bare = pcd.KeynessResult(
        table=pd.DataFrame({"term": ["x"], "g2": [1.0]}),
        method="log_likelihood",
        n_a=100,
        n_b=100,
    )
    with pytest.raises(ValueError, match="A-side corpus"):
        bare.explain("x")


def test_collocation_explain_filters_to_collocate_windows(
    two_outlet_corpus: pcd.Corpus,
) -> None:
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    shift = pcd.compare(a, b).collocation_shift("migrant", window=3, min_count=1)
    concordance = shift.explain("worker", n=5)
    # Every returned line should have 'worker' in either left or right context.
    for _, row in concordance.table.iterrows():
        context = f"{row['left']} {row['right']}"
        assert "worker" in context.split(), (
            f"'worker' missing from window: {context!r}"
        )


def test_collocation_explain_without_source_corpora_raises() -> None:
    bare = pcd.CollocationShiftResult(
        target="x",
        table=pd.DataFrame({"collocate": ["y"], "shift": [0.5]}),
        measure="logDice",
        window=5,
    )
    with pytest.raises(ValueError, match="requires source corpora"):
        bare.explain("y")


def test_collocation_explain_shows_both_sides_when_collocate_in_both(
    two_outlet_corpus: pcd.Corpus,
) -> None:
    # 'the' co-occurs with 'migrant' in both A and B.
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    shift = pcd.compare(a, b).collocation_shift("migrant", window=3, min_count=1)
    concordance = shift.explain("the", n=2)
    assert set(concordance.table["corpus"]) == {"outlet='A'", "outlet='B'"}
