"""Tests for the ``stop_words=`` parameter on ``Comparison.keyness``."""

from __future__ import annotations

import pandas as pd
import pytest

import pycorpdiff as pcd


@pytest.fixture
def function_word_corpus() -> pcd.Corpus:
    """A two-frame corpus where the top-keyness terms include function words."""
    a_docs = [
        "the migrant worker arrived and the migrant family settled here",
        "the migrant community grew the migrant worker thrived",
        "the migrant worker and the migrant family stayed",
    ] * 4
    b_docs = [
        "the migrant criminal threat and the migrant invasion grew",
        "the migrant threat and the migrant crime increased",
        "the migrant invasion of migrant criminal gangs spread",
    ] * 4
    rows = [{"text": d, "frame": "h"} for d in a_docs] + [
        {"text": d, "frame": "c"} for d in b_docs
    ]
    return pcd.from_dataframe(pd.DataFrame(rows), text_col="text", meta_cols=("frame",))


def test_stop_words_filters_listed_terms(function_word_corpus: pcd.Corpus) -> None:
    a = function_word_corpus.slice(frame="h")
    b = function_word_corpus.slice(frame="c")
    stop = {"the", "and", "of"}
    result = pcd.compare(a, b).keyness(min_count=2, stop_words=stop)
    surviving = set(result.table["term"])
    assert surviving.isdisjoint(stop), (
        f"stop_words {stop & surviving} should not appear in keyness output"
    )


def test_stop_words_accepts_list_or_set(function_word_corpus: pcd.Corpus) -> None:
    a = function_word_corpus.slice(frame="h")
    b = function_word_corpus.slice(frame="c")
    by_list = pcd.compare(a, b).keyness(min_count=2, stop_words=["the", "and"])
    by_set = pcd.compare(a, b).keyness(min_count=2, stop_words={"the", "and"})
    assert (
        set(by_list.table["term"].tolist()) == set(by_set.table["term"].tolist())
    )


def test_stop_words_preserves_corpus_totals(function_word_corpus: pcd.Corpus) -> None:
    """Filtering stop words should not change the corpus-size normalisers — the
    *rates* of surviving terms are unchanged from the unfiltered run, just the
    rows are fewer.
    """
    a = function_word_corpus.slice(frame="h")
    b = function_word_corpus.slice(frame="c")
    full = pcd.compare(a, b).keyness(min_count=2)
    stopped = pcd.compare(a, b).keyness(min_count=2, stop_words={"the"})
    assert full.n_a == stopped.n_a
    assert full.n_b == stopped.n_b
    full_table = full.table.set_index("term")
    stop_table = stopped.table.set_index("term")
    for term in stop_table.index:
        if term in full_table.index:
            # G² for non-stop terms should be identical (corpus totals
            # unchanged → expected counts unchanged → G² unchanged).
            assert (
                abs(full_table.loc[term, "g2"] - stop_table.loc[term, "g2"]) < 1e-9
            )


def test_stop_words_records_in_result_params(
    function_word_corpus: pcd.Corpus,
) -> None:
    a = function_word_corpus.slice(frame="h")
    b = function_word_corpus.slice(frame="c")
    result = pcd.compare(a, b).keyness(stop_words={"the", "and"}, min_count=2)
    assert "stop_words" in result.params
    assert set(result.params["stop_words"]) == {"the", "and"}


def test_no_stop_words_means_no_filter(function_word_corpus: pcd.Corpus) -> None:
    a = function_word_corpus.slice(frame="h")
    b = function_word_corpus.slice(frame="c")
    result = pcd.compare(a, b).keyness(min_count=2)
    # 'the' appears in every sentence — it should be in the keyness table.
    assert "the" in set(result.table["term"]) or "the" in set(result.table["term"])


def test_chi_squared_method_works_end_to_end(
    function_word_corpus: pcd.Corpus,
) -> None:
    """Sorting by χ² produces a populated result with the chi_squared column."""
    a = function_word_corpus.slice(frame="h")
    b = function_word_corpus.slice(frame="c")
    result = pcd.compare(a, b).keyness(method="chi_squared", min_count=2)
    assert result.method == "chi_squared"
    assert "chi_squared" in result.table.columns
    # Sorted by |chi_squared| descending.
    diffs = result.table["chi_squared"].abs().diff().dropna()
    assert (diffs <= 1e-9).all()


def test_chi_squared_signs_match_g2_signs(
    function_word_corpus: pcd.Corpus,
) -> None:
    """χ² and G² should agree on the direction of overuse for every term."""
    a = function_word_corpus.slice(frame="h")
    b = function_word_corpus.slice(frame="c")
    ll = pcd.compare(a, b).keyness(min_count=2).table.set_index("term")
    chi = pcd.compare(a, b).keyness(
        method="chi_squared", min_count=2
    ).table.set_index("term")
    for term in chi.index:
        if abs(chi.loc[term, "chi_squared"]) < 1e-9:
            continue
        assert (chi.loc[term, "chi_squared"] > 0) == (ll.loc[term, "g2"] > 0)
