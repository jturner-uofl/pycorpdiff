"""Tests for ``kwic``, ``representative_docs``, and the explain plumbing."""

from __future__ import annotations

import pandas as pd
import pytest

import pycorpdiff as pcd
from pycorpdiff.explain import kwic_compare


def _make_corpus(texts: list[str], outlet: str = "A") -> pcd.Corpus:
    return pcd.from_dataframe(
        pd.DataFrame({"text": texts, "outlet": [outlet] * len(texts)}),
        text_col="text",
        meta_cols=("outlet",),
    )


def test_kwic_returns_expected_schema() -> None:
    corpus = _make_corpus(["the cat sat on the mat", "the dog ran"])
    result = pcd.kwic(corpus, target="the", window=2)
    assert isinstance(result, pcd.ConcordanceResult)
    assert list(result.table.columns) == [
        "corpus",
        "doc_id",
        "position",
        "left",
        "keyword",
        "right",
    ]
    # 'the' appears twice in doc 0 and once in doc 1 → 3 lines.
    assert len(result.table) == 3


def test_kwic_window_extracts_correct_context() -> None:
    corpus = _make_corpus(["alpha beta target gamma delta epsilon"])
    line = pcd.kwic(corpus, target="target", window=2).table.iloc[0]
    assert line["left"] == "alpha beta"
    assert line["right"] == "gamma delta"
    assert line["keyword"] == "target"
    assert int(line["position"]) == 2
    assert int(line["doc_id"]) == 0


def test_kwic_respects_document_boundaries() -> None:
    # Without doc isolation, "alpha" from doc 0 would show up in the
    # context of target in doc 1. With isolation it doesn't.
    corpus = _make_corpus(["alpha target", "target beta"])
    table = pcd.kwic(corpus, target="target", window=5).table
    line_doc1 = table[table["doc_id"] == 1].iloc[0]
    assert line_doc1["left"] == ""
    assert line_doc1["right"] == "beta"


def test_kwic_caps_at_n() -> None:
    corpus = _make_corpus(["the cat sat on the mat with the rat"])
    capped = pcd.kwic(corpus, target="the", n=2).table
    assert len(capped) == 2


def test_kwic_no_matches_returns_empty_table() -> None:
    corpus = _make_corpus(["the cat sat"])
    result = pcd.kwic(corpus, target="dog")
    assert len(result.table) == 0
    # The empty frame should still carry the documented columns so
    # downstream consumers can rely on schema.
    assert list(result.table.columns) == [
        "corpus",
        "doc_id",
        "position",
        "left",
        "keyword",
        "right",
    ]


def test_kwic_rejects_zero_window() -> None:
    corpus = _make_corpus(["the cat"])
    with pytest.raises(ValueError, match="window must be"):
        pcd.kwic(corpus, target="the", window=0)


def test_kwic_label_is_propagated() -> None:
    corpus = _make_corpus(["the cat sat"])
    result = pcd.kwic(corpus, target="the", label="press")
    assert (result.table["corpus"] == "press").all()


def test_kwic_compare_yields_one_set_per_side() -> None:
    a = _make_corpus(["the migrant worker", "the migrant family"], outlet="A")
    b = _make_corpus(["the migrant threat", "the migrant invasion"], outlet="B")
    result = kwic_compare(
        a, b, target="migrant", window=2, n_per_side=2, label_a="A", label_b="B"
    )
    assert set(result.table["corpus"]) == {"A", "B"}
    assert (result.table["corpus"] == "A").sum() == 2
    assert (result.table["corpus"] == "B").sum() == 2


def test_kwic_compare_filters_to_collocate_window() -> None:
    a = _make_corpus(
        ["the migrant worker arrived", "the migrant family settled", "criminal said"],
        outlet="A",
    )
    b = _make_corpus(["the migrant threat grew"], outlet="B")
    # Filter to windows containing 'worker' — only doc 0 of A should match.
    result = kwic_compare(
        a, b, target="migrant", window=3, n_per_side=10, collocate="worker"
    )
    assert len(result.table) == 1
    assert result.table.iloc[0]["doc_id"] == 0
    assert "worker" in result.table.iloc[0]["right"]


def test_representative_docs_orders_by_target_frequency() -> None:
    corpus = _make_corpus(
        [
            "target target target filler",  # 3
            "target filler",  # 1
            "no match here",  # 0
            "target target other",  # 2
        ]
    )
    rep = pcd.representative_docs(corpus, target="target", n=5)
    assert rep["count"].tolist() == [3, 2, 1]
    # Documents without the target are excluded.
    assert (rep["count"] > 0).all()


def test_representative_docs_no_matches_returns_empty() -> None:
    corpus = _make_corpus(["the cat sat"])
    rep = pcd.representative_docs(corpus, target="dog")
    assert len(rep) == 0
    assert list(rep.columns) == ["doc_id", "count", "text"]
