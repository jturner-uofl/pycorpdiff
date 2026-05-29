"""Tests for the vocab / doc_term_counts plumbing on Corpus and CorpusSlice."""

from __future__ import annotations

import pandas as pd

import pycorpdiff as pcd


def test_vocab_sums_across_documents(toy_corpus: pcd.Corpus) -> None:
    vocab = toy_corpus.vocab()
    # "the" appears in all three docs: 2 + 2 + 3 = 7.
    assert int(vocab["the"]) == 7
    assert int(vocab["cat"]) == 2
    assert int(vocab["dog"]) == 2


def test_vocab_min_count_filters(toy_corpus: pcd.Corpus) -> None:
    vocab = toy_corpus.vocab(min_count=3)
    # Only "the" (7) survives a min_count=3 filter on the toy corpus.
    assert list(vocab.index) == ["the"]


def test_total_tokens_matches_dtm_sum(toy_corpus: pcd.Corpus) -> None:
    # Three docs of length 6, 6, 8 — total 20.
    assert toy_corpus.total_tokens() == 20


def test_doc_term_counts_shape_and_dtype(toy_corpus: pcd.Corpus) -> None:
    dtm = toy_corpus.doc_term_counts()
    assert dtm.shape[0] == 3
    assert dtm.dtypes.unique().tolist() == [pd.api.types.pandas_dtype("int64")]
    # Every row's count sum equals the doc's token count.
    assert dtm.iloc[0].sum() == 6
    assert dtm.iloc[2].sum() == 8


def test_corpus_slice_vocab_only_counts_in_slice(toy_corpus: pcd.Corpus) -> None:
    a = toy_corpus.slice(outlet="A")  # docs 0 and 2 only
    vocab = a.vocab()
    # In A: "the" 2 + 3 = 5; "cat" 1 + 1 = 2; "dog" appears in doc 2 only = 1.
    assert int(vocab["the"]) == 5
    assert int(vocab["cat"]) == 2
    assert int(vocab["dog"]) == 1
    # "log" appears only in doc 1 (outlet B) and should be absent.
    assert "log" not in vocab.index


def test_corpus_slice_total_tokens_excludes_unmasked(toy_corpus: pcd.Corpus) -> None:
    a = toy_corpus.slice(outlet="A")
    assert a.total_tokens() == 14  # docs 0 (6 tokens) + 2 (8 tokens)
