"""Tests for :meth:`Corpus.doc_term_counts_sparse`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import sparse

import pycorpdiff as pcd


@pytest.fixture
def toy() -> pcd.Corpus:
    df = pd.DataFrame(
        {
            "text": ["the cat sat", "the dog ran fast", "cat dog cat"],
            "outlet": ["A", "B", "A"],
        }
    )
    return pcd.from_dataframe(df, text_col="text", meta_cols=("outlet",))


def test_returns_csr_matrix_and_sorted_vocab(toy: pcd.Corpus) -> None:
    matrix, vocab = toy.doc_term_counts_sparse()
    assert sparse.isspmatrix_csr(matrix)
    assert vocab == sorted(vocab)


def test_shape_matches_dense(toy: pcd.Corpus) -> None:
    dense = toy.doc_term_counts()
    matrix, vocab = toy.doc_term_counts_sparse()
    assert matrix.shape == dense.shape
    assert vocab == list(dense.columns)


def test_values_agree_with_dense(toy: pcd.Corpus) -> None:
    """Cell-by-cell equality between sparse and dense paths."""
    dense = toy.doc_term_counts()
    matrix, vocab = toy.doc_term_counts_sparse()
    np.testing.assert_array_equal(matrix.toarray(), dense.values)


def test_dtype_is_int64(toy: pcd.Corpus) -> None:
    matrix, _ = toy.doc_term_counts_sparse()
    assert matrix.dtype == np.int64


def test_min_count_drops_low_frequency_terms(toy: pcd.Corpus) -> None:
    """Vocab + columns should both shrink in sync when min_count fires."""
    matrix, vocab = toy.doc_term_counts_sparse(min_count=2)
    # In the toy corpus, only "the" (2), "cat" (3), and "dog" (2) survive.
    assert set(vocab) == {"the", "cat", "dog"}
    assert matrix.shape[1] == 3
    # Column totals should all be >= 2.
    col_totals = np.asarray(matrix.sum(axis=0)).ravel()
    assert (col_totals >= 2).all()


def test_min_count_agrees_with_dense(toy: pcd.Corpus) -> None:
    dense = toy.doc_term_counts(min_count=2)
    matrix, vocab = toy.doc_term_counts_sparse(min_count=2)
    assert vocab == list(dense.columns)
    np.testing.assert_array_equal(matrix.toarray(), dense.values)


def test_works_on_corpus_slice(toy: pcd.Corpus) -> None:
    a = toy.slice(outlet="A")  # docs 0 and 2
    matrix, vocab = a.doc_term_counts_sparse()
    # Outlet A has docs 0 ("the cat sat") and 2 ("cat dog cat") → 2 rows.
    assert matrix.shape[0] == 2
    # Dense parity on the slice.
    dense = a.doc_term_counts()
    np.testing.assert_array_equal(matrix.toarray(), dense.values)
    assert vocab == list(dense.columns)


def test_empty_corpus_returns_empty_matrix() -> None:
    df = pd.DataFrame({"text": []})
    df["text"] = df["text"].astype(str)
    corpus = pcd.from_dataframe(df, text_col="text")
    matrix, vocab = corpus.doc_term_counts_sparse()
    assert matrix.shape == (0, 0)
    assert vocab == []


def test_memory_is_sparse_for_skinny_long_corpus() -> None:
    """A bag of unique words across many docs should produce nnz == n_docs."""
    df = pd.DataFrame({"text": [f"unique{i}" for i in range(100)]})
    corpus = pcd.from_dataframe(df, text_col="text")
    matrix, vocab = corpus.doc_term_counts_sparse()
    assert matrix.shape == (100, 100)
    assert matrix.nnz == 100  # exactly one non-zero per row


def test_can_feed_into_sklearn_style_pipeline(toy: pcd.Corpus) -> None:
    """(csr_matrix, vocab) is the canonical sklearn shape — sanity check."""
    matrix, vocab = toy.doc_term_counts_sparse()
    # Simulate a basic downstream op: tf normalisation per row.
    row_sums = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    # Avoid division-by-zero on empty rows.
    row_sums[row_sums == 0] = 1.0
    inv = sparse.diags(1.0 / row_sums)
    tf = inv @ matrix
    np.testing.assert_allclose(np.asarray(tf.sum(axis=1)).ravel(), 1.0)


@given(
    n_docs=st.integers(min_value=1, max_value=15),
    vocab_pool=st.lists(
        st.text(alphabet="abc", min_size=1, max_size=3),
        min_size=1,
        max_size=8,
        unique=True,
    ),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_property_sparse_dense_agree(
    n_docs: int, vocab_pool: list[str], seed: int
) -> None:
    """For any random toy corpus, sparse and dense DTMs must agree exactly."""
    rng = np.random.default_rng(seed)
    docs = []
    for _ in range(n_docs):
        n_words = int(rng.integers(0, 6))
        words = rng.choice(vocab_pool, size=n_words).tolist() if n_words else []
        docs.append(" ".join(words))
    df = pd.DataFrame({"text": docs})
    corpus = pcd.from_dataframe(df, text_col="text")
    dense = corpus.doc_term_counts()
    matrix, vocab = corpus.doc_term_counts_sparse()
    assert vocab == list(dense.columns)
    np.testing.assert_array_equal(matrix.toarray(), dense.values)
