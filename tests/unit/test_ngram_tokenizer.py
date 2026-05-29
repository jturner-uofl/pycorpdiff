"""Tests for :class:`pycorpdiff.tokenize.NgramTokenizer`."""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

import pycorpdiff as pcd
from pycorpdiff.tokenize import NgramTokenizer, RegexTokenizer


def test_bigrams_simple() -> None:
    tok = NgramTokenizer(n=2)
    assert tok("the cat sat on the mat") == [
        "the_cat",
        "cat_sat",
        "sat_on",
        "on_the",
        "the_mat",
    ]


def test_trigrams_simple() -> None:
    tok = NgramTokenizer(n=3)
    assert tok("the cat sat on the mat") == [
        "the_cat_sat",
        "cat_sat_on",
        "sat_on_the",
        "on_the_mat",
    ]


def test_unigram_passthrough() -> None:
    """n=1 should be equivalent to the base tokenizer."""
    base = RegexTokenizer()
    tok = NgramTokenizer(n=1)
    text = "the cat sat"
    assert tok(text) == base(text)


def test_include_lower_emits_unigrams_and_bigrams() -> None:
    tok = NgramTokenizer(n=2, include_lower=True)
    assert tok("a b c") == ["a", "b", "c", "a_b", "b_c"]


def test_include_lower_for_trigrams() -> None:
    """include_lower=True with n=3 emits 1-, 2-, and 3-grams."""
    tok = NgramTokenizer(n=3, include_lower=True)
    out = tok("a b c d")
    # unigrams
    assert {"a", "b", "c", "d"}.issubset(set(out))
    # bigrams
    assert {"a_b", "b_c", "c_d"}.issubset(set(out))
    # trigrams
    assert {"a_b_c", "b_c_d"}.issubset(set(out))


def test_custom_separator() -> None:
    tok = NgramTokenizer(n=2, sep=" ")
    assert tok("foo bar baz") == ["foo bar", "bar baz"]


def test_n_less_than_one_raises() -> None:
    with pytest.raises(ValueError, match="n must be >= 1"):
        NgramTokenizer(n=0)


def test_short_doc_yields_empty_when_too_few_tokens() -> None:
    """A 1-token document under n=2 should produce no bigrams."""
    assert NgramTokenizer(n=2)("alone") == []
    assert NgramTokenizer(n=3)("two words") == []


def test_wraps_a_custom_base_tokenizer() -> None:
    """The base parameter is honored — uppercase tokens flow through."""
    base = RegexTokenizer(lowercase=False)
    tok = NgramTokenizer(base=base, n=2)
    assert tok("The Cat") == ["The_Cat"]


def test_protocol_compliance() -> None:
    """Should satisfy the Tokenizer Protocol at runtime."""
    from pycorpdiff.tokenize import Tokenizer

    tok = NgramTokenizer()
    assert isinstance(tok, Tokenizer)


def test_threads_through_corpus_doc_term_counts() -> None:
    """A Corpus built with NgramTokenizer should DTM-count n-grams as terms."""
    df = pd.DataFrame({"text": ["the cat sat", "the cat ran"]})
    corpus = pcd.from_dataframe(df, text_col="text")
    bigrammed = corpus.with_tokenizer(NgramTokenizer(n=2))
    vocab = bigrammed.vocab()
    # "the_cat" appears in both docs → 2; "cat_sat" + "cat_ran" → 1 each
    assert int(vocab["the_cat"]) == 2
    assert int(vocab["cat_sat"]) == 1
    assert int(vocab["cat_ran"]) == 1


def test_threads_through_keyness_comparison() -> None:
    """compare(...).keyness() should rank n-grams just like unigrams."""
    a = pd.DataFrame({"text": ["the cat sat"] * 10 + ["the cat ran"] * 1})
    b = pd.DataFrame({"text": ["the cat ran"] * 10 + ["the cat sat"] * 1})
    tok = NgramTokenizer(n=2)
    ca = pcd.from_dataframe(a, text_col="text").with_tokenizer(tok)
    cb = pcd.from_dataframe(b, text_col="text").with_tokenizer(tok)
    result = pcd.compare(ca, cb).keyness().to_df()
    # Expect "cat_sat" to be A-leaning (positive G²) and "cat_ran" B-leaning.
    cat_sat = result.loc[result["term"] == "cat_sat", "g2"].iloc[0]
    cat_ran = result.loc[result["term"] == "cat_ran", "g2"].iloc[0]
    assert cat_sat > 0
    assert cat_ran < 0


def test_repr_and_frozen() -> None:
    """NgramTokenizer is a frozen dataclass — repr stable, instances hashable."""
    tok = NgramTokenizer(n=2)
    assert "NgramTokenizer" in repr(tok)
    # Frozen => mutations raise (dataclasses raise FrozenInstanceError).
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        tok.n = 3  # type: ignore[misc]
    # Frozen => hashable
    assert isinstance(hash(tok), int)


@given(
    n=st.integers(min_value=1, max_value=4),
    tokens=st.lists(
        st.text(alphabet="abc", min_size=1, max_size=3),
        min_size=0,
        max_size=20,
    ),
)
def test_property_ngram_count_invariant(n: int, tokens: list[str]) -> None:
    """For any input of T unigrams, an n-gram run yields max(0, T-n+1) entries."""
    text = " ".join(tokens) if tokens else ""
    tok = NgramTokenizer(n=n)
    out = tok(text)
    # Recompute T from the *base* tokenizer's view of the string to avoid
    # disagreement on what counts as a token (e.g. punctuation stripping).
    base_tokens = RegexTokenizer()(text)
    expected = max(0, len(base_tokens) - n + 1)
    assert len(out) == expected


@given(
    n=st.integers(min_value=2, max_value=4),
    tokens=st.lists(
        st.text(alphabet="abc", min_size=1, max_size=3),
        min_size=2,
        max_size=20,
    ),
)
def test_property_ngrams_made_of_n_unigrams(n: int, tokens: list[str]) -> None:
    """Every emitted n-gram splits back into exactly n base tokens."""
    text = " ".join(tokens)
    base_tokens = RegexTokenizer()(text)
    if len(base_tokens) < n:
        return  # nothing to check
    tok = NgramTokenizer(n=n, sep="|")
    for gram in tok(text):
        assert len(gram.split("|")) == n
