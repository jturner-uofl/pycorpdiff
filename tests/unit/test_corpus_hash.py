"""Tests for ``Corpus.__hash__`` and equality semantics."""

from __future__ import annotations

import pandas as pd

import pycorpdiff as pcd


def _toy(text="the cat sat", outlet="A") -> pcd.Corpus:
    return pcd.from_dataframe(
        pd.DataFrame({"text": [text], "outlet": [outlet]}),
        text_col="text", meta_cols=("outlet",),
    )


def test_hash_is_stable_across_loads() -> None:
    """Same DataFrame content + same tokenizer config → identical hash."""
    a = _toy()
    b = _toy()
    assert hash(a) == hash(b)


def test_hash_changes_with_content() -> None:
    """Different document text → different hash."""
    a = _toy(text="the cat sat")
    b = _toy(text="the dog ran")
    assert hash(a) != hash(b)


def test_hash_changes_with_metadata() -> None:
    """Same text but different outlet → different hash."""
    a = _toy(outlet="A")
    b = _toy(outlet="B")
    assert hash(a) != hash(b)


def test_hash_distinguishes_text_col_config() -> None:
    df = pd.DataFrame({"text": ["one"], "alt": ["two"]})
    a = pcd.Corpus(docs=df, text_col="text")
    b = pcd.Corpus(docs=df, text_col="alt")
    assert hash(a) != hash(b)


def test_corpus_is_dict_key() -> None:
    """The headline use case: memoise analyses keyed by Corpus."""
    a = _toy()
    cache: dict[pcd.Corpus, str] = {a: "first analysis"}
    again = _toy()
    assert cache[again] == "first analysis"


def test_equality_matches_hash() -> None:
    a = _toy()
    b = _toy()
    c = _toy(text="different")
    assert a == b
    assert a != c
    assert (a == c) is False


def test_equality_returns_notimplemented_for_other_types() -> None:
    a = _toy()
    assert (a == 42) is False  # NotImplemented becomes False at the call site
    assert (a == "string") is False


def test_corpus_with_tokenizer_change_changes_hash() -> None:
    """Different tokenizers → different hashes (the tokenizer is part of the
    corpus configuration)."""
    a = _toy()
    custom = pcd.RegexTokenizer(lowercase=False)
    b = a.with_tokenizer(custom)
    assert hash(a) != hash(b)


def test_hash_works_on_hansard_sample() -> None:
    """Verify the bundled sample hashes deterministically across loads."""
    a = pcd.load_hansard_sample()
    b = pcd.load_hansard_sample()
    assert hash(a) == hash(b)
    assert a == b
