"""Smoke tests for the Phase 0 scaffolding.

These exercise the parts of the package that are real in the scaffolding
release: imports, the Corpus constructor, slicing, the regex tokenizer,
and the CSV/parquet readers. Analytical methods are expected to raise
NotImplementedError and are intentionally not exercised here — Phase 1
will replace those tests with real ones.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import pycorpdiff as pcd


def test_version_is_string() -> None:
    assert isinstance(pcd.__version__, str)
    assert pcd.__version__.startswith("0.")


def test_public_api_surface() -> None:
    # The README and CITATION advertise these names; this guards against
    # accidental re-export deletions during refactors.
    expected = {
        "Corpus",
        "CorpusSlice",
        "Comparison",
        "Tokenizer",
        "RegexTokenizer",
        "compare",
        "track",
        "from_dataframe",
        "read_csv",
        "read_parquet",
        "read_txt",
        "KeynessResult",
        "CollocationShiftResult",
        "SemanticShiftResult",
        "TemporalTrajectory",
        "ConcordanceResult",
    }
    assert expected.issubset(set(dir(pcd)))


def test_regex_tokenizer_is_unicode_aware() -> None:
    tok = pcd.RegexTokenizer()
    # Greek + Cyrillic + Latin in one pass; \w+ under re.UNICODE should
    # treat each script's letters as word characters.
    assert tok("hello мир γειά") == ["hello", "мир", "γειά"]


def test_regex_tokenizer_lowercases_by_default() -> None:
    tok = pcd.RegexTokenizer()
    assert tok("Hello World") == ["hello", "world"]


def test_corpus_construction(toy_corpus: pcd.Corpus) -> None:
    assert len(toy_corpus) == 3
    assert toy_corpus.text_col == "text"
    assert "outlet" in toy_corpus.metadata_columns


def test_corpus_rejects_missing_text_col(toy_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="text_col"):
        pcd.Corpus(docs=toy_df, text_col="body")


def test_corpus_rejects_polars_backend(toy_df: pd.DataFrame) -> None:
    # The polars backend is reserved for a later phase; surfacing it as
    # NotImplementedError now keeps the contract honest.
    with pytest.raises(NotImplementedError):
        pcd.Corpus(docs=toy_df, backend="polars")


def test_corpus_slice_scalar_filter(toy_corpus: pcd.Corpus) -> None:
    a = toy_corpus.slice(outlet="A")
    assert len(a) == 2
    assert a.label == "outlet='A'"


def test_corpus_slice_iterable_filter(toy_corpus: pcd.Corpus) -> None:
    s = toy_corpus.slice(outlet=["A", "B"], year=2020)
    assert len(s) == 2


def test_corpus_slice_unknown_column_raises(toy_corpus: pcd.Corpus) -> None:
    with pytest.raises(KeyError):
        toy_corpus.slice(nonexistent="x")


def test_compare_returns_comparison(toy_corpus: pcd.Corpus) -> None:
    a = toy_corpus.slice(outlet="A")
    b = toy_corpus.slice(outlet="B")
    cmp = pcd.compare(a, b)
    assert isinstance(cmp, pcd.Comparison)


def test_compare_before_after_splits_on_date() -> None:
    df = pd.DataFrame(
        {
            "text": ["pre1", "pre2", "post1"],
            "date": pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01"]),
        }
    )
    corpus = pcd.from_dataframe(df, text_col="text")
    cmp = pcd.compare.before_after(corpus, event_date="2021-01-01", time_col="date")
    assert len(cmp.a) == 2
    assert len(cmp.b) == 1


def test_keyness_method_returns_result(toy_corpus: pcd.Corpus) -> None:
    # The toy corpus is too small for min_count=5 to leave anything; drop
    # the threshold to verify keyness completes end-to-end on the plumbing.
    a = toy_corpus.slice(outlet="A")
    b = toy_corpus.slice(outlet="B")
    result = pcd.compare(a, b).keyness(min_count=1)
    assert isinstance(result, pcd.KeynessResult)


def test_read_csv_round_trip(tmp_path: Path, toy_df: pd.DataFrame) -> None:
    path = tmp_path / "toy.csv"
    toy_df.to_csv(path, index=False)
    corpus = pcd.read_csv(path, text_col="text")
    assert len(corpus) == 3


def test_read_parquet_round_trip(tmp_path: Path, toy_df: pd.DataFrame) -> None:
    path = tmp_path / "toy.parquet"
    toy_df.to_parquet(path, index=False)
    corpus = pcd.read_parquet(path, text_col="text")
    assert len(corpus) == 3


def test_track_returns_tracker(toy_corpus: pcd.Corpus) -> None:
    tr = pcd.track(toy_corpus, "cat")
    assert tr.targets == ["cat"]
    # over_time() needs a time column; the toy corpus has 'year' instead of
    # the default 'date'. Passing it explicitly should work end-to-end.
    trajectory = tr.over_time(freq="Y", time_col="year")
    assert isinstance(trajectory, pcd.TemporalTrajectory)
