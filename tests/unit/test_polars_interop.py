"""Tests for polars interop on Corpus and the readers.

Polars is in the optional ``[polars]`` extra. These tests skip cleanly
if it isn't installed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import pycorpdiff as pcd

pl = pytest.importorskip("polars")


def test_corpus_accepts_polars_dataframe() -> None:
    pl_df = pl.DataFrame(
        {
            "text": ["the cat sat on the mat", "the dog ran"],
            "outlet": ["A", "B"],
        }
    )
    corpus = pcd.from_dataframe(pl_df, text_col="text", meta_cols=("outlet",))
    assert len(corpus) == 2
    # The stored docs are pandas internally.
    assert isinstance(corpus.docs, pd.DataFrame)
    assert corpus.docs["outlet"].tolist() == ["A", "B"]


def test_polars_input_preserves_data() -> None:
    pl_df = pl.DataFrame(
        {
            "text": ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"],
            "outlet": ["A", "B", "A"],
            "year": [2020, 2021, 2022],
        }
    )
    corpus = pcd.from_dataframe(
        pl_df, text_col="text", meta_cols=("outlet", "year")
    )
    assert corpus.docs["text"].tolist() == [
        "alpha beta gamma",
        "delta epsilon zeta",
        "eta theta iota",
    ]
    assert corpus.docs["year"].tolist() == [2020, 2021, 2022]


def test_corpus_to_polars_round_trip() -> None:
    pd_df = pd.DataFrame(
        {
            "text": ["the cat sat", "the dog ran"],
            "outlet": ["A", "B"],
        }
    )
    corpus = pcd.from_dataframe(pd_df, text_col="text", meta_cols=("outlet",))
    out = corpus.to_polars()
    assert isinstance(out, pl.DataFrame)
    assert out.shape == (2, 2)
    assert out["text"].to_list() == ["the cat sat", "the dog ran"]


def test_polars_corpus_supports_full_analytical_pipeline() -> None:
    """End-to-end: polars in → keyness out."""
    pl_df = pl.DataFrame(
        {
            "text": [
                "the migrant worker arrived and settled",
                "the migrant family thrived together",
                "the migrant worker found work",
                "the migrant criminal threat grew",
                "the migrant invasion of criminal gangs",
                "the migrant criminal element alarmed",
            ],
            "frame": ["h", "h", "h", "c", "c", "c"],
        }
    )
    corpus = pcd.from_dataframe(pl_df, text_col="text", meta_cols=("frame",))
    result = pcd.compare(
        corpus.slice(frame="h"), corpus.slice(frame="c")
    ).keyness(min_count=1)
    assert isinstance(result, pcd.KeynessResult)
    df = result.table.set_index("term")
    # 'worker' should be A-leaning; 'criminal' should be B-leaning.
    if "worker" in df.index:
        assert df.loc["worker", "g2"] > 0
    if "criminal" in df.index:
        assert df.loc["criminal", "g2"] < 0


def test_read_parquet_use_polars_path(tmp_path: Path) -> None:
    """`read_parquet(use_polars=True)` reads via polars, returns same Corpus."""
    parquet_path = tmp_path / "demo.parquet"
    pd.DataFrame(
        {
            "text": ["alpha beta", "gamma delta", "epsilon zeta"],
            "outlet": ["A", "B", "A"],
        }
    ).to_parquet(parquet_path, index=False)

    pandas_corpus = pcd.read_parquet(
        parquet_path, text_col="text", meta_cols=("outlet",)
    )
    polars_corpus = pcd.read_parquet(
        parquet_path, text_col="text", meta_cols=("outlet",), use_polars=True
    )
    # Same content; the read path was different.
    assert len(pandas_corpus) == len(polars_corpus) == 3
    assert (
        pandas_corpus.docs["text"].tolist() == polars_corpus.docs["text"].tolist()
    )


def test_corpus_slice_to_polars() -> None:
    pl_df = pl.DataFrame(
        {
            "text": ["alpha", "beta", "gamma"],
            "outlet": ["A", "B", "A"],
        }
    )
    corpus = pcd.from_dataframe(pl_df, text_col="text", meta_cols=("outlet",))
    a = corpus.slice(outlet="A")
    out = a.to_polars()
    assert isinstance(out, pl.DataFrame)
    assert out.shape == (2, 2)


def test_corpus_with_no_index_after_polars_input() -> None:
    """polars has no row index; the pandas frame we construct should be
    contiguously 0-indexed."""
    pl_df = pl.DataFrame({"text": [f"doc{i}" for i in range(5)]})
    corpus = pcd.from_dataframe(pl_df, text_col="text")
    assert corpus.docs.index.tolist() == [0, 1, 2, 3, 4]
