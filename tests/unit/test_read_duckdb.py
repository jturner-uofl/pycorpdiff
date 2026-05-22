"""Tests for ``pycorpdiff.read_duckdb``."""

from __future__ import annotations

from pathlib import Path

import pytest

import pycorpdiff as pcd

duckdb = pytest.importorskip("duckdb")


@pytest.fixture
def in_memory_corpus_table() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        """
        CREATE TABLE docs AS
        SELECT * FROM (VALUES
            (1, 'the migrant worker arrived', 'humanising', 2020),
            (2, 'the migrant family settled', 'humanising', 2020),
            (3, 'the migrant criminal threat grew', 'criminalising', 2021),
            (4, 'the migrant invasion of gangs spread', 'criminalising', 2022)
        ) AS t(doc_id, text, frame, year);
        """
    )
    return con


def test_read_duckdb_basic_query(in_memory_corpus_table: duckdb.DuckDBPyConnection) -> None:
    corpus = pcd.read_duckdb(
        in_memory_corpus_table,
        "SELECT text, frame, year FROM docs",
        text_col="text",
        meta_cols=("frame", "year"),
    )
    assert len(corpus) == 4
    assert "frame" in corpus.docs.columns
    assert "year" in corpus.docs.columns


def test_read_duckdb_supports_filtering(
    in_memory_corpus_table: duckdb.DuckDBPyConnection,
) -> None:
    # DuckDB does the row filtering before the rows ever land in pandas.
    corpus = pcd.read_duckdb(
        in_memory_corpus_table,
        "SELECT text, frame FROM docs WHERE frame = 'humanising'",
    )
    assert len(corpus) == 2
    assert (corpus.docs["frame"] == "humanising").all()


def test_read_duckdb_with_named_id_column(
    in_memory_corpus_table: duckdb.DuckDBPyConnection,
) -> None:
    corpus = pcd.read_duckdb(
        in_memory_corpus_table,
        "SELECT doc_id, text FROM docs",
        text_col="text",
        id_col="doc_id",
    )
    assert corpus.id_col == "doc_id"
    assert corpus.docs["doc_id"].tolist() == [1, 2, 3, 4]


def test_read_duckdb_with_sql_params(
    in_memory_corpus_table: duckdb.DuckDBPyConnection,
) -> None:
    # Parameterised queries are the safe way to inject values from
    # user input. DuckDB accepts a positional `?` placeholder.
    corpus = pcd.read_duckdb(
        in_memory_corpus_table,
        "SELECT text, frame FROM docs WHERE year >= ?",
        params=[2021],
    )
    assert len(corpus) == 2
    assert (corpus.docs["frame"] == "criminalising").all()


def test_read_duckdb_missing_text_col_raises(
    in_memory_corpus_table: duckdb.DuckDBPyConnection,
) -> None:
    with pytest.raises(ValueError, match="text_col"):
        pcd.read_duckdb(
            in_memory_corpus_table,
            "SELECT frame FROM docs",  # no text column!
            text_col="text",
        )


def test_read_duckdb_round_trip_through_parquet(
    tmp_path: Path, in_memory_corpus_table: duckdb.DuckDBPyConnection
) -> None:
    """Exercise the path that motivates this reader: read_parquet inside SQL."""
    # First materialise the table to parquet using DuckDB itself.
    parquet_path = tmp_path / "news.parquet"
    in_memory_corpus_table.execute(
        f"COPY (SELECT * FROM docs) TO '{parquet_path}' (FORMAT PARQUET);"
    )
    # Now read from disk via a fresh connection.
    fresh = duckdb.connect()
    corpus = pcd.read_duckdb(
        fresh,
        f"SELECT text, frame, year FROM read_parquet('{parquet_path}')",
        meta_cols=("frame", "year"),
    )
    assert len(corpus) == 4
    fresh.close()


def test_read_duckdb_yields_full_corpus_api(
    in_memory_corpus_table: duckdb.DuckDBPyConnection,
) -> None:
    """The corpus that comes out behaves like any other — slicing,
    vocab, keyness all work end-to-end."""
    corpus = pcd.read_duckdb(
        in_memory_corpus_table,
        "SELECT text, frame FROM docs",
        meta_cols=("frame",),
    )
    a = corpus.slice(frame="humanising")
    b = corpus.slice(frame="criminalising")
    assert len(a) == 2
    assert len(b) == 2
    keyness = pcd.compare(a, b).keyness(min_count=1)
    assert isinstance(keyness, pcd.KeynessResult)
    assert len(keyness.table) > 0


def test_read_duckdb_friendly_error_when_extras_missing(
    monkeypatch: pytest.MonkeyPatch,
    in_memory_corpus_table: duckdb.DuckDBPyConnection,
) -> None:
    """If duckdb can't be imported, the function should point users at extras."""
    import sys

    monkeypatch.setitem(sys.modules, "duckdb", None)
    # Re-import the module so the patched sys.modules takes effect at the
    # function's import-line.
    import importlib

    from pycorpdiff.io import duckdb as duckdb_mod

    importlib.reload(duckdb_mod)
    with pytest.raises(ImportError, match="pycorpdiff\\[duckdb\\]"):
        duckdb_mod.read_duckdb(in_memory_corpus_table, "SELECT text FROM docs")
