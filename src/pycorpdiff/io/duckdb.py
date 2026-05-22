"""Out-of-core corpus querying via DuckDB.

DuckDB is in the optional ``duckdb`` extra. The reader is a thin
shim that runs a SQL query and projects the result into a pandas
DataFrame — DuckDB handles the heavy lifting (out-of-core scans of
parquet, CSV, Arrow tables, SQLite, S3-hosted files) before the data
ever touches pandas.

Use this when your corpus is too large to fit in pandas comfortably
but small enough that the rows you actually need fit after filtering.
"""

from __future__ import annotations

from typing import Any

from ..corpus import Corpus
from ..tokenize import Tokenizer


def read_duckdb(
    connection: Any,
    query: str,
    text_col: str = "text",
    id_col: str | None = None,
    meta_cols: tuple[str, ...] = (),
    tokenizer: Tokenizer | None = None,
    params: list[Any] | dict[str, Any] | None = None,
) -> Corpus:
    """Run a SQL query against a DuckDB connection and wrap as a :class:`Corpus`.

    Parameters
    ----------
    connection
        A :class:`duckdb.DuckDBPyConnection` (the object returned by
        ``duckdb.connect(...)``). Pass ``duckdb.connect()`` for an
        in-memory database, or ``duckdb.connect("path/to/file.duckdb")``
        for an on-disk one. DuckDB also accepts parquet / CSV / Arrow
        directly in SQL via ``read_parquet('path')``.
    query
        SQL that returns rows; must include the text column named by
        ``text_col``. Anything you can express in DuckDB SQL is fine —
        filters, joins, aggregates — the only requirement is that the
        final SELECT yields one row per document.
    text_col
        Name of the column containing document text. Default: ``"text"``.
    id_col
        Optional unique-document-id column.
    meta_cols
        Tuple of metadata column names to surface for slicing. If empty
        (the default), every non-text column becomes metadata.
    tokenizer
        Optional :class:`Tokenizer`. Defaults to :class:`RegexTokenizer`.
    params
        Optional positional or named SQL parameters; forwarded to
        :meth:`duckdb.DuckDBPyConnection.execute`.

    Returns
    -------
    Corpus
        Whose backing DataFrame is the result of the query.

    Examples
    --------
    >>> import duckdb, pycorpdiff as pcd
    >>> con = duckdb.connect()
    >>> corpus = pcd.read_duckdb(           # doctest: +SKIP
    ...     con,
    ...     "SELECT body AS text, outlet, year FROM read_parquet('news/*.parquet') "
    ...     "WHERE year >= 2020",
    ... )
    """
    try:
        import duckdb  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "read_duckdb requires duckdb. Install with: pip install 'pycorpdiff[duckdb]'"
        ) from exc

    cursor = connection.execute(query, params) if params is not None else connection.execute(query)
    df = cursor.df()
    if text_col not in df.columns:
        raise ValueError(
            f"text_col={text_col!r} not found in query result columns "
            f"{list(df.columns)!r}"
        )

    from .readers import from_dataframe

    return from_dataframe(
        df, text_col=text_col, id_col=id_col, meta_cols=meta_cols, tokenizer=tokenizer
    )
