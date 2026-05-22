"""Out-of-core corpus querying via DuckDB.

Importable only when the ``duckdb`` extra is installed.
"""

from __future__ import annotations

from ..corpus import Corpus


def read_duckdb(connection: object, query: str, text_col: str = "text") -> Corpus:
    """Run a SQL query against DuckDB and wrap the result as a Corpus."""
    raise NotImplementedError("read_duckdb() lands post-MVP")
