"""Corpus I/O — readers for txt, csv, parquet, DataFrame, DuckDB."""

from __future__ import annotations

from .duckdb import read_duckdb
from .readers import from_dataframe, read_csv, read_parquet, read_txt

__all__ = ["from_dataframe", "read_csv", "read_duckdb", "read_parquet", "read_txt"]
