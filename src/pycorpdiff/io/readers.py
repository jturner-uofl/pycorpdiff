"""Corpus readers — txt, csv, parquet, in-memory DataFrame."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..corpus import Corpus
from ..tokenize import RegexTokenizer, Tokenizer


def from_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    id_col: str | None = None,
    meta_cols: tuple[str, ...] = (),
    tokenizer: Tokenizer | None = None,
) -> Corpus:
    """Construct a :class:`Corpus` from an in-memory DataFrame."""
    return Corpus(
        docs=df.reset_index(drop=True),
        text_col=text_col,
        id_col=id_col,
        meta_cols=meta_cols,
        tokenizer=tokenizer if tokenizer is not None else RegexTokenizer(),
    )


def read_csv(
    path: str | Path,
    text_col: str = "text",
    id_col: str | None = None,
    meta_cols: tuple[str, ...] = (),
    tokenizer: Tokenizer | None = None,
    **read_csv_kwargs: Any,
) -> Corpus:
    """Read a CSV file into a :class:`Corpus`.

    Extra keyword arguments are forwarded to :func:`pandas.read_csv`.
    """
    df = pd.read_csv(path, **read_csv_kwargs)
    return from_dataframe(
        df, text_col=text_col, id_col=id_col, meta_cols=meta_cols, tokenizer=tokenizer
    )


def read_parquet(
    path: str | Path,
    text_col: str = "text",
    id_col: str | None = None,
    meta_cols: tuple[str, ...] = (),
    tokenizer: Tokenizer | None = None,
    **read_parquet_kwargs: Any,
) -> Corpus:
    """Read a parquet file (or directory of parquet files) into a :class:`Corpus`."""
    df = pd.read_parquet(path, **read_parquet_kwargs)
    return from_dataframe(
        df, text_col=text_col, id_col=id_col, meta_cols=meta_cols, tokenizer=tokenizer
    )


def read_txt(
    path: str | Path,
    encoding: str = "utf-8",
    one_doc_per: str = "file",
    tokenizer: Tokenizer | None = None,
) -> Corpus:
    """Read a single text file into a :class:`Corpus`.

    ``one_doc_per="file"`` treats the entire file as one document.
    ``one_doc_per="line"`` (Phase 1) will treat each non-empty line as a
    separate document.
    """
    if one_doc_per != "file":
        raise NotImplementedError(
            "read_txt(one_doc_per='line') lands in Phase 1; only 'file' is wired up"
        )
    text = Path(path).read_text(encoding=encoding)
    df = pd.DataFrame({"text": [text], "source": [str(path)]})
    return from_dataframe(df, text_col="text", tokenizer=tokenizer)
