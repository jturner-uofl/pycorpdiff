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

    Parameters
    ----------
    path
        Path to a UTF-8 text file (override via ``encoding``).
    one_doc_per
        ``"file"`` treats the whole file as one document. ``"line"``
        treats each non-empty line as its own document — useful for
        per-line corpora like JSONL exports already projected to text,
        or one-utterance-per-line transcripts.
    tokenizer
        Optional :class:`Tokenizer`. Defaults to :class:`RegexTokenizer`.

    Returns
    -------
    Corpus
        Has columns ``text``, ``source`` (the path), and — when
        ``one_doc_per="line"`` — an integer ``line`` column with the
        1-based line number so KWIC results can point back at the
        original file.
    """
    if one_doc_per not in ("file", "line"):
        raise ValueError(
            f"one_doc_per must be 'file' or 'line'; got {one_doc_per!r}"
        )
    text = Path(path).read_text(encoding=encoding)
    if one_doc_per == "file":
        df = pd.DataFrame({"text": [text], "source": [str(path)]})
    else:
        lines = text.splitlines()
        rows = [
            {"text": line, "source": str(path), "line": i + 1}
            for i, line in enumerate(lines)
            if line.strip()
        ]
        df = pd.DataFrame(rows, columns=["text", "source", "line"])
    return from_dataframe(df, text_col="text", tokenizer=tokenizer)
