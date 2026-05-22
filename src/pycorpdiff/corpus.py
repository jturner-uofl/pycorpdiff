"""Core ``Corpus`` and ``CorpusSlice`` data structures.

A :class:`Corpus` wraps a :class:`pandas.DataFrame` of documents plus
metadata. Slicing returns a :class:`CorpusSlice` that shares the parent's
configuration (text column, tokenizer, backend) but presents a
boolean-masked or filtered view. Both objects are immutable frozen
dataclasses; mutations produce new objects.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from .tokenize import RegexTokenizer, Tokenizer

if TYPE_CHECKING:
    from .temporal.slicing import TemporalCorpus


Backend = Literal["pandas", "polars"]


def _doc_term_counts(
    docs: pd.DataFrame,
    text_col: str,
    tokenizer: Tokenizer,
    min_count: int = 1,
) -> pd.DataFrame:
    """Build a docs × term integer count matrix.

    The result is dense (``int64``) and indexed by the parent frame's index.
    Sparse representations are deferred to a later phase — for the corpus
    sizes pycorpdiff targets in MVP scope (medium millions of tokens) dense
    is fast enough and downstream operations stay vectorisable.
    """
    counts_per_doc: list[Counter[str]] = [Counter(tokenizer(t)) for t in docs[text_col]]
    all_terms: list[str] = sorted({term for c in counts_per_doc for term in c})
    term_to_idx: dict[str, int] = {t: i for i, t in enumerate(all_terms)}

    data = np.zeros((len(counts_per_doc), len(all_terms)), dtype=np.int64)
    for i, doc_counts in enumerate(counts_per_doc):
        for term, n in doc_counts.items():
            data[i, term_to_idx[term]] = n

    df = pd.DataFrame(data, columns=all_terms, index=docs.index)
    if min_count > 1:
        df = df.loc[:, df.sum(axis=0) >= min_count]
    return df


@dataclass(frozen=True)
class Corpus:
    """A corpus of documents with optional metadata columns.

    Parameters
    ----------
    docs
        A DataFrame whose rows are documents. Must contain at least the
        text column named by ``text_col``.
    text_col
        Name of the column containing document text.
    id_col
        Name of an optional unique-document-id column.
    meta_cols
        Tuple of column names treated as metadata available for slicing.
        If empty (the default), every non-text column is considered
        metadata.
    tokenizer
        A callable conforming to :class:`pycorpdiff.tokenize.Tokenizer`.
        Defaults to the package's :class:`RegexTokenizer`.
    backend
        ``"pandas"`` (default) or ``"polars"``. Only pandas is wired up
        in the scaffolding release.
    """

    docs: pd.DataFrame
    text_col: str = "text"
    id_col: str | None = None
    meta_cols: tuple[str, ...] = ()
    tokenizer: Tokenizer = field(default_factory=RegexTokenizer)
    backend: Backend = "pandas"

    def __post_init__(self) -> None:
        if self.text_col not in self.docs.columns:
            raise ValueError(
                f"text_col={self.text_col!r} not found in DataFrame columns "
                f"{list(self.docs.columns)!r}"
            )
        if self.id_col is not None and self.id_col not in self.docs.columns:
            raise ValueError(
                f"id_col={self.id_col!r} not found in DataFrame columns "
                f"{list(self.docs.columns)!r}"
            )
        if self.backend not in ("pandas", "polars"):
            raise ValueError(f"backend must be 'pandas' or 'polars', got {self.backend!r}")
        if self.backend == "polars":
            # Polars backend is reserved for a later phase; surfacing it as
            # NotImplementedError now (rather than silently coercing) keeps
            # the contract honest.
            raise NotImplementedError("polars backend is not yet wired up")

    def __len__(self) -> int:
        return len(self.docs)

    @property
    def metadata_columns(self) -> tuple[str, ...]:
        """Effective metadata columns — explicit if given, else inferred."""
        if self.meta_cols:
            return self.meta_cols
        return tuple(c for c in self.docs.columns if c != self.text_col)

    def slice(self, **filters: Any) -> CorpusSlice:
        """Return a :class:`CorpusSlice` filtered on metadata columns.

        Each keyword argument is a column name; the value may be a scalar
        (exact match) or an iterable (membership). All conditions are
        combined with logical AND.
        """
        mask = pd.Series(True, index=self.docs.index)
        for col, value in filters.items():
            if col not in self.docs.columns:
                raise KeyError(f"slice() got unknown column {col!r}")
            if isinstance(value, (list, tuple, set, pd.Series)):
                mask &= self.docs[col].isin(list(value))
            else:
                mask &= self.docs[col] == value
        return CorpusSlice(parent=self, mask=mask, filters=dict(filters))

    def by_time(self, col: str, freq: str = "Y") -> TemporalCorpus:
        """Return a :class:`TemporalCorpus` indexed by time-period.

        ``col`` must be parseable as datetime; ``freq`` is any pandas
        offset alias (``"Y"``, ``"Q"``, ``"M"``, ``"W"``, ``"D"``).
        """
        from .temporal.slicing import TemporalCorpus  # local import to break cycle

        return TemporalCorpus(parent=self, time_col=col, freq=freq)

    def with_tokenizer(self, tokenizer: Tokenizer) -> Corpus:
        """Return a copy of the corpus with a different tokenizer."""
        return replace(self, tokenizer=tokenizer)

    def tokens(self) -> list[list[str]]:
        """Tokenize every document; return one list of tokens per doc."""
        return [self.tokenizer(t) for t in self.docs[self.text_col]]

    def doc_term_counts(self, min_count: int = 1) -> pd.DataFrame:
        """Return a docs × term integer count DataFrame."""
        return _doc_term_counts(self.docs, self.text_col, self.tokenizer, min_count)

    def vocab(self, min_count: int = 1) -> pd.Series:
        """Return a term → total-count Series sorted descending."""
        counts = self.doc_term_counts(min_count=min_count).sum(axis=0)
        return counts.rename("count").sort_values(ascending=False)

    def total_tokens(self) -> int:
        """Total tokens across all documents (before any min_count filter)."""
        return int(self.doc_term_counts(min_count=1).values.sum())


@dataclass(frozen=True)
class CorpusSlice:
    """A boolean-masked view of a :class:`Corpus`.

    Slices behave like corpora for downstream analytical purposes —
    they expose the same ``docs``, ``text_col``, ``tokenizer`` surface —
    but also remember the ``filters`` that produced them, which the
    :class:`pycorpdiff.compare.Comparison` machinery uses to label
    plots and result tables.
    """

    parent: Corpus
    mask: pd.Series
    filters: dict[str, Any]

    @property
    def docs(self) -> pd.DataFrame:
        return self.parent.docs.loc[self.mask]

    @property
    def text_col(self) -> str:
        return self.parent.text_col

    @property
    def id_col(self) -> str | None:
        return self.parent.id_col

    @property
    def tokenizer(self) -> Tokenizer:
        return self.parent.tokenizer

    def __len__(self) -> int:
        return int(self.mask.sum())

    @property
    def label(self) -> str:
        """A short human-readable label derived from the slice's filters."""
        if not self.filters:
            return "slice"
        return ", ".join(f"{k}={v!r}" for k, v in self.filters.items())

    def tokens(self) -> list[list[str]]:
        """Tokenize every document in the slice."""
        return [self.tokenizer(t) for t in self.docs[self.text_col]]

    def doc_term_counts(self, min_count: int = 1) -> pd.DataFrame:
        return _doc_term_counts(self.docs, self.text_col, self.tokenizer, min_count)

    def vocab(self, min_count: int = 1) -> pd.Series:
        counts = self.doc_term_counts(min_count=min_count).sum(axis=0)
        return counts.rename("count").sort_values(ascending=False)

    def total_tokens(self) -> int:
        return int(self.doc_term_counts(min_count=1).values.sum())
